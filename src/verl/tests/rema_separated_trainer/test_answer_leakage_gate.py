from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.rema_separated_trainer.ppo.prefix_probe import (
    answer_round_records,
    apply_prefix_probe_gate,
    collect_prefix_probe_requests,
    compare_worker_final_answers,
    count_planned_subtasks,
    extract_complete_boxed_answer,
    select_console_probe_indices,
    select_stratified_probe_indices,
)
from verl.rema_separated_trainer.ppo.scoped_c3_grpo import estimate_scoped_c3_grpo


@pytest.mark.parametrize("local, final, expected", [
    (r"\frac{1}{2}", "0.5", True),
    ("x+x", "2x", True),
    (r"\sqrt{4}", "2", True),
    ("6", "26", False),
    ("x+1", "y+1", False),
    (r"\text{UNKNOWN}", "26", None),
])
def test_math_equivalence(local, final, expected):
    assert compare_worker_final_answers(
        "REASONING: irrelevant\nLOCAL_RESULT: \\boxed{" + local + "}",
        "Final answer: \\boxed{" + final + "}",
    ) is expected


def test_only_local_result_is_compared_not_reasoning():
    assert compare_worker_final_answers(
        "REASONING: \\boxed{26}\nLOCAL_RESULT: \\boxed{6}",
        "\\boxed{26}",
    ) is False


@pytest.mark.parametrize("worker, final", [
    ("REASONING: \\boxed{26}", "\\boxed{26}"),
    ("LOCAL_RESULT: \\boxed{}", "\\boxed{26}"),
    ("LOCAL_RESULT: \\boxed{\\frac{1}{2}", "\\boxed{0.5}"),
    ("LOCAL_RESULT: \\boxed{6}", ""),
    ("LOCAL_RESULT: \\boxed{6}", "\\boxed{6} then \\boxed{"),
])
def test_missing_or_malformed_answers_are_unknown(worker, final):
    assert compare_worker_final_answers(worker, final) is None


def test_symbolic_errors_are_not_evidence_of_different_answers(monkeypatch):
    import math_verify.grader

    def fail(*args, **kwargs):
        raise ValueError("cannot compare")

    monkeypatch.setattr(math_verify.grader, "sympy_expr_eq", fail)
    assert compare_worker_final_answers("LOCAL_RESULT: \\boxed{6}", "\\boxed{26}") is None


def test_parser_failure_is_unknown_even_when_box_is_balanced(monkeypatch):
    import math_verify

    monkeypatch.setattr(math_verify, "parse", lambda *args, **kwargs: [])
    assert compare_worker_final_answers("LOCAL_RESULT: \\boxed{6}", "\\boxed{26}") is None


def test_comparison_timeout_is_unknown(monkeypatch):
    import math_verify.grader
    from math_verify.utils import TimeoutException

    def fail(*args, **kwargs):
        raise TimeoutException("timeout")

    monkeypatch.setattr(math_verify.grader, "sympy_expr_eq", fail)
    assert compare_worker_final_answers("LOCAL_RESULT: \\boxed{6}", "\\boxed{26}") is None


@pytest.mark.parametrize("ld,count,match,required,reason", [
    (0, 2, 0, True, "eligible"),
    (0, 2, 1, True, "equivalent_answer"),
    (0, 2, float("nan"), True, "comparison_invalid"),
    (0, 1, 0, True, "single_subtask"),
    (1, 3, 0, True, "plan_recoverable"),
    (float("nan"), 3, 0, True, "plan_probe_invalid"),
    (0, 2, float("nan"), False, "eligible"),
    (1, 2, float("nan"), False, "plan_recoverable"),
    (0, 1, float("nan"), False, "single_subtask"),
])
def test_plan_and_action_gate(ld, count, match, required, reason):
    gate = apply_prefix_probe_gate(
        [1., 0.], [ld] * 2, [count] * 2, [match] * 2, [required] * 2,
    )
    allowed = reason == "eligible"
    assert gate.rejection_reasons == [reason, reason]
    assert gate.collaboration_eligible_mask == [allowed, allowed]
    assert gate.outcome_scores == ([1., 0.] if allowed else [0., 0.])


def test_masks_remove_both_signs_without_changing_baseline():
    raw = torch.tensor([1., 1., 0., 0.])
    gate = apply_prefix_probe_gate(
        raw.tolist(), [0.] * 4, [3] * 4, [1., 0., 1., 0.], [True] * 4,
    )
    estimated = estimate_scoped_c3_grpo(
        raw, ["q"] * 4, torch.ones(4, dtype=torch.bool),
        update_mask=torch.tensor(gate.collaboration_eligible_mask),
        normalize=False,
    )
    torch.testing.assert_close(
        estimated.advantage, torch.tensor([2/3, 2/3, -2/3, -2/3]),
    )
    assert estimated.effective_mask.tolist() == [False, True, False, True]


def _record(role, content, task=None, **extra):
    return dict(
        role=role, content=content, executed=True,
        assigned_subtasks=[task] if task else [], **extra,
    )


def _history(local="6", final="26", count=2):
    return [
        _record("decomposer", "PLAN:\n- S1: compute a\n- S2: finish", planned_subtask_count=count),
        _record("worker_stage_1", "LOCAL_RESULT: \\boxed{" + local + "}", "S1"),
        _record("worker_stage_2", "LOCAL_RESULT: \\boxed{" + final + "}", "S2"),
        dict(role="worker_stage_3", content="", executed=False),
    ]


def test_only_plan_is_probed_and_terminal_is_dynamic():
    history = _history()
    requests = collect_prefix_probe_requests(
        [history], ["worker_stage_2"], focal_role="worker_stage_1",
        decomposer_role="decomposer", stage_roles=["worker_stage_1", "worker_stage_2"],
    )
    assert len(requests) == 1
    assert requests[0].source_kind == "decomposer"
    assert requests[0].message == history[0]["content"]


def test_accepted_round_uses_original_plan_and_results():
    original = _history()
    history = original + [
        _record("decomposer", "DECISION: ACCEPT", planned_subtask_count=1),
        dict(role="worker_stage_2", content="copied", executed=False, carried_forward=True),
    ]
    records = answer_round_records(history, "worker_stage_2", "decomposer")
    assert records == original[:3]
    assert count_planned_subtasks(records, "decomposer") == 2


def test_router_count_takes_precedence_over_output_task_labels():
    history = _history(count=1)
    assert count_planned_subtasks(history, "decomposer") == 1


def test_box_extraction_and_validation_sampling():
    assert extract_complete_boxed_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert select_stratified_probe_indices(["a", "a", "a", "b", "c"], 3) == [0, 3, 4]
    assert select_console_probe_indices(["a", "a", "b"], 1) == [0, 2]


def _object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _batch(histories):
    size = len(histories)
    return DataProto.from_dict(
        tensors={"batch_idx": torch.arange(size)},
        non_tensors={
            "history": _object_array(histories),
            "terminal_stage_role": _object_array(["worker_stage_2"] * size),
            "question": _object_array(["hidden original Q"] * size),
            "data_source": _object_array(["math"] * size),
            "reward_model": _object_array([{"ground_truth": "26"}] * size),
            "uid": _object_array(["q"] * size),
        },
    )


def _trainer(focal_role="worker_stage_1", plan_response=r"\boxed{UNKNOWN}"):
    from verl.rema_separated_trainer.ppo.ray_trainer import RayReMASeparatedTrainer

    trainer = RayReMASeparatedTrainer.__new__(RayReMASeparatedTrainer)
    trainer.prefix_probe_enabled = True
    trainer.scoped_c3_grpo_enabled = True
    trainer.prefix_probe_config = {"diagnostic_samples": 0}
    trainer.scoped_c3_grpo_config = {"branch_turn": 0, "normalize_advantages": False}
    trainer._current_train_agent = focal_role
    trainer.global_steps = 1
    trainer._get_hierarchy_config = lambda: {
        "decomposer_role": "decomposer", "score_role": "worker_stage_3",
        "stage_roles": ["worker_stage_1", "worker_stage_2", "worker_stage_3"],
    }
    trainer.reward_fn = SimpleNamespace(
        num_examine=0, score_responses=lambda *args, **kwargs: [1.] * len(args[0]),
    )
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(max_num_turns=1)),
    )
    trainer.probe_calls = []

    def probe(messages):
        trainer.probe_calls.append(messages)
        assert all("hidden original Q" not in message for message in messages)
        return [plan_response] * len(messages), [3] * len(messages), ["stop"] * len(messages)

    trainer._generate_prefix_probe_responses = probe
    return trainer


def test_trainer_probes_plan_once_and_preserves_object_columns_on_concat():
    trainer = _trainer()
    batch = _batch([_history("26"), _history("6"), _history("9", "9")])
    metrics = {}
    trainer._attach_prefix_probe_signals(batch, torch.tensor([1., 1., 0.]), metrics)
    assert len(trainer.probe_calls) == 1
    assert len(trainer.probe_calls[0]) == 1
    assert batch.batch["prefix_probe_collaboration_eligible"].tolist() == [False, True, False]
    assert batch.non_tensor_batch["prefix_probe_worker_match_score"].tolist() == [1., 0., 1.]
    combined = DataProto.concat([batch[:1], batch[1:]])
    assert len(combined) == 3
    assert all(column.dtype == object for column in combined.non_tensor_batch.values())
    assert metrics["reward/leakage/all/raw_accuracy"] == pytest.approx(2/3)
    assert metrics["reward/leakage/all/gated_accuracy"] == pytest.approx(1/3)


@pytest.mark.parametrize("plan_response,count,allowed", [
    (r"\boxed{UNKNOWN}", 2, True),
    (r"\boxed{26}", 2, False),
    ("malformed", 2, False),
    (r"\boxed{UNKNOWN}", 1, False),
])
def test_terminal_exempt_from_comparison_but_not_plan_gate(plan_response, count, allowed):
    trainer = _trainer("worker_stage_2", plan_response)
    batch = _batch([_history(count=count)])
    trainer._attach_prefix_probe_signals(batch, torch.tensor([1.]), {})
    assert batch.non_tensor_batch["prefix_probe_worker_comparisons"].tolist() == [[]]
    assert batch.batch["prefix_probe_collaboration_eligible"].tolist() == [allowed]


def test_validation_checks_all_nonterminal_results_and_logs_example(capsys):
    trainer = _trainer()
    batch = _batch([_history("26"), _history("6")])
    trainer._attach_prefix_probe_signals(batch, torch.ones(2), {}, validation=True)
    trainer._print_validation_leakage_examples(batch)
    assert batch.batch["prefix_probe_gated_outcome_score"].tolist() == [0., 1.]
    output = capsys.readouterr().out
    assert "E=[worker_stage_1:1]" in output
    assert "reason=equivalent_answer" in output
    assert "worker_before" not in output


def test_c3_trainer_keeps_rejected_and_unparseable_actions_as_baseline():
    trainer = _trainer()
    batch = _batch([_history("26"), _history("6"), _history()])
    batch.non_tensor_batch["history"][2][1]["content"] = "no LOCAL_RESULT"
    raw = torch.tensor([1., 0., 0.])
    trainer._attach_prefix_probe_signals(batch, raw, {})
    role = "worker_stage_1"
    batch.non_tensor_batch[f"{role}_action_token_ids"] = _object_array([[2]] * 3)
    batch.non_tensor_batch[f"{role}_conversation_history"] = _object_array([
        [{"role": "user", "content": "shared"}, {"role": "assistant", "content": str(i)}]
        for i in range(3)
    ])
    batch.non_tensor_batch["c3_action_turn"] = _object_array([0] * 3)
    trainer._attach_scoped_c3_grpo_signals(batch, {"acc": raw}, {})
    assert batch.batch["scoped_c3_causal_valid"].tolist() == [True] * 3
    assert batch.batch["scoped_c3_update_mask"].tolist() == [False, True, False]
    for key in ("labels", "step_ids"):
        batch.batch[key] = torch.zeros(3, 2, dtype=torch.long)
    batch.batch["token_level_rewards"] = torch.zeros(3, 2)
    metrics = {}
    trainer._compute_scoped_c3_grpo_advantage(batch, metrics)
    torch.testing.assert_close(batch.batch["advantages"], torch.tensor([[0., 0.], [-.5, -.5], [0., 0.]]))
    assert batch.batch["labels"].tolist() == [[-100, -100], [0, 0], [-100, -100]]
    assert metrics[f"reward/c3/roles/{role}/positive_removed_count"] == 1
    assert metrics[f"reward/c3/roles/{role}/negative_removed_count"] == 1


def test_invalid_math_plan_probe_does_not_become_clean_plan(monkeypatch):
    from verl.rema_separated_trainer.ppo import ray_trainer

    trainer = _trainer(plan_response=r"\boxed{unparseable}")
    monkeypatch.setattr(ray_trainer, "parse_boxed_math_answer", lambda text: None)
    batch = _batch([_history()])
    trainer._attach_prefix_probe_signals(batch, torch.ones(1), {})
    assert batch.non_tensor_batch["prefix_probe_rejection_reason"].tolist() == ["plan_probe_invalid"]


def test_plan_scorer_timeout_is_not_a_clean_measurement():
    trainer = _trainer(plan_response=r"\boxed{26}")

    def timeout(*args, **kwargs):
        assert np.isnan(kwargs["timeout_score"])
        return [kwargs["timeout_score"]]

    trainer.reward_fn.score_responses = timeout
    batch = _batch([_history()])
    trainer._attach_prefix_probe_signals(batch, torch.ones(1), {})
    assert batch.non_tensor_batch["prefix_probe_rejection_reason"].tolist() == ["plan_probe_invalid"]
    assert batch.batch["prefix_probe_collaboration_eligible"].tolist() == [False]


@pytest.mark.parametrize("exception_type", [TimeoutError, "math_verify"])
def test_response_scorer_preserves_timeout_status_for_probes(monkeypatch, exception_type):
    from math_verify.utils import TimeoutException
    from verl.workers.reward_manager import rema

    error = TimeoutException if exception_type == "math_verify" else exception_type

    class Results:
        def __init__(self):
            self.position = 0

        def __next__(self):
            self.position += 1
            if self.position == 1:
                raise error("test timeout")
            if self.position == 2:
                return 1.0
            raise StopIteration

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, *args, **kwargs):
            return SimpleNamespace(result=Results)

    monkeypatch.setattr(rema, "ProcessPool", Pool)
    manager = rema.ReMARewardManager.__new__(rema.ReMARewardManager)
    manager.compute_score = lambda *args: 1.0
    args = (["math"] * 2, [r"\boxed{26}"] * 2, ["26"] * 2)
    assert manager.score_responses(*args) == [0.0, 1.0]
    probe_scores = manager.score_responses(*args, timeout_score=float("nan"))
    assert np.isnan(probe_scores[0]) and probe_scores[1] == 1.0


def test_chunk_metrics_are_weighted_and_replay_contains_new_gates(tmp_path):
    import json

    trainer = _trainer()
    metrics = {}
    first = _batch([_history("26")])
    second = _batch([_history("6"), _history("6")])
    trainer._attach_prefix_probe_signals(first, torch.ones(1), metrics)
    trainer._attach_prefix_probe_signals(second, torch.ones(2), metrics)
    assert metrics["reward/leakage/all/gated_accuracy"] == pytest.approx(2/3)
    assert metrics["reward/leakage/all/worker_match/rate"] == pytest.approx(1/3)
    combined = DataProto.concat([first, second])
    combined.non_tensor_batch["response"] = _object_array([r"\boxed{26}"] * 3)
    combined.non_tensor_batch["finish_reason"] = _object_array(["final_boxed_answer"] * 3)
    combined.batch["worker_stage_2_turn_level_reward"] = torch.ones(3, 1)
    combined.batch["scoped_c3_raw_outcome_score"] = torch.ones(3)
    trainer.config.trainer = SimpleNamespace(default_local_dir=str(tmp_path))
    trainer._get_score_role = lambda: "worker_stage_2"
    trainer._get_rollout_agent_roles = lambda: ["worker_stage_1", "worker_stage_2"]
    trainer._save_train_generations(combined)
    saved = json.loads((tmp_path / "replay_buffer" / "train_step_1.jsonl").read_text())
    assert saved["prefix_probe_rejection_reason"] == ["equivalent_answer", "eligible", "eligible"]
    assert saved["gated_outcome_score"] == [0., 1., 1.]
    assert saved["raw_outcome_score"] == [1., 1., 1.]
    assert "prefix_probe_worker_before_score" not in saved

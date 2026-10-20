"""Accuracy cadence and reward-free validation, using mock generation."""
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

try:
    import verl.hierarchical_rema as api
except ModuleNotFoundError:
    import hierarchical_rema as api

train = importlib.import_module(api.__name__ + ".train")
validation = importlib.import_module(api.__name__ + ".validation")
gfam = importlib.import_module(api.__name__ + ".gfam_reward")
demo = importlib.import_module(api.__name__ + ".demo")


def settings(**changes):
    values = dict(validation_mode="auto", val_diagnostics_every=4,
                  val_num_decompositions=1, val_num_executor_rollouts=1, val_num_selections=1,
                  num_epochs=250, external_validation_every_n_epochs=6)
    return SimpleNamespace(**dict(values, **changes))


def test_modes_preserve_accuracy_events_and_final_diagnostics():
    args = settings()
    assert [validation.validation_mode(args, e) for e in (6, 12, 18, 24, 30, 48, 250)] == [
        "routine", "routine", "routine", "diagnostic", "routine", "diagnostic", "diagnostic"]
    assert validation.validation_mode(settings(val_diagnostics_every=0), 24) == "routine"
    assert validation.validation_mode(settings(val_diagnostics_every=0), 250) == "diagnostic"
    assert validation.validation_mode(settings(external_validation_every_n_epochs=0), 250) == "diagnostic"


def test_multiple_candidates_never_silently_disable_reward_scoring():
    assert validation.validation_mode(settings(val_num_decompositions=2), 6) == "diagnostic"
    assert validation.validation_mode(settings(val_num_executor_rollouts=4), 6) == "diagnostic"
    with pytest.raises(ValueError, match="one plan and one execution"):
        validation.validation_mode(settings(validation_mode="routine", val_num_decompositions=2), 6)
    with pytest.raises(ValueError, match="nonnegative"):
        validation.validation_mode(settings(val_diagnostics_every=-1), 6)


class Tracker:
    def __init__(self):
        self.events = []
    def log(self, values, step):
        self.events.append(values)


def args_for_run(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--backend", "mock", "--num-epochs", "250",
        "--val-num-decompositions", "1", "--val-num-executor-rollouts", "1",
        "--gfam-reward-model-pkl", "not-loaded-in-routine.pkl", "--output-dir", "unused"])
    return train.parse_args()


def run_validation(args, tmp_path, tracking, epoch=6):
    return train.run_external_validation(args=args, epoch_number=epoch,
        val_tasks=[api.TaskExample(task_id="v1", prompt="Solve for x: 2x + 3 = 11.", ground_truth="4",
                                  metadata={"skill_focus": "algebra", "subset": "math"})],
        worker_pool=demo.make_worker_pool(base_model_path="mock"),
        policy_config=api.ControllerPolicyConfig(shared_model_path="mock"),
        output_dir=tmp_path, tracking=tracking, tracking_step=17,
        base_rollout_config=api.RolloutConfig(num_decompositions=1))


def test_routine_never_constructs_scorer_and_preserves_accuracy(monkeypatch, tmp_path):
    args = args_for_run(monkeypatch)
    args.validation_mode = "routine"
    monkeypatch.setattr(gfam, "load_cached_reward_scorer", lambda **kw:
                        pytest.fail("Routine validation must not load/extract Graph+PRM"))
    tracker = Tracker()
    result = run_validation(args, tmp_path / "routine", tracker)
    assert result["mean_best_final_correctness"] == 1
    assert result["verification_coverage"] == 1
    assert result["mean_best_selection_reward"] is None
    assert result["subsets"]["math"]["mean_best_selection_reward"] is None
    assert not result["reward_diagnostics_enabled"]
    assert result["diagnostics"]["worker_outputs"] > 0
    assert result["seconds_per_task"] >= 0
    assert tracker.events[-1] == {"val/accuracy": 1, "val/accuracy/math": 1}
    assert (tmp_path / "validation_accuracy.csv").exists()
    assert not (tmp_path / "routine" / "validation_diagnostics.jsonl").exists()


@pytest.mark.parametrize("overflow", [False, True])
def test_diagnostics_score_same_generation_and_keep_accuracy_on_overflow(monkeypatch, tmp_path, overflow):
    pilot = importlib.import_module("test_graphprm_pilot")
    args = args_for_run(monkeypatch)
    args.validation_mode = "diagnostic"
    scorer = pilot.Scorer(failures={1} if overflow else ())
    monkeypatch.setattr(gfam, "load_cached_reward_scorer", lambda **kw: scorer)
    tracker = Tracker()
    result = run_validation(args, tmp_path / "diagnostic", tracker)
    assert len(scorer.calls) == 1
    assert result["mean_best_final_correctness"] == 1
    assert result["verification_coverage"] == 1
    assert result["reward_diagnostics_enabled"]
    assert tracker.events[-1] == {"val/accuracy": 1, "val/accuracy/math": 1}
    examples = [json.loads(line) for line in (tmp_path / "diagnostic" / "validation_diagnostics.jsonl").read_text().splitlines()]
    assert len(examples) == 1
    assert examples[0]["task"]["ground_truth"] == "4"
    assert examples[0]["decomposition"]["nodes"]
    assert examples[0]["execution"]["executions"]
    if not overflow:
        assert result["diagnostics"]["reward_outcome_disagreement_rate"] == 1
        assert examples[0]["execution"]["reward_model_outputs"]["compiled_rewards"]


def test_audit_uses_real_gate_fields_and_does_not_treat_unverified_as_wrong():
    node = {"reward_before_outcome_gate": .8, "reward": .2}
    output = {"compiled_rewards": {"workers": {"A": node}, "final": node},
              "graph_summary": {"predicted_success": .9, "verified_failure_gate_applied": True}}
    executions = [SimpleNamespace(node_id=name, output_text="3", raw_output_text="3", invalid_reason="")
                  for name in ("A", "B")]
    selection = SimpleNamespace(final_answer="3", executions=executions, reward_model_outputs=output,
                                reward=SimpleNamespace(final_answer_correctness=0))
    plan = SimpleNamespace(decomposition=SimpleNamespace(raw_payload={"validation": {
        "attempt": 1, "errors_before_success": ["bad format"], "fallback_used": True}},
        raw_text="plan", nodes=[], was_hard_truncated=False, final_node_id="B"), selections=[selection])
    audit = validation.ValidationAudit()
    audit.add(SimpleNamespace(decompositions=[plan]), "math")
    report = audit.summary()
    assert report["plan_parse_error_rate"] == 1
    assert report["plan_fallback_rate"] == 1
    assert report["incorrect_mean_positive_credit_before_gate"] == .8
    assert report["incorrect_mean_positive_credit_after_gate"] == .2
    assert report["incorrect_execution_nodes_with_gate_data"] == 2
    assert report["reward_outcome_disagreement_rate"] == 1
    selection.reward.final_answer_correctness = None
    audit.add(SimpleNamespace(decompositions=[plan]))
    assert audit.summary()["outcome_prediction_pairs"] == 1
    assert audit.summary()["scored_incorrect_trajectories"] == 1


def test_routine_unverified_answer_is_not_reported_as_incorrect(monkeypatch, tmp_path):
    import math
    orchestration = importlib.import_module(api.__name__ + ".orchestrator")
    original = orchestration.build_selection_reward
    def unverified(**kwargs):
        result = original(**kwargs)
        result.final_answer_correctness = None
        result.total_reward = None
        result.verification_status = "unverified"
        result.verification_error = {"code": "timeout"}
        return result
    monkeypatch.setattr(orchestration, "build_selection_reward", unverified)
    monkeypatch.setattr(gfam, "load_cached_reward_scorer", lambda **kw: pytest.fail("Unexpected scorer"))
    args = args_for_run(monkeypatch)
    args.validation_mode = "routine"
    result = run_validation(args, tmp_path / "unknown", Tracker())
    assert result["num_verified_tasks"] == 0
    assert result["num_unverified_tasks"] == 1
    assert result["verification_coverage"] == 0
    assert math.isnan(result["mean_best_final_correctness"])
    assert train._validation_primary_metric(result) is None


@pytest.mark.parametrize("setting", ["gnn_prm", "llm_coach", "handcrafted"])
def test_routine_never_constructs_any_reward_provider(monkeypatch, tmp_path, setting):
    providers = importlib.import_module(api.__name__ + ".reward_providers")
    monkeypatch.setattr(providers, "build_reward_provider", lambda *a: pytest.fail("Routine must skip ALL providers"))
    args = args_for_run(monkeypatch)
    args.validation_mode, args.reward_setting = "routine", setting
    result = run_validation(args, tmp_path / setting, Tracker())
    assert result["mean_best_final_correctness"] == 1
    assert not result["reward_diagnostics_enabled"]


def test_handcrafted_diagnostics_do_not_load_graphprm(monkeypatch, tmp_path):
    monkeypatch.setattr(gfam, "load_cached_reward_scorer", lambda **kw: pytest.fail("No Graph+PRM for handcrafted"))
    args = args_for_run(monkeypatch)
    args.validation_mode, args.reward_setting = "diagnostic", "handcrafted"
    result = run_validation(args, tmp_path / "rules", Tracker())
    assert result["reward_diagnostics_enabled"]
    examples = [json.loads(line) for line in (tmp_path / "rules/validation_diagnostics.jsonl").read_text().splitlines()]
    output = examples[0]["execution"]["reward_model_outputs"]
    assert output["source"] == "handcrafted_v1"
    assert output["reward_evidence"]

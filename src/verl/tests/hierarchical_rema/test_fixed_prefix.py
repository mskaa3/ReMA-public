"""Fixed-prefix sampling and action-level credit, tested without model downloads."""
from collections import Counter
from copy import deepcopy
import importlib
import json
from types import SimpleNamespace

import pytest

try:
    import verl.hierarchical_rema as api
except ModuleNotFoundError:
    import hierarchical_rema as api

branch = importlib.import_module(api.__name__ + ".fixed_prefix")
orchestration = importlib.import_module(api.__name__ + ".orchestrator")
recording = importlib.import_module(api.__name__ + ".recording")


def plan(dag=False, single=False):
    dependencies = {"A": [], "B": ["A"], "C": ["B"], "D": ["C"]}
    if dag:
        dependencies = {"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]}
    if single:
        dependencies = {"D": []}
    return api.DecompositionCandidate(
        decomposition_id="plan", summary="A plan", target_quantity="answer",
        final_answer_format_hint="number", final_node_id="D",
        raw_payload={"controller_prompt": "Plan this task"},
        nodes=[api.SubtaskNode(node_id=node, instruction=f"Compute {node}",
                              dependencies=deps, output_key=node)
               for node, deps in dependencies.items()],
    )


class Backend(api.MockHierarchicalBackend):
    def __init__(self, template):
        self.template, self.requests, self.counter = template, [], Counter()
        self.config = SimpleNamespace(worker_batch_size=3)

    def sample_decompositions_batch(self, requests):
        results = []
        for request in requests:
            result = deepcopy(self.template)
            result.decomposition_id = f"plan-{request.decomposition_index}"
            results.append(result)
        return results

    def execute_workers_batch(self, requests):
        results = []
        for request in requests:
            self.requests.append(deepcopy(request))
            self.counter[request.node.node_id] += 1
            output = f"{request.node.node_id}_{self.counter[request.node.node_id]}"
            results.append(api.WorkerExecution(
                node_id=request.node.node_id, worker_id=request.worker.worker_id,
                output_text=output, raw_output_text=f"<worker_result>{output}</worker_result>",
                worker_prompt=f"{request.task.prompt}: {request.node.instruction} "
                              + json.dumps(request.dependency_outputs, sort_keys=True),
                dependency_outputs=dict(request.dependency_outputs),
                entropy=0.0, confidence_reward=0.0, compatibility=1.0,
            ))
        return results


class Scorer:
    source = "gnn_prm"
    rollout_batch_size = 5
    supports_verified_final_correctness = True

    def __init__(self, missing=None, constant=False):
        self.missing, self.constant, self.requests = missing, constant, []

    def score_rollouts(self, requests):
        return [self.score_rollout(**request) for request in requests]

    def score_rollout(self, **request):
        self.requests.append(request)
        selection = request["selection"]
        action, continuation = (0, 0)
        if ":a" in selection.selection_id:
            action, continuation = [int(part[1:]) for part in selection.selection_id.split(":")[-2:]]
        if self.missing == (action, continuation):
            return {"status": "unscored", "error": {"type": "test_overflow"}}
        value = 0.25 if self.constant else action / 10 + continuation / 100
        return {
            "status": "scored",
            "compiled_rewards": {
                "decomposer": {"reward": 0.3}, "final": {"reward": 0.9},
                "workers": {ex.node_id: {"reward": value} for ex in request["executions"]},
            },
        }


def run(monkeypatch, *, focal="B", points=1, dag=False, single=False, mode="fixed_prefix",
        phase=api.AlternatingPhase.EXECUTOR, joint=False, rule="both_signs", missing=None,
        constant=False, actions=8, continuations=4, tasks=1, plans=1, force_focal=True,
        checkpoint="checkpoint", overlap=False):
    monkeypatch.setenv("GFAM_OVERLAP_REWARDS", "true" if overlap else "false")
    if force_focal:
        order = plan(dag, single).topological_order()
        position = order.index(focal)
        monkeypatch.setattr(branch.random.Random, "sample",
                            lambda self, population, k: [(position + i) % len(order) for i in range(k)])
    monkeypatch.setattr(orchestration, "build_selection_reward", lambda **kwargs:
        api.SelectionRewardBreakdown(final_answer_correctness=1.0, confidence_reward=0.0,
                                     compatibility_reward=0.0, total_reward=1.0))
    backend, scorer = Backend(plan(dag, single)), Scorer(missing, constant)
    trainer = api.HierarchicalGRPOTrainer(
        backend=backend, reward_provider=scorer, train_worker_model=True,
        min_worker_grpo_group_size=2, track_workers_history=False,
    )
    pool = api.WorkerPoolConfig(base_model_path=checkpoint, workers=[api.WorkerSpec(
        worker_id="executor", description="math", skills=[], system_prompt="Execute.",
        base_model_path=checkpoint)])
    config = api.RolloutConfig(
        num_decompositions=plans, num_executor_rollouts_per_decomposition=3,
        max_nodes_per_decomposition=4, worker_rollout_mode=mode,
        worker_action_samples=actions, worker_continuations_per_action=continuations,
        worker_branch_points_per_plan=points, worker_update_rule=rule,
    )
    schedule = api.TrainingScheduleConfig(
        mode=api.TrainingMode.JOINT if joint else api.TrainingMode.ALTERNATING,
        alternating_phase=phase)
    results = trainer.orchestrator.run_tasks(
        [api.TaskExample(task_id=f"task-{i}", prompt="Solve.", ground_truth="4")
         for i in range(tasks)], pool, api.ControllerPolicyConfig(shared_model_path="checkpoint"),
        config, schedule, progress_label="epoch=1 batch=1",
    )
    return results, backend, scorer, trainer, pool, config


def test_eight_action_means_and_sample_std_match_reference():
    baseline, delta, advantage = branch.action_advantages([1, .75, .5, .25, 0, 0, 0, 0])
    assert baseline[1] == pytest.approx(.25)
    assert delta[1] == pytest.approx(.5)
    assert advantage == pytest.approx(
        [1.739, 1.107, .474, -.158, -.791, -.791, -.791, -.791], abs=.001)
    assert branch.action_advantages([.5] * 8)[2] == [0] * 8
    with pytest.raises(ValueError):
        branch.action_advantages([0, float("nan")])


@pytest.mark.parametrize("focal,expected", [
    ("A", {"A": 8, "B": 32, "C": 32, "D": 32}),
    ("B", {"A": 1, "B": 8, "C": 32, "D": 32}),
    ("C", {"A": 1, "B": 1, "C": 8, "D": 32}),
    ("D", {"A": 1, "B": 1, "C": 1, "D": 8}),
])
def test_fresh_prefix_and_single_action_copy(monkeypatch, focal, expected):
    results, backend, scorer, *_ = run(monkeypatch, focal=focal)
    result = results[0]
    samples = result.training_batch.worker_samples
    assert backend.counter == expected
    assert len(samples) == 8
    assert {sample.metadata["node_id"] for sample in samples} == {focal}
    assert len({sample.prompt_text for sample in samples}) == 1
    assert len({sample.completion_text for sample in samples}) == 8
    assert result.training_batch.decomposer_samples == []
    assert result.decompositions[0].decomposition_reward is None
    repeats = 1 if focal == "D" else 4
    assert len(scorer.requests) == len(result.decompositions[0].selections) == 8 * repeats
    assert [sample.reward for sample in samples] == pytest.approx(
        [i / 10 + (.015 if repeats == 4 else 0) for i in range(8)])
    assert all(sample.reward != .9 for sample in samples)  # Not the final reward.
    graphs = [request["executions"] for request in scorer.requests]
    assert all(len(graph) == 4 for graph in graphs)
    assert len({id(ex) for graph in graphs for ex in graph}) == len(graphs) * 4
    assert {sample.metadata["advantage_group_size"] for sample in samples} == {8}


def test_dag_join_receives_its_own_branch_outputs(monkeypatch):
    results, _, scorer, *_ = run(monkeypatch, dag=True, focal="B")
    for request in scorer.requests:
        outputs = {ex.node_id: ex.output_text for ex in request["executions"]}
        final = next(ex for ex in request["executions"] if ex.node_id == "D")
        assert final.dependency_outputs == {"B": outputs["B"], "C": outputs["C"]}
    assert len(results[0].training_batch.worker_samples) == 8


def test_terminal_atomic_plan_is_scored_once_per_action(monkeypatch):
    result, backend, scorer, *_ = run(monkeypatch, focal="D", single=True)
    assert backend.counter == {"D": 8}
    assert len(scorer.requests) == 8
    assert len(result[0].training_batch.worker_samples) == 8


def test_joint_planner_uses_only_independent_repeats(monkeypatch):
    result, _, _, *_ = run(monkeypatch, joint=True)
    dec = result[0].decompositions[0]
    assert len(dec.planner_selections) == 3
    assert len(dec.selections) == 3 + 32
    assert dec.decomposition_reward == pytest.approx(.3)
    assert len(result[0].training_batch.worker_samples) == 8
    assert len(result[0].training_batch.decomposer_samples) == 1


def test_planner_phase_never_branches(monkeypatch):
    results, backend, scorer, *_ = run(monkeypatch, phase=api.AlternatingPhase.DECOMPOSER)
    assert backend.counter == dict.fromkeys("ABCD", 3)
    assert len(scorer.requests) == 3
    assert results[0].training_batch.worker_samples == []
    assert results[0].decompositions[0].worker_comparisons == []


@pytest.mark.parametrize("focal,expected", [("B", 4), ("D", 8)])
def test_positive_only_is_after_baseline_and_terminal_keeps_negatives(monkeypatch, focal, expected):
    results, *_ = run(monkeypatch, focal=focal, rule="positive_only_nonterminal")
    samples = results[0].training_batch.worker_samples
    assert len(samples) == expected
    audit = results[0].decompositions[0].worker_comparisons[0]
    assert len(audit["actions"]) == 8
    assert audit["actions"][0]["advantage"] < 0
    assert all(record["score"] is not None for record in audit["actions"])
    if focal != "D":
        assert all(sample.advantage > 0 for sample in samples)


def test_overflow_skips_complete_comparison_not_just_one_suffix(monkeypatch):
    results, *_ = run(monkeypatch, missing=(3, 2))
    assert results[0].training_batch.worker_samples == []
    audit = results[0].decompositions[0].worker_comparisons[0]
    assert audit["reason"] == "incomplete_group"
    assert audit["actions"][3]["score"] is None
    assert len(results[0].decompositions[0].selections) == 32
    assert audit["incomplete_causes"] == ["reward_model:test_overflow"]
    stats = results[0].training_batch.worker_grpo_stats
    assert stats["incomplete_groups_by_cause"] == {"reward_model:test_overflow": 1}
    assert stats["unavailable_continuations_by_cause"] == {"reward_model:test_overflow": 1}


@pytest.mark.parametrize("status", ["verified", "unverified"])
def test_one_unknown_final_outcome_blocks_all_actions(monkeypatch, status):
    results, _, _, trainer, _, config = run(monkeypatch)
    dec = results[0].decompositions[0]
    for rollout in dec.selections[:2]:
        rollout.reward.final_answer_correctness = None
        rollout.reward.verification_status = status
        rollout.reward.verification_error = {"stage": "final_correctness", "code": "prediction_parse_failed"}
    samples, stats = branch.build_samples(trainer.orchestrator, results[0].task, [dec], config, 2)
    assert samples == []
    assert dec.worker_comparisons[0]["actions"][0]["score"] is None
    assert stats["incomplete_groups_by_cause"] == {"final_correctness:prediction_parse_failed": 1}
    assert stats["unavailable_continuations_by_cause"] == {"final_correctness:prediction_parse_failed": 2}


def test_missing_continuation_has_separate_cause(monkeypatch):
    results, _, _, trainer, _, config = run(monkeypatch)
    dec = results[0].decompositions[0]
    dec.selections.pop()
    samples, stats = branch.build_samples(trainer.orchestrator, results[0].task, [dec], config, 2)
    assert samples == []
    assert stats["incomplete_groups_by_cause"] == {"missing_continuation": 1}


def test_optional_auxiliary_unknown_keeps_complete_comparison(monkeypatch):
    results, _, _, trainer, _, config = run(monkeypatch)
    dec = results[0].decompositions[0]
    selection = dec.selections[0]
    selection.reward.auxiliary_verification_status = "unverified"
    selection.reward.auxiliary_verification_errors = [
        {"stage": "answer_containment", "code": "verification_timeout"}]
    selection.executions[0].answer_containment = None
    samples, stats = branch.build_samples(trainer.orchestrator, results[0].task, [dec], config, 2)
    assert len(samples) == 8
    assert stats["num_groups_incomplete"] == 0
    assert all(len(sample.metadata["continuation_rewards"]) == 4 for sample in samples)


def test_required_auxiliary_failure_is_separate_and_counted_in_summary(monkeypatch):
    metrics = importlib.import_module(api.__name__ + ".metrics")
    results, _, _, trainer, _, config = run(monkeypatch)
    dec = results[0].decompositions[0]
    selection = dec.selections[0]
    selection.reward.verification_status = "unverified"
    selection.reward.verification_error = {"stage": "answer_containment", "code": "verification_timeout"}
    samples, stats = branch.build_samples(trainer.orchestrator, results[0].task, [dec], config, 2)
    assert samples == []
    cause = "required_auxiliary:answer_containment:verification_timeout"
    assert stats["incomplete_groups_by_cause"] == {cause: 1}
    results[0].training_batch.worker_grpo_stats = stats
    totals = metrics.rollout_monitoring_totals(results[0])
    assert totals[f"fixed_prefix/incomplete_groups_by_cause/{cause}"] == 1
    assert totals["verified_rollouts"] == 32


def test_uniform_reward_scaling_preserves_normalized_advantages():
    scores = [.2, .4, .8, -.3]
    assert branch.action_advantages(scores)[2] == pytest.approx(
        branch.action_advantages([.25 * value for value in scores])[2], abs=1e-6)


def test_gate_failed_action_still_contributes_to_baselines(monkeypatch):
    results, _, _, trainer, _, config = run(monkeypatch)
    dec = results[0].decompositions[0]
    for rollout in dec.selections:
        if rollout.branch_metadata["action_index"] == 7:
            rollout.reward_model_outputs["training_safety"] = {"excluded": True}
    samples, stats = branch.build_samples(trainer.orchestrator, results[0].task, [dec], config, 2)
    assert len(samples) == 7
    assert stats["num_actions_skipped_gate"] == 1
    scores = [.015 + i / 10 for i in range(8)]
    assert samples[0].metadata["leave_one_out_baseline"] == pytest.approx(sum(scores[1:]) / 7)
    assert dec.worker_comparisons[0]["actions"][7]["score"] == pytest.approx(.715)


def test_zero_signal_group_has_no_policy_copies(monkeypatch):
    results, *_ = run(monkeypatch, constant=True)
    assert results[0].training_batch.worker_samples == []
    assert results[0].decompositions[0].worker_comparisons[0]["status"] == "zero_signal"


def test_task_plan_and_branch_point_identities_stay_separate(monkeypatch):
    results, *_ = run(monkeypatch, points=2, tasks=2, plans=2)
    groups = [sample.group_id for result in results for sample in result.training_batch.worker_samples]
    assert len(set(groups)) == 8
    assert set(Counter(groups).values()) == {8}


def test_mismatched_frozen_input_is_rejected(monkeypatch):
    results, _, _, trainer, _, config = run(monkeypatch)
    dec = results[0].decompositions[0]
    focal = next(ex for ex in dec.selections[0].executions if ex.node_id == "B")
    focal.worker_prompt += " altered"
    with pytest.raises(ValueError, match="input identity"):
        branch.build_samples(trainer.orchestrator, results[0].task, [dec], config, 2)


def test_log_preserves_comparison_prompts_and_actions_in_compact_mode(monkeypatch, tmp_path):
    results, *_ = run(monkeypatch)
    recorder = recording.RolloutRecorder(api.RolloutLoggingConfig(
        output_dir=str(tmp_path), save_all_rollouts=False, compact_mode=True))
    recorder.record_task_rollout(results[0])
    payload = json.loads((tmp_path / "all_rollouts.jsonl").read_text())
    dec = payload["rollout"]["decompositions"][0]
    assert len(dec["selections"]) == 32
    assert dec["worker_comparisons"][0]["prompt_text"]
    assert dec["selections"][0]["executions"][0]["worker_prompt"]
    assert len(dec["worker_comparisons"][0]["actions"]) == 8


def test_default_mode_retains_all_node_training(monkeypatch):
    results, backend, scorer, *_ = run(monkeypatch, mode="repeated_execution")
    assert backend.counter == dict.fromkeys("ABCD", 3)
    assert len(scorer.requests) == 3
    assert len(results[0].training_batch.worker_samples) == 12
    assert results[0].decompositions[0].worker_comparisons == []


@pytest.mark.parametrize("kwargs", [
    {"worker_action_samples": 1}, {"worker_continuations_per_action": 0},
    {"worker_branch_points_per_plan": 0}, {"worker_rollout_mode": "unknown"},
    {"worker_update_rule": "unknown"},
])
def test_bad_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        api.RolloutConfig(**kwargs)


def test_policy_identity_separates_comparisons(monkeypatch):
    first = run(monkeypatch, checkpoint="version1")[0][0]
    second = run(monkeypatch, checkpoint="version2")[0][0]
    first_ids = {sample.group_id for sample in first.training_batch.worker_samples}
    second_ids = {sample.group_id for sample in second.training_batch.worker_samples}
    assert first_ids.isdisjoint(second_ids)


def test_batched_overlapping_scoring_has_identical_action_rewards(monkeypatch):
    sequential = run(monkeypatch)[0][0]
    overlapping = run(monkeypatch, overlap=True)[0][0]
    assert sequential.to_dict() == overlapping.to_dict()


def test_unverified_or_missing_suffix_invalidates_entire_group(monkeypatch):
    results, _, _, trainer, _, config = run(monkeypatch)
    dec = results[0].decompositions[0]
    dec.selections[0].reward.verification_status = "unverified"
    samples, _ = branch.build_samples(trainer.orchestrator, results[0].task, [dec], config, 2)
    assert samples == []
    assert dec.worker_comparisons[0]["reason"] == "incomplete_group"
    # Reevaluate independently with a missing suffix, not a smaller denominator.
    dec.worker_comparisons[0]["status"] = "pending"
    dec.selections.pop(0)
    samples, _ = branch.build_samples(trainer.orchestrator, results[0].task, [dec], config, 2)
    assert samples == []
    assert "missing_or_duplicate_continuation" in dec.worker_comparisons[0]["errors"][0]


def test_incomplete_worker_branch_does_not_discard_joint_planner(monkeypatch):
    results, *_ = run(monkeypatch, joint=True, missing=(3, 2))
    dec = results[0].decompositions[0]
    assert dec.training_eligible
    assert dec.decomposition_reward == pytest.approx(.3)
    assert len(results[0].training_batch.decomposer_samples) == 1
    assert results[0].training_batch.worker_samples == []


def test_seeded_node_selection_is_reproducible(monkeypatch):
    first = run(monkeypatch, points=4, force_focal=False)[0][0]
    second = run(monkeypatch, points=4, force_focal=False)[0][0]
    assert first.to_dict() == second.to_dict()
    assert {group["node_id"] for group in first.decompositions[0].worker_comparisons} == set("ABCD")


def test_configuration_survives_phase_routing_and_cli():
    train = importlib.import_module(api.__name__ + ".train")
    from argparse import ArgumentParser
    parser = ArgumentParser()
    branch.add_rollout_arguments(parser)
    args = parser.parse_args(["--worker-rollout-mode", "fixed_prefix",
                              "--worker-action-samples", "6",
                              "--worker-update-rule", "positive_only_nonterminal"])
    config = api.RolloutConfig(**branch.rollout_arguments(args))
    schedule = api.TrainingScheduleConfig(
        mode=api.TrainingMode.ALTERNATING, alternating_phase=api.AlternatingPhase.EXECUTOR)
    routed = train._rollout_config_for_schedule(config, schedule)
    assert branch.rollout_arguments(routed) == branch.rollout_arguments(config)
    budget = train.rollout_workload_estimate(2, routed, schedule)
    assert budget["num_executor_rollouts"] == 24
    assert budget["worker_branch_groups_per_task"] == routed.num_decompositions


def test_executor_only_branches_are_not_failed_planner_evaluations(monkeypatch):
    train = importlib.import_module(api.__name__ + ".train")
    results, *_ = run(monkeypatch)
    assert train._best_rollout_metrics(results[0])["unscored_plans"] == 0


def test_repeated_mode_positive_only_masks_after_comparison(monkeypatch):
    results, _, _, trainer, pool, config = run(
        monkeypatch, mode="repeated_execution", rule="positive_only_nonterminal")
    dec = results[0].decompositions[0]
    for i, rollout in enumerate(dec.selections):
        for execution in rollout.executions:
            execution.reward_model_reward = float(i)
    batch = trainer.orchestrator._build_training_batch(
        results[0].task, pool, results[0].policy_config, results[0].schedule, [dec], config)
    assert len(batch.worker_samples) == 6  # One positive per intermediate, all three finals.
    assert len([s for s in batch.worker_samples if s.metadata["node_id"] == "D"]) == 3
    assert all(s.advantage > 0 for s in batch.worker_samples if s.metadata["node_id"] != "D")

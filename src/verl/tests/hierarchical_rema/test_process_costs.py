import copy
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import test_fixed_prefix as fixed
import test_reward_providers as providers

api = fixed.api
costs = importlib.import_module(api.__name__ + ".reward_providers.process_costs")
state_module = importlib.import_module(api.__name__ + ".reward_providers.process_cost_state")
factory = importlib.import_module(api.__name__ + ".reward_providers.factory")


def probabilities(good):
    return {"probabilities": {"0": 1 - good, "1": good}}


def weights(**overrides):
    return {**dict.fromkeys(costs.COSTS, .1), **overrides}


def predictions(request, discipline=.8, final=.6, anti_collapse=.9,
                task_fulfillment=.9, final_task_fulfillment=.5,
                planner_discipline=.7, planner_task_fulfillment=.6):
    return {
        "graph": {"anti_collapse": probabilities(anti_collapse), "global_outcome": probabilities(0)},
        "decomposer": {"execution_discipline": probabilities(planner_discipline),
                       "task_fulfillment": probabilities(planner_task_fulfillment)},
        "workers": {str(ex.node_id): {"execution_discipline": probabilities(discipline),
                                     "task_fulfillment": probabilities(task_fulfillment)}
                    for ex in request["executions"]},
        "final": {"execution_discipline": probabilities(final),
                  "task_fulfillment": probabilities(final_task_fulfillment)},
    }


class RawScorer:
    rollout_batch_size = 5

    def __init__(self, missing=None):
        self.requests, self.missing = [], missing

    def predict_rollouts(self, requests):
        self.requests.extend(requests)
        results = []
        for request in requests:
            action, continuation = (0, 0)
            selection_id = request["selection"].selection_id
            if ":a" in selection_id:
                action, continuation = [int(part[1:]) for part in selection_id.split(":")[-2:]]
            if (action, continuation) == self.missing:
                results.append({"status": "unscored", "error": {"code": "context_overflow"}})
            else:
                # The focal action is unchanged, but downstream graph context changes its prediction.
                result = predictions(request, discipline=.1 + action / 10 + continuation / 100,
                                     task_fulfillment=.2 + action / 20 + continuation / 100)
                results.append({"status": "scored", "model_predictions": result,
                                "graph_summary": {"prm_max_length": 4096}})
        return results

    def score_rollouts(self, requests):
        raise AssertionError("Legacy reward compiler must not run")


def test_exact_role_formula_final_head_and_no_double_count():
    request = providers.request()
    lambdas = weights(collapse=.5, discipline=2., task_fulfillment=.4,
                      planner_discipline=.3, planner_task_fulfillment=.2)
    result = costs.compile_process_costs(predictions(request), request,
                                        lambdas, 3)
    rewards, summary = result["compiled_rewards"], result["graph_summary"]
    assert rewards["workers"]["A"]["reward"] == pytest.approx(.56)
    assert rewards["workers"]["F"]["reward"] == pytest.approx(0.)
    assert rewards["final"] == rewards["workers"]["F"]
    assert summary["process_costs"] == pytest.approx({"collapse": .1, "discipline": .3, "task_fulfillment": .3,
                                                       "planner_discipline": .3, "planner_task_fulfillment": .4})
    assert summary["node_task_fulfillment_costs"] == pytest.approx({"A": .1, "F": .5})
    assert rewards["decomposer"]["reward"] == pytest.approx(.06)
    assert rewards["decomposer"]["discipline_cost"] == pytest.approx(.3)
    assert rewards["decomposer"]["task_fulfillment_cost"] == pytest.approx(.4)
    assert rewards["decomposer"]["mean_worker_discipline_cost"] == pytest.approx(.3)
    assert rewards["decomposer"]["mean_worker_task_fulfillment_cost"] == pytest.approx(.3)
    # Model-predicted success still never replaces verified correctness.
    request["verified_final_correctness"] = 0
    failed = costs.compile_process_costs(predictions(request), request,
                                        lambdas, 3)
    assert failed["compiled_rewards"]["workers"]["A"]["reward"] == pytest.approx(-.44)
    assert failed["compiled_rewards"]["decomposer"]["reward"] == pytest.approx(-.94)


@pytest.mark.parametrize("label,cost", [("execution_discipline", "planner_discipline"),
                                       ("task_fulfillment", "planner_task_fulfillment")])
def test_planner_costs_only_change_planner_credit(label, cost):
    request = providers.request()
    before = predictions(request, discipline=1., final=1., task_fulfillment=1.,
                         final_task_fulfillment=1., planner_discipline=1., planner_task_fulfillment=1.)
    after = copy.deepcopy(before)
    after["decomposer"][label] = probabilities(.2)
    lambdas = weights(**{cost: .4})
    good = costs.compile_process_costs(before, request, lambdas, 0)
    bad = costs.compile_process_costs(after, request, lambdas, 0)
    assert good["compiled_rewards"]["workers"] == bad["compiled_rewards"]["workers"]
    assert good["compiled_rewards"]["final"] == bad["compiled_rewards"]["final"]
    assert good["compiled_rewards"]["workers"]["A"]["reward"] == 1.
    assert (good["compiled_rewards"]["decomposer"]["reward"]
            - bad["compiled_rewards"]["decomposer"]["reward"]) == pytest.approx(.32)
    assert bad["graph_summary"]["process_costs"][cost] == pytest.approx(.8)
    # No new direct term may be multiplied or diluted by the number of workers.
    extra = copy.copy(request["executions"][0])
    extra.node_id = "extra"
    request["executions"].append(extra)
    before["workers"]["extra"] = copy.deepcopy(before["workers"]["A"])
    after["workers"]["extra"] = copy.deepcopy(after["workers"]["A"])
    assert costs.compile_process_costs(before, request, lambdas, 0)["compiled_rewards"]["decomposer"] == good["compiled_rewards"]["decomposer"]
    assert costs.compile_process_costs(after, request, lambdas, 0)["compiled_rewards"]["decomposer"] == bad["compiled_rewards"]["decomposer"]


@pytest.mark.parametrize("label", ["execution_discipline", "task_fulfillment"])
def test_missing_or_invalid_planner_predictions_are_not_zero_cost(label):
    request = providers.request()
    missing = predictions(request)
    del missing["decomposer"][label]
    with pytest.raises(KeyError):
        costs.compile_process_costs(missing, request, weights(), 0)
    malformed = predictions(request)
    malformed["decomposer"][label] = {"probabilities": {"0": .2, "1": .9}}
    with pytest.raises(ValueError, match="normalized binary"):
        costs.compile_process_costs(malformed, request, weights(), 0)


def test_local_task_failure_reduces_credit_even_when_final_answer_is_correct():
    request = providers.request()
    lambdas = weights(discipline=.2, task_fulfillment=.4)
    before = predictions(request)
    after = copy.deepcopy(before)
    after["workers"]["A"]["task_fulfillment"] = probabilities(.1)
    good = costs.compile_process_costs(before, request, lambdas, 0)["compiled_rewards"]
    bad = costs.compile_process_costs(after, request, lambdas, 0)["compiled_rewards"]
    assert good["workers"]["A"]["reward"] - bad["workers"]["A"]["reward"] == pytest.approx(.32)
    assert good["decomposer"]["reward"] - bad["decomposer"]["reward"] == pytest.approx(.16)
    assert good["final"] == bad["final"]
    # The final execution uses the final head, not the duplicate worker prediction.
    after["workers"]["F"]["task_fulfillment"] = probabilities(0)
    assert costs.compile_process_costs(after, request, lambdas, 0)["compiled_rewards"] == bad


@pytest.mark.parametrize("empty,reason", [(True, ""), (False, "invalid_protocol")])
def test_protocol_failure_is_explicit_cost_not_old_reward_cap(empty, reason):
    request = providers.request()
    request["executions"][0].invalid_reason = reason
    if empty:
        request["executions"][0].output_text = ""
    compiled = costs.compile_process_costs(predictions(request, discipline=1), request,
                                          weights(discipline=.2, task_fulfillment=.4), 0)
    node = compiled["compiled_rewards"]["workers"]["A"]
    assert node["discipline_cost"] == 1
    assert node["protocol_cost_override"]
    assert node["task_fulfillment_cost"] == pytest.approx(.1)
    assert node["reward"] == pytest.approx(.76)


@pytest.mark.parametrize("outcome", [None, float("nan"), .5, "1", True])
def test_unknown_outcomes_never_use_predicted_success(outcome):
    request = providers.request()
    request["verified_final_correctness"] = outcome
    scorer = RawScorer()
    provider = costs.ProcessCostProvider(scorer, weights())
    result = provider.score_rollout(**request)
    assert result["status"] == "unscored"
    assert result["error"]["code"] == "verified_outcome_required"
    assert scorer.requests == []


def test_batched_raw_predictions_order_overflow_and_snapshot():
    scorer = RawScorer(missing=(1, 0))
    lambdas = weights(collapse=.3, discipline=.2)
    provider = costs.ProcessCostProvider(scorer, lambdas, 4)
    lambdas["discipline"] = 99
    lambdas["task_fulfillment"] = 99
    lambdas["planner_discipline"] = 99
    lambdas["planner_task_fulfillment"] = 99
    requests = [providers.request() for _ in range(3)]
    for index, request in enumerate(requests):
        request["selection"].selection_id = f"prefix:a{index}:c0"
    results = provider.score_rollouts(requests)
    assert [r["status"] for r in results] == ["scored", "unscored", "scored"]
    assert all("verified_final_correctness" not in r for r in scorer.requests)
    assert results[0]["graph_summary"]["process_cost_lambdas"]["discipline"] == .2
    assert results[0]["graph_summary"]["process_cost_lambdas"]["task_fulfillment"] == .1
    assert results[0]["graph_summary"]["process_cost_lambdas"]["planner_discipline"] == .1
    assert results[0]["graph_summary"]["process_cost_lambdas"]["planner_task_fulfillment"] == .1
    assert results[0]["graph_summary"]["process_cost_update_count"] == 4
    assert results[2]["compiled_rewards"]["workers"]["A"]["reward"] > results[0]["compiled_rewards"]["workers"]["A"]["reward"]


def run_mode(monkeypatch, **kwargs):
    monkeypatch.setattr(fixed, "Scorer", lambda missing=None, constant=False:
                        costs.ProcessCostProvider(RawScorer(missing), weights(collapse=.5, discipline=.2)))
    return fixed.run(monkeypatch, **kwargs)


@pytest.mark.parametrize("joint,mode", [(False, "fixed_prefix"), (True, "fixed_prefix"),
                                       (False, "repeated_execution"), (True, "repeated_execution")])
def test_both_rollout_modes_preserve_node_credit_and_planner_means(monkeypatch, joint, mode):
    results, _, _, _, _, _ = run_mode(monkeypatch, joint=joint, mode=mode, overlap=True)
    task = results[0]
    samples = task.training_batch.worker_samples
    plan = task.decompositions[0]
    assert samples
    if mode == "fixed_prefix":
        assert len(samples) == 8
        assert [sample.reward for sample in samples] == pytest.approx([
            1 - .2 * (1 - (.1 + action / 10 + .015)) - .1 * (1 - (.2 + action / 20 + .015))
            for action in range(8)])
        assert len({s.completion_text for s in samples}) == 8
    else:
        assert len(samples) == 4 * 3
        assert {s.metadata["advantage_group_size"] for s in samples} == {3}
    if joint:
        values = [s.reward_model_outputs["compiled_rewards"]["decomposer"]["reward"]
                  for s in plan.planner_selections]
        assert plan.decomposition_reward == pytest.approx(sum(values) / len(values))
    else:
        assert task.training_batch.decomposer_samples == []
    stats = state_module.segment_statistics(results)
    assert stats["coverage"] == 1
    assert stats["mean_costs"]["task_fulfillment"] is not None
    assert stats["mean_costs"]["planner_discipline"] == pytest.approx(.3)
    assert stats["mean_costs"]["planner_task_fulfillment"] == pytest.approx(.4)


def test_task_fulfillment_alone_can_drive_fixed_prefix_node_advantages(monkeypatch):
    monkeypatch.setattr(fixed, "Scorer", lambda missing=None, constant=False:
                        costs.ProcessCostProvider(RawScorer(missing), weights(collapse=0., discipline=0.,
                                                                            task_fulfillment=1.)))
    results, *_ = fixed.run(monkeypatch, mode="fixed_prefix", overlap=True)
    samples = results[0].training_batch.worker_samples
    assert len(samples) == 8
    assert [sample.reward for sample in samples] == pytest.approx([.215 + action / 20 for action in range(8)])
    assert samples[0].advantage < 0 < samples[-1].advantage


def test_terminal_has_one_continuation(monkeypatch):
    results, _, provider, *_ = run_mode(monkeypatch, focal="D", single=True)
    assert len(provider.scorer.requests) == 8
    # Equal final-head scores produce no relative signal, not eight duplicate updates.
    assert len(results[0].decompositions[0].worker_comparisons[0]["actions"]) == 8
    assert results[0].training_batch.worker_grpo_stats["num_groups_zero_signal"] == 1


def test_incomplete_prefix_is_not_partially_averaged(monkeypatch):
    results, *_ = run_mode(monkeypatch, missing=(2, 1))
    assert results[0].training_batch.worker_samples == []
    stats = state_module.segment_statistics(results)
    assert stats["coverage"] == 0
    assert stats["graphs_scored"] == 31
    assert stats["mean_costs"] == dict.fromkeys(costs.COSTS)


def test_legacy_controller_penalties_and_best_aggregation_are_bypassed(monkeypatch):
    results, _, provider, trainer, *_ = run_mode(monkeypatch, joint=True)
    orchestrator = trainer.orchestrator
    orchestrator.controller_format_retry_penalty = 10
    orchestrator.controller_format_fallback_penalty = 20
    orchestrator.controller_fallback_positive_reward_scale = 0
    payload = {"validation": {"attempt": 3, "fallback_used": True}}
    assert orchestrator._controller_training_reward_components(.7, payload, role="decomposer") == (.7, .7, 0.)
    orchestrator.decomposer_reward_aggregation = "best"
    selections = results[0].decompositions[0].planner_selections
    assert orchestrator._aggregate_decomposition_selection_reward(selections, [0., .5, 1.]) == .5
    assert not selections[0].reward_model_outputs["training_safety"]["enabled"]


def statistics(collapse=.4, discipline=.05, task_fulfillment=.6, coverage=1., snapshot=None,
               planner_discipline=.1, planner_task_fulfillment=.1):
    return {"mean_costs": {"collapse": collapse, "discipline": discipline, "task_fulfillment": task_fulfillment,
                          "planner_discipline": planner_discipline, "planner_task_fulfillment": planner_task_fulfillment},
            "coverage": coverage, "graph_coverage": coverage, "mean_correctness": .5,
            "snapshot": snapshot, "fingerprint": "data"}


def test_dual_update_projection_persistence_idempotency_and_restore(tmp_path):
    config = costs.ProcessCostConfig(initial_discipline=.001, learning_rate=.1)
    state = state_module.ProcessCostState(tmp_path / "state.json", config, "checkpoint")
    event = state.finish("segment-1", statistics(snapshot=[0, state.read()["lambdas"]]))
    assert event["after"] == pytest.approx(weights(collapse=.13, discipline=0., task_fulfillment=.15))
    assert state.finish("segment-1", event["statistics"]) == event
    assert state.read()["update_count"] == 1
    restored = state_module.ProcessCostState(tmp_path / "restored.json", config, "checkpoint", state.path)
    assert restored.read()["lambdas"] == event["after"]
    assert restored.read()["update_count"] == 1
    assert restored.read()["events"] == {}
    with pytest.raises(ValueError, match="different rollouts"):
        state.finish("segment-1", statistics(collapse=.7))
    with pytest.raises(ValueError, match="Stale"):
        state.finish("segment-2", event["statistics"])
    with pytest.raises(ValueError, match="different configuration/checkpoint"):
        state_module.ProcessCostState(state.path, config, "wrong-checkpoint")
    with pytest.raises(ValueError, match="different configuration/checkpoint"):
        state_module.ProcessCostState(state.path, replace(config, learning_rate=.2), "checkpoint")


def test_low_coverage_does_not_adapt(tmp_path):
    state = state_module.ProcessCostState(tmp_path / "state.json", costs.ProcessCostConfig(), "cp")
    event = state.finish("low", statistics(coverage=.8))
    assert not event["updated"]
    assert event["before"] == event["after"]
    assert state.read()["update_count"] == 0


@pytest.mark.parametrize("cost", ["task_fulfillment", "planner_discipline", "planner_task_fulfillment"])
@pytest.mark.parametrize("mean,expected", [(.8, .13), (.2, .07)])
def test_multiplier_uses_its_own_target(tmp_path, mean, expected, cost):
    config = costs.ProcessCostConfig(**{"target_" + cost: .5}, learning_rate=.1)
    state = state_module.ProcessCostState(tmp_path / "state.json", config, "cp")
    event = state.finish("segment", statistics(**weights(**{cost: mean})))
    assert event["after"] == pytest.approx(weights(**{cost: expected}))


@pytest.mark.parametrize("version", ["role_aware_process_costs_v1", "role_aware_process_costs_v2"])
def test_old_cost_state_is_not_silently_reused(tmp_path, version):
    config = costs.ProcessCostConfig()
    state = state_module.ProcessCostState(tmp_path / "old.json", config, "cp")
    old = state.read()
    old["version"] = version
    removed = ["planner_discipline", "planner_task_fulfillment"]
    if version.endswith("v1"):
        removed.append("task_fulfillment")
    for name in removed:
        del old["lambdas"][name]
        del old["config"]["initial_" + name]
        del old["config"]["target_" + name]
    state_module.atomic_json(state.path, old)
    assert costs.VERSION == "role_aware_process_costs_v3"
    with pytest.raises(ValueError, match="different configuration/checkpoint"):
        state_module.ProcessCostState(state.path, config, "cp")
    with pytest.raises(ValueError, match="different configuration/checkpoint"):
        state_module.ProcessCostState(tmp_path / "new.json", config, "cp", state.path)
    assert not (tmp_path / "new.json").exists()


def test_statistics_ignore_sample_filters_and_prefer_ordinary_joint_repeats(monkeypatch):
    results, *_ = run_mode(monkeypatch, joint=True)
    baseline = state_module.segment_statistics(results)
    for selection in results[0].decompositions[0].selections:
        selection.reward_model_outputs["training_safety"]["excluded"] = True
        if selection.branch_metadata:
            selection.reward_model_outputs["graph_summary"]["process_costs"] = dict.fromkeys(costs.COSTS, 1.)
    results[0].training_batch.worker_samples = []
    results[0].training_batch.decomposer_samples = []
    assert state_module.segment_statistics(results) == baseline


def test_statistics_equal_tasks_not_number_of_plans(monkeypatch):
    results, *_ = run_mode(monkeypatch, tasks=2, plans=1, mode="repeated_execution")
    for selection in results[0].decompositions[0].selections:
        selection.reward_model_outputs["graph_summary"]["process_costs"] = dict.fromkeys(costs.COSTS, 0.)
    for selection in results[1].decompositions[0].selections:
        selection.reward_model_outputs["graph_summary"]["process_costs"] = dict.fromkeys(costs.COSTS, 1.)
    results[1].decompositions *= 3
    assert state_module.segment_statistics(results)["mean_costs"] == dict.fromkeys(costs.COSTS, .5)


def test_finish_hook_logs_and_saves_only_for_new_setting(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pkl"
    checkpoint.write_bytes(b"test-checkpoint")
    args = SimpleNamespace(reward_setting=costs.SETTING, output_dir=str(tmp_path), gfam_reward_model_pkl=str(checkpoint))
    results, *_ = run_mode(monkeypatch)
    # Match the immutable snapshot actually used for this segment.
    monkeypatch.setenv("REMA_PROCESS_COST_INITIAL_COLLAPSE", ".5")
    monkeypatch.setenv("REMA_PROCESS_COST_INITIAL_DISCIPLINE", ".2")
    logs = []
    tracking = SimpleNamespace(log=lambda values, step: logs.append((values, step)))
    event = state_module.finish_process_cost_segment(args, results, tmp_path / "segment", tracking, 123)
    assert event["updated"]
    assert (tmp_path / "process_costs/state.json").is_file()
    assert (tmp_path / "segment/process_cost_update.json").is_file()
    assert logs[0][1] == 123
    assert logs[0][0]["process_costs/dual_updates"] == 1
    saved = json.loads((tmp_path / "segment/process_cost_state.json").read_text())
    for cost in costs.COSTS:
        assert logs[0][0][f"process_costs/mean/{cost}"] == event["statistics"]["mean_costs"][cost]
        assert logs[0][0][f"process_costs/lambda/{cost}"] == event["after"][cost]
        assert logs[0][0][f"process_costs/target/{cost}"] == .1
        assert saved["lambdas"][cost] == event["after"][cost]
    args.reward_setting = "gnn_prm"
    assert state_module.finish_process_cost_segment(args, [], tmp_path / "unused") is None
    assert not (tmp_path / "unused").exists()


def test_factory_snapshot_is_not_reset_between_training_and_validation(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pkl"
    checkpoint.write_bytes(b"test")
    args = SimpleNamespace(reward_setting=costs.SETTING, output_dir=str(tmp_path),
                           gfam_reward_model_pkl=str(checkpoint), gfam_reward_model_device="cpu",
                           gfam_reward_model_encoder_backend=None, gfam_reward_model_encoder_model=None)
    gfam = importlib.import_module(api.__name__ + ".gfam_reward")
    monkeypatch.setattr(gfam, "load_cached_reward_scorer", lambda **kw:
                        SimpleNamespace(supports_verified_final_correctness=True, graphprm=RawScorer()))
    first = factory.build_reward_provider(args)
    state = state_module.state_for_args(args)
    state.finish("segment", statistics(snapshot=[0, first.lambdas]))
    second = factory.build_reward_provider(args)
    assert first.lambdas == weights()
    assert second.lambdas["collapse"] > first.lambdas["collapse"]
    assert second.lambdas["task_fulfillment"] > first.lambdas["task_fulfillment"]
    second.score_rollout(**providers.request())
    assert state.read()["update_count"] == 1  # Scoring/validation is read-only.
    manifest = factory.reward_manifest(args)
    assert manifest["reward_contract"] == costs.VERSION
    assert "verified_failure_positive_scale" not in manifest
    assert manifest["process_cost_config"]["target_task_fulfillment"] == .1
    assert manifest["process_cost_config"]["initial_task_fulfillment"] == .1
    for cost in ("planner_discipline", "planner_task_fulfillment"):
        assert manifest["process_cost_config"]["target_" + cost] == .1
        assert manifest["process_cost_config"]["initial_" + cost] == .1


def test_launcher_registers_setting_and_preserves_checkpoint_path():
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run(["bash", "-c", 'source "$1"; printf "%s %s %s %s %s %s %s %s" "$REWARD_SETTING" "$REMA_PROCESS_COST_LEARNING_RATE" "$REMA_PROCESS_COST_TARGET_TASK_FULFILLMENT" "$REMA_PROCESS_COST_INITIAL_TASK_FULFILLMENT" "$REMA_PROCESS_COST_TARGET_PLANNER_DISCIPLINE" "$REMA_PROCESS_COST_TARGET_PLANNER_TASK_FULFILLMENT" "$REMA_PROCESS_COST_INITIAL_PLANNER_DISCIPLINE" "$REMA_PROCESS_COST_INITIAL_PLANNER_TASK_FULFILLMENT"',
                             "_", str(root / "scripts/reward-setting.sh")],
                            env={**os.environ, "REWARD_SETTING": costs.SETTING}, capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout == "graph_prm_process_costs 0.01 0.10 0.10 0.10 0.10 0.10 0.10"
    wrapper = (root / "hierarchical-rema-trainer.sh").read_text()
    assert '"$REWARD_SETTING" != "gnn_prm" && "$REWARD_SETTING" != "graph_prm_process_costs"' in wrapper
    assert "REMA_PROCESS_COST_TARGET_TASK_FULFILLMENT=${REMA_PROCESS_COST_TARGET_TASK_FULFILLMENT:-<unused>}" in wrapper
    assert "REMA_PROCESS_COST_INITIAL_TASK_FULFILLMENT=${REMA_PROCESS_COST_INITIAL_TASK_FULFILLMENT:-<unused>}" in wrapper
    for suffix in ("TARGET_PLANNER_DISCIPLINE", "TARGET_PLANNER_TASK_FULFILLMENT",
                   "INITIAL_PLANNER_DISCIPLINE", "INITIAL_PLANNER_TASK_FULFILLMENT"):
        name = "REMA_PROCESS_COST_" + suffix
        assert name + "=${" + name + ":-<unused>}" in wrapper


@pytest.mark.parametrize("cost", ["task_fulfillment", "planner_discipline", "planner_task_fulfillment"])
def test_cost_environment_configuration(monkeypatch, cost):
    monkeypatch.setenv("REMA_PROCESS_COST_TARGET_" + cost.upper(), ".3")
    monkeypatch.setenv("REMA_PROCESS_COST_INITIAL_" + cost.upper(), ".4")
    config = costs.ProcessCostConfig.from_environment()
    assert getattr(config, "target_" + cost) == .3
    assert getattr(config, "initial_" + cost) == .4


@pytest.mark.parametrize("field,value", [("learning_rate", -1), ("target_collapse", 1.1),
                                        ("initial_discipline", float("inf")), ("min_coverage", 0),
                                        ("target_task_fulfillment", 1.1), ("target_task_fulfillment", -1),
                                        ("initial_task_fulfillment", float("nan")),
                                        ("target_planner_discipline", -1),
                                        ("target_planner_task_fulfillment", 1.1),
                                        ("initial_planner_discipline", float("inf")),
                                        ("initial_planner_task_fulfillment", -1)])
def test_config_validation(field, value):
    with pytest.raises(ValueError):
        costs.ProcessCostConfig(**{field: value})


@pytest.mark.parametrize("update_every,expected_updates", [(1, 4), (2, 2)])
def test_integrated_training_updates_once_per_segment_and_records_state(tmp_path, monkeypatch, update_every, expected_updates):
    train = importlib.import_module(api.__name__ + ".train")
    gfam = importlib.import_module(api.__name__ + ".gfam_reward")
    checkpoint = tmp_path / "model.pkl"
    checkpoint.write_bytes(b"fake-checkpoint")
    monkeypatch.setattr(gfam, "load_cached_reward_scorer", lambda **kw:
                        SimpleNamespace(supports_verified_final_correctness=True, graphprm=RawScorer()))
    monkeypatch.setattr(train, "run_offline_policy_training", lambda train_samples, val_samples, config, **kw:
                        {"steps": 1, "output_dir": config.output_dir, "objective": "grpo"})
    monkeypatch.setattr(fixed.orchestration, "build_selection_reward", lambda **kw:
                        api.SelectionRewardBreakdown(final_answer_correctness=1., confidence_reward=0.,
                                                     compatibility_reward=0., total_reward=1.))
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--reward-setting", costs.SETTING, "--gfam-reward-model-pkl", str(checkpoint),
        "--task-source", "demo", "--backend", "mock", "--num-epochs", "2", "--max-tasks", "2",
        "--mode", "joint", "--num-decompositions", "2", "--num-executor-rollouts", "2",
        "--update-every-n-rollout-batches", str(update_every),
        "--rollout-task-batch-size", "1", "--disable-rollout-logging", "--output-dir", str(tmp_path / "run")])
    train.main()
    state = json.loads((tmp_path / "run/process_costs/state.json").read_text())
    assert state["update_count"] == expected_updates
    assert set(state["lambdas"]) == set(costs.COSTS)
    assert state["lambdas"]["task_fulfillment"] > .1
    assert state["lambdas"]["planner_discipline"] > .1
    assert state["lambdas"]["planner_task_fulfillment"] > .1
    assert len(state["events"]) == expected_updates
    snapshots = sorted((tmp_path / "run").glob("epoch_*/train/segment_*/process_cost_state.json"))
    assert [json.loads(path.read_text())["update_count"] for path in snapshots] == list(range(1, expected_updates + 1))
    assert [event["statistics"]["snapshot"][0] for event in state["events"].values()] == list(range(expected_updates))


def test_raw_runtime_api_never_compiles_legacy_rewards(monkeypatch):
    runtime = importlib.import_module(api.__name__ + ".graphprm_v2.runtime")
    scorer = runtime.GraphPRMV2RewardScorer.__new__(runtime.GraphPRMV2RewardScorer)
    seen = []
    scorer.score_records = lambda records, **kwargs: seen.append((records, kwargs)) or []
    scorer.predict_rollouts([providers.request()])
    assert seen[0][1] == {"compile_reward": False}
    assert "SECRET_REFERENCE" not in json.dumps(seen[0][0])

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

try:
    import hierarchical_rema as rema
except ModuleNotFoundError:
    from verl import hierarchical_rema as rema
import importlib

base = importlib.import_module(rema.__name__ + ".reward_providers.base")
factory = importlib.import_module(rema.__name__ + ".reward_providers.factory")
coach = importlib.import_module(rema.__name__ + ".reward_providers.coach")
rules = importlib.import_module(rema.__name__ + ".reward_providers.handcrafted")
demo = importlib.import_module(rema.__name__ + ".demo")


def request():
    nodes = [rema.SubtaskNode("A", "Calculate 2+3"), rema.SubtaskNode("F", "Double the supplied sum", ["A"])]
    return {
        "task": rema.TaskExample("t", "What is twice the sum of 2 and 3?", "SECRET_REFERENCE"),
        "decomposition": rema.DecompositionCandidate("p", "", "twice sum", "integer", nodes, "F"),
        "selection": rema.SelectionCandidate("s", []),
        "executions": [
            rema.WorkerExecution("A", "executor_worker", "5", 0., 0., 0., raw_output_text="REJECTED_AUDIT"),
            rema.WorkerExecution("F", "executor_worker", "10", 0., 0., 0., dependency_outputs={"A": "5"}),
        ],
        "final_answer": "10", "verified_final_correctness": 1.,
    }


def response(req, score=8):
    group = lambda scope: {label: {"score": score, "reason": "visible evidence"} for label in base.LABELS[scope]}
    return {**{scope: group(scope) for scope in ("graph", "decomposer", "final")},
            "workers": {ex.node_id: group("worker") for ex in req["executions"]}}


def make_coach(tmp_path, transport, **kwargs):
    return coach.CoachRewardProvider(model="test-coach", base_url="https://example.invalid/v1",
                                    cache_dir=tmp_path, transport=transport, retries=0, **kwargs)


@pytest.mark.parametrize("setting", ["frozen_prm", "finetuned_prm", "gnn_only"])
def test_placeholders_fail_before_model_loading(setting):
    with pytest.raises(NotImplementedError):
        factory.build_reward_provider(SimpleNamespace(reward_setting=setting))


def test_explicit_setting_no_silent_fallback():
    with pytest.raises(ValueError, match="requires --gfam"):
        factory.build_reward_provider(SimpleNamespace(reward_setting="gnn_prm"))
    with pytest.raises(ValueError, match="Unknown"):
        factory.build_reward_provider(SimpleNamespace(reward_setting="typo"))


def test_handcrafted_does_not_load_models(monkeypatch):
    original = sys.modules.copy()
    provider = factory.build_reward_provider(SimpleNamespace(reward_setting="handcrafted"))
    assert isinstance(provider, rules.HandcraftedRewardProvider)
    assert not any("gfam_reward" in key or "graphprm_v2.runtime" in key for key in set(sys.modules) - set(original))


def test_no_reference_or_raw_rejected_artifact_in_coach_inputs(tmp_path):
    req = request()
    req["executions"][0].output_text = ""
    seen = []
    def transport(payload):
        seen.append(payload)
        return json.dumps(response(req))
    make_coach(tmp_path, transport).score_rollout(**req)
    text = json.dumps(seen)
    assert "SECRET_REFERENCE" not in text and "REJECTED_AUDIT" not in text
    record = json.loads(seen[0]["messages"][1]["content"])
    assert record["workers"][0]["output_text"] == ""
    assert "verified_final_correctness" not in record


@pytest.mark.parametrize("bad", [True, 11, -1, .5, "8", float("nan")])
def test_coach_rejects_invalid_score(tmp_path, bad):
    req = request()
    data = response(req)
    data["workers"]["A"]["causal_utility"]["score"] = bad
    result = make_coach(tmp_path, lambda _: json.dumps(data)).score_rollout(**req)
    assert result["status"] == "unscored"
    assert result["compiled_rewards"] == {}
    assert not list(tmp_path.glob("*.json"))


def test_coach_requires_all_nodes_and_rejects_duplicates():
    req = request()
    row = base.rollout_record(req["task"], req["decomposition"], req["executions"], req["final_answer"])
    data = response(req)
    del data["workers"]["A"]
    with pytest.raises(ValueError, match="node IDs"):
        coach.parse_coach_response(json.dumps(data), row)
    with pytest.raises(ValueError, match="Duplicate"):
        coach.parse_coach_response('{"workers":{},"workers":{}}', row)


def test_coach_retries_and_only_caches_success(tmp_path, monkeypatch):
    req, calls = request(), []
    monkeypatch.setattr(coach.time, "sleep", lambda _: None)
    def transport(payload):
        calls.append(payload)
        return "not JSON" if len(calls) == 1 else json.dumps(response(req))
    provider = make_coach(tmp_path, transport)
    provider.retries = 1
    first = provider.score_rollout(**req)
    second = provider.score_rollout(**req)
    assert len(calls) == 2
    assert first["status"] == "scored"
    assert second["graph_summary"]["coach_cache_hit"]
    # Recompile from cached process scores using fresh verifier evidence.
    req["verified_final_correctness"] = 0.
    failed = provider.score_rollout(**req)
    assert len(calls) == 2
    assert failed["compiled_rewards"]["workers"]["A"]["reward"] == pytest.approx(
        first["compiled_rewards"]["workers"]["A"]["reward"] * .25)
    provider.model = "different-coach"
    provider.score_rollout(**req)
    assert len(calls) == 3


def test_batch_order_failure_isolation(tmp_path):
    requests = [request(), request(), request()]
    for i, req in enumerate(requests):
        req["task"].prompt = str(i)
    def transport(payload):
        i = int(json.loads(payload["messages"][1]["content"])["task"]["prompt"])
        return "invalid" if i == 1 else json.dumps(response(requests[i], 2 + i))
    results = make_coach(tmp_path, transport).score_rollouts(requests)
    assert [r["status"] for r in results] == ["scored", "unscored", "scored"]
    assert results[0]["reward_evidence"]["workers"]["A"]["task_fulfillment"]["score"] == .2
    assert results[2]["reward_evidence"]["workers"]["A"]["task_fulfillment"]["score"] == .4


@pytest.mark.parametrize("setting", ["coach", "handcrafted"])
def test_outcome_gate_preserves_negative_penalties_and_invalid_guard(tmp_path, setting):
    req = request()
    provider = (make_coach(tmp_path, lambda _: json.dumps(response(req, 10)))
                if setting == "coach" else rules.HandcraftedRewardProvider())
    good = provider.score_rollout(**req)
    req["verified_final_correctness"] = 0.
    bad = provider.score_rollout(**req)
    assert bad["compiled_rewards"]["workers"]["A"]["reward"] == pytest.approx(
        good["compiled_rewards"]["workers"]["A"]["reward"] * .25)
    req["executions"][0].output_text = ""
    req["executions"][0].invalid_reason = "bad protocol"
    invalid = provider.score_rollout(**req)
    assert invalid["compiled_rewards"]["workers"]["A"]["reward"] <= 0.


def test_rules_arithmetic_unknown_and_inequalities():
    assert rules.arithmetic_evidence("5", "Calculate 2+3")["score"] == 1.
    assert rules.arithmetic_evidence("7", "Calculate 2+3")["score"] == 0.
    assert rules.arithmetic_evidence("2+3=7", "Solve")["score"] == 0.
    assert rules.arithmetic_evidence("plausible assertion", "Prove a theorem")["score"] == .5
    req = request()
    req["executions"][0].output_text = "x < 3 and y > 2"
    row = base.rollout_record(req["task"], req["decomposition"], req["executions"], "10")
    assert row["workers"][0]["output_text"] == "x < 3 and y > 2"
    with pytest.raises(ValueError):
        rules._arithmetic("2**9999999999")


def test_rules_no_short_output_copy_penalty_and_no_dependency_usage_claim():
    req = request()
    req["executions"][1].output_text = "5"
    req["final_answer"] = "5"
    result = rules.HandcraftedRewardProvider().score_rollout(**req)
    assert result["reward_evidence"]["graph"]["anti_collapse"]["score"] == 1.
    assert "unknown" in result["reward_evidence"]["workers"]["A"]["causal_utility"]["reason"]


def test_graph_adapter_preserves_existing_rewards_exactly():
    req = request()
    existing = rules.HandcraftedRewardProvider().score_rollout(**req)
    existing["source"] = "gfam_v1"
    original = copy.deepcopy(existing)
    class Scorer:
        rollout_batch_size = 4
        def score_rollouts(self, requests):
            return [existing for _ in requests]
    result = factory.GraphPRMProvider(Scorer()).score_rollout(**req)
    assert result["source"] == "gnn_prm"
    assert result["compiled_rewards"] == original["compiled_rewards"]
    assert existing == original
    del existing["compiled_rewards"]["workers"]["A"]
    with pytest.raises(ValueError, match="execution IDs"):
        factory.GraphPRMProvider(Scorer()).score_rollout(**req)


def test_separate_runs_manifest_and_no_secret(tmp_path, monkeypatch):
    args = SimpleNamespace(reward_setting="handcrafted")
    factory.write_reward_manifest(args, tmp_path)
    factory.write_reward_manifest(args, tmp_path)
    args.reward_setting = "llm_coach"
    monkeypatch.setenv("REMA_COACH_MODEL", "coach")
    monkeypatch.setenv("REMA_COACH_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("REMA_COACH_API_KEY", "SECRET_KEY")
    with pytest.raises(ValueError, match="different reward configuration"):
        factory.write_reward_manifest(args, tmp_path)
    assert "SECRET_KEY" not in json.dumps(factory.reward_manifest(args))


def test_shared_orchestrator_node_credit_and_repeats():
    trainer = rema.HierarchicalGRPOTrainer(
        reward_provider=rules.HandcraftedRewardProvider(), train_worker_model=True,
        min_worker_grpo_group_size=1)
    rollout = trainer.run(
        task=rema.TaskExample("t", "Solve for x: 2x + 3 = 11.", "4",
                              metadata={"skill_focus": "algebra", "distractor_answer": "5"}),
        worker_pool=demo.make_worker_pool(base_model_path="mock"),
        policy_config=rema.ControllerPolicyConfig(parameter_sharing=True, shared_model_path="mock"),
        rollout_config=rema.RolloutConfig(num_decompositions=2, num_executor_rollouts_per_decomposition=3),
        schedule=rema.TrainingScheduleConfig(mode=rema.TrainingMode.JOINT))
    assert len(rollout.decompositions) == 2
    for plan in rollout.decompositions:
        assert len(plan.selections) == 3
        rewards = [s.reward_model_outputs["compiled_rewards"]["decomposer"]["reward"] for s in plan.selections]
        assert plan.base_decomposition_reward == pytest.approx(sum(rewards) / 3)
        for sel in plan.selections:
            assert sel.reward.reward_model_source == "handcrafted_v1"
            for ex in sel.executions:
                assert ex.reward_model_reward == sel.reward_model_outputs["compiled_rewards"]["workers"][ex.node_id]["reward"]
    assert rollout.training_batch.worker_samples


def test_launcher_only_selects_one_source():
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts/reward-setting.sh"
    for setting in ("gnn_prm", "handcrafted"):
        result = subprocess.run(["bash", "-c", 'source "$1"; printf "%s" "$REWARD_SETTING"', "_", str(script)],
                                env={**os.environ, "REWARD_SETTING": setting}, capture_output=True, text=True)
        assert result.returncode == 0
        assert result.stdout == setting
    result = subprocess.run(["bash", "-c", 'source "$1"', "_", str(script)],
                            env={**os.environ, "REWARD_SETTING": "frozen_prm"}, capture_output=True, text=True)
    assert result.returncode != 0
    assert "NotImplementedError" in result.stderr


def test_unscored_provider_cannot_fall_back_to_rules_or_partial_plan_mean():
    class FailedProvider:
        source, rollout_batch_size = "llm_coach", 4
        supports_verified_final_correctness = True
        def score_rollouts(self, requests):
            scored = rules.HandcraftedRewardProvider().score_rollouts(requests)
            scored[0] = {"status": "unscored", "error": {"code": "coach_request_failed"}}
            return scored
    trainer = rema.HierarchicalGRPOTrainer(reward_provider=FailedProvider(), train_worker_model=True,
                                          min_worker_grpo_group_size=1)
    rollout = trainer.run(
        task=rema.TaskExample("t", "Solve for x: 2x + 3 = 11.", "4",
                              metadata={"skill_focus": "algebra"}),
        worker_pool=demo.make_worker_pool(base_model_path="mock"),
        policy_config=rema.ControllerPolicyConfig(parameter_sharing=True, shared_model_path="mock"),
        rollout_config=rema.RolloutConfig(num_decompositions=1, num_executor_rollouts_per_decomposition=2),
        schedule=rema.TrainingScheduleConfig(mode=rema.TrainingMode.JOINT))
    plan = rollout.decompositions[0]
    assert plan.decomposition_reward is None
    assert plan.decomposer_advantage is None
    failed = plan.selections[0]
    assert failed.reward.total_reward is None
    assert all(ex.reward_model_reward is None for ex in failed.executions)


def test_manifest_detects_changed_checkpoint_at_same_path(tmp_path):
    checkpoint = tmp_path / "model.pkl"
    checkpoint.write_bytes(b"first-version")
    args = SimpleNamespace(reward_setting="gnn_prm", gfam_reward_model_pkl=str(checkpoint))
    factory.write_reward_manifest(args, tmp_path)
    checkpoint.write_bytes(b"second-version")
    with pytest.raises(ValueError, match="different reward configuration"):
        factory.write_reward_manifest(args, tmp_path)


def test_standalone_rollout_uses_provider_and_records_source(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["demo.py", "--backend", "mock", "--reward-setting", "handcrafted",
                                    "--num-decompositions", "1", "--num-executor-rollouts", "2",
                                    "--output-dir", str(tmp_path), "--print-mode", "none"])
    demo.main()
    manifest = json.loads((tmp_path / "reward_setting.json").read_text())
    assert manifest["reward_setting"] == "handcrafted"
    rows = [json.loads(line) for line in (tmp_path / "all_rollouts.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        for plan in row["rollout"]["decompositions"]:
            assert len(plan["selections"]) == 2
            for selection in plan["selections"]:
                assert selection["reward_model_outputs"]["source"] == "handcrafted_v1"
                assert selection["reward_model_outputs"]["reward_evidence"]

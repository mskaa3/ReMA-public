"""Notation recovery and independence of required versus diagnostic checks."""
import importlib
import json

import pytest

try:
    import verl.hierarchical_rema as api
except ModuleNotFoundError:
    import hierarchical_rema as api

rewarding = importlib.import_module(api.__name__ + ".rewarding")
backend = rewarding._math_verifier_module()


@pytest.mark.parametrize("prediction,reference", [
    ("(1 + sqrt(5))^8 + (1 - sqrt(5))^8", "12032"),
    ("24 * (6 * sqrt(2))^3", r"10368\sqrt{2}"),
    ("5 \u00b1 sqrt(23)", r"\{5+\sqrt{23},5-\sqrt{23}\}"),
    (r"5 \pm \sqrt{23}", r"\{5-\sqrt{23},5+\sqrt{23}\}"),
    ("1519/80 * 99^2 + 81.25 * 99 + c", "c+99*(1519*99/80+325/4)"),
])
def test_rejected_log_notation_is_verified_in_both_directions(prediction, reference):
    assert backend.compute_answer_score(prediction, reference) == 1.0
    assert backend.compute_answer_score(reference, prediction) == 1.0
    assert backend.validate_reference(prediction)


@pytest.mark.parametrize("prediction,reference", [
    ("(1+sqrt(5))^8+(1-sqrt(5))^8", "12033"),
    ("5 \u00b1 sqrt(23)", r"5+\sqrt{23}"),
    ("24*(6*sqrt(2))^3", r"10369\sqrt{2}"),
])
def test_notation_recovery_does_not_make_wrong_answers_correct(prediction, reference):
    assert backend.compute_answer_score(prediction, reference) == 0.0


@pytest.mark.parametrize("artifact", [
    r"\frac{}{2}", "sqrt()", "sqrt(2)+", "5 \u00b1 2 \u00b1 3", "2**(2**100)",
    "__import__('os').system('true')*1", "sqrt(1e10001)", "sqrt(0)+" + "1+" * 100 + "1",
])
def test_unsafe_or_incomplete_artifacts_remain_unknown(artifact):
    with pytest.raises(backend.AnswerVerificationError):
        backend.validate_reference(artifact)


def test_plain_fallback_is_one_attempt_not_generation():
    calls = []
    def empty_parser(text, **kwargs):
        calls.append((text, kwargs))
        return []
    parsed = backend._parse_expression("2+3", empty_parser, lambda: None)
    assert parsed[0].doit() == 5
    assert len(calls) == 1
    assert calls[0][1]["parsing_timeout"] == 0  # Existing parent deadline covers all attempts.


def test_identical_malformed_expressions_do_not_bypass_verification():
    with pytest.raises(backend.AnswerVerificationError):
        rewarding.compute_final_answer_correctness(r"\frac{}{2}", r"\frac{}{2}")


def worker():
    return api.WorkerExecution(node_id="1", worker_id="executor_worker", output_text="2+2",
                               entropy=0.0, confidence_reward=0.0, compatibility=0.0)


@pytest.mark.parametrize("required", [(), ("answer_containment",)])
def test_containment_unknown_is_not_no_violation(monkeypatch, required):
    execution = worker()
    monkeypatch.setattr(rewarding, "is_non_final_answer_leak", lambda *a, **k: False)
    original = rewarding.compute_final_answer_correctness
    def compare(prediction, reference, task_metadata=None):
        if prediction == execution.output_text:
            raise backend.AnswerVerificationError("deadline", "verification_timeout")
        return original(prediction, reference, task_metadata)
    monkeypatch.setattr(rewarding, "compute_final_answer_correctness", compare)
    reward = rewarding.build_selection_reward("4", "4", [execution], api.RewardWeights(),
                                              final_node_id="2", required_auxiliary_checks=required)
    assert reward.final_verification_status == "verified"
    assert reward.final_answer_correctness == 1.0
    assert reward.auxiliary_verification_status == "unverified"
    assert reward.verification_status == ("unverified" if required else "verified")
    assert execution.answer_containment is None
    assert reward.non_final_answer_containment_penalty is None
    assert reward.total_reward is None  # Never invent the legacy containment penalty.
    assert reward.auxiliary_verification_errors[0]["reference_role"] == "generated_final_answer"


def test_containment_checks_other_reference_after_unknown(monkeypatch):
    execution = worker()
    def compare(prediction, reference, task_metadata=None):
        if reference == "bad-final":
            raise backend.AnswerVerificationError("unsupported", "reference_parse_failed")
        return 1.0
    monkeypatch.setattr(rewarding, "compute_final_answer_correctness", compare)
    assert rewarding._check_answer_containment(execution, "bad-final", "4", {}) is None
    assert execution.answer_containment is True
    checks = execution.verification_checks["answer_containment"]["comparisons"]
    assert [item["status"] for item in checks] == ["unverified", "verified"]


def test_independent_provider_scores_and_audits_unknown_auxiliary(monkeypatch, tmp_path):
    demo = importlib.import_module(api.__name__ + ".demo")
    audit = importlib.import_module(api.__name__ + ".audit_answers")
    metrics = importlib.import_module(api.__name__ + ".metrics")
    def timeout(*args, **kwargs):
        raise backend.AnswerVerificationError("deadline", "verification_timeout")
    monkeypatch.setattr(rewarding, "is_non_final_answer_leak", timeout)
    class Independent:
        supports_verified_final_correctness = True
        required_auxiliary_checks = ()
        def score_rollout(self, **request):
            assert request["verified_final_correctness"] in (0, 1)
            return {"status": "scored", "compiled_rewards": {
                "decomposer": {"reward": .4}, "final": {"reward": .4},
                "workers": {ex.node_id: {"reward": .4} for ex in request["executions"]}}}
    trainer = api.HierarchicalGRPOTrainer(gfam_reward_scorer=Independent())
    rollout = trainer.run(
        task=api.TaskExample(task_id="task", prompt="Solve for x: 2x + 3 = 11.", ground_truth="4",
                             metadata={"skill_focus": "algebra", "distractor_answer": "5"}),
        worker_pool=demo.make_worker_pool(base_model_path="mock"),
        policy_config=api.ControllerPolicyConfig(shared_model_path="mock"),
        rollout_config=api.RolloutConfig(num_decompositions=1, num_executor_rollouts_per_decomposition=2),
        schedule=api.TrainingScheduleConfig(mode=api.TrainingMode.JOINT))
    for plan in rollout.decompositions:
        for selection in plan.selections:
            assert selection.reward.total_reward == .4
            assert selection.reward.verification_status == "verified"
            assert selection.reward.auxiliary_verification_status == "unverified"
            assert selection.executions[0].reference_answer_match is None
            assert selection.reward.to_dict()["auxiliary_verification_errors"]
    path = tmp_path / "verification_failures.jsonl"
    audit.record_verification_failures(path, [rollout])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["reward_verification_status"] == "verified" for row in rows)
    assert all(row["auxiliary_errors"] for row in rows)
    totals = metrics.rollout_monitoring_totals(rollout)
    assert totals["verified_rollouts"] == totals["auxiliary_unverified_rollouts"] == 2


def test_providers_explicitly_declare_auxiliary_requirements():
    for module, name in [("factory", "GraphPRMProvider"), ("handcrafted", "HandcraftedRewardProvider"),
                         ("coach", "CoachRewardProvider")]:
        provider = getattr(importlib.import_module(api.__name__ + ".reward_providers." + module), name)
        assert provider.required_auxiliary_checks == ()

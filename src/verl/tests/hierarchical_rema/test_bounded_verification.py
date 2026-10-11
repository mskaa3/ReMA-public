"""Deadlines must stop stuck processes, not merely return while they keep running."""
import importlib
import json
import sys
import time

import pytest

try:
    import verl.hierarchical_rema as api
except ModuleNotFoundError:
    import hierarchical_rema as api

rewarding = importlib.import_module(api.__name__ + ".rewarding")
backend = rewarding._math_verifier_module()
isolated = importlib.import_module(backend.__package__ + ".isolated_verifier")


@pytest.fixture
def fake_verifier(monkeypatch):
    code = '''
import json, sys, signal
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print(json.dumps({"status": "ready"}), flush=True)
for line in sys.stdin:
    operation = json.loads(line)["operation"]
    if operation == "hang":
        while True:
            pass
    if operation == "crash":
        sys.exit(3)
    print(json.dumps({"status": "ok", "value": 1.0}), flush=True)
'''
    monkeypatch.setenv("REMA_VERIFIER_TIMEOUT_SECONDS", "0.1")
    client = isolated.VerifierProcess([sys.executable, "-u", "-c", code], cache_size=2)
    yield client
    client.close()


def test_hung_process_is_killed_unknown_cached_and_next_request_recovers(fake_verifier):
    assert fake_verifier.call("first")["value"] == 1.0
    process = fake_verifier.process
    started = time.monotonic()
    result = fake_verifier.call("hang")
    assert time.monotonic() - started < 3
    assert result["status"] == "unverified"
    assert result["code"] == "verification_timeout"
    assert process.poll() is not None
    assert fake_verifier.process is None
    assert fake_verifier.call("hang") == result
    assert fake_verifier.process is None  # No repeated expensive timeout.
    assert fake_verifier.call("next")["value"] == 1.0
    assert fake_verifier.process.pid != process.pid
    assert len(fake_verifier.cache) == 2


def test_healthy_worker_is_reused(fake_verifier):
    fake_verifier.call("first")
    process = fake_verifier.process
    fake_verifier.call("second")
    assert fake_verifier.process is process


def test_worker_crash_is_not_fabricated_incorrectness(fake_verifier):
    with pytest.raises(RuntimeError, match="without a response"):
        fake_verifier.call("crash")
    assert fake_verifier.process is None


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_timeout_cannot_disable_safety(fake_verifier, monkeypatch, value):
    monkeypatch.setenv("REMA_VERIFIER_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="positive"):
        fake_verifier.call("first")


def test_real_process_still_verifies_equivalent_math_and_rejects_wrong_answers():
    client = isolated.VerifierProcess()
    try:
        assert client.call("compare", "0.5", r"\frac{1}{2}")["value"] == 1.0
        assert client.call("compare", "0.6", r"\frac{1}{2}")["value"] == 0.0
        assert client.call("validate_reference", r"\frac{}{2}")["status"] == "unverified"
    finally:
        client.close()


def test_real_parent_deadline_is_not_converted_to_false(monkeypatch):
    client = isolated.VerifierProcess()
    try:
        client.call("validate_reference", "4")  # Startup is not part of per-check budget.
        monkeypatch.setenv("REMA_VERIFIER_TIMEOUT_SECONDS", "0.000000001")
        monkeypatch.setattr(isolated, "verify_in_process", client.call)
        with pytest.raises(backend.AnswerVerificationError) as exc:
            backend.compute_answer_score("2+2", "4")
        assert exc.value.code == "verification_timeout"
        assert client.process is None
    finally:
        client.close()


def _worker(output="2+2"):
    return api.WorkerExecution(node_id="1", worker_id="executor_worker", output_text=output,
                               entropy=0.0, confidence_reward=0.0, compatibility=0.0)


def test_intermediate_check_is_retained_and_records_mathematical_match():
    worker = _worker()
    reward = rewarding.build_selection_reward("4", "4", [worker], api.RewardWeights(), final_node_id="2")
    assert reward.verification_status == "verified"
    assert worker.reference_answer_match is True
    assert worker.answer_containment is True
    assert worker.verification_checks["reference_answer_match"]["status"] == "verified"
    assert worker.output_text == "2+2"


@pytest.mark.parametrize("check", ["reference_answer_match", "answer_containment"])
@pytest.mark.parametrize("final_answer", ["4", "5"])
def test_auxiliary_timeout_preserves_correctness_but_excludes_reward(monkeypatch, check, final_answer):
    worker = _worker()
    original = rewarding.compute_final_answer_correctness

    def compare(prediction, reference, task_metadata=None):
        if prediction == worker.output_text:
            raise backend.AnswerVerificationError("deadline", "verification_timeout")
        return original(prediction, reference, task_metadata)

    monkeypatch.setattr(rewarding, "compute_final_answer_correctness", compare)
    if check == "answer_containment":
        monkeypatch.setattr(rewarding, "is_non_final_answer_leak", lambda *a, **k: False)
    reward = rewarding.build_selection_reward(final_answer, "4", [worker], api.RewardWeights(), final_node_id="2")
    assert reward.final_answer_correctness == float(final_answer == "4")
    assert reward.total_reward is None
    assert reward.verification_status == "unverified"
    assert reward.verification_error["stage"] == check
    assert getattr(worker, check) is None
    assert worker.verification_checks[check]["code"] == "verification_timeout"
    assert json.loads(json.dumps(worker.to_dict()))[check] is None
    assert worker.output_text == "2+2"  # Never oracle-redact the delivered artifact.


def test_final_timeout_is_unknown_not_zero(monkeypatch):
    def timeout(*args, **kwargs):
        raise backend.AnswerVerificationError("deadline", "verification_timeout")
    monkeypatch.setattr(rewarding, "compute_final_answer_correctness", timeout)
    reward = rewarding.build_selection_reward("2+2", "4", [], api.RewardWeights())
    assert reward.final_answer_correctness is None
    assert reward.total_reward is None
    assert reward.verification_error["code"] == "verification_timeout"


def test_auxiliary_timeout_cannot_reenter_model_scoring_or_policy_updates(monkeypatch, tmp_path):
    demo = importlib.import_module(api.__name__ + ".demo")
    audit = importlib.import_module(api.__name__ + ".audit_answers")

    def timeout(*args, **kwargs):
        raise backend.AnswerVerificationError("deadline", "verification_timeout")

    class NoFallback:
        supports_verified_final_correctness = True

        def score_rollout(self, **kwargs):
            pytest.fail("An incomplete required check cannot fall back to model rewards")

    monkeypatch.setattr(rewarding, "is_non_final_answer_leak", timeout)
    trainer = api.HierarchicalGRPOTrainer(
        train_worker_model=True, min_worker_grpo_group_size=2, gfam_reward_scorer=NoFallback())
    rollout = trainer.run(
        task=api.TaskExample(task_id="task", prompt="Solve for x: 2x + 3 = 11.", ground_truth="4",
                             metadata={"skill_focus": "algebra", "distractor_answer": "5"}),
        worker_pool=demo.make_worker_pool(base_model_path="mock"),
        policy_config=api.ControllerPolicyConfig(shared_model_path="mock"),
        rollout_config=api.RolloutConfig(num_decompositions=2, num_executor_rollouts_per_decomposition=4),
        schedule=api.TrainingScheduleConfig(mode=api.TrainingMode.JOINT),
    )
    assert not rollout.training_batch.worker_samples
    assert not rollout.training_batch.decomposer_samples
    for plan in rollout.decompositions:
        assert plan.decomposition_reward is None
        for selection in plan.selections:
            assert selection.reward.total_reward is None
            assert selection.reward.final_answer_correctness in (0.0, 1.0)
            assert selection.reward_model_outputs["status"] == "unscored"
            assert all(execution.reward_model_reward is None for execution in selection.executions)
    audit.record_verification_failures(tmp_path / "failures.jsonl", [rollout])
    failures = [json.loads(line) for line in (tmp_path / "failures.jsonl").read_text().splitlines()]
    assert len(failures) == 8
    for failure in failures:
        assert failure["final_answer_correctness"] in (0.0, 1.0)
        assert failure["error"]["stage"] == "reference_answer_match"
        assert failure["execution_checks"][0]["output_text"]

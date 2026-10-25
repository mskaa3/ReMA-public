"""A polynomial conversion failure is an unknown check, not a failed job/verdict."""
import importlib
import sys
from textwrap import dedent

import pytest
from sympy import Integer, Symbol
from sympy.core.relational import Relational
from sympy.polys.polyerrors import CoercionFailed, GeneratorsNeeded, PolynomialError

try:
    import verl.hierarchical_rema as api
except ModuleNotFoundError:
    import hierarchical_rema as api

rewarding = importlib.import_module(api.__name__ + ".rewarding")
backend = rewarding._math_verifier_module()
isolated = importlib.import_module(backend.__package__ + ".isolated_verifier")


@pytest.mark.parametrize("error", [GeneratorsNeeded, CoercionFailed, PolynomialError])
@pytest.mark.parametrize("stage", ["set_conversion", "reference_parse", "prediction_parse", "comparison"])
def test_polynomial_failures_become_typed_unknowns(monkeypatch, error, stage):
    def fail(*args, **kwargs):
        raise error("unsupported polynomial representation")

    with pytest.raises(backend.AnswerVerificationError) as caught:
        if stage == "set_conversion":
            monkeypatch.setattr(Relational, "as_set", fail)
            backend._real_solution_set(Symbol("x") > 0)
        elif stage.endswith("_parse"):
            backend._parse_for_verification("4", fail, lambda: None, stage.split("_")[0])
        else:
            monkeypatch.setattr(backend, "_verifier", lambda: (
                lambda *args, **kwargs: [Integer(4)], fail, lambda: None))
            backend._compute_answer_score("2+2", "4")
    assert caught.value.code == stage + "_failed"
    assert error.__name__ in str(caught.value)
    assert isinstance(caught.value.__cause__, error)


def inject_conversion_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise GeneratorsNeeded("Cannot initialize from 'list' without generators")

    monkeypatch.setattr(Relational, "as_set", fail)
    # Keep the real parser and scoring pipeline; inject only the SymPy failure.
    # The process-level test below separately exercises the production IPC path.
    monkeypatch.setattr(backend, "_isolated_call",
                        lambda operation, *args: backend._dispatch_verification(operation, args))


@pytest.mark.parametrize("final_answer,correct", [("[0,1]", 1.0), ("[-1,1]", 0.0)])
@pytest.mark.parametrize("required", [(), ("answer_containment",)])
def test_containment_conversion_failure_preserves_final_verdict(monkeypatch, final_answer, correct, required):
    inject_conversion_failure(monkeypatch)
    execution = api.WorkerExecution(
        node_id="1", worker_id="executor_worker", output_text="x > 0",
        entropy=0.0, confidence_reward=0.0, compatibility=0.0)
    reward = rewarding.build_selection_reward(
        final_answer, "[0,1]", [execution], api.RewardWeights(),
        final_node_id="2", required_auxiliary_checks=required)

    assert reward.final_verification_status == "verified"
    assert reward.final_answer_correctness == correct
    assert reward.auxiliary_verification_status == "unverified"
    assert reward.verification_status == ("unverified" if required else "verified")
    assert reward.total_reward is None  # Legacy reward cannot invent a missing containment cost.
    assert execution.output_text == "x > 0"
    assert execution.answer_containment is None
    assert execution.reference_answer_match is None
    check = execution.verification_checks["answer_containment"]
    assert check["code"] == "set_conversion_failed"
    assert all(item["status"] == "unverified" for item in check["comparisons"])
    assert "GeneratorsNeeded" in check["message"]


@pytest.mark.parametrize("prediction,reference,stage", [
    ("x > 0", "[0,1]", "final_correctness"),
    ("[0,1]", r"\{x\mid x>0\}", "reference_validation"),
])
def test_required_conversion_failure_never_produces_a_reward(monkeypatch, prediction, reference, stage):
    inject_conversion_failure(monkeypatch)
    reward = rewarding.build_selection_reward(prediction, reference, [], api.RewardWeights())
    assert reward.verification_status == "unverified"
    assert reward.final_answer_correctness is None
    assert reward.total_reward is None
    assert reward.verification_error["stage"] == stage
    assert reward.verification_error["code"] == "set_conversion_failed"


@pytest.mark.parametrize("error", [MemoryError, ImportError, OSError, RuntimeError])
def test_non_symbolic_failures_still_surface(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error("infrastructure or unexpected failure")

    monkeypatch.setattr(Relational, "as_set", fail)
    with pytest.raises(error, match="infrastructure or unexpected"):
        backend._real_solution_set(Symbol("x") > 0)


def test_real_verifier_process_survives_polynomial_failure_and_continues():
    # Run the production server with one third-party method fault-injected.
    command = dedent("""
        import runpy, sys
        from unittest.mock import patch
        from sympy.polys.polyerrors import GeneratorsNeeded
        with patch('sympy.core.relational.Relational.as_set', side_effect=GeneratorsNeeded(
                "Cannot initialize from 'list' without generators")):
            runpy.run_path(sys.argv[1], run_name='__main__')
    """)
    client = isolated.VerifierProcess([sys.executable, "-u", "-c", command, isolated.__file__])
    try:
        response = client.call("compare", "x > 0", "[0,1]")
        assert response["status"] == "unverified"
        assert response["code"] == "set_conversion_failed"
        assert "GeneratorsNeeded" in response["message"]
        assert "value" not in response
        process = client.process
        assert process.poll() is None
        assert client.call("compare", "x > 0", "[0,1]") == response  # Cached unknown.
        assert client.call("compare", "0.5", r"\frac{1}{2}")["value"] == 1.0
        assert client.call("compare", "0.6", r"\frac{1}{2}")["value"] == 0.0
        assert client.process is process
        assert process.poll() is None
    finally:
        client.close()

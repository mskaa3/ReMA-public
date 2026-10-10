"""Data failures are auditable unknowns, never fabricated training rewards."""
import importlib
import json
import math
import sys
from threading import Event

import pytest

try:
    import verl.hierarchical_rema as api
except ModuleNotFoundError:
    import hierarchical_rema as api

rewarding = importlib.import_module(api.__name__ + ".rewarding")
train = importlib.import_module(api.__name__ + ".train")
demo = importlib.import_module(api.__name__ + ".demo")
audit = importlib.import_module(api.__name__ + ".audit_answers")
progress_module = importlib.import_module(api.__name__ + ".progress")


@pytest.mark.parametrize("prediction,reference,expected", [
    ("[-2,1)", r"{x|-2\lex<1}", 1.0),
    ("(-2,1)", r"{x|-2\lex<1}", 0.0),
    ("[-2,1]", r"{x|-2\lex<1}", 0.0),
    ("-2", r"{x|-2\lex<1}", 0.0),
    (r"-2\le x<1", r"{x|-2\lex<1}", 1.0),
    (r"\{x\mid -2\leq x<1\}", "[-2,1)", 1.0),
    (r"\frac{\pi}{4}+2-\sqrt{2}", r"\frac \pi4 + 2 - \sqrt2", 1.0),
    (r"2007+\pi/2", r"2007 + \frac\pi 2", 1.0),
    (r"2007+\pi/3", r"2007 + \frac\pi 2", 0.0),
])
def test_supported_reference_notation(prediction, reference, expected):
    assert rewarding.compute_final_answer_correctness(prediction, reference) == expected
    assert rewarding.reference_verification_status(reference)["status"] == "parseable"


@pytest.mark.parametrize("reference", ["", "None", r"\frac{}{2}", r"{x|3k,k\in\mathbb{Z}}"])
def test_bad_reference_becomes_unverified_even_on_exact_string_match(reference):
    reward = rewarding.build_selection_reward(reference, reference, [], api.RewardWeights())
    assert reward.total_reward is None
    assert reward.final_answer_correctness is None
    assert reward.verification_status == "unverified"
    assert reward.verification_error["code"] in {"missing_reference", "reference_parse_failed"}
    assert json.loads(json.dumps(reward.to_dict(), allow_nan=False))["final_answer_correctness"] is None


def test_wrong_prediction_with_valid_reference_is_still_incorrect():
    reward = rewarding.build_selection_reward("5", "4", [], api.RewardWeights())
    assert reward.verification_status == "verified"
    assert reward.final_answer_correctness == 0.0
    assert reward.total_reward is not None


@pytest.mark.parametrize('prediction,reference,expected', [
    (r'-2\le\sin(x)\le1', r'(-\infty,\infty)', 1.),
    (r'-2\le\sin(x)\le1', r'[-1,1]', 0.),
    (r'(-\infty,\infty)', r'\{x\mid -2\le\sin(x)\le1\}', 1.),
    (r'0\le x^2<1', '(-1,1)', 1.),
    (r'0\le x^2<1', '[-1,1]', 0.),
    (r'0<x^2<1', '(-1,1)', 0.),
])
def test_compound_inequality_conversion_avoids_latex2sympy_metadata_bug(prediction, reference, expected):
    reward = rewarding.build_selection_reward(prediction, reference, [], api.RewardWeights())
    assert reward.verification_status == 'verified'
    assert reward.final_answer_correctness == expected


def test_nested_parser_booleans_are_rebuilt_without_changing_logic():
    from latex2sympy2_extended.logic import And as ParserAnd
    from sympy import And, Or, Symbol, Interval
    x = Symbol('x', real=True)
    expression = Or(ParserAnd(x > -2, x < -1), ParserAnd(x > 1, x < 2))
    backend = rewarding._math_verifier_module()
    rebuilt = backend._native_boolean(expression)
    assert all(type(arg) is And for arg in rebuilt.args)
    assert backend._real_solution_set(expression) == Interval.open(-2, -1) | Interval.open(1, 2)


@pytest.mark.parametrize('error', [AttributeError, NotImplementedError, RecursionError])
def test_failed_symbolic_conversion_is_unknown_not_incorrect(monkeypatch, error):
    backend = rewarding._math_verifier_module()
    def broken(expression):
        raise error('unsupported symbolic expression')
    monkeypatch.setattr(backend, '_native_boolean', broken)
    reward = rewarding.build_selection_reward(r'-2\le x<1', '[-2,1)', [], api.RewardWeights())
    assert reward.verification_status == 'unverified'
    assert reward.verification_error['code'] == 'set_conversion_failed'
    assert reward.final_answer_correctness is None
    assert reward.total_reward is None


@pytest.mark.parametrize('role', ['reference', 'prediction', 'comparison'])
def test_external_symbolic_failures_have_auditable_error_codes(monkeypatch, role):
    backend = rewarding._math_verifier_module()
    parse, verify, config = backend._verifier()
    def guarded_parse(text, **kwargs):
        target = '$4$' if role == 'reference' else '$2+2$'
        if role != 'comparison' and text == target:
            raise AttributeError('third-party parser failure')
        return parse(text, **kwargs)
    def guarded_verify(*args, **kwargs):
        if role == 'comparison':
            raise AttributeError('third-party comparison failure')
        return verify(*args, **kwargs)
    monkeypatch.setattr(backend, '_verifier', lambda: (guarded_parse, guarded_verify, config))
    reward = rewarding.build_selection_reward('2+2', '4', [], api.RewardWeights())
    assert reward.verification_status == 'unverified'
    assert reward.verification_error['code'] == ('comparison_failed' if role == 'comparison' else f'{role}_parse_failed')
    assert reward.total_reward is None


def test_missing_dependency_is_not_a_recoverable_data_failure(monkeypatch):
    def unavailable():
        raise RuntimeError("install math-verify")
    monkeypatch.setattr(rewarding._math_verifier_module(), "_verifier", unavailable)
    with pytest.raises(RuntimeError, match="install math-verify"):
        rewarding.build_selection_reward("4", "4", [], api.RewardWeights())


def test_unexpected_verifier_bug_is_not_hidden(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("verifier bug")
    monkeypatch.setattr(rewarding, "compute_final_answer_correctness", broken)
    with pytest.raises(RuntimeError, match="verifier bug"):
        rewarding.build_selection_reward("4", "4", [], api.RewardWeights())


def make_rollout(reference="4", scorer=None):
    trainer = api.HierarchicalGRPOTrainer(
        train_worker_model=True, min_worker_grpo_group_size=2, gfam_reward_scorer=scorer,
    )
    return trainer.run(
        task=api.TaskExample(task_id="task-1", prompt="Solve for x: 2x + 3 = 11.",
                             ground_truth=reference,
                             metadata={"skill_focus": "algebra", "distractor_answer": "5"}),
        worker_pool=demo.make_worker_pool(base_model_path="mock"),
        policy_config=api.ControllerPolicyConfig(shared_model_path="mock"),
        rollout_config=api.RolloutConfig(num_decompositions=2, num_executor_rollouts_per_decomposition=4),
        schedule=api.TrainingScheduleConfig(mode=api.TrainingMode.JOINT),
    )


def test_unverified_rollout_does_not_invoke_reward_model_or_train():
    class NoModelFallback:
        supports_verified_final_correctness = True

        def score_rollout(self, **kwargs):
            pytest.fail("Unverified correctness must not fall back to predicted success")

    rollout = make_rollout(r"\frac{}{2}", NoModelFallback())
    assert not rollout.training_batch.worker_samples
    assert not rollout.training_batch.decomposer_samples
    assert all(plan.decomposition_reward is None for plan in rollout.decompositions)
    for plan in rollout.decompositions:
        for selection in plan.selections:
            assert selection.reward_model_outputs["status"] == "unscored"
            assert selection.reward_model_outputs["compiled_rewards"] == {}
            assert all(execution.reward_model_reward is None for execution in selection.executions)


def test_accuracy_excludes_unknowns_and_reports_coverage_across_batches():
    valid, unknown = make_rollout(), make_rollout(r"\frac{}{2}")
    known_summary = train.epoch_rollout_summary([valid], include_subsets=True)
    unknown_summary = train.epoch_rollout_summary([unknown], include_subsets=True)
    assert math.isnan(unknown_summary["mean_best_final_correctness"])
    assert unknown_summary["verification_coverage"] == 0.0
    assert unknown_summary["unverified_rollouts"] == 8
    combined = train.combine_rollout_summaries(
        json.loads(json.dumps(train._json_safe([known_summary, unknown_summary]), allow_nan=False)),
        include_subsets=True,
    )
    direct = train.epoch_rollout_summary([valid, unknown], include_subsets=True)
    for result in (combined, direct):
        assert result["num_tasks"] == 2
        assert result["num_verified_tasks"] == 1
        assert result["num_unverified_tasks"] == 1
        assert result["verification_coverage"] == 0.5
        assert result["mean_best_final_correctness"] == known_summary["mean_best_final_correctness"]
        assert result["verified_rollouts"] == result["unverified_rollouts"] == 8
        subset = next(iter(result["subsets"].values()))
        assert subset["verification_coverage"] == 0.5


def test_partially_verified_best_of_n_is_not_reported_as_full_accuracy():
    rollout = make_rollout()
    rollout.decompositions[0].selections[0].reward.final_answer_correctness = None
    summary = train.epoch_rollout_summary([rollout])
    assert summary["num_verified_tasks"] == 0
    assert summary["unverified_rollouts"] == 1
    assert math.isnan(summary["mean_best_final_correctness"])


def test_audit_and_failure_artifacts_are_json_safe(tmp_path):
    rollout = make_rollout(r"\frac{}{2}")
    report = audit.audit_tasks([rollout.task], source="test")
    audit.write_audit(tmp_path / "audit.json", [report])
    assert json.loads((tmp_path / "audit.json").read_text())["datasets"][0]["counts"] == {"unverified": 1}
    audit.record_verification_failures(tmp_path / "failures.jsonl", [rollout])
    failures = [json.loads(line) for line in (tmp_path / "failures.jsonl").read_text().splitlines()]
    assert len(failures) == 8
    assert all(item["reference"] == r"\frac{}{2}" for item in failures)


def test_unverified_training_is_skipped_before_loading_models(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(json.dumps({"question": "Solve for x: 2x + 3 = 11.", "answer": r"\frac{}{2}"}) + "\n")
    monkeypatch.setattr(train, "run_offline_policy_training", lambda **kw: pytest.fail("No unverified training"))
    monkeypatch.setattr(train, "make_worker_pool", lambda **kw: pytest.fail("No model setup for empty training"))
    monkeypatch.setattr(sys, "argv", ["train.py", "--task-source", str(dataset),
                                    "--val-task-source", str(dataset), "--backend", "mock",
                                    "--num-epochs", "1",
                                    "--rollout-progress-every", "1", "--output-dir", str(tmp_path / "run")])
    train.main()
    summary = json.loads((tmp_path / "run/training_summary.json").read_text())
    assert summary["status"] == "skipped_no_usable_training_tasks"
    assert summary["epochs"] == []
    assert not (tmp_path / "run/epoch_0001").exists()
    reports = json.loads((tmp_path / "run/answer_reference_audit.json").read_text())["datasets"]
    for report in reports:
        assert report["filtering"]["skipped_rows"] == 1
        assert report["filtering"]["kept_rows"] == 0
        assert report["issues"][0]["action"] == "skipped_before_rollout"
    assert "No usable training examples remain" in capsys.readouterr().out


def test_filter_uses_row_positions_and_retains_unaudited_sources():
    tasks = [api.TaskExample(task_id="duplicate", prompt="Compute 2+2", ground_truth=ref)
             for ref in ("4", "", r"\frac{}{2}")]
    tasks.append(api.TaskExample(task_id="other", prompt="Compute 2+2", ground_truth="4",
                                 metadata={"data_source": "openai/gsm8k"}))
    report = audit.audit_tasks(tasks, source="test")
    kept = audit.skip_unverified_tasks(tasks, report, split="train")
    assert kept == [tasks[0], tasks[3]]
    assert len(tasks) == 4
    assert report["filtering"]["retained_fraction"] == 0.5
    assert "action" not in report["issues"][-1]


@pytest.mark.parametrize("valid_validation", [True, False])
def test_bad_references_never_reach_training_or_validation_rollouts(
    tmp_path, monkeypatch, capsys, valid_validation,
):
    good = {"question": "Solve for x: 2x + 3 = 11.", "answer": "4", "id": "good"}
    bad = [{"question": "Must never be generated", "answer": ref, "id": f"bad-{i}"}
           for i, ref in enumerate(("", "None", r"\frac{}{2}"))]
    train_path, val_path = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train_text = "\n".join(json.dumps(row) for row in [good, *bad]) + "\n"
    val_text = "\n".join(json.dumps(row) for row in ([good] if valid_validation else []) + bad) + "\n"
    train_path.write_text(train_text)
    val_path.write_text(val_text)
    seen = []
    original_run_many = api.HierarchicalGRPOTrainer.run_many

    def checked_run_many(self, *, tasks, **kwargs):
        assert [task.task_id for task in tasks] == ["good"]
        assert all(task.ground_truth == "4" for task in tasks)
        seen.append(kwargs.get("progress_label", ""))
        return original_run_many(self, tasks=tasks, **kwargs)

    monkeypatch.setattr(api.HierarchicalGRPOTrainer, "run_many", checked_run_many)
    monkeypatch.setattr(train, "run_offline_policy_training", lambda **kw: {
        "steps": 1, "output_dir": str(kw["config"].output_dir),
        "num_train_samples": len(kw["train_samples"]), "num_val_samples": len(kw["val_samples"]),
    })
    monkeypatch.setattr(sys, "argv", ["train.py", "--task-source", str(train_path),
                                    "--val-task-source", str(val_path), "--backend", "mock",
                                    "--num-epochs", "1", "--rollout-progress-every", "1",
                                    "--output-dir", str(tmp_path / "run")])
    train.main()
    assert len(seen) == (2 if valid_validation else 1)
    summary = json.loads((tmp_path / "run/training_summary.json").read_text())
    assert summary["num_loaded_tasks"] == 1
    assert summary["num_loaded_val_tasks"] == int(valid_validation)
    assert summary["reference_filtering"]["train"]["skipped_rows"] == 3
    assert summary["reference_filtering"]["validation"]["skipped_rows"] == 3
    rollout = summary["epochs"][0]["rollout_summary"]
    assert rollout["num_tasks"] == 1
    assert rollout["unverified_rollouts"] == 0
    assert not (tmp_path / "run/epoch_0001/verification_failures.jsonl").exists()
    validation_path = tmp_path / "run/epoch_0001/validation_summary.json"
    if valid_validation:
        assert json.loads(validation_path.read_text())["num_tasks"] == 1
    else:
        assert not validation_path.exists()
    assert train_path.read_text() == train_text
    assert val_path.read_text() == val_text
    output = capsys.readouterr().out
    assert "epoch=1/1 phase=joint" in output
    assert "stage=reward_scoring" in output
    assert ("phase=validation" in output) == valid_validation


def test_missing_distribution_metadata_does_not_block_audit(tmp_path, monkeypatch):
    def absent(name):
        raise audit.PackageNotFoundError(name)

    monkeypatch.setattr(audit, "version", absent)
    audit.write_audit(tmp_path / "audit.json", [])
    assert set(json.loads((tmp_path / "audit.json").read_text())["packages"].values()) == {
        "distribution metadata unavailable"
    }


def test_heartbeat_runs_while_scoring_and_stops_on_failure(monkeypatch, capsys):
    reporter = progress_module.StageProgress("reward_scoring", 10, context="epoch=5/8", interval=0.01)
    heartbeat = Event()
    original = reporter._emit

    def emit(status):
        original(status)
        if status == "running":
            heartbeat.set()

    monkeypatch.setattr(reporter, "_emit", emit)
    with pytest.raises(ValueError, match="test"):
        with reporter:
            reporter.advance(unscored=True, unverified=True)
            assert heartbeat.wait(2)
            raise ValueError("test")
    assert not reporter._thread.is_alive()
    output = capsys.readouterr().out
    assert "epoch=5/8" in output
    assert "completed=1/10 unscored=1 unverified=1" in output
    assert "status=failed" in output


def test_quiet_feature_logging_preserves_builder_and_restores_context(capsys):
    logging = importlib.import_module(api.__name__ + ".graphprm_v2.live_logging")
    features = logging.features
    before = features.DEFAULT_SPEC["feature_code_sha256"]
    with pytest.raises(ValueError):
        with logging.feature_progress():
            assert list(features.tqdm([1, 2], desc="quiet-test")) == [1, 2]
            features.print("[features] graphs=1 unique_uncached=1")
            features.print("important error remains visible")
            raise ValueError("restore context")
    features.print("[features] graphs=2 unique_uncached=2")
    output = capsys.readouterr()
    assert "graphs=1" not in output.out
    assert "quiet-test" not in output.err
    assert "important error remains visible" in output.out
    assert "graphs=2" in output.out
    assert features.DEFAULT_SPEC["feature_code_sha256"] == before

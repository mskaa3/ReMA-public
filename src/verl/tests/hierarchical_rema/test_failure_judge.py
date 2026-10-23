"""Result-only audit safety, batching, exports and validation integration (no GPU)."""
from contextlib import contextmanager
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace as NS

import pytest

try:
    import verl.hierarchical_rema as api
except ModuleNotFoundError:
    import hierarchical_rema as api

judge = importlib.import_module(api.__name__ + ".failure_judge")
contract = importlib.import_module(api.__name__ + ".failure_judge_contract")
runtime = importlib.import_module(api.__name__ + ".failure_judge_runtime")


def annotation(value=0):
    return {"categories": {code: {"value": value, "status": "assessed", "explanation": "Visible artifacts checked.",
                                  "evidence": []} for code in contract.CATEGORIES}}


def test_prompt_schema_and_source_crosswalk_use_consecutive_ids():
    ids = [str(index) for index in range(1, 9)]
    assert list(contract.CATEGORIES) == ids
    assert re.findall(r"^([1-8])\. ", contract.PROMPT, re.MULTILINE) == ids
    schema = contract.OUTPUT_SCHEMA["properties"]["categories"]
    assert schema["required"] == ids
    assert list(schema["properties"]) == ids
    assert contract.SOURCE_CATEGORY_IDS == dict(zip(ids, ["1.1", "1.2", "1.3", "1.4", "2.3", "2.4", "2.5", "3.1"]))
    assert not re.search(r"\b[123]\.[1-5]\b", contract.PROMPT)
    assert '"categories":{"1":' in contract.PROMPT
    assert 'Use category IDs "1" through "8" as JSON keys.' in contract.PROMPT
    assert "For any category assigned value 1" in contract.PROMPT
    assert "information never delivered -> 6" in contract.PROMPT
    assert "downstream input -> 4; received but visibly disregarded -> 7" in contract.PROMPT


def test_intro_explains_roles_without_requiring_hidden_reasoning():
    assert contract.PROMPT.startswith("You audit a multi-agent trajectory for failures.\nThe system has a planner/decomposer")
    intro = contract.PROMPT.split("Assess only the supplied task", 1)[0]
    for phrase in ("directed acyclic graph (DAG)", "not compute unknown",
                   "An atomic task need not be split", "only their assigned node instructions",
                   "Only root workers receive the full task", "Only delivered result artifacts",
                   "required format", "unrequested\nexplanations"):
        assert phrase in intro
    assert "reference answers are intentionally absent" in contract.PROMPT


def test_old_category_annotations_are_rejected_under_new_version():
    assert contract.VERSION == "artifact_failures_v2"
    old = {"categories": {contract.SOURCE_CATEGORY_IDS[code]: value
                           for code, value in annotation()["categories"].items()}}
    with pytest.raises(ValueError, match="exactly eight category IDs"):
        contract.validate_annotation(old, {})


def fixture():
    node = NS(node_id="A", instruction="Return x < 3 and y > 2", dependencies=[])
    final = NS(node_id="B", instruction="Using A, return the requested quantity.", dependencies=["A"])
    plan = NS(decomposition_id="plan1", target_quantity="quantity", final_answer_format_hint="integer",
              nodes=[node, final], final_node_id="B", raw_text="<think>SECRET_PLAN</think>",
              raw_payload={"ground_truth": "SECRET_REFERENCE"})
    outputs = [NS(node_id="A", worker_id="executor", output_text="<worker_scratchpad>SECRET_THOUGHT</worker_scratchpad>x < 3 and y > 2",
                  raw_output_text="SECRET_RAW", dependency_outputs={}, reward_model_reward=.99),
               NS(node_id="B", worker_id="executor", output_text="", raw_output_text="SECRET_REJECTED",
                  dependency_outputs={"A": "<think>SECRET_SNAPSHOT</think>x < 3 and y > 2"})]
    selection = NS(selection=NS(selection_id="exec1"), executions=outputs, final_answer="",
                   reward=NS(final_answer_correctness=0), reward_model_outputs={"SECRET_SCORES": 1})
    plan_rollout = NS(decomposition=plan, selections=[selection])
    rollout = NS(task=NS(task_id="t1", prompt="Find a quantity.", ground_truth="SECRET_REFERENCE"),
                 decompositions=[plan_rollout], rollout_config=api.RolloutConfig())
    pool = NS(workers=[NS(worker_id="executor", system_prompt="Execute the assigned operation.")])
    return rollout, pool


def worker_module(monkeypatch):
    root = Path(judge.__file__).parent
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("isolated_failure_judge_worker", root / "failure_judge_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def test_projection_is_result_only_without_reference_or_reward_leakage():
    rollout, pool = fixture()
    plan, = rollout.decompositions
    document = judge.project_trajectory(rollout, plan, plan.selections[0], pool)
    assert "SECRET" not in json.dumps(document)
    assert document["executions"][0]["output"] == "x < 3 and y > 2"
    assert document["executions"][1]["output"] == ""
    assert document["executions"][1]["dependency_inputs"]["A"] == "x < 3 and y > 2"
    assert document["executions"][1]["task_context_expected"] is False
    assert document["plan"]["nodes"][1]["dependencies"] == ["A"]


def test_private_blocks_and_unclosed_blocks_never_escape():
    assert contract.artifact_text("answer <analysis>secret") == "answer"
    assert contract.artifact_text("<THINK>secret</THINK> 41; 900; 41; 900") == "41; 900; 41; 900"
    assert contract.artifact_text("a < b and c > d") == "a < b and c > d"
    assert contract.artifact_text("<think>secret<think>nested</think>still secret</think>answer") == "answer"
    assert contract.artifact_text("<think>secret<analysis>nested</analysis>still secret</think>answer") == "answer"


def test_annotation_evidence_and_unknown_contract():
    doc = {"executions": [{"output": ""}, {"output": "value=5"}]}
    result = annotation()
    result["categories"]["8"].update(value=1, evidence=[{"path": "/executions/0/output", "quote": ""}])
    result["categories"]["4"].update(value=None, status="insufficient_evidence")
    assert contract.parse_annotation(json.dumps(result), doc) == result
    result["categories"]["8"]["evidence"][0]["quote"] = "invented"
    with pytest.raises(ValueError, match="not found"):
        contract.validate_annotation(result, doc)


@pytest.mark.parametrize("code", ["4", "6", "7"])
def test_flow_categories_require_both_sides(code):
    result = annotation()
    result["categories"][code].update(value=1, evidence=[{"path": "/output", "quote": "a"}])
    with pytest.raises(ValueError, match="both sides"):
        contract.validate_annotation(result, {"output": "a"})


@pytest.mark.parametrize("mutation", ["missing", "bool", "unknown_zero", "positive_without_evidence"])
def test_malformed_annotations_rejected(mutation):
    result = annotation()
    if mutation == "missing":
        del result["categories"]["4"]
    elif mutation == "bool":
        result["categories"]["1"]["value"] = True
    elif mutation == "unknown_zero":
        result["categories"]["1"]["status"] = "insufficient_evidence"
    else:
        result["categories"]["1"]["value"] = 1
    with pytest.raises(ValueError):
        contract.validate_annotation(result, {})


def test_duplicate_json_keys_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        contract.parse_annotation('{"categories":{},"categories":{}}', {})


def test_disabled_has_no_model_dependency_or_projection(monkeypatch, tmp_path):
    monkeypatch.setenv("REMA_MAST_ENABLED", "false")
    monkeypatch.setenv("REMA_MAST_BATCH_SIZE", "not-a-number")
    audit = judge.FailureJudgeAudit(None)
    audit.add(None, "unused")
    report = audit.finish(tmp_path, runner=lambda *a: pytest.fail("Disabled judge ran"))
    assert not report["metadata"]["enabled"]
    assert report["rows"] == []


def test_overflow_is_per_record_and_never_truncates(monkeypatch):
    worker = worker_module(monkeypatch)
    seen, emitted = [], []
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen.append(kwargs)
            return list(range(json.loads(messages[1]["content"])["length"]))
    requests = worker.build_requests({"long": {"length": 20}, "ok": {"length": 5}}, Tokenizer(),
                                     {"max_model_len": 15, "max_tokens": 10}, emitted.append)
    assert requests[0][0] == "ok"
    assert requests[0][2] == list(range(5))
    assert emitted[0]["status"] == "context_overflow"
    assert all(item["enable_thinking"] is False for item in seen)


def test_failed_batch_splits_and_failed_annotations_retry_without_duplicates(monkeypatch):
    worker = worker_module(monkeypatch)
    calls, emitted = [], []
    def generate(prompts):
        ids = [p["prompt_token_ids"][0] for p in prompts]
        calls.append(ids)
        if len(ids) > 2:
            raise RuntimeError("batch too large")
        return [{"text": "bad json" if key == 2 else json.dumps(annotation()), "finish_reason": "stop"}
                for key in ids]
    requests = [(str(i), {}, [i]) for i in range(4)]
    worker.judge_batches(requests, generate, 4, 1, emitted.append)
    assert len(emitted) == 4
    assert len({row["input_id"] for row in emitted}) == 4
    by_id = {row["input_id"]: row for row in emitted}
    assert by_id["2"]["status"] == "invalid_annotation"
    assert by_id["2"]["attempts"] == 2
    assert all(by_id[key]["status"] == "ok" for key in ("0", "1", "3"))
    assert calls[-1] == [2]


def test_incomplete_generation_is_not_success(monkeypatch):
    worker = worker_module(monkeypatch)
    results = []
    worker.judge_batches([("a", {}, [1])], lambda p: [{"text": json.dumps(annotation()), "finish_reason": "length"}],
                         4, 0, results.append)
    assert results[0]["status"] == "invalid_annotation"


def test_aggregation_excludes_unavailable_and_equal_weights_tasks():
    def row(task, value=None):
        result = {"task_id": task, "dataset": "math", "status": "ok" if value is not None else "unavailable"}
        if value is not None:
            result["annotation"] = annotation(value)
        return result
    report = judge.aggregate([row("A", 1), row("A", 1), row("B", 0), row("C")])
    label = report["categories"]["1"]
    assert label["rate"] == .5  # Not 2/3 and not 2/4.
    assert label["coverage"] == .75
    assert label["assessed_tasks"] == 2
    assert judge.aggregate([row("C")])["categories"]["1"]["rate"] is None


def test_duplicate_content_restores_each_trajectory_id_and_isolates_failures(tmp_path):
    rollout, pool = fixture()
    audit = judge.FailureJudgeAudit(pool, judge.FailureJudgeConfig(enabled=True))
    audit.add(rollout, "math")
    rollout.decompositions[0].selections[0].selection.selection_id = "exec2"
    audit.add(rollout, "math")
    def runner(config, documents, path):
        assert len(documents) == 1
        return {"results": {key: {"status": "ok", "annotation": annotation()} for key in documents}}
    report = audit.finish(tmp_path, runner)
    assert [row["execution_id"] for row in report["rows"]] == ["exec1", "exec2"]
    assert report["overall"]["categories"]["4"]["assessed_rollouts"] == 2
    audit = judge.FailureJudgeAudit(pool, judge.FailureJudgeConfig(enabled=True))
    audit.add(rollout, "math")
    def unavailable(*args):
        raise RuntimeError("unsupported model")
    report = audit.finish(tmp_path, unavailable)
    assert report["rows"][0]["status"] == "unavailable"
    assert report["overall"]["categories"]["1"]["rate"] is None


def test_coach_handoff_wakes_on_judge_failure():
    state, calls = {"sleeping": False}, []
    def request(base, token, path, post=False):
        calls.append(path)
        if path.startswith("/sleep"):
            state["sleeping"] = True
        if path == "/wake_up":
            state["sleeping"] = False
        return {"is_sleeping": state["sleeping"]}
    env = {"REMA_COACH_BACKEND": "local_vllm", "REMA_COACH_BASE_URL": "http://127.0.0.1:1234/v1"}
    with pytest.raises(ValueError, match="judge failed"):
        with runtime.pause_local_coach(env, request):
            assert state["sleeping"]
            raise ValueError("judge failed")
    assert not state["sleeping"]
    assert "/wake_up" in calls


def test_coach_restore_failure_is_not_swallowed():
    def request(base, token, path, post=False):
        if path == "/wake_up":
            raise RuntimeError("unreachable")
        return {"is_sleeping": path != "/is_sleeping"}
    env = {"REMA_COACH_BACKEND": "local_vllm", "REMA_COACH_BASE_URL": "http://127.0.0.1:1234/v1"}
    # Fail sleep itself; its outcome is uncertain so wake must still be attempted.
    def failing(base, token, path, post=False):
        if path.startswith("/sleep"):
            raise TimeoutError()
        return request(base, token, path, post=post)
    with pytest.raises(runtime.ResourceRestoreError):
        with runtime.pause_local_coach(env, failing):
            pytest.fail("Must not start Gemma before confirmed handoff")


@pytest.mark.parametrize("mode", ["routine", "diagnostic"])
def test_validation_exports_and_accuracy_unchanged(monkeypatch, tmp_path, mode):
    helpers = importlib.import_module("test_validation_modes")
    monkeypatch.setenv("REMA_MAST_ENABLED", "true")
    monkeypatch.setenv("REMA_IDS_ENABLED", "false")
    seen = []
    def runner(config, documents, output_dir):
        seen.extend(documents.values())
        return {"metadata": {"resolved_revision": "a" * 40},
                "results": {key: {"status": "ok", "annotation": annotation()} for key in documents}}
    monkeypatch.setattr(runtime, "run_local_judge", runner)
    args = helpers.args_for_run(monkeypatch)
    args.validation_mode, args.reward_setting = mode, "handcrafted"
    tracker = helpers.Tracker()
    directory = tmp_path / "validation"
    result = helpers.run_validation(args, directory, tracker)
    assert result["accuracy"] == 1
    assert seen and "ground_truth" not in json.dumps(seen)
    assert tracker.events[-1]["val/acc/math"] == 1
    assert tracker.events[-1]["val/failures/context_loss"] == 0
    assert (directory / "validation_failures.jsonl").exists()
    assert (directory / "failure_judge_definition.json").exists()
    definition = json.loads((directory / "failure_judge_definition.json").read_text())
    assert definition["version"] == contract.VERSION
    assert definition["categories"] == contract.CATEGORIES
    assert definition["source_category_ids"] == contract.SOURCE_CATEGORY_IDS
    assert (tmp_path / "failure_judge_history.csv").exists()
    helpers.run_validation(args, directory, tracker)
    rows = [json.loads(line) for line in (tmp_path / "failure_judge_history.jsonl").read_text().splitlines()]
    assert len(rows) == 16  # Eight categories x overall/dataset; no duplicate epoch.
    assert {row["category_id"] for row in rows} == set(contract.CATEGORIES)
    assert all(row["name"] == contract.CATEGORIES[row["category_id"]] for row in rows)


def test_unknown_rates_never_become_clean_wandb_values():
    metrics = importlib.import_module(api.__name__ + ".metrics")
    summary = {"accuracy": .5, "failure_judge": judge.aggregate([
        {"task_id": "a", "dataset": "math", "status": "unavailable"}])}
    assert metrics.validation_tracking_metrics(summary) == {"val/accuracy": .5}


def test_runtime_restores_ids_and_cleans_worker_before_waking_coach(monkeypatch, tmp_path):
    local = importlib.import_module(api.__name__ + ".reward_providers.local_coach")
    gfam = importlib.import_module(api.__name__ + ".gfam_reward")
    events = []
    monkeypatch.setattr(local, "validate_gpu_isolation", lambda env: "3")
    monkeypatch.setattr(gfam, "load_cached_reward_scorer", NS(cache_clear=lambda: events.append("release_graphprm")))
    monkeypatch.setattr(local, "stop_process_group", lambda p, t: events.append("stop_gemma"))
    @contextmanager
    def paused(env):
        events.append("sleep_coach")
        try:
            yield
        finally:
            events.append("wake_coach")
    monkeypatch.setattr(runtime, "pause_local_coach", paused)
    class Process:
        def __init__(self, command, **kwargs):
            events.append("start_gemma")
            env = kwargs["env"]
            assert "VLLM_USE_V1" not in env
            assert "RAY_ADDRESS" not in env
            assert "VLLM_SERVER_DEV_MODE" not in env
            job = json.loads(Path(command[-1]).read_text())
            # Simulate one completed annotation followed by a crash mid-write.
            with Path(job["results_path"]).open("w") as handle:
                handle.write(json.dumps({"input_id": "first", "status": "ok", "annotation": annotation()}) + "\n{")
        def wait(self, timeout):
            return 2
    monkeypatch.setenv("REMA_MAST_ENABLED", "true")
    monkeypatch.setenv("RAY_ADDRESS", "must-not-inherit")
    monkeypatch.setattr(runtime.subprocess, "Popen", Process)
    report = runtime.run_local_judge(judge.FailureJudgeConfig(enabled=True), {"first": {}, "second": {}}, tmp_path)
    assert report["results"]["first"]["status"] == "ok"
    assert report["results"]["second"]["status"] == "unavailable"
    assert events == ["release_graphprm", "sleep_coach", "start_gemma", "stop_gemma", "wake_coach"]


def test_unknown_judgments_do_not_break_validation(monkeypatch, tmp_path):
    helpers = importlib.import_module("test_validation_modes")
    monkeypatch.setenv("REMA_MAST_ENABLED", "true")
    monkeypatch.setenv("REMA_IDS_ENABLED", "false")
    def fail(*args):
        raise RuntimeError("Gemma architecture unsupported in current container")
    monkeypatch.setattr(runtime, "run_local_judge", fail)
    args = helpers.args_for_run(monkeypatch)
    args.validation_mode = "routine"
    tracker = helpers.Tracker()
    result = helpers.run_validation(args, tmp_path / "validation", tracker)
    assert result["accuracy"] == 1
    assert not any(key.startswith("val/failures/") for key in tracker.events[-1])
    saved = json.loads((tmp_path / "validation/validation_failures.jsonl").read_text())
    assert saved["status"] == "unavailable"


def test_coach_sleep_api_is_only_enabled_with_judge(monkeypatch):
    local = importlib.import_module(api.__name__ + ".reward_providers.local_coach")
    assert "VLLM_SERVER_DEV_MODE" not in local.server_environment({}, "token")
    assert local.server_environment({"REMA_MAST_ENABLED": "true"}, "token")["VLLM_SERVER_DEV_MODE"] == "1"

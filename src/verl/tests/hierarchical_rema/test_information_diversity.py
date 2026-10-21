"""IDS formula, DAG mapping, export, and validation integration without downloads."""
import csv
import importlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import numpy as np
import pytest

try:
    import verl.hierarchical_rema as api
except ModuleNotFoundError:
    import hierarchical_rema as api

ids = importlib.import_module(api.__name__ + ".information_diversity")
exports = importlib.import_module(api.__name__ + ".evaluation_exports")
metrics = importlib.import_module(api.__name__ + ".metrics")


def graph(texts, edges):
    return {"node_ids": list(map(str, range(len(texts)))), "texts": texts,
            "edges": edges, "error": None}


def test_paper_formula_identical_and_orthogonal_responses():
    same = graph(["x = 4", "x = 4"], [(0, 1)])
    assert ids.graph_ids(same, {"x = 4": [1, 0]})["ids"] == pytest.approx(0)
    orthogonal = graph(["apple", "orange"], [(0, 1)])
    result = ids.graph_ids(orthogonal, {"apple": [1, 0], "orange": [0, 1]})
    assert result["ids"] == 1
    assert result["weight_sum"] == 2
    assert len(result["pairs"]) == 1


def test_equal_weights_and_no_undocumented_cosine_rescaling():
    value = ids.graph_ids(graph(["apple", "orange"], [(0, 1)]),
                          {"apple": [1, 0], "orange": [.6, .8]})
    assert value["ids"] == pytest.approx(.7)
    value = ids.graph_ids(graph(["apple", "orange"], [(0, 1)]),
                          {"apple": [1, 0], "orange": [-1, 0]})
    assert value["ids"] == pytest.approx(1.5)


def test_weights_use_causality_not_execution_order():
    # A -> B -> C, with an independent D. Direct pairs count twice; A/C once.
    weights = ids.connection_weights(4, [(0, 1), (1, 2)])
    np.testing.assert_array_equal(weights,
        [[0, 2, 1, 0], [2, 0, 2, 0], [1, 2, 0, 0], [0, 0, 0, 0]])
    with pytest.raises(ValueError, match="cyclic"):
        ids.connection_weights(2, [(0, 1), (1, 0)])


def test_numeric_and_math_tokens_are_not_dropped():
    values = ids.tfidf_cosines(["1", "2", "x < 3", "x > 3"])
    assert values[0, 1] == 0
    assert 0 < values[2, 3] < 1
    np.testing.assert_allclose(np.diag(values), 1)


def test_tfidf_matches_reference_library():
    sklearn = pytest.importorskip("sklearn.feature_extraction.text")
    texts = ["x x = 2", "y > 3", "x < 2", "x x = 2"]
    matrix = sklearn.TfidfVectorizer(token_pattern=ids.TOKEN_PATTERN).fit_transform(texts)
    np.testing.assert_allclose(ids.tfidf_cosines(texts), (matrix @ matrix.T).toarray())


def execution(node, output, dependencies=None, raw="secret scratchpad"):
    return NS(node_id=node, output_text=output, dependency_outputs=dependencies or {},
              raw_output_text=raw, worker_prompt="task and ground truth must not enter IDS")


def plan_and_selection():
    plan = NS(nodes=[NS(node_id="A", dependencies=[]), NS(node_id="B", dependencies=["A"])],
              final_node_id="B", decomposition_id="plan")
    selection = NS(executions=[execution("A", "3"), execution("B", "4", {"A": "3"})],
                   final_answer="duplicate final is not another node",
                   reward=NS(final_answer_correctness=1), selection=NS(selection_id="repeat"))
    return plan, selection


def test_mapping_uses_delivered_outputs_and_final_once():
    plan, selection = plan_and_selection()
    result = ids.execution_graph(plan, selection)
    assert result == {"node_ids": ["A", "B"], "texts": ["3", "4"], "edges": [(0, 1)], "error": None}
    selection.executions[0].output_text = ""
    selection.executions[1].dependency_outputs["A"] = ""
    result = ids.execution_graph(plan, selection)
    assert result["texts"] == ["", "4"]  # Never resurrect raw_output_text.
    assert result["error"] is None
    value = ids.graph_ids(result, {})
    assert value["ids"] is None
    assert value["empty_outputs"] == 1


@pytest.mark.parametrize("mutation,reason", [
    ("missing_snapshot", "missing_dependency_snapshot"),
    ("changed_snapshot", "dependency_snapshot_differs_from_output"),
    ("incomplete", "incomplete_or_unmapped_execution"),
    ("duplicate", "duplicate_execution_node_id"),
])
def test_mapping_does_not_invent_edges(mutation, reason):
    plan, selection = plan_and_selection()
    if mutation == "missing_snapshot":
        selection.executions[1].dependency_outputs = {}
    elif mutation == "changed_snapshot":
        selection.executions[1].dependency_outputs["A"] = "different"
    elif mutation == "incomplete":
        selection.executions.pop()
    else:
        selection.executions.append(selection.executions[0])
    assert ids.execution_graph(plan, selection)["error"] == reason


def test_single_node_and_disconnected_graphs_are_undefined_not_diverse():
    for candidate in (graph(["4"], []), graph(["3", "4"], [])):
        assert ids.graph_ids(candidate, {})["ids"] is None
    partial = ids.graph_ids(graph(["a", "", "b"], [(0, 1), (1, 2)]),
                            {"a": [1, 0], "b": [0, 1]})
    assert partial["status"] == "partial"
    assert partial["pair_coverage"] == pytest.approx(1 / 3)
    assert partial["ids"] == 1


def test_task_balancing_and_dataset_identity():
    rows = [{"task_id": "same", "subset": "math", "status": "ok", "ids": 1}] * 8
    rows += [{"task_id": "same", "subset": "gsm", "status": "ok", "ids": 0},
             {"task_id": "bad", "subset": "gsm", "status": "no_pairs", "ids": None}]
    result = ids.aggregate_ids(rows)
    assert result["ids"] == .5
    assert result["task_coverage"] == pytest.approx(2 / 3)
    assert result["rollout_coverage"] == .9


def test_exact_identical_tail_and_unrelated_earlier_output():
    # Last three nodes pass the same text on, even though the graph's score is nonzero.
    result = ids.graph_ids_exact(graph(["intermediate", "42", "42", "42"],
                                      [(0, 1), (1, 2), (2, 3)]))
    assert result["ids_exact"] == pytest.approx(4 / 9)
    assert result["exact_longest_copy_chain"] == ["1", "2", "3"]
    assert result["exact_direct_match_edges"] == [["1", "2"], ["2", "3"]]
    assert result["exact_direct_copy_rate"] == pytest.approx(2 / 3)
    result = ids.graph_ids_exact(graph(["42"] * 3, [(0, 1), (1, 2)]))
    assert result["ids_exact"] == 0
    assert result["exact_direct_copy_rate"] == 1


@pytest.mark.parametrize("left,right,expected", [
    (" 42\n", "42", 0),
    ("1/2", "0.5", 1),
    ("x=4", "x = 4", 1),
    ("x = 4", "x = 5", 1),
    ("a b", "a  b", 1),
    ("x < 3", "x > 3", 1),
    ("X", "x", 1),
])
def test_exact_only_strips_surrounding_whitespace(left, right, expected):
    result = ids.graph_ids_exact(graph([left, right], [(0, 1)]))
    assert result["ids_exact"] == expected


def test_exact_does_not_count_unconnected_duplicates_or_invent_chains():
    result = ids.graph_ids_exact(graph(["7", "7", "9"], [(0, 2)]))
    assert result["ids_exact"] == 1
    assert result["exact_direct_match_edges"] == []
    assert result["exact_longest_copy_chain"] == []
    fork = ids.graph_ids_exact(graph(["42"] * 3, [(0, 1), (0, 2)]))
    assert len(fork["exact_longest_copy_chain"]) == 2
    reversed_order = ids.graph_ids_exact(graph(["42"] * 3, [(2, 1), (1, 0)]))
    assert reversed_order["exact_longest_copy_chain"] == ["2", "1", "0"]


def test_exact_empty_single_invalid_and_partial_graphs():
    for candidate in (graph([""], []), graph(["", ""], [(0, 1)]),
                      graph(["42"], []), graph(["42", "42"], [])):
        result = ids.graph_ids_exact(candidate)
        assert result["ids_exact"] is None
        assert result["exact_status"] == "no_nonempty_connected_pairs"
    partial = ids.graph_ids_exact(graph(["42", "", "42"], [(0, 1), (1, 2)]))
    assert partial["ids_exact"] == 0
    assert partial["exact_status"] == "partial"
    assert partial["pair_coverage"] == pytest.approx(1 / 3)
    assert partial["exact_direct_copy_rate"] is None
    assert partial["exact_longest_copy_chain"] == []
    invalid = {**graph(["42", "42"], [(0, 1)]), "error": "missing_dependency_snapshot"}
    assert ids.graph_ids_exact(invalid)["ids_exact"] is None
    with pytest.raises(ValueError, match="cyclic"):
        ids.graph_ids_exact(graph(["42"] * 2, [(0, 1), (1, 0)]))


def test_exact_task_balancing_is_independent_of_encoder_coverage():
    rows = [{"task_id": "a", "subset": "math", "ids": None, "status": "encoder_unavailable",
             "ids_exact": 1, "exact_status": "ok"}] * 8
    rows += [{"task_id": "b", "subset": "math", "ids": None, "status": "encoder_unavailable",
              "ids_exact": 0, "exact_status": "ok"},
             {"task_id": "c", "subset": "math", "ids": None, "status": "no_pairs",
              "ids_exact": None, "exact_status": "no_pairs"}]
    result = ids.aggregate_ids(rows)
    assert result["ids"] is None
    assert result["ids_exact"] == .5
    assert result["exact_rollout_coverage"] == .9
    assert result["exact_task_coverage"] == pytest.approx(2 / 3)


class FakeEncoder:
    def __init__(self, config):
        self.metadata = {"resolved_revision": "test-fake"}
    def encode(self, texts):
        assert len(texts) == len(set(texts))
        return {text: np.array([1., (len(text) % 7) + 1.]) for text in texts}
    def close(self):
        pass


def rollout():
    plan, selection = plan_and_selection()
    return NS(task=NS(task_id="example"),
              decompositions=[NS(decomposition=plan, selections=[selection])])


def test_audit_batches_texts_and_exports_explicit_encoder_failure():
    audit = ids.InformationDiversityAudit(ids.IDSConfig(), FakeEncoder)
    audit.add(rollout(), "math")
    report = audit.finish()
    assert report["overall"]["eligible_rollouts"] == 1
    assert report["subsets"]["math"]["ids"] == report["overall"]["ids"]
    assert report["rows"][0]["outputs"][0]["chars"] == 1
    assert "texts" not in report["rows"][0]
    class FailingEncoder:
        def __init__(self, config):
            raise OSError("offline model cache missing")
    audit.encoder_factory = FailingEncoder
    report = audit.finish()
    assert report["overall"]["ids"] is None
    assert report["overall"]["rollout_coverage"] == 0
    assert report["rows"][0]["status"] == "encoder_unavailable"
    assert report["overall"]["ids_exact"] == 1
    assert report["overall"]["exact_rollout_coverage"] == 1
    assert report["rows"][0]["exact_status"] == "ok"


def test_exact_survives_invalid_semantic_vectors_and_ignores_raw_scratchpads():
    class InvalidEncoder(FakeEncoder):
        def encode(self, texts):
            return {text: np.array([np.nan, 1.]) for text in texts}
    example = rollout()
    selection = example.decompositions[0].selections[0]
    selection.executions[0].raw_output_text = "scratchpad one; result=42"
    selection.executions[1].raw_output_text = "different scratchpad; result=42"
    selection.executions[0].output_text = "42"
    selection.executions[1].output_text = "42"
    selection.executions[1].dependency_outputs = {"A": "42"}
    audit = ids.InformationDiversityAudit(ids.IDSConfig(), InvalidEncoder)
    audit.add(example, "math")
    report = audit.finish()
    assert report["overall"]["ids"] is None
    assert report["overall"]["ids_exact"] == 0
    assert report["rows"][0]["exact_longest_copy_chain"] == ["A", "B"]


def test_no_encoder_for_disabled_or_pairless_graph():
    def fail(config):
        pytest.fail("Unnecessary BGE load")
    audit = ids.InformationDiversityAudit(ids.IDSConfig(enabled=False), fail)
    audit.add(rollout(), "math")
    assert audit.finish()["rows"] == []
    audit = ids.InformationDiversityAudit(ids.IDSConfig(), fail)
    audit.graphs = [{**graph(["4"], []), "task_id": "x", "subset": "math"}]
    assert audit.finish()["overall"]["ids"] is None
    assert audit.finish()["overall"]["ids_exact"] is None


def test_exports_are_cumulative_idempotent_and_null_safe(tmp_path):
    audit = ids.InformationDiversityAudit(ids.IDSConfig(), FakeEncoder)
    audit.add(rollout(), "math")
    report = audit.finish()
    subset = {"accuracy": .5, "num_tasks": 1, "monitoring_totals": {
        "correct_rollouts": 1, "verified_rollouts": 2, "unverified_rollouts": 1,
        "completed_rollouts": 3}}
    summary = {**subset, "validation_mode": "routine", "subsets": {"math": subset}}
    for epoch in (6, 12, 12):
        exports.write_evaluation_exports(tmp_path / f"epoch_{epoch}", summary, report,
                                         epoch=epoch, tracking_step=17, context={"reward_setting": "gnn_prm"})
    history = [json.loads(line) for line in (tmp_path / "evaluation_history.jsonl").read_text().splitlines()]
    assert len(history) == 4
    assert [row["epoch"] for row in history] == [6, 6, 12, 12]
    assert all(row["accuracy"] == .5 for row in history)
    assert history[0]["unverified_rollouts"] == 1
    assert all(row["ids_exact"] == 1 for row in history)
    assert history[0]["ids_exact_version"] == ids.EXACT_VERSION
    assert history[0]["ids_definition"]["model"] == "BAAI/bge-m3"
    with (tmp_path / "evaluation_history.csv").open() as handle:
        exported = list(csv.DictReader(handle))
        assert len(exported) == 4
        assert all(float(row["ids_exact"]) == 1 for row in exported)
    assert (tmp_path / "epoch_12/validation_ids.jsonl").exists()
    summary["accuracy"] = float("nan")
    exports.write_evaluation_exports(tmp_path / "epoch_18", summary, report,
                                     epoch=18, tracking_step=18, context={})
    overall = next(json.loads(line) for line in (tmp_path / "epoch_18/evaluation_metrics.jsonl").read_text().splitlines()
                   if json.loads(line)["scope"] == "overall")
    assert overall["accuracy"] is None


def test_accuracy_per_dataset_but_only_overall_diversity_scores_are_logged():
    summary = {"accuracy": .7, "information_diversity": {"ids": .3, "ids_exact": .5, "rollout_coverage": .8},
               "subsets": {"math": {"accuracy": .7, "information_diversity": {
                   "ids": .3, "rollout_coverage": .8}}}}
    assert metrics.validation_tracking_metrics(summary) == {
        "val/accuracy": .7, "val/acc/math": .7, "val/ids": .3, "val/ids_exact": .5}
    summary["information_diversity"]["ids"] = None
    assert "val/ids" not in metrics.validation_tracking_metrics(summary)
    assert metrics.validation_tracking_metrics(summary)["val/ids_exact"] == .5
    summary["information_diversity"]["ids_exact"] = None
    assert "val/ids_exact" not in metrics.validation_tracking_metrics(summary)


def test_ids_configuration_defaults_and_invalid_inputs(monkeypatch):
    monkeypatch.delenv("REMA_IDS_ENABLED", raising=False)
    assert ids.IDSConfig.from_env().enabled
    monkeypatch.delenv("REMA_IDS_DEVICE", raising=False)
    assert ids.IDSConfig.from_env().device == "cuda:0"
    monkeypatch.delenv("REMA_IDS_MODEL", raising=False)
    assert ids.IDSConfig.from_env().model == "BAAI/bge-m3"
    monkeypatch.setenv("REMA_IDS_BATCH_SIZE", "0")
    with pytest.raises(ValueError, match="positive"):
        ids.IDSConfig.from_env()


def test_s3_upload_includes_validation_history_and_epoch_artifacts(tmp_path):
    # Exercise the existing upload function with a recording rclone stub.
    launcher = Path(__file__).with_name("hierarchical-rema-trainer.sh")
    if not launcher.exists():
        launcher = Path(__file__).resolve().parents[4] / "hierarchical-rema-trainer.sh"
    source = launcher.read_text()
    function = source.split("upload_epoch_folder_to_s3() {", 1)[1].split(
        "\nsync_epoch_final_models_to_s3()", 1)[0]
    epoch = tmp_path / "epoch_0006"
    epoch.mkdir()
    (tmp_path / "validation_accuracy.csv").write_text("epoch,math\n6,1\n")
    script = """
set -eu
LOCAL_OUTPUT_DIR="$1"
S3_OUTPUT_PATH="remote:bucket/run"
S3_EPOCHS_PATH="remote:bucket/run/epochs"
rclone() { printf '%s\\n' "$*"; }
sync_epoch_final_models_to_s3() { return 0; }
upload_epoch_folder_to_s3() {""" + function + """
upload_epoch_folder_to_s3 "$LOCAL_OUTPUT_DIR/epoch_0006"
"""
    result = subprocess.run(["bash", "-s", "--", str(tmp_path)], input=script,
                            text=True, capture_output=True, check=True)
    assert "--include /evaluation_history.csv" in result.stdout
    assert "--include /evaluation_history.jsonl" in result.stdout
    assert "remote:bucket/run/epochs/epoch_0006" in result.stdout


@pytest.mark.parametrize("enabled,device,reservation,success", [
    ("true", "cuda:0", "1", True),
    ("true", "cuda:1", "1", False),
    ("true", "cuda:0", "0", False),
    ("TRUE", "cuda:0", "0", False),
    ("true", "cpu", "0", True),
    ("false", "cuda:0", "0", True),
])
def test_launcher_keeps_ids_off_policy_gpus(enabled, device, reservation, success):
    launcher = Path(__file__).with_name("hierarchical-rema-trainer.sh")
    if not launcher.exists():
        launcher = Path(__file__).resolve().parents[4] / "hierarchical-rema-trainer.sh"
    source = launcher.read_text()
    start = source.index('if [[ "$RUN_KIND" == "train" && "$REMA_IDS_ENABLED"')
    end = source.index("\nCOACH_SUPERVISOR=", start)
    environment = {**os.environ, "RUN_KIND": "train", "REMA_IDS_ENABLED": enabled,
                   "REMA_IDS_DEVICE": device, "GFAM_REWARD_GPU_RESERVATION": reservation}
    result = subprocess.run(["bash", "-s"], input=source[start:end], text=True,
                            capture_output=True, env=environment)
    assert (result.returncode == 0) is success


@pytest.mark.parametrize("mode", ["routine", "diagnostic"])
def test_validation_integration_same_accuracy_events_no_new_rollouts(monkeypatch, tmp_path, mode):
    validation_tests = importlib.import_module("test_validation_modes")
    train = validation_tests.train
    monkeypatch.setenv("REMA_IDS_ENABLED", "true")
    monkeypatch.setattr(ids, "BGEResponseEncoder", FakeEncoder)
    args = validation_tests.args_for_run(monkeypatch)
    args.validation_mode, args.reward_setting = mode, "handcrafted"
    tracker = validation_tests.Tracker()
    summary = validation_tests.run_validation(args, tmp_path / "epoch_6", tracker)
    assert summary["accuracy"] == 1
    assert summary["information_diversity"]["total_rollouts"] == 1
    assert summary["information_diversity"]["eligible_rollouts"] == 1
    assert tracker.events[-1]["val/ids"] == summary["information_diversity"]["ids"]
    assert tracker.events[-1]["val/ids_exact"] == summary["information_diversity"]["ids_exact"]
    assert "val/ids_exact/math" not in tracker.events[-1]
    assert "val/ids/math" not in tracker.events[-1]
    assert "val/ids_coverage" not in tracker.events[-1]
    assert (tmp_path / "evaluation_history.csv").exists()
    assert len((tmp_path / "epoch_6/validation_ids.jsonl").read_text().splitlines()) == 1


def test_frozen_bge_cls_and_chunk_pooling_preserves_all_tokens_and_cpu_rng(monkeypatch):
    import torch
    import transformers

    class Tokenizer:
        model_max_length = 8192
        def num_special_tokens_to_add(self, pair=False):
            return 2
        def encode(self, text, **kwargs):
            return [int(part) for part in text.split()]
        def prepare_for_model(self, tokens, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": [0, *tokens, 2],
                    "attention_mask": [1] * (len(tokens) + 2)}
        def pad(self, payloads, **kwargs):
            length = max(len(p["input_ids"]) for p in payloads)
            return {key: torch.tensor([p[key] + [int(key == "input_ids")] * (length - len(p[key])) for p in payloads])
                    for key in payloads[0]}

    def model(*args, **kwargs):
        torch.manual_seed(23)
        return transformers.XLMRobertaModel(transformers.XLMRobertaConfig(
            hidden_size=8, num_attention_heads=2, intermediate_size=16,
            num_hidden_layers=1, vocab_size=100, max_position_embeddings=10, pad_token_id=1))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", model)
    before = torch.random.get_rng_state().clone()
    encoder = ids.BGEResponseEncoder(ids.IDSConfig(batch_size=2, device="cpu"))
    assert torch.equal(before, torch.random.get_rng_state())
    assert not encoder.model.training
    assert all(not param.requires_grad for param in encoder.model.parameters())
    assert encoder.window == 8  # XLM-R's padding offset leaves only 8 valid positions.
    texts = ["3 4 5 6 7 8 9 10 11 12 13 14 15", "3 4"]
    result = encoder.encode(texts)
    assert encoder.metadata["content_tokens"] == 15
    assert encoder.metadata["chunked_responses"] == 1
    assert all(vector.shape == (8,) for vector in result.values())
    for vector in result.values():
        assert np.linalg.norm(vector) == pytest.approx(1)
    tokenizer = encoder.tokenizer
    with torch.inference_mode():
        pieces = []
        for tokens in ([3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14], [15]):
            payload = tokenizer.prepare_for_model(tokens, truncation=False)
            batch = tokenizer.pad([payload])
            hidden = encoder.model(**batch).last_hidden_state
            cls = torch.nn.functional.normalize(hidden[:, 0], dim=-1)[0].numpy()
            pieces.append(len(tokens) * cls)
        expected = sum(pieces)
        np.testing.assert_allclose(result[texts[0]], expected / np.linalg.norm(expected), atol=1e-6)
        payload = tokenizer.prepare_for_model([3, 4], truncation=False)
        hidden = encoder.model(**tokenizer.pad([payload])).last_hidden_state
        cls = torch.nn.functional.normalize(hidden[:, 0], dim=-1)[0].numpy()
        np.testing.assert_allclose(result[texts[1]], cls, atol=1e-6)
    serial_encoder = ids.BGEResponseEncoder(ids.IDSConfig(batch_size=1, device="cpu"))
    serial = serial_encoder.encode(texts)
    for text in texts:
        np.testing.assert_allclose(result[text], serial[text], atol=1e-6)
    encoder.close()
    serial_encoder.close()

"""Durable numerical validation histories, independent of W&B availability."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(child) for child in value]
    return value


def _atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The S3 watcher must never see a partially rewritten history.
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".evaluation-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _jsonl(path, rows):
    def write(handle):
        for row in rows:
            handle.write(json.dumps(_safe(row), ensure_ascii=False, allow_nan=False) + "\n")
    _atomic(path, write)


def write_evaluation_exports(output_dir, summary, ids_report, *, epoch, tracking_step, context,
                             failure_report=None):
    output_dir = Path(output_dir)
    metadata = ids_report["metadata"]
    definition = {key: value for key, value in metadata.items()
                  if key not in ("unique_responses", "chunked_responses", "content_tokens")}
    definition_id = hashlib.sha256(
        json.dumps(definition, sort_keys=True).encode("utf-8")).hexdigest()
    common = {
        "epoch": epoch, "tracking_step": tracking_step,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_mode": summary["validation_mode"],
        "ids_definition_id": definition_id, "ids_definition": definition,
        **context,
    }
    rows = []
    targets = [("overall", "__overall__", summary)]
    targets.extend(("dataset", name, values) for name, values in sorted(summary.get("subsets", {}).items()))
    for scope, dataset, values in targets:
        ids = ids_report["overall"] if scope == "overall" else ids_report["subsets"].get(dataset, {})
        totals = values.get("monitoring_totals", {})
        rows.append({**common, "scope": scope, "dataset": dataset,
                     "accuracy": values.get("accuracy"),  # Same denominator as W&B.
                     "accuracy_definition": "correct_rollouts / verified_rollouts",
                     "correct_rollouts": totals.get("correct_rollouts", 0),
                     "verified_rollouts": totals.get("verified_rollouts", 0),
                     "unverified_rollouts": totals.get("unverified_rollouts", 0),
                     "completed_rollouts": totals.get("completed_rollouts", 0),
                     "num_tasks": values.get("num_tasks", 0),
                     "ids": ids.get("ids"),
                     "ids_eligible_tasks": ids.get("eligible_tasks", 0),
                     "ids_eligible_rollouts": ids.get("eligible_rollouts", 0),
                     "ids_task_coverage": ids.get("task_coverage"),
                     "ids_rollout_coverage": ids.get("rollout_coverage"),
                     "ids_status_counts": ids.get("status_counts", {}),
                     "ids_exact": ids.get("ids_exact"),
                     "ids_exact_version": metadata.get("exact", {}).get("version"),
                     "ids_exact_eligible_tasks": ids.get("exact_eligible_tasks", 0),
                     "ids_exact_eligible_rollouts": ids.get("exact_eligible_rollouts", 0),
                     "ids_exact_task_coverage": ids.get("exact_task_coverage"),
                     "ids_exact_rollout_coverage": ids.get("exact_rollout_coverage"),
                     "ids_exact_status_counts": ids.get("exact_status_counts", {}),
                     "elapsed_seconds": summary.get("elapsed_seconds")})
    _jsonl(output_dir / "evaluation_metrics.jsonl", rows)
    _jsonl(output_dir / "validation_ids.jsonl",
           [{**common, **row} for row in ids_report["rows"]])
    _atomic(output_dir / "ids_definition.json",
            lambda handle: json.dump(_safe(metadata), handle, indent=2, allow_nan=False))

    history_path = output_dir.parent / "evaluation_history.jsonl"
    history = []
    if history_path.exists():
        with history_path.open(encoding="utf-8") as handle:
            history = [json.loads(line) for line in handle if line.strip()]
    # Re-evaluating a round replaces that round, rather than silently duplicating it.
    history = [row for row in history if row["epoch"] != epoch] + rows
    history.sort(key=lambda row: (row["epoch"], row["scope"], row["dataset"]))
    _jsonl(history_path, history)
    columns = ("epoch", "tracking_step", "evaluated_at_utc", "validation_mode",
               "reward_setting", "scope", "dataset", "accuracy", "correct_rollouts",
               "verified_rollouts", "unverified_rollouts", "completed_rollouts", "num_tasks",
               "ids", "ids_eligible_tasks", "ids_eligible_rollouts", "ids_task_coverage",
               "ids_rollout_coverage", "ids_exact", "ids_exact_version", "ids_exact_eligible_tasks",
               "ids_exact_eligible_rollouts", "ids_exact_task_coverage", "ids_exact_rollout_coverage",
               "ids_definition_id", "elapsed_seconds")

    def write_csv(handle):
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_safe(history))
    _atomic(output_dir.parent / "evaluation_history.csv", write_csv)
    if failure_report and failure_report["metadata"]["enabled"]:
        _write_failure_exports(output_dir, failure_report, common)


def _write_failure_exports(output_dir, report, common):
    metadata = report["metadata"]
    # Do not repeat the full prompt in every rollout/history row.
    identity = {key: value for key, value in metadata.items() if key not in ("prompt", "runtime_error")}
    identity_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    _atomic(output_dir / "failure_judge_definition.json",
            lambda handle: json.dump(_safe(metadata), handle, indent=2))
    base = {key: common[key] for key in ("epoch", "tracking_step", "evaluated_at_utc", "validation_mode")}
    base.update(judge_definition_id=identity_id, model=metadata["model"],
                revision=metadata.get("resolved_revision"), prompt_sha256=metadata["prompt_sha256"])
    _jsonl(output_dir / "validation_failures.jsonl", [{**base, **row} for row in report["rows"]])
    rows = []
    for dataset, summary in [("__overall__", report["overall"]), *sorted(report["subsets"].items())]:
        for code, values in summary["categories"].items():
            rows.append({**base, "dataset": dataset, "category_id": code, **values})
    _jsonl(output_dir / "failure_judge_metrics.jsonl", rows)
    path = output_dir.parent / "failure_judge_history.jsonl"
    old = []
    if path.exists():
        with path.open() as handle:
            old = [json.loads(line) for line in handle if line.strip()]
    history = [row for row in old if row["epoch"] != common["epoch"]] + rows
    history.sort(key=lambda row: (row["epoch"], row["dataset"], row["category_id"]))
    _jsonl(path, history)

    def write(handle):
        columns = (*base, "dataset", "category_id", "name", "rate", "positive_rollouts",
                   "assessed_rollouts", "assessed_tasks", "total_rollouts", "coverage")
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    _atomic(output_dir.parent / "failure_judge_history.csv", write)

"""Audit reference parseability before costly rollouts; never guess missing answers.

Standalone: python -m hierarchical_rema.audit_answers --task-source data/MATH/train_lv3to5_8k.parquet --output reference_audit.json
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import time

from .rewarding import reference_verification_status


def audit_tasks(tasks, *, source, progress_every=1000):
    counts = Counter()
    issues = []
    started = time.monotonic()
    print(f"[answer-audit] source={source} references={len(tasks)} start", flush=True)
    for index, task in enumerate(tasks, 1):
        result = reference_verification_status(task.ground_truth, task.metadata)
        counts[result["status"]] += 1
        if result["status"] != "parseable":
            issues.append({"row_index": index - 1, "task_id": task.task_id,
                           "prompt": task.prompt, "reference": task.ground_truth, **result})
        if progress_every and index % progress_every == 0:
            print(f"[answer-audit] source={source} checked={index}/{len(tasks)} "
                  f"unverified={counts['unverified']}", flush=True)
    result = {"source": str(source), "references": len(tasks), "counts": dict(counts),
              "issues": issues, "elapsed_seconds": round(time.monotonic() - started, 2)}
    print(f"[answer-audit] source={source} checked={len(tasks)}/{len(tasks)} "
          f"counts={dict(counts)} elapsed={result['elapsed_seconds']:.1f}s", flush=True)
    return result


def skip_unverified_tasks(tasks, report, *, split):
    """Filter by audited row position, not potentially duplicated task IDs."""
    if len(tasks) != report["references"]:
        raise ValueError("Reference audit does not match the loaded dataset")
    skipped = set()
    for issue in report["issues"]:
        if issue["status"] == "unverified":
            skipped.add(issue["row_index"])
            issue["action"] = "skipped_before_rollout"
    kept = [task for index, task in enumerate(tasks) if index not in skipped]
    report["filtering"] = {
        "split": split, "input_rows": len(tasks), "kept_rows": len(kept),
        "skipped_rows": len(skipped),
        "retained_fraction": len(kept) / len(tasks) if tasks else 0.0,
    }
    print(f"[answer-audit] split={split} input={len(tasks)} kept={len(kept)} "
          f"skipped_unverified={len(skipped)} before rollout generation", flush=True)
    return kept


def _package_versions():
    packages = {}
    for name in ("math-verify", "latex2sympy2_extended", "sympy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "distribution metadata unavailable"
    return packages


def write_audit(path, reports):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audit": "reference_parseability_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "note": "Parseable does not prove the reference is correct or that all equivalent generated notation will be recognized. Unverified references must not be counted as incorrect answers.",
        "datasets": reports,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    print(f"[answer-audit] report={path.resolve()}", flush=True)


def record_verification_failures(path, rollouts):
    failures = []
    for rollout in rollouts:
        for plan in rollout.decompositions:
            for selection in plan.selections:
                if selection.reward.verification_status == "unverified":
                    failures.append({
                        "task_id": rollout.task.task_id, "prompt": rollout.task.prompt,
                        "reference": rollout.task.ground_truth,
                        "plan_id": plan.decomposition.decomposition_id,
                        "rollout_id": selection.selection.selection_id,
                        "final_answer": selection.final_answer,
                        "error": selection.reward.verification_error,
                    })
    if failures:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-source", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--task-id-key", default="id")
    args = parser.parse_args()
    from .train import load_tasks
    reports = []
    for source in args.task_source:
        tasks = load_tasks(source, "auto", args.prompt_key, args.answer_key, args.task_id_key, 0)
        reports.append(audit_tasks(tasks, source=source))
    write_audit(args.output, reports)


if __name__ == "__main__":
    main()

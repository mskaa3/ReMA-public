"""Extract a matched-step paper evidence bundle without importing training code."""

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def read_run(path, step):
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = re.sub(r"^\(TaskRunner pid=\d+\)[ \t]?", "", text, flags=re.M)
    text = re.sub(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d[ \t]", "", text, flags=re.M)
    audit_lines = [line for line in text.splitlines()
                   if re.match(rf"\[leakage/val\] step={step}\b", line)]
    if not audit_lines:
        raise ValueError(f"No validation audit for step {step}: {path}")
    audit = {key: float(value) for key, value in
             re.findall(r"([\w_]+)=(-?\d+(?:\.\d+)?)", audit_lines[-1])}
    metric_lines = [line for line in text.splitlines()
                    if re.match(rf"step:{step} -", line)]
    metrics = {}
    for line in metric_lines:
        metrics.update({key: float(value) for key, value in
                        re.findall(r"([^\s:]+):(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", line)})
    examples = {}
    for match in re.finditer(r"^\[question\] (.*)$", text, re.M):
        end = text.find("[question]", match.end())
        block = text[match.end():end if end >= 0 else len(text)]
        tag = re.search(r"^\[leakage/(val|all)\] step=(\d+)", block, re.M)
        if not tag or tag[1] != "val" or int(tag[2]) != step:
            continue
        history_match = re.search(r"^\[history\] (.*?\}\])\n", block, re.M | re.S)
        gate_match = re.search(r"^\[leakage/example\] (.*)$", block, re.M)
        if not history_match or not gate_match:
            continue
        # Console pretty-printing places dictionaries on adjacent lines without commas.
        history = ast.literal_eval(re.sub(r"\}\n\s*\{", "},\n{", history_match[1]))
        question = " ".join((match[1] + block.split("[ground_truth]")[0]).split())
        gate = {key: float(value) for key, value in
                re.findall(r"([\w_]+)=(-?\d+(?:\.\d+)?)", gate_match[1])}
        if question in examples:
            raise ValueError(f"Duplicate printed question at step {step}: {question}")
        examples[question] = {"question": question, "gate": gate,
                              "history": history, "gate_line": gate_match[0]}
    return {
        "source": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
        "step": step, "final_audit": audit,
        "validation_metrics": {k: v for k, v in metrics.items() if k.startswith("val/")},
        "complete_benchmark_metrics_available": any(k.startswith("val/acc/") for k in metrics),
        "actor_update_step": metrics.get("train/actor_update_step", audit.get("actor_update_step")),
    }, examples, audit_lines[-1], metric_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "evidence")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = {"notes": [
        "Baseline: retained ANSI-cleaned copy of run 6061597; original full log is unavailable.",
        "Treatment: user-provided October 11 excerpt, identified as run 6068468.",
        "Both final audit summaries refer to the configured 128-trajectory audit, not full-suite accuracy.",
        "The treatment step-30 metric line is truncated; missing benchmark results are not inferred.",
        "Runs differ in eligibility handling AND group handling; not an isolated intervention.",
        "Audit selection prioritizes console examples and fills the remaining budget by subset strata.",
    ]}
    runs, examples = {}, {}
    for name, path in (("baseline", args.baseline), ("eligibility", args.eligibility)):
        runs[name], examples[name], summary, metrics = read_run(path, args.step)
        (args.output / f"{name}-step{args.step}-log.txt").write_text(
            summary + "\n" + "\n".join(metrics) + "\n")
    common = sorted(set(examples["baseline"]) & set(examples["eligibility"]))
    matched = []
    for question in common:
        matched.append({"question": question, **{
            name: {"raw_score": examples[name][question]["gate"]["raw_score"],
                   "gated_score": examples[name][question]["gate"]["gated_score"],
                   "subtasks": int(examples[name][question]["gate"]["subtasks"])}
            for name in runs}})
    bundle["runs"] = runs
    bundle["matched_printed_examples"] = {"count": len(common)}
    for name in runs:
        bundle["matched_printed_examples"][name] = {
            "raw_correct": sum(row[name]["raw_score"] for row in matched),
            "gate_accepted_correct": sum(row[name]["gated_score"] for row in matched),
            "subtask_counts": dict(sorted(Counter(row[name]["subtasks"] for row in matched).items())),
        }
    selected = {name: [example for question, example in examples[name].items()
                       if any(term in question for term in
                              ("newspaper ad", "Janet had", "median", "Grayson recycles"))]
                for name in runs}
    for filename, obj in (("step30.json", bundle), ("matched-questions.json", matched),
                          ("qualitative-examples.json", selected)):
        (args.output / filename).write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps({"audit": {name: run["final_audit"] for name, run in runs.items()},
                      "matched": bundle["matched_printed_examples"]}, indent=2))


if __name__ == "__main__":
    main()

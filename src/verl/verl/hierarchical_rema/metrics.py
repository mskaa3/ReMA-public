"""Compact monitoring views; full diagnostics remain in JSON/JSONL artifacts."""
from collections import Counter
import math


TRAIN_METRICS = (
    "loss", "approx_kl", "entropy", "clipfrac", "reward_mean", "reward_std",
    "advantage_std", "grad_norm", "lr",
)


def learner_tracking_metrics(record, prefix=""):
    values = dict(record)
    if "reward_mean" not in values and "mean_reward" in values:
        values["reward_mean"] = values["mean_reward"]
    result = {f"{prefix}train/{key}": float(values[key])
              for key in TRAIN_METRICS if key in values}
    if prefix in ("shared_worker/", "shared_controller/") and "optimizer_updates" in values:
        result[f"{prefix}optimizer_updates"] = int(values["optimizer_updates"])
    return result


def moment_metrics(totals):
    """Population moments across all retained rows/ranks in one optimizer update."""
    count, reward_sum, reward_sq, advantage_sum, advantage_sq = totals
    if not count:
        return {}
    reward_mean, advantage_mean = reward_sum / count, advantage_sum / count
    return {
        "reward_mean": reward_mean,
        "reward_std": math.sqrt(max(0.0, reward_sq / count - reward_mean ** 2)),
        "advantage_std": math.sqrt(max(0.0, advantage_sq / count - advantage_mean ** 2)),
    }


def rollout_monitoring_totals(rollout):
    totals = Counter()
    for plan in rollout.decompositions:
        payload = plan.decomposition.raw_payload or {}
        validation = payload.get("validation") or payload.get("_validation") or {}
        totals["plans"] += 1
        totals["plans_with_validation_metadata"] += bool(validation)
        totals["fallback_plans"] += bool(validation.get("fallback_used"))
        for selection in plan.selections:
            totals["completed_rollouts"] += 1
            outcome = selection.reward.final_answer_correctness
            if outcome in (0, 1):
                totals["verified_rollouts"] += 1
                totals["correct_rollouts"] += outcome
            else:
                totals["unverified_rollouts"] += 1
            reward = selection.reward.total_reward
            if reward is not None and math.isfinite(reward):
                # The orchestrator routes the post-gate final reward here.
                totals["scored_rollouts"] += 1
                totals["final_reward_sum"] += reward
                if outcome == 0:
                    totals["scored_incorrect_rollouts"] += 1
                    totals["incorrect_final_reward_sum"] += reward
            else:
                totals["unscored_rollouts"] += 1
            totals["safety_excluded_rollouts"] += bool(selection.training_excluded)
            for execution in selection.executions:
                totals["worker_outputs"] += 1
                totals["invalid_worker_outputs"] += bool(execution.invalid_reason)
    return dict(totals)


def _ratio(totals, numerator, denominator):
    count = totals.get(denominator, 0)
    return totals.get(numerator, 0) / count if count else float("nan")


def merge_monitoring(summaries):
    totals = Counter()
    for summary in summaries:
        # Counter.update preserves negative reward sums, unlike Counter addition.
        totals.update(summary.get("monitoring_totals", {}))
    return {"monitoring_totals": dict(totals),
            "accuracy": _ratio(totals, "correct_rollouts", "verified_rollouts")}


def rollout_tracking_metrics(summary):
    totals = summary["monitoring_totals"]
    ratios = {
        "accuracy": ("correct_rollouts", "verified_rollouts"),
        "final_reward_mean": ("final_reward_sum", "scored_rollouts"),
        "incorrect_final_reward_mean": ("incorrect_final_reward_sum", "scored_incorrect_rollouts"),
        "plan_fallback_rate": ("fallback_plans", "plans_with_validation_metadata"),
        "invalid_worker_output_rate": ("invalid_worker_outputs", "worker_outputs"),
        "unscored_rate": ("unscored_rollouts", "completed_rollouts"),
        "safety_excluded_rate": ("safety_excluded_rollouts", "completed_rollouts"),
    }
    return {f"rollout/{key}": _ratio(totals, *fields) for key, fields in ratios.items()}


def validation_tracking_metrics(summary):
    return {"val/accuracy": summary["accuracy"],
            **{f"val/accuracy/{name}": subset["accuracy"]
               for name, subset in summary.get("subsets", {}).items()}}


def warn_verification_coverage(summary, context):
    totals = summary["monitoring_totals"]
    missing = totals.get("unverified_rollouts", 0)
    if missing:
        print(f"[hierarchical-rema][WARNING] {context}: {missing}/"
              f"{totals['completed_rollouts']} rollouts unverified; accuracy excludes them, "
              "not counted as incorrect. See saved summary for coverage.", flush=True)


def allocated_gpu_count(environ):
    """Use the Slurm allocation, not Ray's smaller policy-only GPU pool."""
    total = environ.get("SLURM_GPUS", "")
    if total.isdigit():
        return int(total)
    per_node = (environ.get("SLURM_GPUS_PER_NODE") or environ.get("SLURM_GPUS_ON_NODE", "")).split(":")[-1]
    nodes = environ.get("SLURM_JOB_NUM_NODES") or environ.get("SLURM_NNODES", "")
    if per_node.isdigit() and nodes.isdigit():
        return int(per_node) * int(nodes)
    return None


def performance_metrics(seconds, completed_rollouts, allocated_gpus=None):
    # Measured from round start through generation/scoring/learning, before validation.
    result = {"perf/round_seconds": seconds}
    if seconds > 0:
        result["perf/rollouts_per_second"] = completed_rollouts / seconds
    if completed_rollouts > 0 and allocated_gpus is not None and allocated_gpus > 0:
        result["perf/gpu_hours_per_1000_rollouts"] = seconds * allocated_gpus / 3600 * 1000 / completed_rollouts
    return result

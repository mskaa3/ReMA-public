"""Quarantine suspicious reward-model outcomes without rewriting their scores."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class SaturatedFailureFilter:
    enabled: bool = True
    threshold: float = 0.95

    def __post_init__(self):
        if not math.isfinite(self.threshold) or not 0 < self.threshold <= 1:
            raise ValueError("GFAM_SATURATED_FAILURE_THRESHOLD must be finite and in (0, 1]")

    @classmethod
    def from_environment(cls):
        enabled = os.environ.get("GFAM_SKIP_SATURATED_FAILURES", "true").strip().lower()
        if enabled not in {"true", "false", "1", "0"}:
            raise ValueError("GFAM_SKIP_SATURATED_FAILURES must be true/false or 1/0")
        return cls(enabled in {"true", "1"},
                   float(os.environ.get("GFAM_SATURATED_FAILURE_THRESHOLD", "0.95")))

    def evaluate(self, *, task, decomposition, executions, reward, compiled_rewards):
        audit = {"filter": "verified_failure_saturated_intermediates_v1",
                 "enabled": self.enabled, "threshold": self.threshold, "excluded": False,
                 "score_field": "reward_before_outcome_gate",
                 "verified_final_correctness": reward.final_answer_correctness}
        if (not self.enabled or reward.verification_status != "verified"
                or task.ground_truth is None or not str(task.ground_truth).strip()
                or reward.final_answer_correctness != 0):
            return audit
        intermediate = [ex for ex in executions if ex.node_id != decomposition.final_node_id]
        if not intermediate:
            return audit
        workers = compiled_rewards.get("workers", {})
        scores = {}
        for execution in intermediate:
            payload = workers.get(execution.node_id, {})
            applied = payload.get("reward")
            if isinstance(applied, (int, float)) and applied < 0:
                # A protocol penalty is already negative credit, not the saturated
                # positive-credit pattern this filter is intended to quarantine.
                audit["not_excluded_reason"] = "negative_intermediate_reward"
                return audit
            # Do not mistake the failure-discounted reward for the model's original credit.
            value = payload.get("reward_before_outcome_gate")
            if value is None:
                audit["not_evaluated_reason"] = "missing_pre_gate_score"
                return audit
            try:
                value = float(value)
            except (TypeError, ValueError):
                audit["not_evaluated_reason"] = "invalid_pre_gate_score"
                return audit
            if not math.isfinite(value):
                audit["not_evaluated_reason"] = "nonfinite_pre_gate_score"
                return audit
            scores[execution.node_id] = value
        audit["intermediate_scores"] = scores
        audit["excluded"] = all(value >= self.threshold for value in scores.values())
        if audit["excluded"]:
            audit["reason"] = "verified_failure_with_saturated_intermediate_rewards"
        return audit

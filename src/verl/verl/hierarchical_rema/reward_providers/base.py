"""Reward contract independent of generation, policy learning and evaluation."""
from __future__ import annotations

import math
from typing import Protocol, Sequence

from ..graphprm_v2.contract import canonical_record
from ..graphprm_v2.rewards import compile_rewards

LABELS = {
    "graph": ("global_outcome", "anti_collapse"),
    "decomposer": ("task_fulfillment", "execution_discipline"),
    "worker": ("task_fulfillment", "execution_discipline", "causal_utility"),
    "final": ("task_fulfillment", "execution_discipline", "causal_utility"),
}
COMPILED_SOURCES = frozenset(("gfam_v1", "gnn_prm", "llm_coach", "handcrafted_v1"))


class RewardProvider(Protocol):
    source: str
    rollout_batch_size: int
    supports_verified_final_correctness: bool

    def score_rollout(self, **request) -> dict: ...
    def score_rollouts(self, requests: Sequence[dict]) -> list[dict]: ...


def rollout_record(task, decomposition, executions, final_answer):
    """Allowlisted delivered artifacts; no reference, raw rejected text or rewards."""
    return canonical_record({
        "task": {"prompt": task.prompt},
        "decomposition": {
            "subtasks": [node.to_dict() for node in decomposition.nodes],
            "final_node_id": str(decomposition.final_node_id),
            "target_quantity": decomposition.target_quantity,
            "final_answer_format_hint": decomposition.final_answer_format_hint,
        },
        "trajectory": {"final_answer": final_answer, "final_node_id": str(decomposition.final_node_id)},
        "workers": [{
            "node_id": str(ex.node_id), "worker_id": str(ex.worker_id), "output_text": ex.output_text,
            "dependency_outputs": dict(ex.dependency_outputs), "invalid_reason": ex.invalid_reason,
        } for ex in executions],
    })


def evidence(value, reason):
    """Normalized rubric score, not a calibrated class probability."""
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Rubric score must be finite and in [0, 1]")
    return {"score": float(value), "reason": str(reason)}


def compile_evidence(source, scores, record, bad_class_penalty, failure_scale, verified):
    """Keep the same label weights, outcome gates and protocol checks across sources.

    The compiler's internal probability-shaped adapter is never exported as model
    predictions: heuristic and coach scores are not classifier probabilities.
    """
    def group(values, scope):
        if set(values) != set(LABELS[scope]):
            raise ValueError(f"Wrong {scope} reward criteria")
        return {name: {"probabilities": {"0": 1 - item["score"], "1": item["score"]}}
                for name, item in values.items()}

    predictions = {scope: group(scores[scope], scope) for scope in ("graph", "decomposer", "final")}
    predictions["workers"] = {node: group(values, "worker") for node, values in scores["workers"].items()}
    compiled = compile_rewards(predictions, record, bad_class_penalty,
                               verified_final_correctness=verified,
                               verified_failure_positive_scale=failure_scale)
    summary = compiled["graph_summary"]
    summary.update(reward_provider=source, score_semantics="normalized_rubric_not_probability")
    summary["estimated_success_score"] = summary.pop("predicted_success")
    summary["anti_collapse_score"] = summary.pop("predicted_anti_collapse")
    return {
        "source": source, "backend": source, "status": "scored",
        "compiled_rewards": compiled["node_rewards"], "graph_summary": summary,
        "reward_evidence": scores, "model_predictions": {}, "prm_step_scores": [],
    }


def validate_result(result, request):
    """Never silently substitute legacy rewards for missing/invalid node scores."""
    if result.get("status") == "unscored":
        if not result.get("error"):
            raise ValueError("Unscored reward requires a failure reason")
        return result
    compiled = result.get("compiled_rewards", {})
    expected = {str(ex.node_id) for ex in request["executions"]}
    if set(compiled.get("workers", {})) != expected:
        raise ValueError("Reward provider returned missing or unexpected execution IDs")
    for payload in [compiled.get("decomposer", {}), compiled.get("final", {}),
                    *compiled["workers"].values()]:
        value = payload.get("reward")
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Reward provider returned a missing/nonfinite reward")
    return result

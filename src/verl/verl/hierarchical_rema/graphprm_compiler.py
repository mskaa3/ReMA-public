from __future__ import annotations
import argparse
import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from . import graphprm_core as base

def validate_bad_class_penalty(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("bad_class_penalty must be finite and non-negative")
    return value


def binary_label_utility(centered_score: float, bad_class_penalty: float = 1.0) -> float:
    """Expected utility with good=+1 and bad=-lambda; leave probabilities intact."""
    penalty = validate_bad_class_penalty(bad_class_penalty)
    if penalty == 1.0:
        return float(centered_score)
    good_probability = (float(centered_score) + 1.0) / 2.0
    return good_probability - penalty * (1.0 - good_probability)


def compile_rewards_from_scores(compiler_inputs: dict[str, Any]) -> dict[str, Any]:
    example: base.GraphExample = compiler_inputs["example"]
    graph_scores = compiler_inputs["graph_scores"]
    decomposer_scores = compiler_inputs["decomposer_scores"]
    worker_scores = compiler_inputs["worker_scores"]
    final_scores = compiler_inputs["final_scores"]
    final_anchor_score = float(compiler_inputs["final_anchor_score"])
    bad_class_penalty = validate_bad_class_penalty(compiler_inputs.get("bad_class_penalty", 1.0))
    record = example.source_record
    final_worker_id = base.effective_final_node_id(record)

    def success(score_map: dict[str, float], key: str) -> float:
        return binary_label_utility(score_map.get(key, 0.0), bad_class_penalty)

    def lack(score_map: dict[str, float], key: str) -> float:
        # Penalties/gates use original probabilities, not the asymmetric utility.
        return base.absence_strength_from_centered(float(score_map.get(key, 0.0)))

    def node_quality(score_map: dict[str, float], *, has_utility: bool) -> float:
        if has_utility:
            return (
                0.42 * success(score_map, "task_fulfillment")
                + 0.33 * success(score_map, "execution_discipline")
                + 0.25 * success(score_map, "causal_utility")
            )
        return (
            0.58 * success(score_map, "task_fulfillment")
            + 0.42 * success(score_map, "execution_discipline")
        )

    def node_badness(score_map: dict[str, float], *, has_utility: bool) -> float:
        if has_utility:
            return base.clamp01(
                0.42 * lack(score_map, "task_fulfillment")
                + 0.33 * lack(score_map, "execution_discipline")
                + 0.25 * lack(score_map, "causal_utility")
            )
        return base.clamp01(
            0.58 * lack(score_map, "task_fulfillment")
            + 0.42 * lack(score_map, "execution_discipline")
        )

    final_failure_severity = base.clamp01((1.0 - final_anchor_score) / 2.0)
    anti_collapse_lack = base.absence_strength_from_centered(float(graph_scores.get("anti_collapse", 0.0)))

    transition_weights: dict[tuple[str, str], float] = {}
    for edge in record.get("graph", {}).get("used_dependency_edges", []):
        src = str(edge.get("from_node_id") or "")
        dst = str(edge.get("to_node_id") or "")
        if src and dst:
            transition_weights[(src, dst)] = 0.75

    downstream_graph: dict[str, list[tuple[str, float]]] = {}
    for (src, dst), weight in transition_weights.items():
        downstream_graph.setdefault(src, []).append((dst, weight))

    def max_path_weight(start: str, goal: str) -> float:
        if start == goal:
            return 1.0
        frontier = [(start, 1.0)]
        visited = {start: 1.0}
        best = 0.0
        while frontier:
            current, current_weight = frontier.pop()
            for neighbor, edge_weight in downstream_graph.get(current, []):
                new_weight = current_weight * edge_weight
                if new_weight <= visited.get(neighbor, 0.0):
                    continue
                visited[neighbor] = new_weight
                if neighbor == goal:
                    best = max(best, new_weight)
                frontier.append((neighbor, new_weight))
        return best

    worker_badness = {
        node_id: node_badness(scores, has_utility=True)
        for node_id, scores in worker_scores.items()
    }

    def downstream_consequence(node_id: str) -> float:
        numerator = 0.0
        denominator = 0.0
        for target_id, badness in worker_badness.items():
            if target_id == node_id:
                continue
            weight = max_path_weight(node_id, target_id)
            if weight <= 0:
                continue
            numerator += weight * badness
            denominator += weight
        final_weight = max_path_weight(node_id, final_worker_id)
        if final_weight > 0:
            numerator += 1.25 * final_weight * node_badness(final_scores, has_utility=True)
            denominator += 1.25 * final_weight
        return numerator / denominator if denominator > 0 else 0.0

    def positive_cap(min_cap: float, failure_weight: float, collapse_weight: float) -> float:
        return base.clamp01(
            max(
                min_cap,
                1.0 - failure_weight * final_failure_severity - collapse_weight * anti_collapse_lack,
            )
        )

    def apply_cap(value: float, cap: float) -> float:
        return value if value <= 0.0 else cap * value

    node_rewards: dict[str, Any] = {
        "decomposer": {"node_id": "d"},
        "workers": {},
        "final": {"node_id": "f"},
    }

    q_decomposer = node_quality(decomposer_scores, has_utility=False)
    decomposer_fault = node_badness(decomposer_scores, has_utility=False)
    decomposer_branch_consequence = base.mean_or_zero(
        [downstream_consequence(node_id) for node_id in worker_scores]
    )
    decomposer_pre = (
        0.90 * q_decomposer
        - 0.16 * final_failure_severity
        - 0.20 * anti_collapse_lack
        - 0.14 * decomposer_fault * decomposer_branch_consequence
    )
    decomposer_cap = positive_cap(0.35, 0.45, 0.30)
    node_rewards["decomposer"].update(
        {
            "quality": q_decomposer,
            "downstream_consequence": decomposer_branch_consequence,
            "global_failure_penalty": 0.16 * final_failure_severity,
            "anti_collapse_penalty": 0.20 * anti_collapse_lack,
            "reward_pre_cap": decomposer_pre,
            "positive_cap": decomposer_cap,
            "reward_raw": apply_cap(decomposer_pre, decomposer_cap),
        }
    )

    reach_to_final = {node_id: max_path_weight(node_id, final_worker_id) for node_id in worker_scores}
    for node_id, scores in worker_scores.items():
        q_worker = node_quality(scores, has_utility=True)
        badness = node_badness(scores, has_utility=True)
        d_u = downstream_consequence(node_id)
        final_worker_penalty = 0.0
        if node_id == final_worker_id:
            final_worker_penalty = 0.30 * max(
                node_badness(final_scores, has_utility=True),
                final_failure_severity,
            )
        reward_pre = (
            0.88 * q_worker
            - 0.28 * d_u
            - 0.22 * final_failure_severity * reach_to_final.get(node_id, 0.0) * badness
            - final_worker_penalty
        )
        cap = positive_cap(0.08, 0.88, 0.05) if node_id == final_worker_id else positive_cap(0.22, 0.70, 0.05)
        node_rewards["workers"][node_id] = {
            "node_id": node_id,
            "quality": q_worker,
            "badness": badness,
            "downstream_consequence": d_u,
            "is_final_worker": node_id == final_worker_id,
            "final_stage_penalty": final_worker_penalty,
            "reward_pre_cap": reward_pre,
            "positive_cap": cap,
            "reward_raw": apply_cap(reward_pre, cap),
        }

    q_final = node_quality(final_scores, has_utility=True)
    final_badness = node_badness(final_scores, has_utility=True)
    final_pre = 0.92 * q_final - 0.35 * final_failure_severity * final_badness
    final_cap = positive_cap(0.05, 0.95, 0.05)
    node_rewards["final"].update(
        {
            "quality": q_final,
            "badness": final_badness,
            "downstream_consequence": 0.0,
            "reward_pre_cap": final_pre,
            "positive_cap": final_cap,
            "reward_raw": apply_cap(final_pre, final_cap),
        }
    )

    node_rewards["decomposer"]["reward"] = node_rewards["decomposer"]["reward_raw"]
    node_rewards["final"]["reward"] = node_rewards["final"]["reward_raw"]
    for payload in node_rewards["workers"].values():
        payload["reward"] = payload["reward_raw"]

    return {
        "node_rewards": node_rewards,
        "graph_summary": {
            "bad_class_penalty": bad_class_penalty,
            "label_utility_positive_threshold": bad_class_penalty / (1.0 + bad_class_penalty),
            "global_outcome_score": graph_scores.get("global_outcome", 0.0),
            "anti_collapse_score": graph_scores.get("anti_collapse", 0.0),
            "final_anchor_score": final_anchor_score,
            "final_failure_severity": final_failure_severity,
            "anti_collapse_lack": anti_collapse_lack,
        },
    }

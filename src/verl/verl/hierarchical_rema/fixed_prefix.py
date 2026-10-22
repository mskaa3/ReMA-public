"""Fixed-input action comparisons, without recursive rollout branching."""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any

from .backends import WorkerExecutionRequest
from .schema import ControllerTrainingSample
from .progress import StageProgress


def add_rollout_arguments(parser):
    parser.add_argument("--worker-rollout-mode", choices=["repeated_execution", "fixed_prefix"],
                        default="repeated_execution")
    parser.add_argument("--worker-action-samples", type=int, default=8)
    parser.add_argument("--worker-continuations-per-action", type=int, default=4)
    parser.add_argument("--worker-branch-points-per-plan", type=int, default=1)
    parser.add_argument("--worker-update-rule", choices=["both_signs", "positive_only_nonterminal"],
                        default="both_signs")
    parser.add_argument("--worker-branch-seed", type=int, default=42)


def rollout_arguments(args):
    return {name: getattr(args, name) for name in (
        "worker_rollout_mode", "worker_action_samples", "worker_continuations_per_action",
        "worker_branch_points_per_plan", "worker_update_rule", "worker_branch_seed",
    )}


def identity(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def action_advantages(scores, epsilon=1e-8):
    """LOO advantages normalized by sample std, after continuation averaging."""
    if len(scores) < 2 or not all(math.isfinite(score) for score in scores):
        raise ValueError("A fixed-prefix comparison requires at least two finite action scores")
    total = math.fsum(scores)
    baselines = [(total - score) / (len(scores) - 1) for score in scores]
    deltas = [score - baseline for score, baseline in zip(scores, baselines)]
    mean = math.fsum(deltas) / len(deltas)
    std = math.sqrt(math.fsum((delta - mean) ** 2 for delta in deltas) / (len(deltas) - 1))
    advantages = [delta / (std + epsilon) for delta in deltas] if std > epsilon else [0.0] * len(deltas)
    return baselines, deltas, advantages


@dataclass
class Prefix:
    state: Any
    stop: int
    comparison: dict


def request_for(state, worker_pool):
    node = state.decomposition.nodes_by_id()[state.topo_order[state.next_node_index]]
    assignment = state.selection.assignment_for(node.node_id)
    return WorkerExecutionRequest(
        task=state.task, decomposition=state.decomposition, node=node,
        worker=worker_pool.workers_by_id()[assignment.worker_id],
        dependency_outputs={key: state.outputs[key] for key in node.dependencies},
        compatibility=assignment.compatibility,
    )


def execute_requests(orchestrator, states, worker_pool):
    requests = [request_for(state, worker_pool) for state in states]
    executions = orchestrator.backend.execute_workers_batch(requests)
    if len(executions) != len(states):
        raise ValueError("Fixed-prefix backend returned misaligned executions")
    for state, request, execution in zip(states, requests, executions):
        if (execution.node_id != request.node.node_id
                or execution.worker_id != request.worker.worker_id
                or execution.dependency_outputs != request.dependency_outputs):
            raise ValueError("Fixed-prefix backend changed node, executor, or dependency inputs")
        state.executions.append(execution)
        state.outputs[execution.node_id] = execution.output_text
        state.next_node_index += 1


def generate_branches(orchestrator, tasks, plans, worker_pool, config, worker_performance,
                      state_factory, progress_label, round_index, selection_offset):
    """Generate all prefixes, all focal actions, then independently complete suffixes."""
    prefixes, comparisons = [], {}
    for task_index, task_plans in enumerate(plans):
        for plan_index, plan in enumerate(task_plans):
            bucket = comparisons.setdefault((task_index, plan_index), [])
            if orchestrator._uses_fallback_path(plan.raw_payload):
                bucket.append({"status": "skipped", "reason": "fallback_plan"})
                continue
            order = plan.topological_order()
            seed = identity([config.worker_branch_seed, progress_label, round_index,
                             tasks[task_index].task_id, plan.decomposition_id, plan.to_dict()])
            positions = random.Random(seed).sample(
                range(len(order)), min(config.worker_branch_points_per_plan, len(order)))
            for point, position in enumerate(positions):
                group = {
                    "group_id": f"prefix:{seed}:{point}", "node_id": order[position],
                    "sampling_seed": seed, "branch_point": point,
                    "expected_actions": config.worker_action_samples,
                    "expected_continuations": (
                        1 if order[position] == plan.final_node_id
                        else config.worker_continuations_per_action),
                    "update_rule": config.worker_update_rule,
                    "score_source": "provider_node_reward", "status": "pending",
                }
                bucket.append(group)
                state = state_factory(
                    task_index=task_index, decomposition_index=plan_index, selection_index=0,
                    task=tasks[task_index], decomposition=plan, topo_order=order,
                    selection=orchestrator._build_executor_rollout_candidate(
                        decomposition=plan, worker_pool=worker_pool,
                        worker_performance=worker_performance, rollout_index=0))
                prefixes.append(Prefix(state, position, group))

    chunk_size = max(1, int(getattr(getattr(orchestrator.backend, "config", None),
                                    "worker_batch_size", 16)))
    with StageProgress("frozen_prefix", sum(prefix.stop for prefix in prefixes),
                       context=progress_label) as progress:
        while True:
            ready = [prefix.state for prefix in prefixes if prefix.state.next_node_index < prefix.stop]
            if not ready:
                break
            for start in range(0, len(ready), chunk_size):
                batch = ready[start:start + chunk_size]
                execute_requests(orchestrator, batch, worker_pool)
                for _ in batch:
                    progress.advance()

    actions = []
    for prefix in prefixes:
        request = request_for(prefix.state, worker_pool)
        group = prefix.comparison
        group["executor_policy"] = {
            "worker_id": request.worker.worker_id,
            "base_model_path": request.worker.base_model_path or worker_pool.base_model_path,
            "lora_adapter_path": request.worker.lora_adapter_path,
        }
        group["system_prompt"] = request.worker.system_prompt
        group["dependency_outputs"] = dict(request.dependency_outputs)
        group["prefix_node_ids"] = [item.node_id for item in prefix.state.executions]
        group["prefix_outputs"] = dict(prefix.state.outputs)
        for action_index in range(config.worker_action_samples):
            actions.append((prefix, action_index, deepcopy(prefix.state)))
    with StageProgress("focal_actions", len(actions), context=progress_label) as progress:
        for start in range(0, len(actions), chunk_size):
            batch = actions[start:start + chunk_size]
            execute_requests(orchestrator, [item[2] for item in batch], worker_pool)
            for _ in batch:
                progress.advance()

    states = []
    for prefix, action_index, action in actions:
        group = prefix.comparison
        execution = action.executions[-1]
        prompt = execution.worker_prompt
        if not prompt:
            raise ValueError("Fixed-prefix sampling requires the actual rendered worker prompt")
        if "prompt_text" in group and group["prompt_text"] != prompt:
            raise ValueError("Fixed-prefix actions received different worker prompts")
        group["prompt_text"] = prompt
        group["prefix_hash"] = identity({
            "prompt": prompt, "system_prompt": group["system_prompt"],
            "prefix_outputs": group["prefix_outputs"], "policy": group["executor_policy"],
        })
        group["group_id"] = (
            f"prefix:{group['sampling_seed']}:{group['branch_point']}:{group['prefix_hash']}")
        for continuation in range(group["expected_continuations"]):
            # Scoring mutates execution metadata. No prefix/action object is shared
            # between completed graphs, even when its generated text is identical.
            state = deepcopy(action)
            state.selection_index = selection_offset + len(states)
            state.selection.selection_id = f"{group['group_id']}:a{action_index}:c{continuation}"
            state.branch_metadata = {
                "group_id": group["group_id"], "prefix_hash": group["prefix_hash"],
                "node_id": group["node_id"], "action_index": action_index,
                "continuation_index": continuation,
                "expected_actions": group["expected_actions"],
                "expected_continuations": group["expected_continuations"],
            }
            states.append(state)

    scored = orchestrator._execute_selections_batch(states, worker_pool, progress_label) if states else {}
    result = {}
    for state in states:
        key = (state.task_index, state.decomposition_index, state.selection_index)
        result.setdefault(key[:2], []).append(scored[key])
    return result, comparisons


def _unavailable_causes(rollout):
    reward = rollout.reward
    error = reward.verification_error or {}
    code = str(error.get("code") or "unknown")
    if reward.final_answer_correctness not in (0.0, 1.0):
        stage = "reference_validation" if error.get("stage") == "reference_validation" else "final_correctness"
        return [f"{stage}:{code}"]
    if reward.verification_status == "unverified":
        return [f"required_auxiliary:{error.get('stage', 'unknown')}:{code}"]
    if reward.total_reward is None:
        model_error = rollout.reward_model_outputs.get("error") or {}
        return [f"reward_model:{model_error.get('code') or model_error.get('type') or 'unavailable'}"]
    if not math.isfinite(reward.total_reward):
        return ["nonfinite_rollout_reward"]
    return []


def build_samples(orchestrator, task, decompositions, config, min_group_size):
    samples, sizes = [], []
    stats = {
        "num_groups_total": 0, "num_groups_used": 0, "num_groups_skipped": 0,
        "num_samples_total": 0, "num_samples_used": 0, "num_samples_skipped": 0,
        "num_groups_incomplete": 0, "num_groups_zero_signal": 0,
        "num_actions_skipped_gate": 0, "num_actions_skipped_sign": 0,
        "incomplete_groups_by_cause": {}, "unavailable_continuations_by_cause": {},
    }
    for decomposition in decompositions:
        by_group = {}
        for rollout in decomposition.selections:
            if rollout.branch_metadata:
                by_group.setdefault(rollout.branch_metadata["group_id"], []).append(rollout)
        for comparison in decomposition.worker_comparisons:
            if comparison["status"] == "skipped":
                continue
            stats["num_groups_total"] += 1
            count, repeats = comparison["expected_actions"], comparison["expected_continuations"]
            stats["num_samples_total"] += count
            actions = {index: [] for index in range(count)}
            errors = []
            details = []
            for rollout in by_group.get(comparison["group_id"], []):
                meta = rollout.branch_metadata
                if meta["action_index"] not in actions:
                    raise ValueError("Unexpected fixed-prefix action identity")
                actions[meta["action_index"]].append(rollout)
            records = []
            for index, rollouts in actions.items():
                if (len(rollouts) != repeats
                        or {r.branch_metadata["continuation_index"] for r in rollouts} != set(range(repeats))):
                    errors.append(f"action_{index}:missing_or_duplicate_continuation")
                    indices = [r.branch_metadata["continuation_index"] for r in rollouts]
                    causes = []
                    if set(range(repeats)) - set(indices):
                        causes.append("missing_continuation")
                    if len(indices) != len(set(indices)):
                        causes.append("duplicate_continuation")
                    if set(indices) - set(range(repeats)):
                        causes.append("unexpected_continuation")
                    details.append({"action_index": index, "causes": causes,
                                    "expected_continuations": repeats, "observed_indices": indices})
                    continue
                rewards, executions, gates = [], [], []
                for rollout in sorted(rollouts, key=lambda r: r.branch_metadata["continuation_index"]):
                    causes = _unavailable_causes(rollout)
                    if causes:
                        errors.append(f"action_{index}:unscored_or_unverified")
                        details.append({"action_index": index,
                                        "continuation_index": rollout.branch_metadata["continuation_index"],
                                        "rollout_id": rollout.selection.selection_id, "causes": causes})
                        continue
                    matched = [ex for ex in rollout.executions if ex.node_id == comparison["node_id"]]
                    if len(matched) != 1:
                        raise ValueError("Missing or duplicate focal node in completed graph")
                    execution = matched[0]
                    if (execution.worker_prompt != comparison["prompt_text"]
                            or execution.dependency_outputs != comparison["dependency_outputs"]
                            or execution.worker_id != comparison["executor_policy"]["worker_id"]
                            or rollout.branch_metadata["prefix_hash"] != comparison["prefix_hash"]):
                        raise ValueError("Fixed-prefix comparison input identity changed")
                    reward = orchestrator._worker_training_reward(rollout, execution, allow_excluded=True)
                    if not math.isfinite(reward):
                        errors.append(f"action_{index}:nonfinite_reward")
                        details.append({"action_index": index,
                                        "continuation_index": rollout.branch_metadata["continuation_index"],
                                        "rollout_id": rollout.selection.selection_id,
                                        "causes": ["nonfinite_focal_reward"]})
                        continue
                    rewards.append(reward)
                    executions.append(execution)
                    if rollout.training_excluded:
                        gates.append("training_safety")
                    if execution.final_answer_leak and execution.node_id != decomposition.decomposition.final_node_id:
                        gates.append("final_answer_leak")
                    if not str(execution.raw_output_text or "").strip():
                        gates.append("empty_completion")
                if executions and len({ex.raw_output_text for ex in executions}) != 1:
                    raise ValueError("A sampled action changed across its continuations")
                records.append({
                    "action_index": index, "continuation_rewards": rewards,
                    "continuation_ids": [r.selection.selection_id for r in rollouts],
                    "score": math.fsum(rewards) / repeats if len(rewards) == repeats else None,
                    "gate_reasons": sorted(set(gates)),
                    "completion_text": executions[0].raw_output_text if executions else "",
                })
            comparison["actions"] = records
            if errors or count < min_group_size:
                comparison.update(status="skipped", reason="incomplete_group" if errors else "minimum_group_size",
                                  errors=errors, incomplete_details=details)
                # A group counts once per cause; several causes may affect it.
                causes = sorted({cause for detail in details for cause in detail["causes"]})
                comparison["incomplete_causes"] = causes
                for cause in causes:
                    counts = stats["incomplete_groups_by_cause"]
                    counts[cause] = counts.get(cause, 0) + 1
                counts = Counter(stats["unavailable_continuations_by_cause"])
                for detail in details:
                    if "rollout_id" in detail:
                        counts.update(detail["causes"])
                stats["unavailable_continuations_by_cause"] = dict(counts)
                stats["num_groups_skipped"] += 1
                stats["num_samples_skipped"] += count
                stats["num_groups_incomplete"] += bool(errors)
                continue
            scores = [record["score"] for record in records]
            baselines, deltas, advantages = action_advantages(scores)
            zero_signal = not any(advantages)
            comparison.update(status="zero_signal" if zero_signal else "scored",
                              action_scores=scores)
            stats["num_groups_zero_signal"] += zero_signal
            used = 0
            terminal = comparison["node_id"] == decomposition.decomposition.final_node_id
            for record, baseline, delta, advantage in zip(records, baselines, deltas, advantages):
                reason = (
                    "gate" if record["gate_reasons"] else "zero_signal" if zero_signal else
                    "nonpositive_advantage" if config.worker_update_rule == "positive_only_nonterminal"
                    and not terminal and advantage <= 0 else None
                )
                record.update(baseline=baseline, raw_advantage=delta, advantage=advantage,
                              update_eligible=reason is None, update_skip_reason=reason)
                if reason:
                    stats["num_samples_skipped"] += 1
                    stats["num_actions_skipped_gate"] += reason == "gate"
                    stats["num_actions_skipped_sign"] += reason == "nonpositive_advantage"
                    continue
                policy = comparison["executor_policy"]
                samples.append(ControllerTrainingSample(
                    role="worker", policy_id="shared_worker", group_id=comparison["group_id"],
                    prompt_text=comparison["prompt_text"], completion_text=record["completion_text"],
                    reward=record["score"], advantage=advantage,
                    metadata={
                        "model_path": policy["base_model_path"], "worker_id": policy["worker_id"],
                        "node_id": comparison["node_id"],
                        "decomposition_id": decomposition.decomposition.decomposition_id,
                        "selection_id": record["continuation_ids"][0],
                        "advantage_group_kind": "fixed_prefix_action_means",
                        "advantage_group_size": count, "advantage_group_executor_policy": policy,
                        "prefix_hash": comparison["prefix_hash"], "update_rule": config.worker_update_rule,
                        "action_index": record["action_index"],
                        "continuation_ids": record["continuation_ids"],
                        "continuation_rewards": record["continuation_rewards"],
                        "leave_one_out_baseline": baseline, "raw_advantage": delta,
                    }))
                used += 1
            stats["num_samples_used"] += used
            stats["num_groups_used" if used else "num_groups_skipped"] += 1
            if used:
                sizes.append(count)
    stats["mean_group_size_used"] = math.fsum(sizes) / len(sizes) if sizes else 0.0
    return samples, stats

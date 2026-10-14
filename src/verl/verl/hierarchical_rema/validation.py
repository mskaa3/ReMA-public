"""Validation scheduling and observational diagnostics, not reward policy."""
from collections import Counter, defaultdict
import math


def validation_mode(args, epoch_number):
    mode = getattr(args, "validation_mode", "auto")
    every = int(getattr(args, "val_diagnostics_every", 4))
    if every < 0:
        raise ValueError("val_diagnostics_every must be nonnegative")
    candidates = args.val_num_decompositions * (args.val_num_executor_rollouts or args.val_num_selections)
    if mode == "routine" and candidates != 1:
        raise ValueError("Routine validation requires exactly one plan and one execution per task")
    if mode != "auto":
        return mode
    if candidates != 1 or epoch_number == args.num_epochs:
        return "diagnostic"
    interval = int(args.external_validation_every_n_epochs)
    event = epoch_number // interval if interval > 0 else 1
    return "diagnostic" if every and event and event % every == 0 else "routine"


def hide_uncomputed_rewards(summary):
    """Do not present legacy handcrafted values as Graph+PRM measurements."""
    for target in [summary, *summary.get("subsets", {}).values()]:
        for key in target:
            if key.startswith("mean_") and "reward" in key:
                target[key] = None
        target["reward_metric_task_counts"] = {}


class ValidationAudit:
    def __init__(self):
        self.counts = Counter()
        self.sums = Counter()
        self.subsets = defaultdict(ValidationAudit)

    def add(self, rollout, subset=None):
        if subset is not None:
            self.subsets[subset].add(rollout)
        c, sums = self.counts, self.sums
        for plan in rollout.decompositions:
            c["plans"] += 1
            payload = plan.decomposition.raw_payload or {}
            validation = payload.get("validation") or payload.get("_validation") or {}
            c["plans_with_validation_metadata"] += bool(validation)
            errors = validation.get("errors_before_success") or validation.get("errors") or []
            c["plans_with_parse_errors"] += bool(errors or validation.get("unparseable_batch_output"))
            c["fallback_plans"] += bool(validation.get("fallback_used"))
            c["retried_plans"] += bool(validation.get("attempt", 0))
            c["locally_repaired_plans"] += bool(validation.get("local_salvage_used") or
                                               validation.get("final_node_repair"))
            c["hard_truncated_plans"] += bool(plan.decomposition.was_hard_truncated)
            sums["plan_nodes"] += len(plan.decomposition.nodes)
            sums["planner_output_chars"] += len(plan.decomposition.raw_text or "")
            for selection in plan.selections:
                c["trajectories"] += 1
                sums["final_answer_chars"] += len(selection.final_answer or "")
                for execution in selection.executions:
                    c["worker_outputs"] += 1
                    c["invalid_worker_outputs"] += bool(execution.invalid_reason)
                    sums["worker_output_chars"] += len(execution.output_text or "")
                    sums["raw_worker_output_chars"] += len(execution.raw_output_text or execution.output_text or "")
                output = selection.reward_model_outputs or {}
                compiled = output.get("compiled_rewards") or {}
                if not compiled or output.get("status") == "unscored":
                    continue
                c["scored_trajectories"] += 1
                graph = output.get("graph_summary") or {}
                outcome = selection.reward.final_answer_correctness
                probability = graph.get("predicted_success")
                if outcome in (0, 1) and isinstance(probability, (float, int)) and math.isfinite(probability):
                    c["outcome_prediction_pairs"] += 1
                    c["outcome_prediction_disagreements"] += int(probability >= .5) != int(outcome)
                c["outcome_gated_trajectories"] += bool(graph.get("verified_failure_gate_applied"))
                if outcome == 0:
                    c["scored_incorrect_trajectories"] += 1
                    final_reward = (compiled.get("final") or {}).get("reward")
                    if isinstance(final_reward, (int, float)) and math.isfinite(final_reward):
                        c["incorrect_with_final_reward"] += 1
                        c["incorrect_positive_final_reward"] += final_reward > 0
                    # Count each executed node once, using the same final/worker
                    # routing as training rather than double-counting the final.
                    nodes = [compiled.get("final") or {} if ex.node_id == plan.decomposition.final_node_id
                             else (compiled.get("workers") or {}).get(ex.node_id, {})
                             for ex in selection.executions]
                    for node in nodes:
                        before, after = node.get("reward_before_outcome_gate"), node.get("reward")
                        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (before, after)):
                            c["incorrect_execution_nodes_with_gate_data"] += 1
                            sums["incorrect_positive_credit_before_gate"] += max(before, 0)
                            sums["incorrect_positive_credit_after_gate"] += max(after, 0)

    def summary(self):
        c, sums = self.counts, self.sums
        def ratio(value, denominator):
            return value / c[denominator] if c[denominator] else None
        return {
            **dict(c),
            "plan_validation_metadata_coverage": ratio(c["plans_with_validation_metadata"], "plans"),
            "plan_parse_error_rate": ratio(c["plans_with_parse_errors"], "plans_with_validation_metadata"),
            "plan_fallback_rate": ratio(c["fallback_plans"], "plans_with_validation_metadata"),
            "plan_retry_rate": ratio(c["retried_plans"], "plans_with_validation_metadata"),
            "plan_local_repair_rate": ratio(c["locally_repaired_plans"], "plans_with_validation_metadata"),
            "invalid_worker_output_rate": ratio(c["invalid_worker_outputs"], "worker_outputs"),
            "mean_plan_nodes": ratio(sums["plan_nodes"], "plans"),
            "mean_planner_output_chars": ratio(sums["planner_output_chars"], "plans"),
            "mean_worker_output_chars": ratio(sums["worker_output_chars"], "worker_outputs"),
            "mean_raw_worker_output_chars": ratio(sums["raw_worker_output_chars"], "worker_outputs"),
            "mean_final_answer_chars": ratio(sums["final_answer_chars"], "trajectories"),
            "reward_outcome_disagreement_rate": ratio(c["outcome_prediction_disagreements"], "outcome_prediction_pairs"),
            "incorrect_positive_final_reward_rate": ratio(c["incorrect_positive_final_reward"], "incorrect_with_final_reward"),
            "incorrect_mean_positive_credit_before_gate": ratio(sums["incorrect_positive_credit_before_gate"], "incorrect_execution_nodes_with_gate_data"),
            "incorrect_mean_positive_credit_after_gate": ratio(sums["incorrect_positive_credit_after_gate"], "incorrect_execution_nodes_with_gate_data"),
        }


def diagnostic_examples(rollouts):
    for rollout in rollouts:
        for plan_index, plan in enumerate(rollout.decompositions):
            for repeat_index, selection in enumerate(plan.selections):
                yield {"task": rollout.task.to_dict(), "plan_index": plan_index,
                       "execution_repeat_index": repeat_index, "decomposition": plan.decomposition.to_dict(),
                       "execution": selection.to_dict()}

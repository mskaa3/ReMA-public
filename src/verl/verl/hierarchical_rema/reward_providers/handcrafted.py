"""Label-aligned structural/behavioral rules with conservative semantic evidence."""
import ast
from collections import Counter
from fractions import Fraction
import operator
import re

from .base import compile_evidence, evidence, rollout_record, validate_result

VERSION = "handcrafted_v1"


def _normal(text):
    return " ".join(str(text).strip().lower().split())


def _arithmetic(text):
    """Bounded arithmetic only: no eval, symbolic solving, powers or arbitrary code."""
    if len(text) > 160 or not re.fullmatch(r"[\d\s.+*/()\-]+", text):
        raise ValueError("Unsupported arithmetic")
    tree = ast.parse(text.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 60:
        raise ValueError("Arithmetic too complex")
    ops = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (float, int):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("Unsupported arithmetic")
    return visit(tree.body)


def arithmetic_evidence(output, instruction):
    checked = 0
    for line in output.splitlines():
        parts = line.strip().strip("$").split("=")
        if len(parts) < 2:
            continue
        try:
            values = [_arithmetic(part) for part in parts]
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            continue
        checked += 1
        if len(set(values)) != 1:
            return evidence(0., "false_numeric_equality")
    match = re.fullmatch(r"\s*(?:calculate|compute|evaluate)\s+([\d\s.+*/()\-]+?)[.?]?\s*",
                         instruction, re.I)
    if match:
        try:
            answer = output.strip().strip("$").split("=")[-1]
            correct = _arithmetic(answer) == _arithmetic(match[1])
            return evidence(float(correct), "verified_atomic_arithmetic")
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            pass
    if checked:
        return evidence(.75, f"{checked}_valid_equalities_task_completion_not_proven")
    return evidence(.5, "mathematical_completion_unknown")


class HandcraftedRewardProvider:
    source = "handcrafted_v1"
    supports_verified_final_correctness = True
    required_auxiliary_checks = ()
    rollout_batch_size = 64

    def __init__(self, bad_class_penalty=1., failure_scale=.25):
        self.bad_class_penalty = bad_class_penalty
        self.failure_scale = failure_scale

    def score_rollout(self, *, task, decomposition, selection, executions,
                      final_answer, verified_final_correctness=None):
        row = rollout_record(task, decomposition, executions, final_answer)
        tasks = {t["node_id"]: t for t in row["decomposition"]["subtasks"]}
        workers = {w["node_id"]: w for w in row["workers"]}
        final_id = row["decomposition"]["final_node_id"]
        signatures = Counter(_normal(w["output_text"]) for w in workers.values() if w["output_text"])
        reachable, pending = {final_id}, [final_id]
        while pending:
            for dep in tasks[pending.pop()]["dependencies"]:
                if dep in tasks and dep not in reachable:
                    reachable.add(dep)
                    pending.append(dep)
        plan_flags = []
        if row["dependency_audit"]["invalid_declared_plan"]:
            plan_flags.append("invalid_dependency_graph")
        if len(reachable) < len(tasks):
            plan_flags.append("nodes_disconnected_from_final")
        instructions = [_normal(t["instruction"]) for t in tasks.values()]
        if len(set(instructions)) < len(instructions):
            plan_flags.append("duplicate_subtask_instructions")
        for node, subtask in tasks.items():
            if node != final_id and re.search(r"\b(?:give|return|output)\s+(?:the\s+)?final\s+answer\b",
                                             subtask["instruction"], re.I):
                plan_flags.append("non_final_answer_instruction")
        collapse_flags = list(plan_flags)
        # A copied finalization artifact may be legitimate formatting. Require
        # repeated intermediate prose, not merely identical numerical results.
        redundant = {node for node, w in workers.items() if node != final_id
                     and len(w["output_text"].split()) >= 8 and signatures[_normal(w["output_text"])] > 1}
        if redundant:
            collapse_flags.append("repeated_intermediate_artifacts_proxy")
        safe = max(0., 1. - .35 * len(set(collapse_flags)))
        scores = {
            "graph": {
                "global_outcome": evidence(verified_final_correctness if verified_final_correctness is not None else .5,
                                           "verified_final_outcome" if verified_final_correctness is not None else "unknown"),
                "anti_collapse": evidence(safe, ", ".join(collapse_flags) or "no_structural_collapse_detected"),
            },
            "decomposer": {
                "task_fulfillment": evidence(.25 if plan_flags else .5,
                                             "structural_defect" if plan_flags else "semantic_plan_sufficiency_unknown"),
                "execution_discipline": evidence(max(0., 1 - .35 * len(set(plan_flags))),
                                                 ", ".join(plan_flags) or "valid_plan_structure"),
            },
            "workers": {},
        }
        for node, worker in workers.items():
            text = worker["output_text"]
            invalid = bool(worker.get("invalid_reason")) or not text.strip()
            missing = set(tasks[node]["dependencies"]) - set(worker["provided_inputs"])
            repeated = len(text.split()) >= 30 and len(set(text.split())) / len(text.split()) < .15
            flags = ["invalid_delivered_artifact"] if invalid else []
            if repeated:
                flags.append("extreme_repetition")
            if re.search(r"\b(?:ignore (?:previous|above) instructions|assign (?:me )?(?:a )?(?:score|reward))\b", text, re.I):
                flags.append("reward_or_instruction_manipulation")
            fulfillment = evidence(0., "missing_or_invalid_result") if invalid else arithmetic_evidence(text, tasks[node]["instruction"])
            if node == final_id and verified_final_correctness is not None:
                fulfillment = evidence(verified_final_correctness, "verified_final_outcome")
            # Provision and reachability are possible influence, not actual use.
            utility = evidence(.5, "on_final_dependency_path_actual_causal_use_unknown")
            if invalid or node not in reachable:
                utility = evidence(0., "no_valid_artifact_on_final_dependency_path")
            elif node in redundant or repeated:
                utility = evidence(.25, "redundancy_proxy_not_counterfactual_proof")
            elif missing:
                utility = evidence(.5, "missing_supplied_context_not_automatically_worker_fault")
            scores["workers"][node] = {
                "task_fulfillment": fulfillment,
                "execution_discipline": evidence(0. if invalid else max(0., 1 - .5 * len(flags)),
                                                  ", ".join(flags) or "no_visible_protocol_violation"),
                "causal_utility": utility,
            }
        scores["final"] = dict(scores["workers"][final_id])
        result = compile_evidence(self.source, scores, row, self.bad_class_penalty,
                                  self.failure_scale, verified_final_correctness)
        result["graph_summary"]["heuristic_version"] = VERSION
        return validate_result(result, {"executions": executions})

    def score_rollouts(self, requests):
        return [self.score_rollout(**request) for request in requests]

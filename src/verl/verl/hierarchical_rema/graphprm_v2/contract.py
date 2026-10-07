"""One input contract for dataset migration, graph training and live scoring."""
import copy
from collections import Counter
from graphlib import TopologicalSorter, CycleError

from . import FEATURE_SCHEMA
from .graphprm_inputs import normalize_text_record, clean_result_text


def canonical_record(record, legacy_context_policy="unknown"):
    if legacy_context_policy not in ("unknown", "reconstruct_declared"):
        raise ValueError("Unknown legacy context policy")
    row = normalize_text_record(record)
    tasks = {str(t["node_id"]): t for t in row["decomposition"]["subtasks"]}
    workers = {str(w["node_id"]): w for w in row["workers"]}
    if len(tasks) != len(row["decomposition"]["subtasks"]) or len(workers) != len(row["workers"]):
        raise ValueError("Duplicate subtask/execution node IDs")
    declared, provided = [], []
    invalid_plan = False
    for node, task in tasks.items():
        task["node_id"] = node
        task["dependencies"] = sorted(set(map(str, task.get("dependencies", []))))
        for dep in task["dependencies"]:
            if dep not in tasks or dep == node:
                invalid_plan = True
            declared.append({"from_node_id": dep, "to_node_id": node})
    try:
        list(TopologicalSorter({n: t['dependencies'] for n, t in tasks.items()}).static_order())
    except CycleError:
        invalid_plan = True
    for node, worker in workers.items():
        if node not in tasks:
            raise ValueError(f"Execution {node} has no subtask")
        worker["node_id"] = node
        worker["subtask"] = tasks[node]
        worker["declared_dependencies"] = tasks[node]["dependencies"]
        if "provided_inputs" in worker:
            inputs = worker["provided_inputs"]
            status = worker.get("dependency_context_status", "observed")
        elif "dependency_outputs" in worker:
            # Presence matters: an explicitly recorded {} means no inputs.
            inputs, status = worker["dependency_outputs"], "observed"
        elif legacy_context_policy == "reconstruct_declared" and not invalid_plan:
            inputs = {dep: workers[dep]["output_text"] for dep in task_dependencies(worker) if dep in workers}
            status = "reconstructed_assumption"
        else:
            inputs, status = {}, "unknown"
        if not isinstance(inputs, dict) or status not in ("observed", "reconstructed_assumption", "unknown"):
            raise ValueError("Invalid provided-input snapshot or context status")
        if status == 'unknown' and inputs:
            raise ValueError('Unknown context cannot also contain supplied-input snapshots')
        worker["provided_inputs"] = {str(k): clean_result_text(v) for k, v in inputs.items()}
        worker["dependency_context_status"] = status
        # These fields have ambiguous historical meanings; they are never features.
        for key in ("dependency_used", "downstream_used_by", "upstream_context"):
            worker.pop(key, None)
        for dep in worker["provided_inputs"]:
            if dep not in workers or dep == node:
                raise ValueError(f"Provided input {dep} -> {node} has no valid execution")
            provided.append({"from_node_id": dep, "to_node_id": node})
    recipients = Counter(edge["from_node_id"] for edge in provided)
    for node, worker in workers.items():
        worker["downstream_recipient_count"] = recipients[node]
        others = [w for n, w in workers.items() if n != node]
        worker['recipient_context_coverage'] = (sum(w['dependency_context_status'] != 'unknown' for w in others)
                                                  / len(others) if others else 1.)
    row["graph"] = {"declared_dependency_edges": declared, "provided_dependency_edges": provided}
    row["feature_schema"] = FEATURE_SCHEMA
    row['dependency_audit'] = {'invalid_declared_plan': invalid_plan}
    final_id = str(row["decomposition"].get("final_node_id") or row["trajectory"].get("final_node_id") or "")
    if final_id not in workers:
        raise ValueError(f"Final execution {final_id!r} is missing")
    row["decomposition"]["final_node_id"] = final_id
    for node, worker in workers.items():
        worker["is_final_node"] = node == final_id
    return row


def task_dependencies(worker):
    return worker["declared_dependencies"]


def model_record(record):
    """Strict feature allowlist: no labels, source IDs, reference or judge metadata."""
    row = canonical_record(record)
    plan = row["decomposition"]
    subtask_fields = ('node_id', 'instruction', 'dependencies', 'required_skills', 'required_skills_note')
    subtasks = [{k: copy.deepcopy(t[k]) for k in subtask_fields if k in t} for t in plan['subtasks']]
    by_id = {str(t['node_id']): t for t in subtasks}
    worker_fields = ('node_id', 'declared_dependencies', 'provided_inputs', 'dependency_context_status',
                     'downstream_recipient_count', 'recipient_context_coverage', 'output_text', 'is_final_node')
    return {
        "feature_schema": FEATURE_SCHEMA,
        "task": {"prompt": row["task"]["prompt"]},
        "trajectory": {"final_answer": row["trajectory"].get("final_answer", ""), "final_node_id": plan["final_node_id"]},
        "decomposition": {**{k: copy.deepcopy(plan.get(k, "")) for k in (
            "final_node_id", "target_quantity", "final_answer_format_hint")}, 'subtasks': subtasks},
        "workers": [{**{k: copy.deepcopy(w[k]) for k in worker_fields},
                     'subtask': copy.deepcopy(by_id[w['node_id']])} for w in row["workers"]],
        "graph": row["graph"],
    }

from __future__ import annotations

import ast
import json
import re
from copy import deepcopy
from json import JSONDecodeError
from typing import Any, Dict, List

from .schema import (
    DecompositionCandidate,
    RolloutConfig,
    SelectionCandidate,
    SubtaskNode,
    WorkerAssignment,
    WorkerPoolConfig,
)


class StructuredOutputError(ValueError):
    pass


KNOWN_CONTROLLER_TAGS = (
    "decomposition_json",
    "selection_json",
    "answer_json",
    "answer",
    "json",
)

KNOWN_DECOMPOSITION_TAGS = (
    "decomposition_plan",
    "decomposition_json",
    "decomposition",
    "answer_json",
    "answer",
    "json",
)

KNOWN_SELECTION_TAGS = (
    "selection_plan",
    "selection_json",
    "selection",
    "answer_json",
    "answer",
    "json",
)

def _extract_tagged_content(text: str, tags: tuple[str, ...] = KNOWN_CONTROLLER_TAGS) -> str:
    for tag in tags:
        pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", flags=re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return text


def _normalize_structured_text(text: str, tags: tuple[str, ...]) -> str:
    stripped = _extract_tagged_content(text.strip(), tags=tags)
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    return stripped.strip()


def _parse_csv_field(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    text = str(raw_value).strip()
    if not text or text.lower() in {"none", "null", "[]", "n/a"}:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]

def format_decomposition_plan(candidate: DecompositionCandidate) -> str:
    lines = [
        "<decomposition_plan>",
        f"DECOMPOSITION_ID: {candidate.decomposition_id}",
        f"SUMMARY: {candidate.summary}",
        f"FINAL_NODE_ID: {candidate.final_node_id}",
    ]
    for node in candidate.nodes:
        lines.extend(
            [
                f"NODE: {node.node_id}",
                f"INSTRUCTION: {node.instruction}",
                f"DEPENDENCIES: {', '.join(node.dependencies) if node.dependencies else 'none'}",
                f"REQUIRED_SKILLS: {', '.join(node.required_skills) if node.required_skills else 'none'}",
                f"OUTPUT_KEY: {node.output_key}",
            ]
        )
    lines.append("</decomposition_plan>")
    return "\n".join(lines)


def format_selection_plan(candidate: SelectionCandidate) -> str:
    lines = [
        "<selection_plan>",
        f"SELECTION_ID: {candidate.selection_id}",
    ]
    for assignment in candidate.assignments:
        lines.append(f"{assignment.node_id} -> {assignment.worker_id}")
    lines.append("</selection_plan>")
    return "\n".join(lines)


def _extract_key_value_payload(text: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    current_node: Dict[str, Any] | None = None
    nodes: List[Dict[str, Any]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    def finalize_node() -> None:
        nonlocal current_node
        if current_node is not None and current_node.get("node_id"):
            nodes.append(current_node)
        current_node = None

    for line in lines:
        if line.startswith("<") and line.endswith(">"):
            continue
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip().upper()
        raw_value = raw_value.strip()
        if key == "NODE":
            finalize_node()
            current_node = {"node_id": raw_value}
            continue
        if current_node is not None and key in {"INSTRUCTION", "DEPENDENCIES", "REQUIRED_SKILLS", "SKILLS", "OUTPUT_KEY"}:
            if key == "INSTRUCTION":
                current_node["instruction"] = raw_value
            elif key in {"REQUIRED_SKILLS", "SKILLS"}:
                current_node["required_skills"] = _parse_csv_field(raw_value)
            elif key == "DEPENDENCIES":
                current_node["dependencies"] = _parse_csv_field(raw_value)
            elif key == "OUTPUT_KEY":
                current_node["output_key"] = raw_value
            continue
        if key == "DECOMPOSITION_ID":
            payload["decomposition_id"] = raw_value
        elif key == "SUMMARY":
            payload["summary"] = raw_value
        elif key in {"FINAL_NODE_ID", "FINAL_NODE"}:
            payload["final_node_id"] = raw_value

    finalize_node()
    if nodes:
        payload["nodes"] = nodes
    return payload


def _parse_decomposition_plan(text: str) -> Dict[str, Any]:
    normalized = _normalize_structured_text(text, tags=KNOWN_DECOMPOSITION_TAGS)
    payload = _extract_key_value_payload(normalized)
    if payload.get("nodes"):
        payload.setdefault("summary", "Compact decomposition.")
        if not payload.get("final_node_id"):
            payload["final_node_id"] = payload["nodes"][-1]["node_id"]
        return payload
    raise StructuredOutputError("Could not parse a decomposition plan from controller output")


def _parse_selection_plan(text: str) -> Dict[str, Any]:
    normalized = _normalize_structured_text(text, tags=KNOWN_SELECTION_TAGS)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    payload: Dict[str, Any] = {"assignments": []}
    assignment_pattern = re.compile(
        r"^(?:[-*]\s*)?(?:\d+[.)]\s*)?(?:ASSIGN(?:MENT)?\s*:?\s*)?(?P<node_id>.+?)\s*->\s*(?P<worker_id>[^|]+?)"
        r"(?:\s*\|\s*compatibility\s*[:=]\s*(?P<compatibility>[^|]+?))?"
        r"(?:\s*\|\s*rationale\s*[:=]\s*(?P<rationale>.*))?$",
        flags=re.IGNORECASE,
    )

    for line in lines:
        if line.startswith("<") and line.endswith(">"):
            continue
        if line.upper().startswith("SELECTION_ID:"):
            payload["selection_id"] = line.split(":", 1)[1].strip()
            continue
        match = assignment_pattern.match(line)
        if not match:
            continue
        compatibility_raw = match.group("compatibility")
        try:
            compatibility = float(compatibility_raw) if compatibility_raw is not None else 0.0
        except ValueError:
            compatibility = 0.0
        rationale = (match.group("rationale") or "Selected for this node.").strip()
        payload["assignments"].append(
            {
                "node_id": match.group("node_id").strip(),
                "worker_id": match.group("worker_id").strip(),
                "compatibility": compatibility,
                "rationale": rationale,
            }
        )

    if payload["assignments"]:
        payload.setdefault("selection_id", "selection")
        return payload
    raise StructuredOutputError("Could not parse a selection plan from controller output")


def extract_json_dict(text: str) -> Dict[str, Any]:
    stripped = _normalize_structured_text(text, tags=KNOWN_CONTROLLER_TAGS)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[idx:])
        except JSONDecodeError:
            try:
                obj = ast.literal_eval(stripped[idx:])
            except Exception:
                continue
        if isinstance(obj, dict):
            return obj
    raise StructuredOutputError("Could not find a valid JSON object in controller output")


def extract_decomposition_payload(text: str) -> Dict[str, Any]:
    try:
        return extract_json_dict(text)
    except StructuredOutputError:
        return _parse_decomposition_plan(text)


def extract_selection_payload(text: str) -> Dict[str, Any]:
    try:
        return extract_json_dict(text)
    except StructuredOutputError:
        return _parse_selection_plan(text)


def compute_dag_hops(nodes: List[SubtaskNode]) -> int:
    node_map = {node.node_id: node for node in nodes}
    depth: Dict[str, int] = {}

    def visit(node_id: str) -> int:
        if node_id in depth:
            return depth[node_id]
        node = node_map[node_id]
        if not node.dependencies:
            depth[node_id] = 1
        else:
            depth[node_id] = 1 + max(visit(dep) for dep in node.dependencies)
        return depth[node_id]

    return max((visit(node.node_id) for node in nodes), default=0)


def _synthetic_final_node(
    kept_nodes: List[SubtaskNode],
    reason: str,
    synthetic_id: str,
) -> SubtaskNode:
    dependency_ids = [node.node_id for node in kept_nodes if node.node_id]
    if dependency_ids:
        depended_on = {dep for node in kept_nodes for dep in node.dependencies}
        dependency_ids = [node.node_id for node in kept_nodes if node.node_id not in depended_on]
    required_skills = sorted({skill for node in kept_nodes for skill in node.required_skills})
    return SubtaskNode(
        node_id=synthetic_id,
        instruction=(
            "The original decomposition exceeded the execution limit. "
            f"{reason}. Use the available partial outputs to provide the best final answer now."
        ),
        dependencies=dependency_ids,
        required_skills=required_skills,
        output_key="final_answer",
    )


def _truncate_to_node_budget(
    candidate: DecompositionCandidate,
    budget: int,
    reason: str,
) -> DecompositionCandidate:
    if budget <= 0:
        budget = 1
    if len(candidate.nodes) <= budget:
        return candidate

    node_map = candidate.nodes_by_id()
    order = candidate.topological_order()
    keep_prefix = order[: max(budget - 1, 0)]
    kept_nodes = [deepcopy(node_map[node_id]) for node_id in keep_prefix]
    synthetic_final = _synthetic_final_node(
        kept_nodes=kept_nodes,
        reason=reason,
        synthetic_id=f"{candidate.decomposition_id}_hard_limit_final",
    )
    new_nodes = kept_nodes + [synthetic_final]
    return DecompositionCandidate(
        decomposition_id=candidate.decomposition_id,
        summary=f"{candidate.summary} [TRUNCATED]",
        nodes=new_nodes,
        final_node_id=synthetic_final.node_id,
        num_hops=candidate.num_hops,
        effective_num_hops=compute_dag_hops(new_nodes),
        soft_penalty=candidate.soft_penalty,
        was_hard_truncated=True,
        raw_text=candidate.raw_text,
        raw_payload=dict(candidate.raw_payload),
    )


def apply_decomposition_limits(
    candidate: DecompositionCandidate,
    rollout_config: RolloutConfig,
) -> DecompositionCandidate:
    original_hops = compute_dag_hops(candidate.nodes)
    candidate.num_hops = original_hops
    candidate.effective_num_hops = original_hops

    if rollout_config.soft_max_hops is not None and original_hops > rollout_config.soft_max_hops:
        exceedance = original_hops - rollout_config.soft_max_hops
        candidate.soft_penalty = rollout_config.soft_hop_penalty * (
            exceedance ** rollout_config.soft_hop_penalty_power
        )

    limited_candidate = candidate
    if len(limited_candidate.nodes) > rollout_config.max_nodes_per_decomposition:
        limited_candidate = _truncate_to_node_budget(
            limited_candidate,
            budget=rollout_config.max_nodes_per_decomposition,
            reason=f"Node count exceeded {rollout_config.max_nodes_per_decomposition}",
        )

    if (
        rollout_config.hard_max_hops is not None
        and limited_candidate.effective_num_hops > rollout_config.hard_max_hops
    ):
        limited_candidate = _truncate_to_node_budget(
            limited_candidate,
            budget=rollout_config.hard_max_hops,
            reason=f"Hop count exceeded {rollout_config.hard_max_hops}",
        )

    limited_candidate.effective_num_hops = compute_dag_hops(limited_candidate.nodes)
    return limited_candidate


def validate_decomposition_payload(
    payload: Dict[str, Any],
    rollout_config: RolloutConfig,
    fallback_id: str,
) -> DecompositionCandidate:
    decomposition_id = str(payload.get("decomposition_id") or fallback_id)
    summary = str(payload.get("summary") or "No summary provided.")
    final_node_id = str(payload.get("final_node_id") or "")
    nodes_payload = payload.get("nodes")
    if not isinstance(nodes_payload, list) or not nodes_payload:
        raise StructuredOutputError("Decomposer output must contain a non-empty 'nodes' list")

    nodes: List[SubtaskNode] = []
    seen_node_ids = set()
    for idx, node_payload in enumerate(nodes_payload):
        if not isinstance(node_payload, dict):
            raise StructuredOutputError(f"Node {idx} is not a JSON object")
        node_id = str(node_payload.get("node_id") or "")
        if not node_id:
            raise StructuredOutputError(f"Node {idx} is missing 'node_id'")
        if node_id in seen_node_ids:
            raise StructuredOutputError(f"Duplicate node_id '{node_id}' in decomposition")
        seen_node_ids.add(node_id)
        instruction = str(node_payload.get("instruction") or "").strip()
        if not instruction:
            raise StructuredOutputError(f"Node {node_id} is missing 'instruction'")
        dependencies = _parse_csv_field(node_payload.get("dependencies"))
        if not isinstance(dependencies, list) or not all(isinstance(dep, str) for dep in dependencies):
            raise StructuredOutputError(f"Node {node_id} has invalid 'dependencies'")
        required_skills = _parse_csv_field(node_payload.get("required_skills"))
        if not isinstance(required_skills, list) or not all(
            isinstance(skill, str) for skill in required_skills
        ):
            raise StructuredOutputError(f"Node {node_id} has invalid 'required_skills'")
        output_key = str(node_payload.get("output_key") or f"{node_id}_output")
        nodes.append(
            SubtaskNode(
                node_id=node_id,
                instruction=instruction,
                dependencies=list(dependencies),
                required_skills=list(required_skills),
                output_key=output_key,
            )
        )

    if not final_node_id:
        final_node_id = nodes[-1].node_id

    candidate = DecompositionCandidate(
        decomposition_id=decomposition_id,
        summary=summary,
        nodes=nodes,
        final_node_id=final_node_id,
    )
    candidate.topological_order()
    return apply_decomposition_limits(candidate, rollout_config)


def validate_selection_payload(
    payload: Dict[str, Any],
    decomposition: DecompositionCandidate,
    worker_pool: WorkerPoolConfig,
) -> SelectionCandidate:
    selection_id = str(payload.get("selection_id") or f"{decomposition.decomposition_id}-selection")
    assignments_payload = payload.get("assignments")
    if not isinstance(assignments_payload, list) or not assignments_payload:
        raise StructuredOutputError("Selector output must contain a non-empty 'assignments' list")

    valid_node_ids = {node.node_id for node in decomposition.nodes}
    valid_worker_ids = set(worker_pool.workers_by_id().keys())
    assignments: List[WorkerAssignment] = []
    seen_node_ids = set()
    for idx, assignment_payload in enumerate(assignments_payload):
        if not isinstance(assignment_payload, dict):
            raise StructuredOutputError(f"Assignment {idx} is not a JSON object")
        node_id = str(assignment_payload.get("node_id") or "")
        worker_id = str(assignment_payload.get("worker_id") or "")
        if node_id not in valid_node_ids:
            raise StructuredOutputError(f"Assignment {idx} references unknown node_id '{node_id}'")
        if worker_id not in valid_worker_ids:
            raise StructuredOutputError(f"Assignment {idx} references unknown worker_id '{worker_id}'")
        if node_id in seen_node_ids:
            raise StructuredOutputError(f"Duplicate assignment for node_id '{node_id}'")
        seen_node_ids.add(node_id)
        rationale = str(assignment_payload.get("rationale") or f"Selected for node {node_id}.").strip()
        compatibility_raw = assignment_payload.get("compatibility", 0.0)
        try:
            compatibility = float(compatibility_raw)
        except (TypeError, ValueError) as exc:
            raise StructuredOutputError(
                f"Assignment for node '{node_id}' has invalid compatibility"
            ) from exc
        compatibility = max(0.0, min(1.0, compatibility))
        assignments.append(
            WorkerAssignment(
                node_id=node_id,
                worker_id=worker_id,
                rationale=rationale,
                compatibility=compatibility,
            )
        )

    missing_nodes = valid_node_ids - seen_node_ids
    if missing_nodes:
        raise StructuredOutputError(
            f"Selector output is missing assignments for nodes: {sorted(missing_nodes)}"
        )

    return SelectionCandidate(selection_id=selection_id, assignments=assignments)


def build_fallback_decomposition(
    task_id: str,
    raw_text: str,
    error_message: str,
    rollout_config: RolloutConfig,
) -> DecompositionCandidate:
    payload = {
        "decomposition_id": f"{task_id}-fallback-decomposition",
        "summary": "Fallback decomposition due to invalid controller output.",
        "final_node_id": "fallback_final",
        "nodes": [
            {
                "node_id": "fallback_final",
                "instruction": "Provide the best final answer directly.",
                "dependencies": [],
                "required_skills": [],
                "output_key": "final_answer",
            }
        ],
        "validation": {
            "fallback_used": True,
            "error": error_message,
            "raw_text": raw_text,
        },
    }
    candidate = validate_decomposition_payload(
        payload=payload,
        rollout_config=rollout_config,
        fallback_id=payload["decomposition_id"],
    )
    candidate.raw_payload = payload
    candidate.raw_text = format_decomposition_plan(candidate)
    return candidate


def build_fallback_selection(
    decomposition: DecompositionCandidate,
    worker_pool: WorkerPoolConfig,
    raw_text: str,
    error_message: str,
) -> SelectionCandidate:
    default_worker = worker_pool.workers[0]
    payload = {
        "selection_id": f"{decomposition.decomposition_id}-fallback-selection",
        "assignments": [
            {
                "node_id": node.node_id,
                "worker_id": default_worker.worker_id,
                "rationale": "Fallback selection due to invalid controller output.",
                "compatibility": 0.0,
            }
            for node in decomposition.nodes
        ],
        "validation": {
            "fallback_used": True,
            "error": error_message,
            "raw_text": raw_text,
        },
    }
    candidate = validate_selection_payload(
        payload=payload,
        decomposition=decomposition,
        worker_pool=worker_pool,
    )
    candidate.raw_payload = payload
    candidate.raw_text = format_selection_plan(candidate)
    return candidate

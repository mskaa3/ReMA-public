import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class PrefixProbeRequest:
    """One isolated role message to test for answer recoverability."""

    sample_index: int
    source_role: str
    source_kind: str
    message: str


@dataclass(frozen=True)
class PrefixProbeGate:
    """Leakage-adjusted outcome and validity for one focal role."""

    outcome_scores: List[float]
    valid_mask: List[bool]
    upstream_clean_mask: List[bool]


def apply_prefix_probe_gate(
    raw_scores: Sequence[float],
    decomposer_scores: Sequence[float],
    worker_scores: Sequence[float],
    upstream_worker_scores: Sequence[float],
    terminal_roles: Sequence[str],
    *,
    focal_role: str,
    decomposer_role: str,
    selector_role: str,
    stage_roles: Sequence[str],
) -> PrefixProbeGate:
    """Build role-local outcomes from isolated-message leakage probes.

    A decomposer is credited only when its plan does not reveal a recoverable
    answer. Downstream actions are excluded when the upstream plan is leaky,
    because their fixed prefix is already contaminated. A non-terminal worker
    additionally loses credit when its own isolated message reveals the final
    answer. The terminal worker is exempt from that own-message check.
    """

    size = len(raw_scores)
    if not (
        len(decomposer_scores) == size
        and len(worker_scores) == size
        and len(upstream_worker_scores) == size
        and len(terminal_roles) == size
    ):
        raise ValueError("All prefix-probe gate inputs must have equal lengths")

    stage_role_set = set(stage_roles)
    outcomes = []
    valid = []
    upstream_clean = []
    for raw, plan_score, worker_score, upstream_score, terminal_role in zip(
        raw_scores,
        decomposer_scores,
        worker_scores,
        upstream_worker_scores,
        terminal_roles,
    ):
        raw = float(raw)
        plan_available = math.isfinite(float(plan_score))
        plan_clean = plan_available and float(plan_score) <= 0.0
        prior_workers_clean = (
            not math.isfinite(float(upstream_score))
            or float(upstream_score) <= 0.0
        )
        clean_prefix = plan_clean and prior_workers_clean
        upstream_clean.append(clean_prefix)

        if focal_role == decomposer_role:
            valid.append(plan_available)
            outcomes.append(raw if plan_clean else 0.0)
            continue

        if focal_role == selector_role:
            valid.append(plan_clean)
            outcomes.append(raw if plan_clean else 0.0)
            continue

        if focal_role in stage_role_set:
            if not clean_prefix:
                valid.append(False)
                outcomes.append(0.0)
                continue
            if focal_role == str(terminal_role):
                valid.append(True)
                outcomes.append(raw)
                continue

            worker_available = math.isfinite(float(worker_score))
            worker_clean = worker_available and float(worker_score) <= 0.0
            valid.append(worker_available)
            outcomes.append(raw if worker_clean else 0.0)
            continue

        valid.append(True)
        outcomes.append(raw)

    return PrefixProbeGate(
        outcome_scores=outcomes,
        valid_mask=valid,
        upstream_clean_mask=upstream_clean,
    )


def _latest_executed_message(
    history: Iterable[Dict],
    role: str,
) -> Optional[str]:
    latest = None
    for record in history:
        if not isinstance(record, dict) or record.get("role") != role:
            continue
        if record.get("executed", True) is False:
            continue
        content = record.get("content", "")
        if isinstance(content, str) and content.strip():
            latest = content.strip()
    return latest


def collect_prefix_probe_requests(
    histories: Sequence[Iterable[Dict]],
    terminal_roles: Sequence[str],
    *,
    focal_role: str,
    decomposer_role: str,
    stage_roles: Sequence[str],
) -> List[PrefixProbeRequest]:
    """Collect plan, focal, and causally upstream messages for probing.

    The terminal worker is deliberately excluded because revealing the final
    answer is its assigned responsibility.
    """

    if len(histories) != len(terminal_roles):
        raise ValueError("histories and terminal_roles must have equal lengths")

    requests = []
    stage_role_set = set(stage_roles)
    focal_is_worker = focal_role in stage_role_set
    stage_index = {
        role: index for index, role in enumerate(stage_roles)
    }
    for sample_index, (history, terminal_role) in enumerate(
        zip(histories, terminal_roles)
    ):
        plan = _latest_executed_message(history, decomposer_role)
        if plan is not None:
            requests.append(PrefixProbeRequest(
                sample_index=sample_index,
                source_role=decomposer_role,
                source_kind="decomposer",
                message=plan,
            ))

        if not focal_is_worker:
            continue

        focal_index = stage_index[focal_role]
        terminal_index = stage_index.get(str(terminal_role), len(stage_roles))
        for worker_role in stage_roles[:terminal_index]:
            worker_index = stage_index[worker_role]
            if worker_index > focal_index:
                continue
            worker_message = _latest_executed_message(history, worker_role)
            if worker_message is None:
                continue
            source_kind = (
                "nonterminal_worker"
                if worker_role == focal_role
                else "upstream_worker"
            )
            requests.append(PrefixProbeRequest(
                sample_index=sample_index,
                source_role=worker_role,
                source_kind=source_kind,
                message=worker_message,
            ))

    return requests

"""Terminal-instruction gates, plan diagnostics, and answer equivalence."""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from verl.rema_separated_trainer.ppo.local_results import (
    extract_complete_boxed_answer,
    extract_local_result,
)


@dataclass(frozen=True)
class PrefixProbeRequest:
    sample_index: int
    source_role: str
    source_kind: str
    message: str


@dataclass(frozen=True)
class PrefixProbeGate:
    """Diagnostic gated outcomes and update eligibility, not C3 rewards."""

    outcome_scores: List[float]
    valid_mask: List[bool]
    collaboration_eligible_mask: List[bool]
    plan_eligible_mask: List[bool]
    rejection_reasons: List[str]


def has_complete_boxed_answer(response: str) -> bool:
    return extract_complete_boxed_answer(response) is not None


def parse_boxed_math_answer(response: str):
    """Parse a complete mathematical box without fallback strings."""

    from math_verify import parse
    from math_verify.parser import LatexExtractionConfig
    from math_verify.utils import TimeoutException

    answer = extract_complete_boxed_answer(response)
    if answer is None or ''.join(answer.lower().split()) in {
        'unknown', r'\text{unknown}', r'\mathrm{unknown}',
    }:
        return None
    try:
        parsed = parse(
            "\\boxed{" + answer + "}",
            extraction_config=[LatexExtractionConfig()],
            fallback_mode="no_fallback",
            extraction_mode="first_match",
        )
        if not parsed or any(isinstance(value, str) for value in parsed):
            return None
        return parsed
    except (Exception, TimeoutException):
        return None


def compare_worker_final_answers(worker_output: str, terminal_output: str) -> Optional[bool]:
    """Compare LOCAL_RESULT with the terminal box; None means unknown."""

    from math_verify.grader import sympy_expr_eq
    from math_verify.utils import TimeoutException, timeout

    candidate = parse_boxed_math_answer(extract_local_result(worker_output))
    reference = parse_boxed_math_answer(terminal_output)
    if candidate is None or reference is None:
        return None

    # Math-Verify 0.7 verify() swallows errors/timeouts as False. Call its
    # symbolic comparator with an outer timeout so those remain unknown.
    @timeout(timeout_seconds=5)
    def equivalent():
        return any(
            sympy_expr_eq(gold, pred, float_rounding=6, numeric_precision=15, strict=True)
            for gold in reference for pred in candidate
        )

    try:
        return bool(equivalent())
    except (Exception, TimeoutException):
        return None


def answer_round_records(history: Iterable[Dict], terminal_role: str, decomposer_role: str):
    """Use the round that generated the answer, ignoring carried-forward copies."""

    records = [record for record in history if isinstance(record, dict)]
    terminal_indices = [
        index for index, record in enumerate(records)
        if record.get("role") == terminal_role
        and record.get("executed", True) is not False
    ]
    end = terminal_indices[-1] + 1 if terminal_indices else len(records)
    starts = [
        index for index, record in enumerate(records[:end])
        if record.get("role") == decomposer_role
        and record.get("executed", True) is not False
    ]
    return records[starts[-1]:end] if starts else []


def count_planned_subtasks(records: Sequence[Dict], decomposer_role: str) -> int:
    """Prefer the executed router's count over re-parsing the generated plan."""

    plan = next((record for record in records if record.get("role") == decomposer_role), {})
    if "planned_subtask_count" in plan:
        return max(int(plan["planned_subtask_count"]), 0)
    # Compatibility with histories recorded before the router exposed counts.
    ids = set()
    for record in records:
        for item in record.get("assigned_subtasks", []) or []:
            label = item[0] if isinstance(item, (list, tuple)) else item
            ids.add(str(label).upper())
    return len(ids)


def collect_prefix_probe_requests(
    histories: Sequence[Iterable[Dict]],
    terminal_roles: Sequence[str],
    *,
    focal_role: Optional[str],
    decomposer_role: str,
    stage_roles: Sequence[str],
) -> List[PrefixProbeRequest]:
    """Probe the recorded terminal input and the whole plan separately.

    Missing terminal metadata is unknown, not a clean measurement. Never
    reconstruct it from the plan or include earlier worker outputs.
    """

    if len(histories) != len(terminal_roles):
        raise ValueError("histories and terminal_roles must have equal lengths")
    requests = []
    for index, (history, terminal_role) in enumerate(zip(histories, terminal_roles)):
        records = answer_round_records(history, str(terminal_role), decomposer_role)
        plan = next((r.get("content", "") for r in records if r.get("role") == decomposer_role), "")
        if isinstance(plan, str) and plan.strip():
            requests.append(PrefixProbeRequest(index, decomposer_role, "decomposer", plan.strip()))
        terminal = next((r for r in records if r.get("role") == terminal_role
                         and r.get("executed", True) is not False), {})
        instruction = terminal.get("terminal_probe_input")
        if isinstance(instruction, str) and instruction.strip():
            requests.append(PrefixProbeRequest(index, str(terminal_role), "terminal", instruction))
    return requests


def apply_prefix_probe_gate(
    raw_scores: Sequence[float],
    terminal_scores: Sequence[float],
    subtask_counts: Sequence[int],
    comparison_scores: Sequence[float],
    comparison_required: Sequence[bool],
) -> PrefixProbeGate:
    """Apply G(plan) and M(action), leaving raw C3 rewards to the caller.

    Validation supplies max(E_k) across all non-terminal workers, with NaN
    if any required comparison failed. Training supplies the focal E_k only.
    Terminal actions need no comparison but still require a valid negative
    terminal-instruction probe. Whole-plan L_D never enters this gate.
    """

    size = len(raw_scores)
    if any(len(values) != size for values in (
        terminal_scores, subtask_counts, comparison_scores, comparison_required,
    )):
        raise ValueError("All leakage gate inputs must have equal lengths")
    valid, eligible, plans, outcomes, reasons = [], [], [], [], []
    for raw, lt, count, match, required in zip(
        raw_scores, terminal_scores, subtask_counts, comparison_scores, comparison_required,
    ):
        plan_valid = math.isfinite(float(lt))
        compare_valid = not required or math.isfinite(float(match))
        plan_ok = int(count) >= 2 and plan_valid and float(lt) <= 0.0
        if int(count) < 2:
            reason = "single_subtask"
        elif not plan_valid:
            reason = "terminal_probe_invalid"
        elif float(lt) > 0.0:
            reason = "terminal_recoverable"
        elif not compare_valid:
            reason = "comparison_invalid"
        elif required and float(match) > 0.0:
            reason = "equivalent_answer"
        else:
            reason = "eligible"
        allowed = reason == "eligible"
        valid.append(plan_valid and compare_valid)
        plans.append(plan_ok)
        eligible.append(allowed)
        outcomes.append(float(raw) if allowed else 0.0)
        reasons.append(reason)
    return PrefixProbeGate(outcomes, valid, eligible, plans, reasons)


def select_stratified_probe_indices(labels: Sequence[str], max_samples: int) -> List[int]:
    """Select a deterministic round-robin sample across validation subsets."""

    if max_samples <= 0 or len(labels) == 0:
        return []
    if len(labels) <= max_samples:
        return list(range(len(labels)))
    buckets: Dict[str, List[int]] = {}
    for index, label in enumerate(labels):
        buckets.setdefault(str(label), []).append(index)
    selected = []
    depth = 0
    while len(selected) < max_samples:
        added = False
        for bucket in buckets.values():
            if depth >= len(bucket):
                continue
            selected.append(bucket[depth])
            added = True
            if len(selected) == max_samples:
                break
        if not added:
            break
        depth += 1
    return sorted(selected)


def select_console_probe_indices(labels: Sequence[str], samples_per_label: int) -> List[int]:
    """Mirror the reward manager's first-N examples per data source."""

    if samples_per_label <= 0:
        return []
    counts: Dict[str, int] = {}
    selected = []
    for index, label in enumerate(labels):
        label = str(label)
        count = counts.get(label, 0)
        if count >= samples_per_label:
            continue
        selected.append(index)
        counts[label] = count + 1
    return selected

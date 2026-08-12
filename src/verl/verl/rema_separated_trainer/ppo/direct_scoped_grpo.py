"""Direct scoped GRPO advantage construction.

CPCR supplies a role-local utility score. GRPO normalizes that score within
each rollout group, and the scope gate is applied only afterwards so that a
gated positive advantage cannot become positive again through centering.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import torch


FULL_TASK_SCOPE_INSTRUCTION = (
    "- Solve the complete original problem and derive the final answer."
)

WORKER_OUTPUT_FORMAT = (
    "Output exactly:\n"
    "REASONING:\n"
    "<step-by-step reasoning>\n\n"
    "LOCAL_RESULT: \\boxed{<result>}"
)


@dataclass(frozen=True)
class DirectScopedGRPOEstimate:
    """Per-sample scalar advantages before and after scope gating."""

    base_advantage: torch.Tensor
    scoped_advantage: torch.Tensor
    effective_mask: torch.Tensor


def build_full_task_scope_counterfactual(
    chat: Sequence[dict],
    assigned_subtasks_text: str,
) -> list[dict] | None:
    """Replace the latest local assignment with a full-task assignment."""

    return build_scope_assignment_counterfactual(
        chat,
        assigned_subtasks_text,
        FULL_TASK_SCOPE_INSTRUCTION,
    )


def build_scope_assignment_counterfactual(
    chat: Sequence[dict],
    assigned_subtasks_text: str,
    replacement_assignment_text: str,
) -> list[dict] | None:
    """Replace only the latest assignment while preserving all other context."""

    if (
        not chat
        or not assigned_subtasks_text
        or not replacement_assignment_text
        or chat[-1].get("role") != "user"
    ):
        return None

    user_content = chat[-1].get("content")
    if not isinstance(user_content, str):
        return None
    assignment_start = user_content.rfind(assigned_subtasks_text)
    if assignment_start < 0:
        return None

    counterfactual_content = (
        user_content[:assignment_start]
        + replacement_assignment_text
        + user_content[assignment_start + len(assigned_subtasks_text):]
    )
    return [
        dict(message)
        for message in chat[:-1]
    ] + [{
        "role": "user",
        "content": counterfactual_content,
    }]


def build_verifier_scope_counterfactuals(
    chat: Sequence[dict],
    question: str,
    assigned_subtasks_text: str,
) -> dict[str, list[dict]] | None:
    """Build clean role and dependency counterfactuals for a verifier action."""

    if (
        not chat
        or chat[-1].get("role") != "user"
        or not str(question).strip()
        or not assigned_subtasks_text.strip()
    ):
        return None

    factual_user_content = chat[-1].get("content")
    if not isinstance(factual_user_content, str):
        return None
    assignment_start = factual_user_content.rfind(assigned_subtasks_text)
    if assignment_start < 0:
        return None

    shared_prefix = [dict(message) for message in chat[:-1]]
    context_with_candidate = factual_user_content[:assignment_start].rstrip()
    dependency_hint = (
        "Use previous LOCAL_RESULTs from the work above when they are relevant. "
        "Check earlier subtasks when an inconsistency matters."
    )
    context_with_candidate = context_with_candidate.replace(
        dependency_hint,
        "",
    ).rstrip()
    solve_from_scratch = (
        f"{context_with_candidate}\n\n"
        "Solve the complete problem independently from scratch. Do not verify, "
        "reuse, or discuss the previous candidate shown above. Derive the answer yourself.\n\n"
        f"{WORKER_OUTPUT_FORMAT}"
    )
    dependency_removed = (
        f"Reference problem:\n{question}\n\n"
        "The previous S1 LOCAL_RESULT is unavailable.\n\n"
        f"{assigned_subtasks_text}\n\n"
        "Act as the verifier for the assigned subtask. Check the candidate against "
        "the reference problem, repair errors or omitted cases, and state the corrected result.\n\n"
        f"{WORKER_OUTPUT_FORMAT}"
    )
    return {
        "solve_from_scratch": shared_prefix + [{
            "role": "user",
            "content": solve_from_scratch,
        }],
        "dependency_removed": shared_prefix + [{
            "role": "user",
            "content": dependency_removed,
        }],
    }


def build_dependency_removed_scope_counterfactual(
    chat: Sequence[dict],
    question: str,
    assigned_subtasks_text: str,
) -> list[dict] | None:
    """Remove prior worker results while preserving the local assignment."""

    if (
        not chat
        or chat[-1].get("role") != "user"
        or not str(question).strip()
        or not assigned_subtasks_text.strip()
    ):
        return None

    factual_user_content = chat[-1].get("content")
    if not isinstance(factual_user_content, str):
        return None
    assignment_start = factual_user_content.rfind(assigned_subtasks_text)
    if assignment_start < 0:
        return None

    assignment_suffix = factual_user_content[
        assignment_start + len(assigned_subtasks_text):
    ]
    counterfactual_content = (
        f"Reference problem:\n{question}\n\n"
        "No previous LOCAL_RESULT is available.\n\n"
        f"{assigned_subtasks_text}{assignment_suffix}"
    )
    return [dict(message) for message in chat[:-1]] + [{
        "role": "user",
        "content": counterfactual_content,
    }]


def estimate_direct_scoped_grpo(
    cpcr_scores: torch.Tensor,
    group_ids: Sequence[object],
    valid_mask: torch.Tensor,
    scope_gate: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> DirectScopedGRPOEstimate:
    """Normalize reliable CPCR scores by group, then gate positive advantages.

    A group contributes only when it contains at least two valid samples with
    nonzero score variance. Invalid samples never enter the normalization
    statistics and receive zero policy-gradient advantage.
    """

    if cpcr_scores.ndim != 1:
        raise ValueError("cpcr_scores must be one-dimensional")
    if valid_mask.shape != cpcr_scores.shape:
        raise ValueError("valid_mask must match cpcr_scores")
    if scope_gate.shape != cpcr_scores.shape:
        raise ValueError("scope_gate must match cpcr_scores")
    if len(group_ids) != cpcr_scores.shape[0]:
        raise ValueError("group_ids must match cpcr_scores")

    device = cpcr_scores.device
    dtype = cpcr_scores.dtype
    valid_mask = valid_mask.to(device=device, dtype=torch.bool)
    scope_gate = scope_gate.to(device=device, dtype=dtype).clamp(0.0, 1.0)

    base_advantage = torch.zeros_like(cpcr_scores)
    scoped_advantage = torch.zeros_like(cpcr_scores)
    effective_mask = torch.zeros_like(valid_mask)

    indices_by_group = defaultdict(list)
    for sample_idx, group_id in enumerate(group_ids):
        if bool(valid_mask[sample_idx].item()):
            indices_by_group[group_id].append(sample_idx)

    with torch.no_grad():
        for sample_indices in indices_by_group.values():
            if len(sample_indices) < 2:
                continue
            index_tensor = torch.tensor(sample_indices, dtype=torch.long, device=device)
            group_scores = cpcr_scores[index_tensor]
            if not bool(torch.isfinite(group_scores).all().item()):
                continue

            group_std = group_scores.std(unbiased=True)
            if not bool(torch.isfinite(group_std).item()) or float(group_std.item()) <= epsilon:
                continue

            normalized = (
                group_scores - group_scores.mean()
            ) / (group_std + float(epsilon))
            gates = scope_gate[index_tensor]
            gated = torch.minimum(normalized, torch.zeros_like(normalized))
            gated = gated + gates * torch.maximum(normalized, torch.zeros_like(normalized))

            base_advantage[index_tensor] = normalized
            scoped_advantage[index_tensor] = gated
            effective_mask[index_tensor] = True

    return DirectScopedGRPOEstimate(
        base_advantage=base_advantage,
        scoped_advantage=scoped_advantage,
        effective_mask=effective_mask,
    )

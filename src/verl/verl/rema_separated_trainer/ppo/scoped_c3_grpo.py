"""Fixed-prefix C3 advantages for role-local GRPO updates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ScopedC3Estimate:
    """Per-sample fixed-prefix causal advantages."""

    advantage: torch.Tensor
    effective_mask: torch.Tensor
    eligible_success_group_mask: torch.Tensor


def estimate_scoped_c3_grpo(
    outcome_scores: torch.Tensor,
    group_ids: Sequence[object],
    valid_mask: torch.Tensor,
    *,
    update_mask: torch.Tensor | None = None,
    positive_only_mask: torch.Tensor | None = None,
    require_eligible_success: bool = False,
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> ScopedC3Estimate:
    """Compute fixed-prefix leave-one-out credit.

    Every effective group contains at least two valid alternatives generated
    from one shared prefix. ``update_mask`` may exclude an action from policy
    optimization without removing it from the leave-one-out baseline.
    Positive-only actions are masked after normalization; ``advantage`` retains
    the signed factual credit for diagnostics. Success means a positive binary
    outcome with an eligible positive advantage, not just a positive raw reward.
    """

    if outcome_scores.ndim != 1:
        raise ValueError("outcome_scores must be one-dimensional")
    if valid_mask.shape != outcome_scores.shape:
        raise ValueError("valid_mask must match outcome_scores")
    if update_mask is not None and update_mask.shape != outcome_scores.shape:
        raise ValueError("update_mask must match outcome_scores")
    if positive_only_mask is not None and positive_only_mask.shape != outcome_scores.shape:
        raise ValueError("positive_only_mask must match outcome_scores")
    if len(group_ids) != outcome_scores.shape[0]:
        raise ValueError("group_ids must match outcome_scores")

    device = outcome_scores.device
    valid_mask = valid_mask.to(device=device, dtype=torch.bool)
    if update_mask is None:
        update_mask = valid_mask
    else:
        update_mask = update_mask.to(device=device, dtype=torch.bool)
    if positive_only_mask is None:
        positive_only_mask = torch.zeros_like(valid_mask)
    else:
        positive_only_mask = positive_only_mask.to(device=device, dtype=torch.bool)

    advantage = torch.zeros_like(outcome_scores)
    effective_mask = torch.zeros_like(valid_mask)
    eligible_success_group_mask = torch.zeros_like(valid_mask)

    indices_by_group = defaultdict(list)
    for sample_idx, group_id in enumerate(group_ids):
        if bool(valid_mask[sample_idx].item()):
            indices_by_group[group_id].append(sample_idx)

    with torch.no_grad():
        for sample_indices in indices_by_group.values():
            if len(sample_indices) < 2:
                continue
            index_tensor = torch.tensor(
                sample_indices,
                dtype=torch.long,
                device=device,
            )
            scores = outcome_scores[index_tensor]
            if not bool(torch.isfinite(scores).all().item()):
                continue

            leave_one_out = scores - (
                (scores.sum() - scores) / float(len(sample_indices) - 1)
            )
            signal_std = leave_one_out.std(unbiased=True)
            if (
                not bool(torch.isfinite(signal_std).item())
                or float(signal_std.item()) <= epsilon
            ):
                continue
            if normalize:
                leave_one_out = leave_one_out / (
                    signal_std + float(epsilon)
                )

            advantage[index_tensor] = leave_one_out
            eligible_success = (
                update_mask[index_tensor]
                & (scores > 0)
                & (leave_one_out > float(epsilon))
            )
            has_eligible_success = bool(eligible_success.any().item())
            eligible_success_group_mask[index_tensor] = has_eligible_success
            if require_eligible_success and not has_eligible_success:
                continue
            effective_mask[index_tensor] = (
                update_mask[index_tensor]
                & (leave_one_out.abs() > float(epsilon))
                & (~positive_only_mask[index_tensor] | (leave_one_out > 0))
            )

    return ScopedC3Estimate(
        advantage=advantage,
        effective_mask=effective_mask,
        eligible_success_group_mask=eligible_success_group_mask,
    )

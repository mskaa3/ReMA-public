"""Cross-prefix credit estimation from already sampled rollout groups.

The estimator is intentionally independent from rollout generation.  Callers
provide a matrix whose ``(k, l)`` entry is the log-likelihood of action/suffix
``l`` under prefix ``k``.  CPCR then forms balance-heuristic MIS weights and a
leave-one-out counterfactual baseline for every factual rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class CPCREstimate:
    """Per-rollout CPCR statistics for one prompt group and one role."""

    baseline: torch.Tensor
    advantage: torch.Tensor
    normalized_weights: torch.Tensor
    effective_sample_size: torch.Tensor
    fallback_mask: torch.Tensor
    valid_target_mask: torch.Tensor


def estimate_cpcr(
    log_likelihoods: torch.Tensor,
    factual_outcomes: torch.Tensor,
    *,
    transported_outcomes: Optional[torch.Tensor] = None,
    pair_mask: Optional[torch.Tensor] = None,
    min_effective_sample_size: float = 2.0,
    log_weight_clip: Optional[float] = 20.0,
    fallback: str = "group_loo",
) -> CPCREstimate:
    """Estimate role credit using cross-prefix MIS and leave-one-out baselines.

    Args:
        log_likelihoods: Square ``[K, K]`` matrix. Entry ``[k, l]`` scores the
            sampled action/suffix from rollout ``l`` under rollout ``k``'s
            prefix. Invalid pairs may be ``-inf``.
        factual_outcomes: Factual terminal outcome ``R_k`` for each rollout.
        transported_outcomes: Optional ``[K, K]`` matrix with verifier scores
            for each stitched pair. When omitted, candidate rollout outcome
            ``R_l`` is used. This is exact for a transported terminal suffix
            whose final response is unchanged; callers must mask unsupported
            or structurally incompatible suffixes.
        pair_mask: Optional boolean matrix of usable target-candidate pairs.
        min_effective_sample_size: Use the fallback baseline below this ESS.
        log_weight_clip: Symmetric clipping of log MIS ratios. ``None`` disables
            clipping.
        fallback: ``group_loo`` or ``zero``.
    """

    if log_likelihoods.ndim != 2 or log_likelihoods.shape[0] != log_likelihoods.shape[1]:
        raise ValueError("log_likelihoods must be a square [K, K] matrix")
    group_size = log_likelihoods.shape[0]
    if factual_outcomes.shape != (group_size,):
        raise ValueError(f"factual_outcomes must have shape ({group_size},)")
    if fallback not in {"group_loo", "zero"}:
        raise ValueError(f"Unsupported CPCR fallback: {fallback}")

    device = log_likelihoods.device
    dtype = log_likelihoods.dtype
    factual_outcomes = factual_outcomes.to(device=device, dtype=dtype)

    finite_mask = torch.isfinite(log_likelihoods)
    if pair_mask is None:
        pair_mask = finite_mask
    else:
        if pair_mask.shape != log_likelihoods.shape:
            raise ValueError("pair_mask must match log_likelihoods")
        pair_mask = pair_mask.to(device=device, dtype=torch.bool) & finite_mask

    # A rollout is never used as its own counterfactual candidate.
    off_diagonal = ~torch.eye(group_size, dtype=torch.bool, device=device)
    pair_mask = pair_mask & off_diagonal

    if transported_outcomes is None:
        transported_outcomes = factual_outcomes.unsqueeze(0).expand(group_size, -1)
    else:
        if transported_outcomes.shape != log_likelihoods.shape:
            raise ValueError("transported_outcomes must match log_likelihoods")
        transported_outcomes = transported_outcomes.to(device=device, dtype=dtype)

    # Balance heuristic: proposal density for candidate l is the uniform
    # mixture of its likelihood under all available prefixes m.
    mixture_mask = finite_mask
    safe_log_likelihoods = log_likelihoods.masked_fill(~mixture_mask, -torch.inf)
    mixture_count = mixture_mask.sum(dim=0)
    log_mixture = torch.logsumexp(safe_log_likelihoods, dim=0) - torch.log(
        mixture_count.clamp_min(1).to(dtype)
    )
    log_ratios = log_likelihoods - log_mixture.unsqueeze(0)
    if log_weight_clip is not None:
        clip = max(float(log_weight_clip), 0.0)
        log_ratios = log_ratios.clamp(min=-clip, max=clip)
    log_ratios = log_ratios.masked_fill(~pair_mask, -torch.inf)

    valid_target_mask = pair_mask.any(dim=1)
    log_normalizer = torch.logsumexp(log_ratios, dim=1, keepdim=True)
    normalized_weights = torch.where(
        valid_target_mask.unsqueeze(1),
        torch.exp(log_ratios - log_normalizer),
        torch.zeros_like(log_ratios),
    )
    effective_sample_size = torch.where(
        valid_target_mask,
        normalized_weights.square().sum(dim=1).clamp_min(1e-12).reciprocal(),
        torch.zeros(group_size, device=device, dtype=dtype),
    )

    weighted_baseline = (normalized_weights * transported_outcomes).sum(dim=1)
    # Fall back only over candidates that belong to the same scored support.
    # This matters when callers align pairs by round or subsample suffixes.
    fallback_counts = pair_mask.sum(dim=1)
    fallback_sums = (
        pair_mask.to(dtype) * factual_outcomes.unsqueeze(0)
    ).sum(dim=1)
    loo_baseline = torch.where(
        fallback_counts > 0,
        fallback_sums / fallback_counts.clamp_min(1).to(dtype),
        torch.zeros_like(factual_outcomes),
    )
    fallback_baseline = loo_baseline if fallback == "group_loo" else torch.zeros_like(loo_baseline)

    fallback_mask = (~valid_target_mask) | (
        effective_sample_size < float(min_effective_sample_size)
    )
    baseline = torch.where(fallback_mask, fallback_baseline, weighted_baseline)
    advantage = factual_outcomes - baseline

    return CPCREstimate(
        baseline=baseline,
        advantage=advantage,
        normalized_weights=normalized_weights,
        effective_sample_size=effective_sample_size,
        fallback_mask=fallback_mask,
        valid_target_mask=valid_target_mask,
    )

"""Learned partial reward decoupling utilities.

This module keeps the first PRD reward-composer implementation deliberately
small: it routes a fixed set of reward sources to each role and produces one
effective reward per role.  The training script can learn this router offline,
while the reward manager can load a checkpoint and blend it with the current
manual reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch import nn


PRD_REWARD_SOURCE_NAMES: List[str] = [
    "decomposer_plan_parseable_gate",
    "hierarchy_utilization_gate",
    "decomposer_unique_local_result_rate",
    "decomposer_dependency_usage_rate",
    "decomposer_repair_success",
    "selector_assignment_completeness",
    "selector_assignment_precision",
    "selector_assignment_recall",
    "selector_assignment_final_present",
    "selector_worker_valid_local_result_rate",
    "worker_unique_local_result_rate",
    "worker_downstream_used_rate",
    "worker_later_worker_used_rate",
    "final_worker_result_usage_rate",
    "final_consistency_with_worker_results",
    "penalty_meta_boxed",
    "penalty_worker_finish",
    "penalty_worker_empty_assigned",
    "penalty_worker_missing_local_result",
    "penalty_worker_subtask_overreach",
    "penalty_worker_duplicate_result",
    "penalty_selector_extra_assignment",
    "penalty_selector_missing_assignment",
    "penalty_selector_duplicate_assignment",
    "penalty_selector_missing_final",
    "penalty_selector_empty_output",
    "penalty_final_ignores_worker_results",
    "penalty_planner_repeat",
    "penalty_planner_excess_subtask",
]


PRD_ROLE_FEATURE_NAMES: List[str] = [
    "is_decomposer",
    "is_selector",
    "is_worker",
    "is_final",
    "is_planner",
    "role_index_norm",
    "manual_role_bonus",
    "manual_role_penalty",
    "manual_role_score",
]


@dataclass(frozen=True)
class PRDComposerSpec:
    source_names: Sequence[str] = tuple(PRD_REWARD_SOURCE_NAMES)
    role_feature_names: Sequence[str] = tuple(PRD_ROLE_FEATURE_NAMES)


def build_prd_source_tensor(
    source_values: Mapping[str, float],
    source_names: Sequence[str] = PRD_REWARD_SOURCE_NAMES,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert a source-value mapping to a stable-order tensor."""

    return torch.tensor(
        [float(source_values.get(name, 0.0)) for name in source_names],
        device=device,
        dtype=dtype,
    )


def build_prd_role_feature_tensor(
    agent_roles: Sequence[str],
    score_role: Optional[str],
    worker_roles: Iterable[str],
    *,
    role_bonuses: Optional[Mapping[str, float]] = None,
    role_penalties: Optional[Mapping[str, float]] = None,
    manual_role_scores: Optional[Mapping[str, float]] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build simple role features for the PRD router.

    The last three manual fields are useful for imitation bootstrapping.  They
    can be set to zero when training a composer from counterfactual labels.
    """

    worker_roles = set(worker_roles)
    role_bonuses = role_bonuses or {}
    role_penalties = role_penalties or {}
    manual_role_scores = manual_role_scores or {}
    denom = max(len(agent_roles) - 1, 1)
    rows = []
    for idx, role in enumerate(agent_roles):
        is_decomposer = 1.0 if role == "decomposer" else 0.0
        is_selector = 1.0 if role == "selector" else 0.0
        is_worker = 1.0 if role in worker_roles else 0.0
        is_final = 1.0 if role == score_role else 0.0
        is_planner = 1.0 if role in {"decomposer", "selector"} else 0.0
        rows.append(
            [
                is_decomposer,
                is_selector,
                is_worker,
                is_final,
                is_planner,
                float(idx) / float(denom),
                float(role_bonuses.get(role, 0.0)),
                float(role_penalties.get(role, 0.0)),
                float(manual_role_scores.get(role, 0.0)),
            ]
        )
    return torch.tensor(rows, device=device, dtype=dtype)


class PRDRewardComposer(nn.Module):
    """A lightweight reward-source router.

    Inputs:
        source_values: ``[batch, num_sources]``
        role_features: ``[batch, num_roles, role_feature_dim]``

    Outputs:
        ``role_scores``: ``[batch, num_roles]``
        ``routing``: ``[batch, num_roles, num_sources]``
    """

    def __init__(
        self,
        num_sources: int,
        role_feature_dim: int,
        hidden_dim: int = 128,
        source_embed_dim: int = 32,
        global_context_dim: Optional[int] = None,
        routing_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if routing_activation not in {"sigmoid", "softmax"}:
            raise ValueError(f"Unsupported routing activation: {routing_activation}")
        self.num_sources = int(num_sources)
        self.role_feature_dim = int(role_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.source_embed_dim = int(source_embed_dim)
        self.global_context_dim = int(global_context_dim or hidden_dim)
        self.routing_activation = routing_activation

        self.source_embedding = nn.Embedding(self.num_sources, self.source_embed_dim)
        self.global_encoder = nn.Sequential(
            nn.Linear(self.num_sources, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.global_context_dim),
            nn.GELU(),
        )
        self.source_encoder = nn.Sequential(
            nn.Linear(self.source_embed_dim + 1 + self.global_context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.role_encoder = nn.Sequential(
            nn.Linear(self.role_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.routing_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + self.global_context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, source_values: torch.Tensor, role_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if source_values.dim() != 2:
            raise ValueError(f"source_values must be [batch, sources], got {tuple(source_values.shape)}")
        if role_features.dim() != 3:
            raise ValueError(f"role_features must be [batch, roles, features], got {tuple(role_features.shape)}")
        if source_values.shape[1] != self.num_sources:
            raise ValueError(f"Expected {self.num_sources} sources, got {source_values.shape[1]}")
        if role_features.shape[2] != self.role_feature_dim:
            raise ValueError(f"Expected role feature dim {self.role_feature_dim}, got {role_features.shape[2]}")

        batch_size = source_values.shape[0]
        source_ids = torch.arange(self.num_sources, device=source_values.device)
        source_ids = source_ids.unsqueeze(0).expand(batch_size, -1)
        global_context = self.global_encoder(source_values)
        source_emb = self.source_embedding(source_ids)
        source_input = torch.cat(
            [
                source_emb,
                source_values.unsqueeze(-1),
                global_context.unsqueeze(1).expand(-1, self.num_sources, -1),
            ],
            dim=-1,
        )
        source_hidden = self.source_encoder(source_input)
        role_hidden = self.role_encoder(role_features)

        num_roles = role_features.shape[1]
        pair_hidden = torch.cat(
            [
                role_hidden.unsqueeze(2).expand(-1, -1, self.num_sources, -1),
                source_hidden.unsqueeze(1).expand(-1, num_roles, -1, -1),
                global_context.unsqueeze(1).unsqueeze(2).expand(-1, num_roles, self.num_sources, -1),
            ],
            dim=-1,
        )
        routing_logits = self.routing_head(pair_hidden).squeeze(-1)
        if self.routing_activation == "softmax":
            routing = torch.softmax(routing_logits, dim=-1)
        else:
            routing = torch.sigmoid(routing_logits)

        role_scores = (routing * source_values.unsqueeze(1)).sum(dim=-1)
        return {"role_scores": role_scores, "routing": routing, "routing_logits": routing_logits}

    def checkpoint_payload(
        self,
        *,
        source_names: Sequence[str] = PRD_REWARD_SOURCE_NAMES,
        role_feature_names: Sequence[str] = PRD_ROLE_FEATURE_NAMES,
    ) -> Dict[str, object]:
        return {
            "state_dict": self.state_dict(),
            "num_sources": self.num_sources,
            "role_feature_dim": self.role_feature_dim,
            "hidden_dim": self.hidden_dim,
            "source_embed_dim": self.source_embed_dim,
            "global_context_dim": self.global_context_dim,
            "routing_activation": self.routing_activation,
            "source_names": list(source_names),
            "role_feature_names": list(role_feature_names),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, map_location: str | torch.device = "cpu") -> "PRDRewardComposer":
        payload = torch.load(checkpoint_path, map_location=map_location)
        model = cls(
            num_sources=int(payload["num_sources"]),
            role_feature_dim=int(payload["role_feature_dim"]),
            hidden_dim=int(payload.get("hidden_dim", 128)),
            source_embed_dim=int(payload.get("source_embed_dim", 32)),
            global_context_dim=int(payload.get("global_context_dim", payload.get("hidden_dim", 128))),
            routing_activation=str(payload.get("routing_activation", "sigmoid")),
        )
        model.load_state_dict(payload["state_dict"])
        return model

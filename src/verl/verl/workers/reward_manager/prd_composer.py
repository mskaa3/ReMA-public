"""Learned partial reward decoupling utilities.

This module keeps the first PRD reward-composer implementation deliberately
small: it routes a fixed set of reward sources to each role and produces one
effective reward per role.  The training script can learn this router offline,
while the reward manager can load a checkpoint and blend it with the current
manual reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


ROLE_PRD_FEATURE_NAMES: List[str] = [
    "is_decomposer",
    "is_selector",
    "is_worker",
    "is_final",
    "role_index_norm",
    "decomposer_plan_parseable_gate",
    "hierarchy_utilization_gate",
    "decomposer_dependency_usage_rate",
    "decomposer_repair_success",
    "planner_repeat_penalty",
    "planner_excess_subtask_penalty",
    "selector_assignment_completeness",
    "selector_assignment_precision",
    "selector_assignment_recall",
    "selector_assignment_final_present",
    "selector_empty_output",
    "selector_extra_assignment_penalty",
    "selector_missing_assignment_penalty",
    "selector_missing_final_penalty",
    "worker_unique_local_result_rate",
    "worker_downstream_used_rate",
    "worker_later_worker_used_rate",
    "worker_missing_local_result_penalty",
    "worker_duplicate_result_penalty",
    "worker_subtask_overreach_penalty",
    "final_worker_result_usage_rate",
    "final_consistency_with_worker_results",
    "final_ignores_worker_results_penalty",
]


PRD_SOURCE_STAGE: Dict[str, int] = {
    "decomposer_plan_parseable_gate": 0,
    "hierarchy_utilization_gate": 0,
    "decomposer_unique_local_result_rate": 0,
    "decomposer_dependency_usage_rate": 0,
    "decomposer_repair_success": 0,
    "selector_assignment_completeness": 1,
    "selector_assignment_precision": 1,
    "selector_assignment_recall": 1,
    "selector_assignment_final_present": 1,
    "selector_worker_valid_local_result_rate": 1,
    "worker_unique_local_result_rate": 2,
    "worker_downstream_used_rate": 2,
    "worker_later_worker_used_rate": 2,
    "final_worker_result_usage_rate": 3,
    "final_consistency_with_worker_results": 3,
    "penalty_meta_boxed": 0,
    "penalty_worker_finish": 2,
    "penalty_worker_empty_assigned": 2,
    "penalty_worker_missing_local_result": 2,
    "penalty_worker_subtask_overreach": 2,
    "penalty_worker_duplicate_result": 2,
    "penalty_selector_extra_assignment": 1,
    "penalty_selector_missing_assignment": 1,
    "penalty_selector_duplicate_assignment": 1,
    "penalty_selector_missing_final": 1,
    "penalty_selector_empty_output": 1,
    "penalty_final_ignores_worker_results": 3,
    "penalty_planner_repeat": 0,
    "penalty_planner_excess_subtask": 0,
}


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


def _role_stage(role: str, score_role: Optional[str], worker_roles: Iterable[str]) -> int:
    worker_roles = set(worker_roles)
    if role == "decomposer":
        return 0
    if role == "selector":
        return 1
    if role == score_role:
        return 3
    if role in worker_roles:
        return 2
    return 2


def build_role_prd_graph_prior_tensors(
    agent_roles: Sequence[str],
    score_role: Optional[str],
    worker_roles: Iterable[str],
    *,
    mode: str = "none",
    soft_distance_penalty: float = 1.0,
    reverse_distance_penalty: float = 3.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build optional priors for role-to-role credit attention.

    Rows are credited target roles and columns are context/source roles.  A
    downstream role can provide evidence for upstream credit, e.g. final usage
    can credit a worker, while hard mode masks reverse protocol edges.
    """

    if mode not in {"none", "soft", "hard"}:
        raise ValueError(f"Unsupported role PRD graph prior mode: {mode}")
    if mode == "none":
        return None, None

    worker_roles = set(worker_roles)
    role_stages = [
        _role_stage(role, score_role, worker_roles)
        for role in agent_roles
    ]
    bias_rows = []
    mask_rows = []
    for target_stage in role_stages:
        bias_row = []
        mask_row = []
        for source_stage in role_stages:
            if target_stage <= source_stage:
                distance = source_stage - target_stage
                allowed = 1.0
            else:
                distance = reverse_distance_penalty + target_stage - source_stage
                allowed = 0.0
            bias_row.append(-float(soft_distance_penalty) * float(distance))
            mask_row.append(allowed)
        bias_rows.append(bias_row)
        mask_rows.append(mask_row)

    attention_bias = torch.tensor(bias_rows, device=device, dtype=dtype)
    attention_mask = torch.tensor(mask_rows, device=device, dtype=dtype) if mode == "hard" else None
    return attention_bias, attention_mask


def build_prd_graph_prior_tensors(
    agent_roles: Sequence[str],
    score_role: Optional[str],
    worker_roles: Iterable[str],
    source_names: Sequence[str] = PRD_REWARD_SOURCE_NAMES,
    *,
    mode: str = "none",
    soft_distance_penalty: float = 1.0,
    reverse_distance_penalty: float = 3.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build optional graph priors for role/source reward routing.

    The graph follows the protocol order:
    decomposer -> selector -> non-final workers -> final.

    A source owned by a downstream stage can credit upstream stages because
    upstream decisions could have caused that downstream outcome.  Reverse
    edges are discouraged in soft mode and removed in hard mode.
    """

    if mode not in {"none", "soft", "hard"}:
        raise ValueError(f"Unsupported PRD graph prior mode: {mode}")
    if mode == "none":
        return None, None

    role_stages = [
        _role_stage(role, score_role, worker_roles)
        for role in agent_roles
    ]
    source_stages = [
        int(PRD_SOURCE_STAGE.get(source_name, 2))
        for source_name in source_names
    ]
    bias_rows = []
    mask_rows = []
    for role_stage in role_stages:
        bias_row = []
        mask_row = []
        for source_stage in source_stages:
            if role_stage <= source_stage:
                distance = source_stage - role_stage
                allowed = 1.0
            else:
                distance = reverse_distance_penalty + role_stage - source_stage
                allowed = 0.0
            bias_row.append(-float(soft_distance_penalty) * float(distance))
            mask_row.append(allowed)
        bias_rows.append(bias_row)
        mask_rows.append(mask_row)

    routing_bias = torch.tensor(bias_rows, device=device, dtype=dtype)
    routing_mask = torch.tensor(mask_rows, device=device, dtype=dtype) if mode == "hard" else None
    return routing_bias, routing_mask


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
        routing_floor: float = 0.0,
    ) -> None:
        super().__init__()
        if routing_activation not in {"sigmoid", "softmax"}:
            raise ValueError(f"Unsupported routing activation: {routing_activation}")
        if not 0.0 <= float(routing_floor) < 1.0:
            raise ValueError(f"routing_floor must be in [0, 1), got {routing_floor}")
        self.num_sources = int(num_sources)
        self.role_feature_dim = int(role_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.source_embed_dim = int(source_embed_dim)
        self.global_context_dim = int(global_context_dim or hidden_dim)
        self.routing_activation = routing_activation
        self.routing_floor = float(routing_floor)

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

    def forward(
        self,
        source_values: torch.Tensor,
        role_features: torch.Tensor,
        routing_bias: Optional[torch.Tensor] = None,
        routing_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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
        if routing_bias is not None:
            while routing_bias.dim() < routing_logits.dim():
                routing_bias = routing_bias.unsqueeze(0)
            routing_logits = routing_logits + routing_bias.to(
                device=routing_logits.device,
                dtype=routing_logits.dtype,
            )
        if routing_mask is not None:
            while routing_mask.dim() < routing_logits.dim():
                routing_mask = routing_mask.unsqueeze(0)
            routing_mask = routing_mask.to(device=routing_logits.device, dtype=routing_logits.dtype)
            if self.routing_activation == "softmax":
                routing_logits = routing_logits.masked_fill(routing_mask <= 0, -1e9)
        if self.routing_activation == "softmax":
            routing = torch.softmax(routing_logits, dim=-1)
        else:
            routing = torch.sigmoid(routing_logits)
            if self.routing_floor > 0.0:
                routing = self.routing_floor + (1.0 - self.routing_floor) * routing
        if routing_mask is not None and self.routing_activation != "softmax":
            routing = routing * routing_mask

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
            "routing_floor": self.routing_floor,
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
            routing_floor=float(payload.get("routing_floor", 0.0)),
        )
        model.load_state_dict(payload["state_dict"])
        return model


class RolePRDCreditRouter(nn.Module):
    """A role-to-role credit router closer to PRD-style role relevance.

    Inputs:
        role_features: ``[batch, num_roles, role_feature_dim]``

    Outputs:
        ``role_scores``: ``[batch, num_roles]``
        ``routing``: ``[batch, num_roles, num_roles]`` where rows are credited
        roles and columns are context roles attended to for that credit.
    """

    def __init__(
        self,
        role_feature_dim: int,
        hidden_dim: int = 128,
        routing_activation: str = "softmax",
        routing_floor: float = 0.0,
        score_activation: str = "identity",
    ) -> None:
        super().__init__()
        if routing_activation not in {"sigmoid", "softmax"}:
            raise ValueError(f"Unsupported routing activation: {routing_activation}")
        if score_activation not in {"identity", "tanh"}:
            raise ValueError(f"Unsupported score activation: {score_activation}")
        if not 0.0 <= float(routing_floor) < 1.0:
            raise ValueError(f"routing_floor must be in [0, 1), got {routing_floor}")
        self.role_feature_dim = int(role_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.routing_activation = routing_activation
        self.routing_floor = float(routing_floor)
        self.score_activation = score_activation

        self.role_encoder = nn.Sequential(
            nn.Linear(self.role_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        role_features: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if role_features.dim() != 3:
            raise ValueError(f"role_features must be [batch, roles, features], got {tuple(role_features.shape)}")
        if role_features.shape[2] != self.role_feature_dim:
            raise ValueError(f"Expected role feature dim {self.role_feature_dim}, got {role_features.shape[2]}")

        hidden = self.role_encoder(role_features)
        q = self.query(hidden)
        k = self.key(hidden)
        v = self.value(hidden)
        scale = float(self.hidden_dim) ** -0.5
        routing_logits = torch.matmul(q, k.transpose(-1, -2)) * scale
        if attention_bias is not None:
            while attention_bias.dim() < routing_logits.dim():
                attention_bias = attention_bias.unsqueeze(0)
            routing_logits = routing_logits + attention_bias.to(
                device=routing_logits.device,
                dtype=routing_logits.dtype,
            )
        if attention_mask is not None:
            while attention_mask.dim() < routing_logits.dim():
                attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.to(device=routing_logits.device, dtype=routing_logits.dtype)
            if self.routing_activation == "softmax":
                routing_logits = routing_logits.masked_fill(attention_mask <= 0, -1e9)

        if self.routing_activation == "softmax":
            routing = torch.softmax(routing_logits, dim=-1)
        else:
            routing = torch.sigmoid(routing_logits)
            if self.routing_floor > 0.0:
                routing = self.routing_floor + (1.0 - self.routing_floor) * routing
        if attention_mask is not None and self.routing_activation != "softmax":
            routing = routing * attention_mask

        context = torch.matmul(routing, v)
        role_scores = self.score_head(torch.cat([hidden, context], dim=-1)).squeeze(-1)
        if self.score_activation == "tanh":
            role_scores = torch.tanh(role_scores)
        return {"role_scores": role_scores, "routing": routing, "routing_logits": routing_logits}

    def checkpoint_payload(
        self,
        *,
        role_feature_names: Sequence[str] = ROLE_PRD_FEATURE_NAMES,
    ) -> Dict[str, object]:
        return {
            "state_dict": self.state_dict(),
            "model_type": "role",
            "role_feature_dim": self.role_feature_dim,
            "hidden_dim": self.hidden_dim,
            "routing_activation": self.routing_activation,
            "routing_floor": self.routing_floor,
            "score_activation": self.score_activation,
            "role_feature_names": list(role_feature_names),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, map_location: str | torch.device = "cpu") -> "RolePRDCreditRouter":
        payload = torch.load(checkpoint_path, map_location=map_location)
        model = cls(
            role_feature_dim=int(payload["role_feature_dim"]),
            hidden_dim=int(payload.get("hidden_dim", 128)),
            routing_activation=str(payload.get("routing_activation", "softmax")),
            routing_floor=float(payload.get("routing_floor", 0.0)),
            score_activation=str(payload.get("score_activation", "identity")),
        )
        model.load_state_dict(payload["state_dict"])
        return model

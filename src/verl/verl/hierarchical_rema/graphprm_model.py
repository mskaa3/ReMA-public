from __future__ import annotations
import argparse
import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from . import graphprm_core as base
from . import graphprm_sequence as qwen_seq
from .graphprm_inputs import (
    clean_result_text, delivered_result, escape_prm_field, check_rendered_markers,
    marker_alignment,
)
BINARY_LABEL_SCHEMA = True
_GRAPHPRM_LAYOUT = None

@dataclass(frozen=True)
class GraphPRMFeatureLayout:
    mode: str
    bge_dim: int
    prm_hidden_dim: int
    aux_dim: int
    aux_names: tuple[str, ...]

    @property
    def total_dim(self) -> int:
        return self.bge_dim + self.prm_hidden_dim + self.aux_dim

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "bge_dim": self.bge_dim,
            "prm_hidden_dim": self.prm_hidden_dim,
            "aux_dim": self.aux_dim,
            "aux_names": list(self.aux_names),
            "total_dim": self.total_dim,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "GraphPRMFeatureLayout":
        return cls(
            mode=str(payload["mode"]),
            bge_dim=int(payload["bge_dim"]),
            prm_hidden_dim=int(payload.get("prm_hidden_dim", 0)),
            aux_dim=int(payload.get("aux_dim", 0)),
            aux_names=tuple(str(item) for item in payload.get("aux_names", [])),
        )


def set_graphprm_layout(layout: GraphPRMFeatureLayout) -> None:
    global _GRAPHPRM_LAYOUT
    _GRAPHPRM_LAYOUT = layout


def graphprm_layout_from_encoder_spec(encoder_spec: dict[str, Any]) -> GraphPRMFeatureLayout | None:
    payload = encoder_spec.get("feature_layout")
    if isinstance(payload, dict):
        return GraphPRMFeatureLayout.from_json(payload)
    return None


def current_graphprm_layout(text_dim: int) -> GraphPRMFeatureLayout:
    if _GRAPHPRM_LAYOUT is not None and _GRAPHPRM_LAYOUT.total_dim == text_dim:
        return _GRAPHPRM_LAYOUT
    # Fallback for defensive checkpoint loading: behave like a BGE-only model.
    return GraphPRMFeatureLayout(
        mode="fallback_single_text_feature",
        bge_dim=text_dim,
        prm_hidden_dim=0,
        aux_dim=0,
        aux_names=(),
    )


def truncate_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + " ... [truncated]"


def render_graphprm_sequence(
    record: dict[str, Any],
    *,
    tokenizer: Any | None,
    edge_mode: str,
    max_problem_chars: int,
    max_step_chars: int,
    max_dependency_chars: int,
    include_final_marker: bool,
) -> dict[str, Any]:
    subtasks = qwen_seq.get_subtasks(record)
    workers = qwen_seq.get_workers(record)
    nodes = list(subtasks.keys()) or list(workers.keys())
    edges = qwen_seq.choose_edges(record, edge_mode)
    order, cyclic = qwen_seq.topological_order(nodes, edges)
    order = [node_id for node_id in order if node_id in workers]

    incoming: dict[str, list[str]] = {node_id: [] for node_id in order}
    for source, target in edges:
        if target in incoming:
            incoming[target].append(source)
    for node_id in incoming:
        incoming[node_id] = sorted(set(incoming[node_id]), key=qwen_seq.node_sort_key)

    def safe(value):
        return escape_prm_field(value, tokenizer)

    def result(worker, limit):
        return safe(truncate_text(delivered_result(worker), limit))

    prompt = safe(truncate_text((record.get("task") or {}).get("prompt", ""), max_problem_chars))
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {"role": "user", "content": prompt},
    ]

    assistant_parts: list[str] = []
    targets: list[dict[str, Any]] = []
    node_to_step = {node_id: index for index, node_id in enumerate(order, start=1)}

    for index, node_id in enumerate(order, start=1):
        worker = workers[node_id]
        subtask = subtasks.get(node_id) or worker.get("subtask") or {}
        dependency_ids = incoming.get(node_id) or [str(dep) for dep in worker.get("declared_dependencies") or []]
        dependency_ids = [node for node in dependency_ids if node in node_to_step and node_to_step[node] < index]
        dependency_phrase = qwen_seq.format_dependency_phrase(dependency_ids, node_to_step)
        lines = [f"Step {index} [node {safe(node_id)}, {dependency_phrase}]:"]
        instruction = safe(truncate_text(
            subtask.get("instruction") or worker.get("subtask_instruction") or "",
            max_step_chars // 2,
        ))
        if instruction:
            lines.append(f"Task: {instruction}")
        if dependency_ids:
            lines.append("Relevant previous results:")
            for dep_id in dependency_ids:
                dep_worker = workers.get(dep_id, {})
                dep_result = result(dep_worker, max_dependency_chars)
                lines.append(f"- Step {node_to_step[dep_id]} result: {dep_result}")
        lines.append(f"Result: {result(worker, max_step_chars)}")
        assistant_parts.append("\n".join(lines) + "\n<extra_0>")
        targets.append(
            {
                "scope": "worker",
                "node_id": node_id,
                "step_index": index,
                "is_final_worker": bool(worker.get("is_final_node")),
            }
        )

    if include_final_marker:
        final_answer = safe(truncate_text(clean_result_text((record.get("trajectory") or {}).get("final_answer", "")), max_step_chars))
        final_worker_id = base.effective_final_node_id(record)
        dependency_ids = [final_worker_id] if final_worker_id in node_to_step else []
        dependency_phrase = qwen_seq.format_dependency_phrase(dependency_ids, node_to_step)
        final_index = len(targets) + 1
        lines = [f"Step {final_index} [final answer, {dependency_phrase}]:"]
        lines.append("Task: Synthesize the available worker results into the final answer.")
        if dependency_ids:
            dep_worker = workers.get(final_worker_id, {})
            lines.append("Relevant previous results:")
            lines.append(f"- Step {node_to_step[final_worker_id]} result: {result(dep_worker, max_dependency_chars)}")
        lines.append(f"Result: {final_answer}")
        assistant_parts.append("\n".join(lines) + "\n<extra_0>")
        targets.append(
            {
                "scope": "final",
                "node_id": "f",
                "step_index": final_index,
                "is_final_worker": False,
            }
        )

    messages.append({"role": "assistant", "content": "\n\n".join(assistant_parts)})
    if tokenizer is not None:
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            rendered = "\n\n".join(
                [
                    messages[0]["content"],
                    f"User: {messages[1]['content']}",
                    f"Assistant: {messages[2]['content']}",
                ]
            )
    else:
        rendered = "\n\n".join(
            [
                messages[0]["content"],
                f"User: {messages[1]['content']}",
                f"Assistant: {messages[2]['content']}",
            ]
        )

    check_rendered_markers(rendered, targets)
    branch_nodes = qwen_seq.Counter(source for source, _ in edges)
    merge_nodes = qwen_seq.Counter(target for _, target in edges)
    return {
        "prompt": rendered,
        "targets": targets,
        "node_order": order,
        "edges": edges,
        "cyclic": cyclic,
        "branch_node_count": sum(1 for count in branch_nodes.values() if count > 1),
        "merge_node_count": sum(1 for count in merge_nodes.values() if count > 1),
        "isolated_node_count": max(0, len(nodes) - len({node for edge in edges for node in edge})),
    }


def marker_features_from_outputs(
    *,
    outputs: Any,
    encoded: Any,
    step_sep_id: int,
    include_hidden: bool,
    include_score: bool,
) -> list[dict[str, Any]]:
    logits = outputs[0] if isinstance(outputs, (tuple, list)) else getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("Qwen PRM output does not expose logits.")
    if logits.ndim != 3 or logits.shape[-1] < 2:
        raise RuntimeError(f"Expected Qwen PRM logits [batch, seq, classes], got {tuple(logits.shape)}.")

    hidden_states = getattr(outputs, "hidden_states", None)
    last_hidden = None
    if include_hidden:
        if not hidden_states:
            raise RuntimeError("Qwen PRM output did not include hidden_states. Cannot build hidden ablation.")
        last_hidden = hidden_states[-1]

    probabilities = torch.softmax(logits.detach().float().cpu(), dim=-1)
    attention = encoded.get("attention_mask")
    attention = attention.detach().cpu().bool() if attention is not None else torch.ones_like(encoded["input_ids"], dtype=torch.bool).cpu()
    masks = (encoded["input_ids"].detach().cpu() == step_sep_id) & attention
    hidden_cpu = last_hidden.detach().cpu() if last_hidden is not None else None

    rows: list[dict[str, Any]] = []
    for batch_index, sample_mask in enumerate(masks):
        positions = torch.nonzero(sample_mask, as_tuple=False).flatten()
        unpadded_ids = encoded["input_ids"][batch_index].detach().cpu()[attention[batch_index]]
        unpadded_positions = torch.nonzero(unpadded_ids == step_sep_id, as_tuple=False).flatten().tolist()
        sample_scores: list[float] = []
        sample_hidden_rows: list[Tensor] = []
        if include_score:
            sample_scores = [float(value) for value in probabilities[batch_index, positions, 1].tolist()]
        if include_hidden and hidden_cpu is not None:
            for position in positions.tolist():
                sample_hidden_rows.append(hidden_cpu[batch_index, int(position)].to(torch.float16))
        rows.append(
            {
                "scores": sample_scores,
                "hidden_states": torch.stack(sample_hidden_rows, dim=0) if sample_hidden_rows else None,
                "marker_count": int(positions.numel()),
                "marker_positions": unpadded_positions,
            }
        )
    return rows


class FrozenQwenMarkerFeatureExtractor:
    def __init__(self, args: argparse.Namespace, *, include_hidden: bool, include_score: bool) -> None:
        self.args = args
        self.include_hidden = include_hidden
        self.include_score = include_score
        self.tokenizer = qwen_seq.load_tokenizer(args.prm_model)
        step_tokens = self.tokenizer.encode("<extra_0>", add_special_tokens=False)
        if len(step_tokens) != 1:
            raise SystemExit(f"Expected <extra_0> to be one token for {args.prm_model}, got {step_tokens}.")
        self.step_sep_id = int(step_tokens[0])
        self.model = None
        self.input_device: str | torch.device = args.prm_device

    def load_model(self) -> None:
        if self.model is not None:
            return
        prm_args = argparse.Namespace(
            model=self.args.prm_model,
            device=self.args.prm_device,
            device_map=self.args.prm_device_map,
            torch_dtype=self.args.prm_torch_dtype,
        )
        self.model, self.input_device = qwen_seq.load_qwen_prm(prm_args, self.tokenizer)

    def render(self, record: dict[str, Any]) -> dict[str, Any]:
        return render_graphprm_sequence(
            record,
            tokenizer=self.tokenizer,
            edge_mode=self.args.prm_edge_mode,
            max_problem_chars=self.args.prm_max_problem_chars,
            max_step_chars=self.args.prm_max_step_chars,
            max_dependency_chars=self.args.prm_max_dependency_chars,
            include_final_marker=self.args.include_final_prm_marker,
        )

    def alignment(self, render):
        return marker_alignment(self.tokenizer, render["prompt"], render["targets"], self.args.prm_max_length)

    def extract_batch(self, prompts: list[str]) -> list[dict[str, Any]]:
        self.load_model()
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.args.prm_max_length),
        ).to(self.input_device)
        with torch.no_grad():
            outputs = self.model(
                **encoded,
                use_cache=False,
                output_hidden_states=self.include_hidden,
                return_dict=True,
            )
        return marker_features_from_outputs(
            outputs=outputs,
            encoded=encoded,
            step_sep_id=self.step_sep_id,
            include_hidden=self.include_hidden,
            include_score=self.include_score,
        )


def build_fused_feature_tensor(
    *,
    blueprint: base.GraphBlueprint,
    bge_embeddings: Tensor,
    prm_payload: dict[str, Any] | None,
    layout: GraphPRMFeatureLayout,
) -> Tensor:
    rows = [bge_embeddings.float()]
    if layout.prm_hidden_dim > 0:
        prm_hidden = torch.zeros(
            (bge_embeddings.shape[0], layout.prm_hidden_dim),
            dtype=torch.float32,
        )
    else:
        prm_hidden = None
    if layout.aux_dim > 0:
        aux = torch.zeros((bge_embeddings.shape[0], layout.aux_dim), dtype=torch.float32)
    else:
        aux = None

    if prm_payload is not None and (prm_hidden is not None or aux is not None):
        targets = prm_payload.get("targets") or []
        scores = prm_payload.get("scores") or []
        hidden_states = prm_payload.get("hidden_states")
        for marker_index, target in enumerate(targets):
            scope = target.get("scope")
            if scope == "worker":
                node_index = blueprint.worker_indices_by_node_id.get(str(target.get("node_id")))
            elif scope == "final":
                node_index = blueprint.final_index
            else:
                node_index = None
            if node_index is None:
                continue
            has_marker = marker_index < int(prm_payload.get("marker_count", 0))
            if prm_hidden is not None and isinstance(hidden_states, Tensor) and marker_index < hidden_states.shape[0]:
                source = hidden_states[marker_index].float()
                if source.shape[-1] == layout.prm_hidden_dim:
                    prm_hidden[node_index] = source
                    has_marker = True
            if aux is not None:
                prm_score = (
                    qwen_seq.score_value(scores[marker_index])
                    if marker_index < len(scores)
                    else None
                )
                values: dict[str, float] = {
                    "has_prm_marker": 1.0 if has_marker and prm_score is not None else 0.0,
                    "prm_step_quality_score": prm_score if prm_score is not None else 0.0,
                    "is_final_prm_marker": 1.0 if scope == "final" else 0.0,
                }
                for aux_index, name in enumerate(layout.aux_names):
                    aux[node_index, aux_index] = values.get(name, 0.0)

    if prm_hidden is not None:
        rows.append(prm_hidden)
    if aux is not None:
        rows.append(aux)
    return torch.cat(rows, dim=-1)


class GraphPRMHybridModel(nn.Module):
    def __init__(
        self,
        *,
        text_dim: int,
        metadata_dim: int,
        hidden_dim: int,
        message_passing_layers: int,
        dropout: float,
        graph_architecture: str = "relational",
        gat_heads: int = 4,
        node_head_context: str = "ego",
    ) -> None:
        super().__init__()
        if node_head_context not in base.NODE_HEAD_CONTEXTS:
            raise ValueError(f"Unsupported node_head_context={node_head_context!r}.")
        if graph_architecture not in base.GRAPH_ARCHITECTURES:
            raise ValueError(f"Unsupported graph_architecture={graph_architecture!r}.")

        self.layout = current_graphprm_layout(text_dim)
        self.hidden_dim = hidden_dim
        self.label_classes = 2 if BINARY_LABEL_SCHEMA else 3
        self.bge_norm = nn.LayerNorm(self.layout.bge_dim)
        self.bge_proj = nn.Linear(self.layout.bge_dim, hidden_dim)
        self.prm_norm = nn.LayerNorm(self.layout.prm_hidden_dim) if self.layout.prm_hidden_dim > 0 else None
        self.prm_proj = nn.Linear(self.layout.prm_hidden_dim, hidden_dim) if self.layout.prm_hidden_dim > 0 else None
        self.aux_proj = (
            nn.Sequential(
                nn.Linear(self.layout.aux_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.layout.aux_dim > 0
            else None
        )
        gate_input_dim = hidden_dim * 2 + self.layout.aux_dim
        self.prm_gate = nn.Linear(gate_input_dim, hidden_dim) if self.prm_proj is not None else None
        self.meta_proj = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_type_embedding = nn.Embedding(len(base.NODE_TYPES), hidden_dim)
        if graph_architecture == "gat":
            layer_factory = base.RelationalGraphAttentionLayer
        else:
            layer_factory = base.RelationalMessagePassingLayer
        self.layers = nn.ModuleList(
            layer_factory(hidden_dim, len(base.EDGE_TYPES), dropout, int(gat_heads))
            if graph_architecture == "gat"
            else layer_factory(hidden_dim, len(base.EDGE_TYPES), dropout)
            for _ in range(message_passing_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.node_head_context = node_head_context
        self.graph_pool_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.local_pool_proj = nn.Linear(hidden_dim * 3, hidden_dim) if node_head_context == "ego" else None
        self.graph_label_head = nn.Linear(hidden_dim, len(base.GRAPH_LABELS) * self.label_classes)
        self.decomposer_head = nn.Linear(hidden_dim, len(base.DECOMPOSER_LABELS) * self.label_classes)
        self.worker_head = nn.Linear(hidden_dim, len(base.WORKER_LABELS) * self.label_classes)
        self.final_head = nn.Linear(hidden_dim, len(base.FINAL_LABELS) * self.label_classes)
        self.primary_stage_head = nn.Linear(hidden_dim, len(base.PRIMARY_FAILURE_STAGES))
        self.final_anchor_head = nn.Linear(hidden_dim, 1)
        self.ranking_head = nn.Linear(hidden_dim, 1)

    def split_node_features(self, text: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None]:
        start = 0
        bge = text[:, start : start + self.layout.bge_dim]
        start += self.layout.bge_dim
        prm = None
        aux = None
        if self.layout.prm_hidden_dim > 0:
            prm = text[:, start : start + self.layout.prm_hidden_dim]
            start += self.layout.prm_hidden_dim
        if self.layout.aux_dim > 0:
            aux = text[:, start : start + self.layout.aux_dim]
        return bge, prm, aux

    def initial_node_hidden(self, text: Tensor) -> Tensor:
        bge, prm, aux = self.split_node_features(text)
        bge_repr = F.gelu(self.bge_proj(self.bge_norm(bge)))
        hidden = bge_repr
        if self.prm_proj is not None and self.prm_norm is not None and prm is not None:
            prm_repr = F.gelu(self.prm_proj(self.prm_norm(prm)))
            gate_parts = [bge_repr, prm_repr]
            if aux is not None:
                gate_parts.append(aux)
            gate = torch.sigmoid(self.prm_gate(torch.cat(gate_parts, dim=-1)))
            hidden = hidden + gate * prm_repr
        if self.aux_proj is not None and aux is not None:
            hidden = hidden + self.aux_proj(aux)
        return hidden

    def validate_example_indices(self, example: base.GraphExample) -> None:
        node_type_ids = example.node_type_ids
        if node_type_ids.numel() > 0:
            min_node_type = int(node_type_ids.min().item())
            max_node_type = int(node_type_ids.max().item())
            upper = int(self.node_type_embedding.num_embeddings)
            if min_node_type < 0 or max_node_type >= upper:
                raise RuntimeError(
                    "GraphPRM node_type_ids are out of range for the current model. "
                    f"Observed min={min_node_type}, max={max_node_type}, embedding_size={upper}."
                )
        edge_index = example.edge_index
        if edge_index.numel() > 0:
            min_edge = int(edge_index.min().item())
            max_edge = int(edge_index.max().item())
            node_count = int(example.text_embeddings.shape[0])
            if min_edge < 0 or max_edge >= node_count:
                raise RuntimeError(
                    "GraphPRM edge_index contains out-of-range node references. "
                    f"Observed min={min_edge}, max={max_edge}, node_count={node_count}."
                )

    def node_head_representation(
        self,
        hidden: Tensor,
        edge_index: Tensor,
        node_index: int,
    ) -> Tensor:
        center = hidden[node_index]
        if self.node_head_context != "ego" or self.local_pool_proj is None or edge_index.numel() == 0:
            return center

        src_matches = edge_index[0] == node_index
        dst_matches = edge_index[1] == node_index
        neighbor_indices = torch.cat(
            [edge_index[1, src_matches], edge_index[0, dst_matches]],
            dim=0,
        )
        if neighbor_indices.numel() == 0:
            return center
        neighbor_indices = torch.unique(neighbor_indices)
        neighbor_indices = neighbor_indices[neighbor_indices != node_index]
        if neighbor_indices.numel() == 0:
            return center

        ego_hidden = hidden[neighbor_indices]
        mean_pool = ego_hidden.mean(dim=0)
        max_pool = ego_hidden.max(dim=0).values
        return F.gelu(self.local_pool_proj(torch.cat([center, mean_pool, max_pool], dim=-1)))

    def forward(self, example: base.GraphExample, device: torch.device) -> dict[str, Any]:
        self.validate_example_indices(example)
        text = example.text_embeddings.to(device)
        metadata = example.metadata_features.to(device)
        node_type_ids = example.node_type_ids.to(device)
        edge_index = example.edge_index.to(device)
        edge_type_ids = example.edge_type_ids.to(device)

        hidden = (
            self.initial_node_hidden(text)
            + self.meta_proj(metadata)
            + self.node_type_embedding(node_type_ids)
        )
        hidden = self.dropout(F.gelu(hidden))
        for layer in self.layers:
            hidden = layer(hidden, edge_index, edge_type_ids)

        mean_pool = hidden.mean(dim=0)
        max_pool = hidden.max(dim=0).values
        graph_embedding = F.gelu(self.graph_pool_proj(torch.cat([mean_pool, max_pool], dim=-1)))
        decomposer_repr = self.node_head_representation(hidden, edge_index, example.decomposer_index)
        final_repr = self.node_head_representation(hidden, edge_index, example.final_index)

        def expose_logits(raw: Tensor, count: int) -> Tensor:
            raw = raw.view(count, self.label_classes)
            if self.label_classes == 2:
                # Keep the legacy evaluator's three-column interface while
                # training only two real classes: bad, good. The middle class
                # is permanently unavailable in binary mode.
                blocked = torch.full((count, 1), -1e4, dtype=raw.dtype, device=raw.device)
                return torch.cat([raw[:, :1], blocked, raw[:, 1:]], dim=-1)
            return raw

        graph_label_logits = expose_logits(self.graph_label_head(graph_embedding), len(base.GRAPH_LABELS))
        decomposer_logits = expose_logits(self.decomposer_head(decomposer_repr), len(base.DECOMPOSER_LABELS))
        final_logits = expose_logits(self.final_head(final_repr), len(base.FINAL_LABELS))

        worker_logits: dict[str, Tensor] = {}
        for node_id, index in example.worker_indices_by_node_id.items():
            worker_repr = self.node_head_representation(hidden, edge_index, index)
            worker_logits[node_id] = expose_logits(self.worker_head(worker_repr), len(base.WORKER_LABELS))

        return {
            "hidden": hidden,
            "graph_embedding": graph_embedding,
            "graph_label_logits": graph_label_logits,
            "decomposer_logits": decomposer_logits,
            "worker_logits": worker_logits,
            "final_logits": final_logits,
            "primary_stage_logits": self.primary_stage_head(graph_embedding),
            "final_anchor_logit": self.final_anchor_head(graph_embedding).squeeze(-1),
            "ranking_score": self.ranking_head(graph_embedding).squeeze(-1),
        }


def instantiate_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> GraphPRMHybridModel:
    encoder_spec = checkpoint.get("encoder_spec") or {}
    layout = graphprm_layout_from_encoder_spec(encoder_spec)
    if layout is None:
        state_dict_for_layout = checkpoint["model_state_dict"]
        inferred_bge_dim = int(state_dict_for_layout["bge_proj.weight"].shape[1])
        inferred_prm_dim = (
            int(state_dict_for_layout["prm_proj.weight"].shape[1])
            if "prm_proj.weight" in state_dict_for_layout
            else 0
        )
        inferred_aux_dim = (
            int(state_dict_for_layout["aux_proj.0.weight"].shape[1])
            if "aux_proj.0.weight" in state_dict_for_layout
            else 0
        )
        layout = GraphPRMFeatureLayout(
            mode="inferred_from_state_dict",
            bge_dim=inferred_bge_dim,
            prm_hidden_dim=inferred_prm_dim,
            aux_dim=inferred_aux_dim,
            aux_names=tuple(f"aux_{index}" for index in range(inferred_aux_dim)),
        )
    set_graphprm_layout(layout)
    config = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]
    metadata_dim = int(state_dict["meta_proj.0.weight"].shape[1])
    model = GraphPRMHybridModel(
        text_dim=layout.total_dim,
        metadata_dim=metadata_dim,
        hidden_dim=int(config["hidden_dim"]),
        message_passing_layers=int(config["message_passing_layers"]),
        dropout=float(config["dropout"]),
        graph_architecture=str(config.get("graph_architecture", "relational")),
        gat_heads=int(config.get("gat_heads", 4)),
        node_head_context=str(config.get("node_head_context", "ego")),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

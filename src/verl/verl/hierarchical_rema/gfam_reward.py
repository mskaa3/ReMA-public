from __future__ import annotations

import copy
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .schema import (
    DecompositionCandidate,
    SelectionCandidate,
    TaskExample,
    WorkerExecution,
)

GRAPH_LABELS = (
    "G_final_correct",
    "G_cascade_present",
    "G_hierarchy_bypassed",
)
DECOMPOSER_LABELS = (
    "D_coverage",
    "D_dependency_correct",
    "D_subtasks_solvable",
    "D_role_drift",
    "D_under_decomposition",
)
SELECTOR_LABELS = (
    "S_worker_match",
    "S_bad_routing_caused_failure",
)
WORKER_LABELS = (
    "W_subtask_solved",
    "W_used_dependencies",
    "W_format_correct",
    "W_reasoning_or_fact_error",
    "W_role_drift",
    "W_nonfinal_solved_final",
    "W_trivial_finalization",
    "W_contaminated_by_upstream",
)
FINAL_LABELS = (
    "F_aggregation_error",
    "F_verification_error",
    "F_answer_missing_or_invalid",
)
PRIMARY_FAILURE_STAGES = (
    "decomposition",
    "selection",
    "worker",
    "final",
    "verification",
    "none",
    "unclear",
)
NODE_HEAD_CONTEXTS = ("center", "ego")
MISSINGISH_ANSWER_TEXTS = {
    "",
    "none",
    "null",
    "n/a",
    "na",
    "unknown",
    "no answer",
    "no final answer",
    "final answer",
    "produce the final answer",
    "return the final answer",
}
NODE_TYPES = (
    "question",
    "decomposer",
    "subtask",
    "selector",
    "worker",
    "final",
)
ROLE_TYPES = NODE_TYPES
EDGE_TYPES_BASE = (
    "question_to_decomposer",
    "decomposes_to",
    "declared_dep",
    "decomposition_available_to_selector",
    "selects_for_subtask",
    "assigned_to",
    "task_input",
    "uses_output",
    "contributes_final",
)
EDGE_TYPES = EDGE_TYPES_BASE + tuple(f"{name}_rev" for name in EDGE_TYPES_BASE)
NODE_TYPE_TO_ID = {name: idx for idx, name in enumerate(NODE_TYPES)}
ROLE_TYPE_TO_ID = {name: idx for idx, name in enumerate(ROLE_TYPES)}
EDGE_TYPE_TO_ID = {name: idx for idx, name in enumerate(EDGE_TYPES)}
DEFAULT_HASH_DIM = 384


def stable_hash(text: str) -> int:
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def yes_strength_from_centered(value: float) -> float:
    return clamp01((float(value) + 1.0) / 2.0)


def absence_strength_from_centered(value: float) -> float:
    return clamp01((1.0 - float(value)) / 2.0)


def is_missingish_text(value: Any) -> bool:
    normalized = normalize_text(value).lower()
    if not normalized:
        return True
    if normalized in MISSINGISH_ANSWER_TEXTS:
        return True
    if normalized.startswith("final answer:"):
        suffix = normalized.split(":", 1)[1].strip()
        return suffix in MISSINGISH_ANSWER_TEXTS or not suffix
    return False


def get_final_worker_record(record: dict[str, Any]) -> dict[str, Any] | None:
    final_node_id = record.get("trajectory", {}).get("final_node_id")
    for worker in record.get("workers", []):
        if worker.get("node_id") == final_node_id or worker.get("is_final_node"):
            return worker
    return None


def record_missing_final_answer_signal(record: dict[str, Any]) -> float:
    final_answer_missing = is_missingish_text(record.get("trajectory", {}).get("final_answer"))
    final_worker = get_final_worker_record(record)
    worker_output_missing = True
    if final_worker is not None:
        worker_output_missing = is_missingish_text(final_worker.get("output_text"))
    return 1.0 if final_answer_missing or worker_output_missing else 0.0


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BaseSentenceEncoder:
    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError


class HashingSentenceEncoder(BaseSentenceEncoder):
    def __init__(self, embedding_dim: int = DEFAULT_HASH_DIM) -> None:
        super().__init__(embedding_dim=embedding_dim)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        vectors = []
        for text in texts:
            tokens = re.findall(r"[A-Za-z0-9_]+", text.lower()) or ["<empty>"]
            vector = torch.zeros(self.embedding_dim, dtype=torch.float32)
            for token in tokens:
                token_hash = stable_hash(token)
                index = token_hash % self.embedding_dim
                sign = -1.0 if ((token_hash >> 1) & 1) else 1.0
                vector[index] += sign
            norm = vector.norm(p=2)
            if norm > 0:
                vector /= norm
            vectors.append(vector)
        return torch.stack(vectors, dim=0)


class SentenceTransformersEncoder(BaseSentenceEncoder):
    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        sample = self.model.encode(["sample"], convert_to_tensor=True, normalize_embeddings=True)
        super().__init__(embedding_dim=int(sample.shape[-1]))

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).detach().cpu()


class TransformersMeanPoolEncoder(BaseSentenceEncoder):
    def __init__(self, model_name: str, device: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        super().__init__(embedding_dim=int(self.model.config.hidden_size))

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        batches = []
        with torch.no_grad():
            for start in range(0, len(texts), 16):
                batch_texts = texts[start : start + 16]
                tokens = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                outputs = self.model(**tokens)
                mask = tokens["attention_mask"].unsqueeze(-1)
                summed = (outputs.last_hidden_state * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp_min(1)
                pooled = F.normalize(summed / counts, p=2, dim=-1)
                batches.append(pooled.detach().cpu())
        return torch.cat(batches, dim=0)


def build_sentence_encoder(
    backend: str,
    model_name: str,
    device: torch.device,
) -> BaseSentenceEncoder:
    if backend in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformersEncoder(model_name=model_name, device=str(device))
        except Exception:
            if backend == "sentence-transformers":
                raise
    if backend in ("auto", "transformers"):
        try:
            return TransformersMeanPoolEncoder(model_name=model_name, device=str(device))
        except Exception:
            if backend == "transformers":
                raise
    return HashingSentenceEncoder()


@dataclass
class GraphExample:
    source_record: dict[str, Any]
    text_embeddings: torch.Tensor
    node_type_ids: torch.Tensor
    role_ids: torch.Tensor
    worker_bucket_ids: torch.Tensor
    metadata_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type_ids: torch.Tensor
    decomposer_index: int
    final_index: int
    selector_indices_by_node_id: dict[str, int]
    worker_indices_by_node_id: dict[str, int]


@dataclass
class LoadedRewardModel:
    checkpoint_path: Path
    checkpoint: dict[str, Any]
    encoder: BaseSentenceEncoder
    model: GFAMSmallModel
    worker_bucket_count: int
    device: torch.device


def build_metadata_vector(
    *,
    node_type: str,
    depth: int,
    indegree: int,
    outdegree: int,
    is_final_node: bool,
    dependency_count: int,
    downstream_count: int,
) -> list[float]:
    return [
        depth / 8.0,
        indegree / 8.0,
        outdegree / 8.0,
        1.0 if is_final_node else 0.0,
        dependency_count / 8.0,
        downstream_count / 8.0,
        1.0 if node_type == "worker" else 0.0,
        1.0 if node_type == "subtask" else 0.0,
    ]


def worker_bucket_id(worker_id: str | None, bucket_count: int) -> int:
    if not worker_id:
        return 0
    return 1 + (stable_hash(worker_id) % max(1, bucket_count - 1))


def edge_key(from_node_id: str, to_node_id: str) -> str:
    return f"{from_node_id}->{to_node_id}"


def selector_assignment_node_ids(record: dict[str, Any]) -> list[str]:
    node_ids: list[str] = []
    seen: set[str] = set()
    for item in record.get("selection", {}).get("assignments", []):
        node_id = str(item.get("node_id") or "").strip()
        if not node_id or node_id in seen:
            continue
        node_ids.append(node_id)
        seen.add(node_id)
    return node_ids


def _build_role_texts(record: dict[str, Any]) -> dict[str, str]:
    question_text = normalize_text(record["task"]["prompt"])
    decomposition_text = normalize_text(
        record["decomposition"].get("raw_text") or record["decomposition"].get("summary")
    )
    final_text = normalize_text(f"final_answer={record['trajectory'].get('final_answer')}")
    return {
        "question": question_text,
        "decomposer": decomposition_text,
        "final": final_text,
    }


def build_graph_example_for_inference(
    record: dict[str, Any],
    encoder: BaseSentenceEncoder,
    worker_bucket_count: int,
) -> GraphExample:
    shared_texts = _build_role_texts(record)
    subtask_map = {item["node_id"]: item for item in record["decomposition"]["subtasks"]}
    used_edges = {
        edge_key(item["from_node_id"], item["to_node_id"]): item
        for item in record["graph"].get("used_dependency_edges", [])
    }

    node_texts: list[str] = []
    node_type_ids: list[int] = []
    role_ids: list[int] = []
    worker_buckets: list[int] = []
    metadata_rows: list[list[float]] = []
    subtask_indices_by_node_id: dict[str, int] = {}
    selector_indices_by_node_id: dict[str, int] = {}
    worker_indices_by_node_id: dict[str, int] = {}

    question_index = len(node_texts)
    node_texts.append(shared_texts["question"])
    node_type_ids.append(NODE_TYPE_TO_ID["question"])
    role_ids.append(ROLE_TYPE_TO_ID["question"])
    worker_buckets.append(0)
    metadata_rows.append(
        build_metadata_vector(
            node_type="question",
            depth=0,
            indegree=0,
            outdegree=1,
            is_final_node=False,
            dependency_count=0,
            downstream_count=0,
        )
    )

    decomposer_index = len(node_texts)
    node_texts.append(shared_texts["decomposer"])
    node_type_ids.append(NODE_TYPE_TO_ID["decomposer"])
    role_ids.append(ROLE_TYPE_TO_ID["decomposer"])
    worker_buckets.append(0)
    metadata_rows.append(
        build_metadata_vector(
            node_type="decomposer",
            depth=1,
            indegree=1,
            outdegree=len(subtask_map) + 1,
            is_final_node=False,
            dependency_count=0,
            downstream_count=len(subtask_map),
        )
    )

    ordered_subtasks = sorted(record["decomposition"]["subtasks"], key=lambda item: item["node_id"])
    for subtask in ordered_subtasks:
        idx = len(node_texts)
        subtask_indices_by_node_id[subtask["node_id"]] = idx
        skills = subtask.get("required_skills", [])
        subtask_text = normalize_text(
            " ".join(
                [
                    f"instruction={subtask.get('instruction')}",
                    f"required_skills={','.join(skills)}",
                    f"dependencies={','.join(subtask.get('dependencies', []))}",
                ]
            )
        )
        node_texts.append(subtask_text)
        node_type_ids.append(NODE_TYPE_TO_ID["subtask"])
        role_ids.append(ROLE_TYPE_TO_ID["subtask"])
        worker_buckets.append(0)
        metadata_rows.append(
            build_metadata_vector(
                node_type="subtask",
                depth=2,
                indegree=len(subtask.get("dependencies", [])) + 1,
                outdegree=1,
                is_final_node=subtask["node_id"] == record["trajectory"]["final_node_id"],
                dependency_count=len(subtask.get("dependencies", [])),
                downstream_count=0,
            )
        )

    assignment_map = {
        str(item.get("node_id")): item
        for item in record.get("selection", {}).get("assignments", [])
        if str(item.get("node_id") or "").strip()
    }
    selector_node_ids = sorted(
        assignment_map,
        key=lambda node_id: (
            0,
            int(node_id),
        ) if node_id.isdigit() else (1, node_id),
    )
    if not selector_node_ids and record.get("selection", {}).get("raw_text"):
        selector_node_ids = ["__aggregate__"]

    for selector_node_id in selector_node_ids:
        idx = len(node_texts)
        selector_indices_by_node_id[selector_node_id] = idx
        assignment = assignment_map.get(selector_node_id, {})
        subtask = subtask_map.get(selector_node_id, {})
        selector_text = normalize_text(
            " ".join(
                [
                    f"subtask={subtask.get('instruction') or selector_node_id}",
                    f"required_skills={','.join(subtask.get('required_skills', []))}",
                    f"dependencies={','.join(subtask.get('dependencies', []))}",
                    f"chosen_worker={assignment.get('worker_id')}",
                    f"selection_plan={record.get('selection', {}).get('raw_text') if selector_node_id == '__aggregate__' else ''}",
                ]
            )
        )
        node_texts.append(selector_text)
        node_type_ids.append(NODE_TYPE_TO_ID["selector"])
        role_ids.append(ROLE_TYPE_TO_ID["selector"])
        worker_buckets.append(0)
        metadata_rows.append(
            build_metadata_vector(
                node_type="selector",
                depth=3,
                indegree=2 if selector_node_id in subtask_indices_by_node_id else 1,
                outdegree=1 if selector_node_id in assignment_map else 0,
                is_final_node=False,
                dependency_count=len(subtask.get("dependencies", [])),
                downstream_count=1 if selector_node_id in assignment_map else 0,
            )
        )

    ordered_workers = sorted(record["workers"], key=lambda item: item["node_id"])
    for worker in ordered_workers:
        idx = len(node_texts)
        worker_indices_by_node_id[worker["node_id"]] = idx
        upstream_summary = "; ".join(
            f"{item.get('node_id')}={normalize_text(item.get('used_value'))}"
            for item in worker.get("upstream_context", [])
        )
        worker_text = normalize_text(
            " ".join(
                [
                    f"subtask={worker['subtask'].get('instruction')}",
                    f"worker_id={worker.get('worker_id')}",
                    f"output={worker.get('output_text')}",
                    f"raw_output={worker.get('raw_output_text')}",
                    f"upstream={upstream_summary}",
                ]
            )
        )
        node_texts.append(worker_text)
        node_type_ids.append(NODE_TYPE_TO_ID["worker"])
        role_ids.append(ROLE_TYPE_TO_ID["worker"])
        worker_buckets.append(worker_bucket_id(worker.get("worker_id"), worker_bucket_count))
        metadata_rows.append(
            build_metadata_vector(
                node_type="worker",
                depth=4,
                indegree=len(worker.get("declared_dependencies", [])) + 2,
                outdegree=len(worker.get("downstream_used_by", [])) + int(worker.get("is_final_node", False)),
                is_final_node=bool(worker.get("is_final_node")),
                dependency_count=len(worker.get("declared_dependencies", [])),
                downstream_count=len(worker.get("downstream_used_by", [])),
            )
        )

    final_index = len(node_texts)
    node_texts.append(shared_texts["final"])
    node_type_ids.append(NODE_TYPE_TO_ID["final"])
    role_ids.append(ROLE_TYPE_TO_ID["final"])
    worker_buckets.append(0)
    metadata_rows.append(
        build_metadata_vector(
            node_type="final",
            depth=5,
            indegree=1,
            outdegree=0,
            is_final_node=True,
            dependency_count=1,
            downstream_count=0,
        )
    )

    edge_rows: list[tuple[int, int, int]] = []

    def add_directed_edge(src: int, dst: int, edge_type: str) -> None:
        edge_rows.append((src, dst, EDGE_TYPE_TO_ID[edge_type]))
        edge_rows.append((dst, src, EDGE_TYPE_TO_ID[f"{edge_type}_rev"]))

    add_directed_edge(question_index, decomposer_index, "question_to_decomposer")

    for subtask in ordered_subtasks:
        subtask_index = subtask_indices_by_node_id[subtask["node_id"]]
        add_directed_edge(decomposer_index, subtask_index, "decomposes_to")
        for dependency in subtask.get("dependencies", []):
            dependency_index = subtask_indices_by_node_id.get(dependency)
            if dependency_index is not None:
                add_directed_edge(dependency_index, subtask_index, "declared_dep")
        selector_index = selector_indices_by_node_id.get(subtask["node_id"])
        if selector_index is not None:
            add_directed_edge(decomposer_index, selector_index, "decomposition_available_to_selector")
            add_directed_edge(subtask_index, selector_index, "selects_for_subtask")

    aggregate_selector_index = selector_indices_by_node_id.get("__aggregate__")
    if aggregate_selector_index is not None:
        add_directed_edge(decomposer_index, aggregate_selector_index, "decomposition_available_to_selector")

    for worker in ordered_workers:
        worker_index = worker_indices_by_node_id[worker["node_id"]]
        subtask_index = subtask_indices_by_node_id[worker["node_id"]]
        selector_index = selector_indices_by_node_id.get(worker["node_id"])
        if selector_index is None:
            selector_index = aggregate_selector_index
        if selector_index is not None:
            add_directed_edge(selector_index, worker_index, "assigned_to")
        add_directed_edge(subtask_index, worker_index, "task_input")
        for dependency in worker.get("declared_dependencies", []):
            dependency_index = worker_indices_by_node_id.get(dependency)
            if dependency_index is not None and edge_key(dependency, worker["node_id"]) not in used_edges:
                add_directed_edge(dependency_index, worker_index, "declared_dep")

    for used in record["graph"].get("used_dependency_edges", []):
        src_index = worker_indices_by_node_id.get(used["from_node_id"])
        dst_index = worker_indices_by_node_id.get(used["to_node_id"])
        if src_index is not None and dst_index is not None:
            add_directed_edge(src_index, dst_index, "uses_output")

    final_worker_index = worker_indices_by_node_id.get(record["trajectory"]["final_node_id"])
    if final_worker_index is not None:
        add_directed_edge(final_worker_index, final_index, "contributes_final")

    edge_index_tensor = torch.tensor(
        [[src, dst] for src, dst, _ in edge_rows],
        dtype=torch.long,
    ).t().contiguous()
    edge_type_tensor = torch.tensor(
        [edge_type for _, _, edge_type in edge_rows],
        dtype=torch.long,
    )

    return GraphExample(
        source_record=record,
        text_embeddings=encoder.encode_texts(node_texts),
        node_type_ids=torch.tensor(node_type_ids, dtype=torch.long),
        role_ids=torch.tensor(role_ids, dtype=torch.long),
        worker_bucket_ids=torch.tensor(worker_buckets, dtype=torch.long),
        metadata_features=torch.tensor(metadata_rows, dtype=torch.float32),
        edge_index=edge_index_tensor,
        edge_type_ids=edge_type_tensor,
        decomposer_index=decomposer_index,
        final_index=final_index,
        selector_indices_by_node_id=selector_indices_by_node_id,
        worker_indices_by_node_id=worker_indices_by_node_id,
    )


def expected_score_from_probs(probabilities: Tensor) -> Tensor:
    values = torch.tensor([0.0, 1.0, 2.0], dtype=probabilities.dtype, device=probabilities.device)
    return (probabilities * values).sum(dim=-1) / 2.0


def centered_score_from_probs(probabilities: Tensor) -> Tensor:
    values = torch.tensor([-1.0, 0.0, 1.0], dtype=probabilities.dtype, device=probabilities.device)
    return (probabilities * values).sum(dim=-1)


def compile_rewards_from_scores(compiler_inputs: dict[str, Any]) -> dict[str, Any]:
    example: GraphExample = compiler_inputs["example"]
    graph_scores = compiler_inputs["graph_scores"]
    decomposer_scores = compiler_inputs["decomposer_scores"]
    raw_selector_scores = compiler_inputs["selector_scores"]
    worker_scores = compiler_inputs["worker_scores"]
    final_scores = compiler_inputs["final_scores"]
    final_anchor_score = float(compiler_inputs["final_anchor_score"])
    record = example.source_record
    final_worker_id = record["trajectory"]["final_node_id"]
    selector_node_ids = selector_assignment_node_ids(record)

    if raw_selector_scores and all(label_name in raw_selector_scores for label_name in SELECTOR_LABELS):
        selector_scores_by_node_id = {
            node_id: dict(raw_selector_scores)
            for node_id in (selector_node_ids or ["__aggregate__"])
        }
    else:
        selector_scores_by_node_id = {
            str(node_id): dict(score_map)
            for node_id, score_map in raw_selector_scores.items()
        }
    if not selector_scores_by_node_id:
        selector_scores_by_node_id = {
            node_id: {label_name: 0.0 for label_name in SELECTOR_LABELS}
            for node_id in (selector_node_ids or ["__aggregate__"])
        }

    def success(score_map: dict[str, float], key: str) -> float:
        return float(score_map.get(key, 0.0))

    def error(score_map: dict[str, float], key: str) -> float:
        return yes_strength_from_centered(float(score_map.get(key, 0.0)))

    def lack(score_map: dict[str, float], key: str) -> float:
        return absence_strength_from_centered(float(score_map.get(key, 0.0)))

    observed_missing_final = record_missing_final_answer_signal(record)
    label_missing_final = error(final_scores, "F_answer_missing_or_invalid")
    missing_final_signal = max(observed_missing_final, label_missing_final)

    q_decomposer = (
        +0.40 * success(decomposer_scores, "D_coverage")
        + 0.30 * success(decomposer_scores, "D_dependency_correct")
        + 0.25 * success(decomposer_scores, "D_subtasks_solvable")
        - 0.25 * error(decomposer_scores, "D_role_drift")
        - 0.35 * error(decomposer_scores, "D_under_decomposition")
    )

    worker_local: dict[str, dict[str, float]] = {}
    for node_id, scores in worker_scores.items():
        subtask_solved = success(scores, "W_subtask_solved")
        used_dependencies = success(scores, "W_used_dependencies")
        format_correct = success(scores, "W_format_correct")
        reasoning_error = error(scores, "W_reasoning_or_fact_error")
        role_drift = error(scores, "W_role_drift")
        nonfinal_solved_final = error(scores, "W_nonfinal_solved_final")
        trivial_finalization = error(scores, "W_trivial_finalization")
        contaminated = error(scores, "W_contaminated_by_upstream")

        q_worker = (
            +0.50 * subtask_solved
            + 0.25 * used_dependencies
            + 0.15 * format_correct
            - 0.35 * reasoning_error
            - 0.25 * role_drift
            - 0.20 * nonfinal_solved_final
            - 0.35 * trivial_finalization
        )
        own_fault = max(
            reasoning_error,
            role_drift,
            absence_strength_from_centered(subtask_solved),
            nonfinal_solved_final,
            trivial_finalization,
        )
        fault_weight = own_fault * (1.0 - 0.70 * contaminated)
        badness = max(
            0.0,
            0.60 * absence_strength_from_centered(subtask_solved)
            + 0.20 * reasoning_error
            + 0.15 * role_drift
            + 0.20 * trivial_finalization
            + 0.15 * nonfinal_solved_final,
        )
        worker_local[node_id] = {
            "quality": q_worker,
            "protect": contaminated,
            "fault_weight": fault_weight,
            "badness": badness,
            "trivial_finalization": trivial_finalization,
        }

    aggregation_error = error(final_scores, "F_aggregation_error")
    verification_error = error(final_scores, "F_verification_error")
    final_answer_missing_or_invalid = missing_final_signal
    final_badness = max(
        0.0,
        0.45 * aggregation_error
        + 0.30 * verification_error
        + 0.45 * final_answer_missing_or_invalid,
    )
    final_quality = (
        0.45 * success(graph_scores, "G_final_correct")
        - 0.35 * aggregation_error
        - 0.25 * verification_error
        - 0.40 * final_answer_missing_or_invalid
    )
    final_fault = max(
        aggregation_error,
        verification_error,
        final_answer_missing_or_invalid,
    )

    transition_weights: dict[tuple[str, str], float] = {}
    for edge in record.get("graph", {}).get("used_dependency_edges", []):
        from_node_id = str(edge.get("from_node_id") or "")
        to_node_id = str(edge.get("to_node_id") or "")
        if from_node_id and to_node_id:
            transition_weights[(from_node_id, to_node_id)] = 0.75

    downstream_graph: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (src, dst), weight in transition_weights.items():
        if weight > 0:
            downstream_graph[src].append((dst, weight))

    def max_path_weight(start: str, goal: str) -> float:
        if start == goal:
            return 1.0
        frontier = [(start, 1.0)]
        best = 0.0
        visited: dict[str, float] = {start: 1.0}
        while frontier:
            current, current_weight = frontier.pop()
            for neighbor, edge_weight in downstream_graph.get(current, []):
                new_weight = current_weight * edge_weight
                if new_weight <= visited.get(neighbor, 0.0):
                    continue
                visited[neighbor] = new_weight
                if neighbor == goal:
                    best = max(best, new_weight)
                frontier.append((neighbor, new_weight))
        return best

    def downstream_consequence(node_id: str) -> float:
        numerator = 0.0
        denominator = 0.0
        for target_id in worker_local:
            if target_id == node_id:
                continue
            weight = max_path_weight(node_id, target_id)
            if weight <= 0:
                continue
            numerator += weight * worker_local[target_id]["badness"]
            denominator += weight
        final_weight = max_path_weight(node_id, final_worker_id)
        if final_weight > 0:
            final_weight *= 1.35
            numerator += final_weight * final_badness
            denominator += final_weight
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    reach_to_final = {node_id: max_path_weight(node_id, final_worker_id) for node_id in worker_local}

    node_rewards: dict[str, Any] = {
        "decomposer": {"node_id": "d"},
        "selector": {"node_id": "s"},
        "selector_decisions": {},
        "workers": {},
        "final": {"node_id": "f"},
    }

    decomposer_fault = max(
        error(decomposer_scores, "D_role_drift"),
        lack(decomposer_scores, "D_subtasks_solvable"),
        lack(decomposer_scores, "D_dependency_correct"),
        error(decomposer_scores, "D_under_decomposition"),
    )

    final_failure_severity = clamp01((1.0 - final_anchor_score) / 2.0)

    def controller_branch_weight(node_id: str) -> float:
        return max(0.25, float(reach_to_final.get(node_id, 0.0)))

    worker_branch_consequences = {
        node_id: downstream_consequence(node_id) * controller_branch_weight(node_id)
        for node_id in worker_local
    }

    selector_assignment_nodes = [
        node_id for node_id in selector_node_ids if node_id in worker_local
    ]
    if not selector_assignment_nodes:
        selector_assignment_nodes = list(worker_local)

    def positive_cap(min_cap: float, failure_weight: float, invalid_weight: float = 0.0) -> float:
        return clamp01(
            max(
                min_cap,
                1.0
                - failure_weight * final_failure_severity
                - invalid_weight * final_answer_missing_or_invalid,
            )
        )

    def apply_positive_cap(value: float, cap: float) -> float:
        if value <= 0.0:
            return value
        return cap * value

    node_rewards["decomposer"]["quality"] = q_decomposer
    decomposer_branch_consequence = mean_or_zero(list(worker_branch_consequences.values()))

    decomposer_global_penalty = (
        0.10 * final_failure_severity
        + 0.06 * final_answer_missing_or_invalid
    )
    decomposer_traceable_penalty = 0.18 * decomposer_fault * decomposer_branch_consequence
    decomposer_success_bonus = 0.05 * success(graph_scores, "G_final_correct")

    node_rewards["decomposer"]["downstream_consequence"] = decomposer_branch_consequence
    node_rewards["decomposer"]["global_failure_penalty"] = decomposer_global_penalty
    node_rewards["decomposer"]["traceable_branch_penalty"] = decomposer_traceable_penalty

    decomposer_pre_cap = (
        0.85 * q_decomposer
        - decomposer_global_penalty
        - decomposer_traceable_penalty
        + decomposer_success_bonus
    )

    decomposer_cap = positive_cap(0.45, 0.35, 0.03)
    node_rewards["decomposer"]["reward_pre_cap"] = decomposer_pre_cap
    node_rewards["decomposer"]["positive_cap"] = decomposer_cap
    node_rewards["decomposer"]["reward_raw"] = apply_positive_cap(decomposer_pre_cap, decomposer_cap)

    assignment_worker_map = {
        str(item.get("node_id")): item.get("worker_id")
        for item in record.get("selection", {}).get("assignments", [])
    }
    selector_payloads: list[dict[str, Any]] = []
    for selector_node_id, selector_scores in selector_scores_by_node_id.items():
        q_selector = (
            +0.50 * success(selector_scores, "S_worker_match")
            - 0.40 * error(selector_scores, "S_bad_routing_caused_failure")
        )
        selector_fault = max(
            error(selector_scores, "S_bad_routing_caused_failure"),
            lack(selector_scores, "S_worker_match"),
        )
        selector_branch_consequence = worker_branch_consequences.get(selector_node_id, 0.0)
        selector_global_penalty = (
            0.07 * final_failure_severity
            + 0.04 * final_answer_missing_or_invalid
        )
        selector_traceable_penalty = 0.24 * selector_fault * selector_branch_consequence
        selector_success_bonus = 0.04 * success(graph_scores, "G_final_correct")
        selector_pre_cap = (
            0.85 * q_selector
            - selector_global_penalty
            - selector_traceable_penalty
            + selector_success_bonus
        )
        selector_cap = positive_cap(0.35, 0.45, 0.05)
        selector_reward_raw = apply_positive_cap(selector_pre_cap, selector_cap)
        payload = {
            "node_id": selector_node_id,
            "assigned_worker_id": assignment_worker_map.get(selector_node_id),
            "quality": q_selector,
            "downstream_consequence": selector_branch_consequence,
            "global_failure_penalty": selector_global_penalty,
            "traceable_branch_penalty": selector_traceable_penalty,
            "reward_pre_cap": selector_pre_cap,
            "positive_cap": selector_cap,
            "reward_raw": selector_reward_raw,
        }
        node_rewards["selector_decisions"][selector_node_id] = payload
        selector_payloads.append(payload)

    if selector_payloads:
        node_rewards["selector"]["decision_count"] = len(selector_payloads)
        node_rewards["selector"]["quality"] = mean_or_zero([item["quality"] for item in selector_payloads])
        node_rewards["selector"]["downstream_consequence"] = mean_or_zero(
            [item["downstream_consequence"] for item in selector_payloads]
        )
        node_rewards["selector"]["global_failure_penalty"] = mean_or_zero(
            [item["global_failure_penalty"] for item in selector_payloads]
        )
        node_rewards["selector"]["traceable_branch_penalty"] = mean_or_zero(
            [item["traceable_branch_penalty"] for item in selector_payloads]
        )
        node_rewards["selector"]["reward_pre_cap"] = mean_or_zero(
            [item["reward_pre_cap"] for item in selector_payloads]
        )
        node_rewards["selector"]["positive_cap"] = mean_or_zero(
            [item["positive_cap"] for item in selector_payloads]
        )
        node_rewards["selector"]["reward_raw"] = mean_or_zero(
            [item["reward_raw"] for item in selector_payloads]
        )
    else:
        node_rewards["selector"]["decision_count"] = 0
        node_rewards["selector"]["quality"] = 0.0
        node_rewards["selector"]["downstream_consequence"] = 0.0
        node_rewards["selector"]["global_failure_penalty"] = 0.0
        node_rewards["selector"]["traceable_branch_penalty"] = 0.0
        node_rewards["selector"]["reward_pre_cap"] = 0.0
        node_rewards["selector"]["positive_cap"] = 1.0
        node_rewards["selector"]["reward_raw"] = 0.0

    for node_id, stats in worker_local.items():
        d_u = downstream_consequence(node_id)
        final_worker_penalty = 0.0
        if node_id == final_worker_id:
            final_worker_penalty = 0.40 * max(final_badness, final_answer_missing_or_invalid) * (
                1.0 - 0.50 * stats["protect"]
            )
        reward_pre_cap = (
            0.80 * stats["quality"]
            - 0.30 * d_u
            - 0.20 * final_failure_severity * reach_to_final.get(node_id, 0.0) * stats["fault_weight"]
            - final_worker_penalty
        )
        worker_cap = (
            positive_cap(0.08, 0.92, 0.20)
            if node_id == final_worker_id
            else positive_cap(0.25, 0.75, 0.05)
        )
        node_rewards["workers"][node_id] = {
            "node_id": node_id,
            "quality": stats["quality"],
            "downstream_consequence": d_u,
            "inherited_protection": stats["protect"],
            "fault_weight": stats["fault_weight"],
            "final_stage_penalty": final_worker_penalty,
            "is_final_worker": node_id == final_worker_id,
            "reward_pre_cap": reward_pre_cap,
            "positive_cap": worker_cap,
            "reward_raw": apply_positive_cap(reward_pre_cap, worker_cap),
        }

    final_pre_cap = (
        0.85 * final_quality
        - 0.35 * final_failure_severity * final_fault
    )
    final_cap = positive_cap(0.05, 0.95, 0.25)
    node_rewards["final"]["quality"] = final_quality
    node_rewards["final"]["downstream_consequence"] = 0.0
    node_rewards["final"]["reward_pre_cap"] = final_pre_cap
    node_rewards["final"]["positive_cap"] = final_cap
    node_rewards["final"]["reward_raw"] = apply_positive_cap(final_pre_cap, final_cap)

    node_rewards["decomposer"]["reward"] = node_rewards["decomposer"]["reward_raw"]
    node_rewards["selector"]["reward"] = node_rewards["selector"]["reward_raw"]
    for payload in node_rewards["selector_decisions"].values():
        payload["reward"] = payload["reward_raw"]
    node_rewards["final"]["reward"] = node_rewards["final"]["reward_raw"]
    for payload in node_rewards["workers"].values():
        payload["reward"] = payload["reward_raw"]

    return {
        "node_rewards": node_rewards,
        "graph_summary": {
            "graph_final_correct_score": graph_scores.get("G_final_correct", 0.0),
            "cascade_score": graph_scores.get("G_cascade_present", 0.0),
            "hierarchy_bypass_score": graph_scores.get("G_hierarchy_bypassed", 0.0),
            "final_anchor_score": final_anchor_score,
            "final_failure_severity": final_failure_severity,
            "final_answer_missing_or_invalid_score": final_answer_missing_or_invalid,
        },
    }


def compile_rewards_from_predictions(example: GraphExample, outputs: dict[str, Any]) -> dict[str, Any]:
    def probs_to_scores(logits: Tensor, label_names: tuple[str, ...]) -> dict[str, float]:
        probabilities = torch.softmax(logits, dim=-1)
        scores = centered_score_from_probs(probabilities).detach().cpu().tolist()
        return {name: float(score) for name, score in zip(label_names, scores)}

    graph_scores = probs_to_scores(outputs["graph_label_logits"], GRAPH_LABELS)
    decomposer_scores = probs_to_scores(outputs["decomposer_logits"], DECOMPOSER_LABELS)
    selector_scores = {
        node_id: probs_to_scores(logits, SELECTOR_LABELS)
        for node_id, logits in outputs["selector_logits"].items()
    }
    final_scores = probs_to_scores(outputs["final_logits"], FINAL_LABELS)
    worker_scores = {
        node_id: probs_to_scores(logits, WORKER_LABELS)
        for node_id, logits in outputs["worker_logits"].items()
    }
    final_anchor_prob = float(torch.sigmoid(outputs["final_anchor_logit"]).detach().cpu())
    compiler_inputs = {
        "graph_scores": graph_scores,
        "decomposer_scores": decomposer_scores,
        "selector_scores": selector_scores,
        "worker_scores": worker_scores,
        "final_scores": final_scores,
        "final_anchor_score": 2.0 * final_anchor_prob - 1.0,
        "example": example,
    }
    return compile_rewards_from_scores(compiler_inputs)


class RelationalMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, relation_count: int, dropout: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.rel_linears = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(relation_count)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, edge_index: Tensor, edge_type_ids: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return hidden
        device = hidden.device
        message_sum = torch.zeros_like(hidden, device=device)
        message_count = torch.zeros(hidden.shape[0], 1, dtype=hidden.dtype, device=device)
        for relation_id, relation_linear in enumerate(self.rel_linears):
            mask = edge_type_ids == relation_id
            if not torch.any(mask):
                continue
            relation_edges = edge_index[:, mask]
            src = relation_edges[0]
            dst = relation_edges[1]
            messages = relation_linear(hidden[src])
            message_sum.index_add_(0, dst, messages)
            ones = torch.ones((dst.shape[0], 1), dtype=hidden.dtype, device=device)
            message_count.index_add_(0, dst, ones)
        aggregated = message_sum / message_count.clamp_min(1.0)
        updated = F.gelu(self.self_linear(hidden) + aggregated)
        return self.norm(hidden + self.dropout(updated))


class GFAMSmallModel(nn.Module):
    def __init__(
        self,
        *,
        text_dim: int,
        metadata_dim: int,
        hidden_dim: int,
        message_passing_layers: int,
        dropout: float,
        worker_bucket_count: int,
        node_head_context: str = "ego",
    ) -> None:
        super().__init__()
        if node_head_context not in NODE_HEAD_CONTEXTS:
            raise ValueError(
                f"Unsupported node_head_context={node_head_context!r}; "
                f"expected one of {NODE_HEAD_CONTEXTS}."
            )
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.meta_proj = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_type_embedding = nn.Embedding(len(NODE_TYPES), hidden_dim)
        self.role_embedding = nn.Embedding(len(ROLE_TYPES), hidden_dim)
        self.worker_embedding = nn.Embedding(worker_bucket_count, hidden_dim)
        self.layers = nn.ModuleList(
            RelationalMessagePassingLayer(hidden_dim, len(EDGE_TYPES), dropout)
            for _ in range(message_passing_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.node_head_context = node_head_context
        self.graph_pool_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.local_pool_proj = (
            nn.Linear(hidden_dim * 3, hidden_dim) if node_head_context == "ego" else None
        )
        self.graph_label_head = nn.Linear(hidden_dim, len(GRAPH_LABELS) * 3)
        self.decomposer_head = nn.Linear(hidden_dim, len(DECOMPOSER_LABELS) * 3)
        self.selector_head = nn.Linear(hidden_dim, len(SELECTOR_LABELS) * 3)
        self.worker_head = nn.Linear(hidden_dim, len(WORKER_LABELS) * 3)
        self.final_head = nn.Linear(hidden_dim, len(FINAL_LABELS) * 3)
        self.primary_stage_head = nn.Linear(hidden_dim, len(PRIMARY_FAILURE_STAGES))
        self.final_anchor_head = nn.Linear(hidden_dim, 1)
        self.ranking_head = nn.Linear(hidden_dim, 1)

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

    def forward(self, example: GraphExample, device: torch.device) -> dict[str, Any]:
        text = example.text_embeddings.to(device)
        metadata = example.metadata_features.to(device)
        node_type_ids = example.node_type_ids.to(device)
        role_ids = example.role_ids.to(device)
        worker_bucket_ids = example.worker_bucket_ids.to(device)
        edge_index = example.edge_index.to(device)
        edge_type_ids = example.edge_type_ids.to(device)

        hidden = (
            self.text_proj(text)
            + self.meta_proj(metadata)
            + self.node_type_embedding(node_type_ids)
            + self.role_embedding(role_ids)
            + self.worker_embedding(worker_bucket_ids)
        )
        hidden = self.dropout(F.gelu(hidden))
        for layer in self.layers:
            hidden = layer(hidden, edge_index, edge_type_ids)

        mean_pool = hidden.mean(dim=0)
        max_pool = hidden.max(dim=0).values
        graph_embedding = F.gelu(self.graph_pool_proj(torch.cat([mean_pool, max_pool], dim=-1)))
        decomposer_repr = self.node_head_representation(hidden, edge_index, example.decomposer_index)
        final_repr = self.node_head_representation(hidden, edge_index, example.final_index)

        graph_label_logits = self.graph_label_head(graph_embedding).view(len(GRAPH_LABELS), 3)
        decomposer_logits = self.decomposer_head(decomposer_repr).view(len(DECOMPOSER_LABELS), 3)
        final_logits = self.final_head(final_repr).view(len(FINAL_LABELS), 3)

        selector_logits: dict[str, Tensor] = {}
        for node_id, index in example.selector_indices_by_node_id.items():
            selector_repr = self.node_head_representation(hidden, edge_index, index)
            selector_logits[node_id] = self.selector_head(selector_repr).view(len(SELECTOR_LABELS), 3)

        worker_logits: dict[str, Tensor] = {}
        for node_id, index in example.worker_indices_by_node_id.items():
            worker_repr = self.node_head_representation(hidden, edge_index, index)
            worker_logits[node_id] = self.worker_head(worker_repr).view(len(WORKER_LABELS), 3)

        return {
            "hidden": hidden,
            "graph_embedding": graph_embedding,
            "graph_label_logits": graph_label_logits,
            "decomposer_logits": decomposer_logits,
            "selector_logits": selector_logits,
            "worker_logits": worker_logits,
            "final_logits": final_logits,
            "primary_stage_logits": self.primary_stage_head(graph_embedding),
            "final_anchor_logit": self.final_anchor_head(graph_embedding).squeeze(-1),
            "ranking_score": self.ranking_head(graph_embedding).squeeze(-1),
        }


def build_encoder_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
    backend_override: str | None,
    model_override: str | None,
) -> BaseSentenceEncoder:
    encoder_spec = checkpoint.get("encoder_spec", {})
    backend = backend_override or encoder_spec.get("backend", "hashing")
    model_name = model_override or encoder_spec.get("model_name")
    if backend == "hashing":
        return build_sentence_encoder("hashing", model_name or "", device)
    if not model_name:
        raise RuntimeError(
            "GFAM checkpoint does not contain an encoder model name; pass an override."
        )
    encoder = build_sentence_encoder(backend, model_name, device)
    expected_dim = checkpoint["model_state_dict"]["text_proj.weight"].shape[1]
    if int(encoder.embedding_dim) != int(expected_dim):
        raise RuntimeError(
            f"GFAM encoder embedding dim mismatch: checkpoint expects {expected_dim}, "
            f"but encoder produced {encoder.embedding_dim}."
        )
    return encoder


def current_label_schema() -> dict[str, tuple[str, ...]]:
    return {
        "graph": GRAPH_LABELS,
        "decomposer": DECOMPOSER_LABELS,
        "selector": SELECTOR_LABELS,
        "worker": WORKER_LABELS,
        "final": FINAL_LABELS,
        "primary_failure_stages": PRIMARY_FAILURE_STAGES,
    }


def current_head_dimensions() -> dict[str, int]:
    return {
        "graph_label_head": len(GRAPH_LABELS) * 3,
        "decomposer_head": len(DECOMPOSER_LABELS) * 3,
        "selector_head": len(SELECTOR_LABELS) * 3,
        "worker_head": len(WORKER_LABELS) * 3,
        "final_head": len(FINAL_LABELS) * 3,
        "primary_stage_head": len(PRIMARY_FAILURE_STAGES),
    }


def current_scoring_contract() -> dict[str, Any]:
    return {
        "checkpoint_schema_changes": {
            "added_decomposer_labels": ["D_under_decomposition"],
            "added_worker_labels": ["W_trivial_finalization"],
            "added_final_labels": ["F_answer_missing_or_invalid"],
            "removed_selector_labels": ["S_dependency_readiness"],
            "removed_worker_labels": ["W_contaminates_downstream"],
            "removed_edge_label_family": True,
        },
        "reward_probability_mapping": {
            "type": "centered",
            "formula": "centered_score = p(class=2) - p(class=0)",
        },
        "graph_edges": {
            "supervision": "structural_only",
            "note": "Used dependency edges remain as graph structure for message passing and propagation.",
        },
        "reward_export": {
            "raw_field": "reward_raw",
            "normalized_field": "reward_normalized",
            "default_reward_field_before_normalization": "reward",
        },
        "node_head_context": {
            "default": "ego",
            "available": list(NODE_HEAD_CONTEXTS),
            "legacy_checkpoint_fallback": "center",
        },
    }


def resolved_node_head_context_from_checkpoint(checkpoint: dict[str, Any]) -> str:
    return str(checkpoint.get("config", {}).get("node_head_context", "center"))


def instantiate_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> GFAMSmallModel:
    config = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]
    text_dim = int(state_dict["text_proj.weight"].shape[1])
    metadata_dim = int(state_dict["meta_proj.0.weight"].shape[1])
    model = GFAMSmallModel(
        text_dim=text_dim,
        metadata_dim=metadata_dim,
        hidden_dim=int(config["hidden_dim"]),
        message_passing_layers=int(config["message_passing_layers"]),
        dropout=float(config["dropout"]),
        worker_bucket_count=int(config["worker_bucket_count"]),
        node_head_context=resolved_node_head_context_from_checkpoint(checkpoint),
    ).to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        expected_dims = current_head_dimensions()
        checkpoint_dims = {
            "graph_label_head": int(state_dict["graph_label_head.weight"].shape[0]),
            "decomposer_head": int(state_dict["decomposer_head.weight"].shape[0]),
            "selector_head": int(state_dict["selector_head.weight"].shape[0]),
            "worker_head": int(state_dict["worker_head.weight"].shape[0]),
            "final_head": int(state_dict["final_head.weight"].shape[0]),
            "primary_stage_head": int(state_dict["primary_stage_head.weight"].shape[0]),
        }
        mismatch_hints = []
        for head_name, expected_dim in expected_dims.items():
            observed_dim = checkpoint_dims.get(head_name)
            if observed_dim is None or observed_dim == expected_dim:
                continue
            if head_name.endswith("_head") and observed_dim % 3 == 0 and expected_dim % 3 == 0:
                mismatch_hints.append(
                    f"{head_name}: checkpoint has {observed_dim // 3} labels, "
                    f"current code expects {expected_dim // 3}"
                )
            else:
                mismatch_hints.append(
                    f"{head_name}: checkpoint dim={observed_dim}, current code expects dim={expected_dim}"
                )
        raise RuntimeError(
            "Checkpoint/schema mismatch while loading GFAMSmallModel.\n"
            "This usually means the inference code still uses an older GFAM label schema.\n"
            f"Current label schema:\n{json.dumps(current_label_schema(), ensure_ascii=True, indent=2)}\n"
            "Observed head mismatches:\n"
            + "\n".join(f"  - {item}" for item in mismatch_hints)
            + f"\nOriginal load_state_dict error:\n{exc}"
        ) from exc
    model.eval()
    return model


def latest_checkpoint_path() -> Path:
    alias_candidates = sorted(Path("processed/gfam_runs").glob("**/best_overall_model.pkl"))
    if alias_candidates:
        return max(alias_candidates, key=lambda path: path.stat().st_mtime)
    candidates = sorted(Path("processed/gfam_runs").glob("**/best_model.pkl"))
    if not candidates:
        raise FileNotFoundError("No GFAM checkpoint found under processed/gfam_runs.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_reward_model_bundle(
    *,
    model_pkl: str | Path | None = None,
    device_name: str = "auto",
    encoder_backend: str | None = None,
    encoder_model: str | None = None,
) -> LoadedRewardModel:
    device = resolve_device(device_name)
    checkpoint_path = Path(model_pkl).expanduser() if model_pkl else latest_checkpoint_path()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = build_encoder_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
        backend_override=encoder_backend,
        model_override=encoder_model,
    )
    model = instantiate_model_from_checkpoint(checkpoint=checkpoint, device=device)
    return LoadedRewardModel(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        encoder=encoder,
        model=model,
        worker_bucket_count=int(checkpoint["config"]["worker_bucket_count"]),
        device=device,
    )


def tensor_label_prediction(logits: torch.Tensor, label_names: tuple[str, ...]) -> dict[str, Any]:
    probabilities = torch.softmax(logits, dim=-1)
    centered_scores = centered_score_from_probs(probabilities).detach().cpu().tolist()
    yes_probabilities = probabilities[:, 2].detach().cpu().tolist()
    hard_labels = logits.argmax(dim=-1).detach().cpu().tolist()
    return {
        "hard_labels": {name: int(value) for name, value in zip(label_names, hard_labels)},
        "centered_label_scores": {
            name: float(value) for name, value in zip(label_names, centered_scores)
        },
        "label_truth_scores": {
            name: float(value) for name, value in zip(label_names, centered_scores)
        },
        "yes_probabilities": {
            name: float(value) for name, value in zip(label_names, yes_probabilities)
        },
    }


def summarize_predictions(example: GraphExample, outputs: dict[str, Any]) -> dict[str, Any]:
    del example
    summary = {
        "graph": tensor_label_prediction(outputs["graph_label_logits"], GRAPH_LABELS),
        "decomposer": tensor_label_prediction(outputs["decomposer_logits"], DECOMPOSER_LABELS),
        "selectors": {},
        "final": tensor_label_prediction(outputs["final_logits"], FINAL_LABELS),
        "workers": {},
    }
    summary["graph"]["primary_failure_stage"] = PRIMARY_FAILURE_STAGES[
        int(outputs["primary_stage_logits"].argmax().detach().cpu())
    ]
    summary["graph"]["final_correct_probability"] = float(
        torch.sigmoid(outputs["final_anchor_logit"]).detach().cpu()
    )
    summary["graph"]["ranking_score"] = float(outputs["ranking_score"].detach().cpu())

    for node_id, logits in outputs["selector_logits"].items():
        summary["selectors"][node_id] = tensor_label_prediction(logits, SELECTOR_LABELS)
    for node_id, logits in outputs["worker_logits"].items():
        summary["workers"][node_id] = tensor_label_prediction(logits, WORKER_LABELS)
    return summary


def flatten_compiled_rewards(compiled_rewards: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_row(role: str, payload: dict[str, Any]) -> None:
        rows.append({"role": role, **payload})

    add_row("decomposer", compiled_rewards["decomposer"])
    add_row("selector", compiled_rewards["selector"])
    for node_id in sorted(compiled_rewards.get("selector_decisions", {})):
        add_row("selector_decision", compiled_rewards["selector_decisions"][node_id])
    for node_id in sorted(compiled_rewards["workers"]):
        add_row("worker", compiled_rewards["workers"][node_id])
    add_row("final", compiled_rewards["final"])
    return rows


def build_scored_row(
    record: dict[str, Any],
    compiled: dict[str, Any],
    predicted_labels: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "trajectory_id": record.get("trajectory_id"),
        "task_id": record.get("task", {}).get("task_id"),
        "metadata": {
            "folder_filename_id": record.get("metadata", {}).get("folder_filename_id"),
        },
        "graph_summary": compiled["graph_summary"],
        "compiled_rewards": compiled["node_rewards"],
        "predicted_labels": predicted_labels,
    }
    row["compiled_rewards_flat"] = flatten_compiled_rewards(row["compiled_rewards"])
    return row


def score_rollout_record(
    bundle: LoadedRewardModel,
    record: dict[str, Any],
) -> dict[str, Any]:
    example = build_graph_example_for_inference(
        record=record,
        encoder=bundle.encoder,
        worker_bucket_count=bundle.worker_bucket_count,
    )
    with torch.no_grad():
        outputs = bundle.model(example, device=bundle.device)
    compiled = compile_rewards_from_predictions(example, outputs)
    predicted_labels = summarize_predictions(example, outputs)
    return build_scored_row(record, compiled, predicted_labels)


def normalize_reward_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_role_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    global_role_values: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        task_id = row["task_id"]
        rewards = row["compiled_rewards"]
        for role in ("decomposer", "final"):
            raw = float(rewards[role]["reward_raw"])
            task_role_values[(task_id, role)].append(raw)
            global_role_values[role].append(raw)
        selector_decision_rewards = rewards.get("selector_decisions", {})
        if selector_decision_rewards:
            for payload in selector_decision_rewards.values():
                raw = float(payload["reward_raw"])
                task_role_values[(task_id, "selector")].append(raw)
                global_role_values["selector"].append(raw)
        else:
            raw = float(rewards["selector"]["reward_raw"])
            task_role_values[(task_id, "selector")].append(raw)
            global_role_values["selector"].append(raw)
        for payload in rewards["workers"].values():
            raw = float(payload["reward_raw"])
            task_role_values[(task_id, "worker")].append(raw)
            global_role_values["worker"].append(raw)

    def stats(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 1.0
        if len(values) == 1:
            return values[0], 1.0
        mean_value = statistics.fmean(values)
        std_value = max(statistics.pstdev(values), 1e-6)
        return mean_value, std_value

    task_role_stats = {key: stats(values) for key, values in task_role_values.items()}
    global_role_stats = {role: stats(values) for role, values in global_role_values.items()}

    normalized_rows = []
    for row in rows:
        copied = copy.deepcopy(row)
        task_id = copied["task_id"]
        rewards = copied["compiled_rewards"]

        for role in ("decomposer", "final"):
            payload = rewards[role]
            raw = float(payload["reward_raw"])
            if len(task_role_values[(task_id, role)]) >= 2:
                mean_value, std_value = task_role_stats[(task_id, role)]
                scope = "task_role"
            else:
                mean_value, std_value = global_role_stats[role]
                scope = "global_role"
            payload["reward_normalized"] = max(-3.0, min(3.0, (raw - mean_value) / std_value))
            payload["reward"] = payload["reward_normalized"]
            payload["normalization_scope"] = scope

        selector_payloads = list(rewards.get("selector_decisions", {}).values())
        if selector_payloads:
            selector_normalized_values = []
            for payload in selector_payloads:
                raw = float(payload["reward_raw"])
                if len(task_role_values[(task_id, "selector")]) >= 2:
                    mean_value, std_value = task_role_stats[(task_id, "selector")]
                    scope = "task_role"
                else:
                    mean_value, std_value = global_role_stats["selector"]
                    scope = "global_role"
                normalized = max(-3.0, min(3.0, (raw - mean_value) / std_value))
                payload["reward_normalized"] = normalized
                payload["reward"] = normalized
                payload["normalization_scope"] = scope
                selector_normalized_values.append(normalized)
            rewards["selector"]["reward_normalized"] = mean_or_zero(selector_normalized_values)
            rewards["selector"]["reward"] = rewards["selector"]["reward_normalized"]
            rewards["selector"]["normalization_scope"] = "decision_mean"
        else:
            payload = rewards["selector"]
            raw = float(payload["reward_raw"])
            if len(task_role_values[(task_id, "selector")]) >= 2:
                mean_value, std_value = task_role_stats[(task_id, "selector")]
                scope = "task_role"
            else:
                mean_value, std_value = global_role_stats["selector"]
                scope = "global_role"
            payload["reward_normalized"] = max(-3.0, min(3.0, (raw - mean_value) / std_value))
            payload["reward"] = payload["reward_normalized"]
            payload["normalization_scope"] = scope

        for payload in rewards["workers"].values():
            raw = float(payload["reward_raw"])
            if len(task_role_values[(task_id, "worker")]) >= 2:
                mean_value, std_value = task_role_stats[(task_id, "worker")]
                scope = "task_role"
            else:
                mean_value, std_value = global_role_stats["worker"]
                scope = "global_role"
            payload["reward_normalized"] = max(-3.0, min(3.0, (raw - mean_value) / std_value))
            payload["reward"] = payload["reward_normalized"]
            payload["normalization_scope"] = scope

        normalized_rows.append(copied)
    return normalized_rows


def score_rollout_records(
    bundle: LoadedRewardModel,
    records: list[dict[str, Any]],
    *,
    normalize_rewards: bool = False,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    del show_progress
    exported_rows = [score_rollout_record(bundle, record) for record in records]
    if normalize_rewards:
        exported_rows = normalize_reward_rows(exported_rows)
        for row in exported_rows:
            row["compiled_rewards_flat"] = flatten_compiled_rewards(row["compiled_rewards"])
    return exported_rows


def _dependency_was_used(execution: WorkerExecution, dependency_output: str) -> bool:
    if execution.dependency_used:
        return True
    normalized_dependency = normalize_text(dependency_output)
    if not normalized_dependency:
        return False
    haystack = normalize_text(execution.raw_output_text or execution.output_text)
    return bool(haystack and normalized_dependency in haystack)


def _build_inference_record(
    task: TaskExample,
    decomposition: DecompositionCandidate,
    selection: SelectionCandidate,
    executions: list[WorkerExecution],
    final_answer: str,
) -> dict[str, Any]:
    node_map = decomposition.nodes_by_id()
    assignments = [
        {
            "node_id": assignment.node_id,
            "worker_id": assignment.worker_id,
            "compatibility": assignment.compatibility,
        }
        for assignment in selection.assignments
    ]
    used_edges: list[dict[str, str]] = []
    upstream_context_by_node: dict[str, list[dict[str, str]]] = {}
    for execution in executions:
        contexts = []
        for dependency_id, dependency_output in execution.dependency_outputs.items():
            if _dependency_was_used(execution, dependency_output):
                used_edges.append(
                    {
                        "from_node_id": dependency_id,
                        "to_node_id": execution.node_id,
                    }
                )
                contexts.append(
                    {
                        "node_id": dependency_id,
                        "worker_id": next(
                            (
                                candidate.worker_id
                                for candidate in executions
                                if candidate.node_id == dependency_id
                            ),
                            "",
                        ),
                        "output_text": dependency_output,
                        "used_value": dependency_output,
                    }
                )
        upstream_context_by_node[execution.node_id] = contexts

    downstream_used_by: dict[str, list[str]] = {}
    for edge in used_edges:
        downstream_used_by.setdefault(edge["from_node_id"], []).append(edge["to_node_id"])

    workers_payload = []
    for execution in executions:
        node = node_map[execution.node_id]
        assignment = selection.assignment_for(execution.node_id)
        workers_payload.append(
            {
                "node_id": execution.node_id,
                "worker_id": execution.worker_id,
                "assignment": {
                    "node_id": assignment.node_id,
                    "worker_id": assignment.worker_id,
                    "compatibility": assignment.compatibility,
                },
                "subtask": {
                    "node_id": node.node_id,
                    "instruction": node.instruction,
                    "dependencies": list(node.dependencies),
                    "required_skills": list(node.required_skills),
                    "required_skills_note": node.required_skills_note,
                    "output_key": node.output_key,
                },
                "declared_dependencies": list(node.dependencies),
                "dependency_outputs": dict(execution.dependency_outputs),
                "upstream_context": upstream_context_by_node.get(execution.node_id, []),
                "downstream_used_by": downstream_used_by.get(execution.node_id, []),
                "compatibility": execution.compatibility,
                "confidence_reward": execution.confidence_reward,
                "entropy": execution.entropy,
                "output_text": execution.output_text,
                "raw_output_text": execution.raw_output_text,
                "is_final_node": execution.node_id == decomposition.final_node_id,
            }
        )

    declared_edges = []
    for node in decomposition.nodes:
        for dependency_id in node.dependencies:
            declared_edges.append(
                {
                    "from_node_id": dependency_id,
                    "to_node_id": node.node_id,
                }
            )

    return {
        "trajectory_id": f"{task.task_id}::{decomposition.decomposition_id}::{selection.selection_id}",
        "task": {
            "task_id": task.task_id,
            "prompt": task.prompt,
            "ground_truth": task.ground_truth,
        },
        "trajectory": {
            "decomposition_id": decomposition.decomposition_id,
            "selection_id": selection.selection_id,
            "final_answer": final_answer,
            "final_node_id": decomposition.final_node_id,
            "legacy_final_correct": 0.0,
            "score": 0.0,
        },
        "decomposition": {
            "raw_text": decomposition.raw_text,
            "summary": decomposition.summary,
            "final_node_id": decomposition.final_node_id,
            "subtasks": [
                {
                    "node_id": node.node_id,
                    "instruction": node.instruction,
                    "dependencies": list(node.dependencies),
                    "required_skills": list(node.required_skills),
                    "required_skills_note": node.required_skills_note,
                    "output_key": node.output_key,
                }
                for node in decomposition.nodes
            ],
        },
        "selection": {
            "raw_text": selection.raw_text,
            "assignments": assignments,
        },
        "graph": {
            "declared_dependency_edges": declared_edges,
            "used_dependency_edges": used_edges,
            "final_node_id": decomposition.final_node_id,
        },
        "workers": workers_payload,
        "split": "inference",
    }


@dataclass
class GFAMRewardScorer:
    checkpoint_path: str
    device: str = "auto"
    encoder_backend: Optional[str] = None
    encoder_model: Optional[str] = None

    def __post_init__(self) -> None:
        self.bundle = load_reward_model_bundle(
            model_pkl=self.checkpoint_path,
            device_name=self.device,
            encoder_backend=self.encoder_backend,
            encoder_model=self.encoder_model,
        )
        self.device_obj = self.bundle.device
        self.checkpoint = self.bundle.checkpoint
        self.encoder = self.bundle.encoder
        self.model = self.bundle.model
        self.worker_bucket_count = self.bundle.worker_bucket_count
        self.checkpoint_path = str(self.bundle.checkpoint_path)

    def score_rollout(
        self,
        *,
        task: TaskExample,
        decomposition: DecompositionCandidate,
        selection: SelectionCandidate,
        executions: list[WorkerExecution],
        final_answer: str,
    ) -> dict[str, Any]:
        record = _build_inference_record(
            task=task,
            decomposition=decomposition,
            selection=selection,
            executions=executions,
            final_answer=final_answer,
        )
        scored = score_rollout_record(self.bundle, record)
        return {
            "source": "gfam_v1",
            "record": record,
            **scored,
        }

from __future__ import annotations
import argparse
import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F
def truncate_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + " ... [truncated]"


def node_sort_key(node_id: str) -> tuple[int, int | str]:
    text = str(node_id)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def score_value(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    if score < 0.0 or score > 1.0:
        return None
    return score


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def resolve_torch_dtype(value: str, torch_module: Any) -> Any | None:
    if value == "auto":
        return None
    if value == "bfloat16":
        return torch_module.bfloat16
    if value == "float16":
        return torch_module.float16
    if value == "float32":
        return torch_module.float32
    return None


def get_subtasks(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    subtasks: dict[str, dict[str, Any]] = {}
    for subtask in (record.get("decomposition") or {}).get("subtasks") or []:
        node_id = subtask.get("node_id")
        if node_id is not None:
            subtasks[str(node_id)] = subtask
    return subtasks


def get_workers(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    workers: dict[str, dict[str, Any]] = {}
    for worker in record.get("workers") or []:
        node_id = worker.get("node_id")
        if node_id is not None:
            workers[str(node_id)] = worker
    return workers


def choose_edges(record: dict[str, Any], edge_mode: str) -> list[tuple[str, str]]:
    graph = record.get("graph") or {}
    declared = graph.get("declared_dependency_edges") or []
    if edge_mode not in ('declared', 'provided'):
        raise ValueError('Dependency usage cannot be inferred from delivery')
    raw_edges = graph.get('provided_dependency_edges', []) if edge_mode == 'provided' else declared
    edges: set[tuple[str, str]] = set()
    for edge in raw_edges:
        source = edge.get("from_node_id")
        target = edge.get("to_node_id")
        if source is None or target is None:
            continue
        source_text = str(source)
        target_text = str(target)
        if source_text != target_text:
            edges.add((source_text, target_text))
    return sorted(edges, key=lambda item: (node_sort_key(item[0]), node_sort_key(item[1])))


def topological_order(
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> tuple[list[str], bool]:
    original_index = {node_id: idx for idx, node_id in enumerate(nodes)}
    node_set = set(nodes)
    for source, target in edges:
        node_set.add(source)
        node_set.add(target)

    indegree = {node_id: 0 for node_id in node_set}
    adjacency = {node_id: [] for node_id in node_set}
    for source, target in edges:
        adjacency[source].append(target)
        indegree[target] += 1

    queue = deque(
        sorted(
            [node_id for node_id, degree in indegree.items() if degree == 0],
            key=lambda item: (original_index.get(item, 10**9), node_sort_key(item)),
        )
    )
    order: list[str] = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for neighbor in sorted(
            adjacency[node_id],
            key=lambda item: (original_index.get(item, 10**9), node_sort_key(item)),
        ):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
        queue = deque(
            sorted(
                queue,
                key=lambda item: (original_index.get(item, 10**9), node_sort_key(item)),
            )
        )

    cyclic = len(order) != len(node_set)
    if cyclic:
        remaining = [node_id for node_id in nodes if node_id not in order]
        order.extend(remaining)
    return [node_id for node_id in order if node_id in node_set], cyclic


def format_dependency_phrase(
    dependency_ids: list[str],
    node_to_step: dict[str, int],
) -> str:
    step_refs = [f"Step {node_to_step[node_id]}" for node_id in dependency_ids if node_id in node_to_step]
    if not step_refs:
        return "no recorded provided inputs"
    if len(step_refs) == 1:
        return f"provided {step_refs[0]}"
    return "provided " + ", ".join(step_refs[:-1]) + f" and {step_refs[-1]}"


def worker_result(worker: dict[str, Any], max_chars: int) -> str:
    return truncate_text(
        worker.get("output_text")
        or worker.get("worker_output_text")
        or worker.get("raw_output_text")
        or "",
        max_chars,
    )


def load_qwen_prm(args: argparse.Namespace, tokenizer: Any) -> tuple[Any, str]:
    try:
        import torch
        from transformers import AutoConfig, AutoModel
    except Exception as exc:
        raise SystemExit(
            "Qwen PRM rescoring requires torch and transformers. Install project requirements first."
        ) from exc

    device = resolve_device(args.device)
    torch_dtype = resolve_torch_dtype(args.torch_dtype, torch)
    device_map = args.device_map
    if args.device == "auto" and device_map is None:
        device_map = "auto"

    revision = getattr(args, 'revision', None)
    config = AutoConfig.from_pretrained(args.model, revision=revision, trust_remote_code=True)
    if not hasattr(config, "pad_token_id") or getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tokenizer.pad_token_id
    if hasattr(config, "use_cache"):
        config.use_cache = False

    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "config": config, "revision": revision}
    if device_map:
        load_kwargs["device_map"] = device_map
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    model, loading = AutoModel.from_pretrained(args.model, output_loading_info=True, **load_kwargs)
    missing = [k for k in loading.get('missing_keys', []) if not k.endswith('inv_freq')]
    if missing or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise RuntimeError(f'Incomplete PRM checkpoint load: {loading}')
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "config"):
        inner_model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(inner_model.config, "use_cache"):
            inner_model.config.use_cache = False
    if not device_map:
        model.to(device)
    model.eval()

    input_device = device
    if device_map:
        try:
            input_device = str(next(model.parameters()).device)
        except StopIteration:
            input_device = device
    return model, input_device


def load_tokenizer(model_name: str, revision=None) -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise SystemExit("Qwen PRM rescoring requires transformers.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer

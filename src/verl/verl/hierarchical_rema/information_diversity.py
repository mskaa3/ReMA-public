"""GEMMAS Algorithm 1, with an explicit, versioned ReMA execution-DAG mapping.

Observational only: never used for rewards, advantages, or checkpoint selection.
See INFORMATION_DIVERSITY.md for the reproduction choices left open by the paper.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import math
import os
import re

import numpy as np


VERSION = "gemmas_ids_rema_bge_m3_v2"
EXACT_VERSION = "rema_ids_exact_v1"
TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"


@dataclass(frozen=True)
class IDSConfig:
    enabled: bool = True
    model: str = "BAAI/bge-m3"
    revision: str = "main"
    device: str = "cuda:0"
    batch_size: int = 16

    @classmethod
    def from_env(cls):
        enabled = os.getenv("REMA_IDS_ENABLED", "true").lower()
        if enabled not in ("true", "false", "1", "0"):
            raise ValueError("REMA_IDS_ENABLED must be true or false")
        result = cls(enabled=enabled in ("true", "1"),
                     model=os.getenv("REMA_IDS_MODEL", cls.model),
                     revision=os.getenv("REMA_IDS_REVISION", cls.revision),
                     device=os.getenv("REMA_IDS_DEVICE", cls.device),
                     batch_size=int(os.getenv("REMA_IDS_BATCH_SIZE", "16")))
        if result.batch_size < 1:
            raise ValueError("REMA_IDS_BATCH_SIZE must be positive")
        return result

    def metadata(self):
        return {
            **asdict(self), "version": VERSION,
            "paper": "https://arxiv.org/pdf/2507.13190",
            "syntactic_weight": 0.5, "semantic_weight": 0.5,
            "nodes": "worker invocations, final worker included once",
            "responses": "delivered output_text only; no prompt/raw/reference text",
            "spatial": "observed dependency_outputs keys",
            "temporal": "transitive causal reachability of spatial edges",
            "tfidf": {"fit": "nonempty responses within each rollout",
                      "token_pattern": TOKEN_PATTERN, "lowercase": True,
                      "smooth_idf": True, "sublinear_tf": False, "norm": "l2"},
            "semantic_pooling": "L2-normalized CLS; token-weighted normalized chunk mean above 8192 tokens",
            "semantic_dtype": "float32",
            "cosine": "raw cosine; only numerical roundoff clipped to [-1,1]",
            "aggregation": "mean rollout IDS within task, then mean over eligible tasks",
            "candidate_policy": "all validation candidates, no correctness/reward selection",
            "exact": {"version": EXACT_VERSION, "normalization": "strip surrounding whitespace only",
                      "formula": "weighted mean of unequal result-text pairs; same DAG weights as IDS",
                      "range": [0, 1], "encoder_required": False,
                      "interpretation": "textual repetition, not mathematical equivalence or proven redundancy"},
        }


def execution_graph(plan, selection):
    """Copy small evaluation inputs before the much larger rollout is released."""
    executions = selection.executions
    ids = [str(ex.node_id) for ex in executions]
    texts = [str(ex.output_text or "") for ex in executions]
    graph = {"node_ids": ids, "texts": texts, "edges": [], "error": None}
    if len(set(ids)) != len(ids):
        graph["error"] = "duplicate_execution_node_id"
        return graph
    positions = {node_id: i for i, node_id in enumerate(ids)}
    planned = {str(node.node_id): node for node in plan.nodes}
    if set(ids) != set(planned) or str(plan.final_node_id) not in positions:
        graph["error"] = "incomplete_or_unmapped_execution"
        return graph
    for j, ex in enumerate(executions):
        provided = getattr(ex, "dependency_outputs", None)
        if not isinstance(provided, dict) or (planned[ids[j]].dependencies and not provided):
            # Empty legacy defaults cannot establish what the worker received.
            graph["error"] = "missing_dependency_snapshot"
            return graph
        for upstream, snapshot in provided.items():
            upstream = str(upstream)
            if upstream not in positions or upstream == ids[j]:
                graph["error"] = "invalid_dependency_snapshot"
                return graph
            i = positions[upstream]
            if str(snapshot or "") != texts[i]:
                graph["error"] = "dependency_snapshot_differs_from_output"
                return graph
            graph["edges"].append((i, j))
    return graph


def connection_weights(size, edges):
    """S = direct delivery; T = causal reachability, not arbitrary list order."""
    spatial = np.zeros((size, size), dtype=bool)
    for i, j in edges:
        spatial[i, j] = True
    temporal = spatial.copy()
    for k in range(size):
        temporal |= temporal[:, k, None] & temporal[None, k, :]
    if np.diag(temporal).any():
        raise ValueError("cyclic_delivery_graph")
    return (spatial | spatial.T).astype(float) + (temporal | temporal.T).astype(float)


def _pair_inputs(graph):
    if graph["error"]:
        return None, [], {"status": graph["error"]}
    texts = graph["texts"]
    weights = connection_weights(len(texts), graph["edges"])
    all_pairs = [(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))
                 if weights[i, j] > 0]
    pairs = [(i, j) for i, j in all_pairs if texts[i].strip() and texts[j].strip()]
    base = {"eligible_pairs": len(pairs), "connected_pairs": len(all_pairs),
            "empty_outputs": sum(not text.strip() for text in texts),
            "pair_coverage": len(pairs) / len(all_pairs) if all_pairs else None}
    if not pairs:
        return weights, [], {**base, "status": "no_nonempty_connected_pairs"}
    return weights, pairs, {**base, "status": "ok" if len(pairs) == len(all_pairs) else "partial"}


def graph_ids(graph, embeddings):
    """Return undefined with a reason, never a fabricated 0 or 1."""
    weights, pairs, base = _pair_inputs(graph)
    if not pairs:
        return {**base, "ids": None, "pairs": []}
    texts = graph["texts"]

    active = [i for i, text in enumerate(texts) if text.strip()]
    syntax = tfidf_cosines([texts[i] for i in active])
    index = {original: local for local, original in enumerate(active)}
    vectors = np.stack([embeddings[texts[i]] for i in active]).astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or np.any(norms <= 0):
        raise ValueError("nonfinite_or_zero_semantic_embedding")
    vectors /= norms[:, None]
    semantics = np.clip(vectors @ vectors.T, -1, 1)
    records = []
    for i, j in pairs:
        syn = float(np.clip(syntax[index[i], index[j]], 0, 1))
        sem = float(semantics[index[i], index[j]])
        records.append({"source": graph["node_ids"][i], "target": graph["node_ids"][j],
                        "weight": float(weights[i, j]), "tfidf_cosine": syn,
                        "semantic_cosine": sem, "diversity": 1 - .5 * (syn + sem)})
    denominator = sum(pair["weight"] for pair in records)
    score = sum(pair["weight"] * pair["diversity"] for pair in records) / denominator
    return {**base, "ids": score, "weight_sum": denominator, "pairs": records}


def graph_ids_exact(graph):
    """Encoder-free companion: exact delivered-result equality, not correctness."""
    weights, pairs, base = _pair_inputs(graph)
    status = base.pop("status")
    if not pairs:
        return {**base, "ids_exact": None, "exact_status": status, "exact_pairs": []}
    texts = [text.strip() for text in graph["texts"]]
    records = [{"source": graph["node_ids"][i], "target": graph["node_ids"][j],
                "weight": float(weights[i, j]), "equal": texts[i] == texts[j]}
               for i, j in pairs]
    denominator = sum(pair["weight"] for pair in records)
    score = sum(pair["weight"] for pair in records if not pair["equal"]) / denominator
    direct = sorted({(i, j) for i, j in graph["edges"] if texts[i] and texts[j]})
    matching = [(i, j) for i, j in direct if texts[i] == texts[j]]
    incoming = defaultdict(list)
    for i, j in matching:
        incoming[j].append(i)
    paths = {}

    def path_to(node):
        if node not in paths:
            prefix = max((path_to(parent) for parent in incoming[node]), key=len, default=[])
            paths[node] = prefix + [node]
        return paths[node]

    # _pair_inputs has already rejected cycles; input list order need not be topological.
    longest = max((path_to(j) for _, j in matching), key=len, default=[])
    return {**base, "ids_exact": score, "exact_status": status,
            "exact_weight_sum": denominator, "exact_pairs": records,
            "exact_direct_match_edges": [[graph["node_ids"][i], graph["node_ids"][j]]
                                         for i, j in matching],
            "exact_direct_eligible_edges": len(direct),
            "exact_direct_copy_rate": len(matching) / len(direct) if direct else None,
            "exact_longest_copy_chain": [graph["node_ids"][i] for i in longest]}


def tfidf_cosines(texts):
    """Smoothed-IDF/raw-TF/L2 cosine, fitted per graph without an extra dependency."""
    counts = [Counter(re.findall(TOKEN_PATTERN, text.lower())) for text in texts]
    vocabulary = sorted(set().union(*(count.keys() for count in counts)))
    if not vocabulary:
        raise ValueError("empty_tfidf_vocabulary")
    matrix = np.array([[count[word] for word in vocabulary] for count in counts], dtype=float)
    idf = np.log((1 + len(texts)) / (1 + (matrix > 0).sum(axis=0))) + 1
    matrix *= idf
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return matrix @ matrix.T


class BGEResponseEncoder:
    """Frozen BGE-M3 dense CLS embeddings; no silent truncation of long results."""
    def __init__(self, config):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, revision=config.revision, trust_remote_code=False)
        # Model construction must not perturb the policy's CPU RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.model = AutoModel.from_pretrained(
                config.model, revision=config.revision, trust_remote_code=False,
                torch_dtype=torch.float32)
        if self.model.config.model_type != "xlm-roberta":
            raise ValueError("This IDS version requires BGE-M3 (XLM-RoBERTa); set REMA_IDS_MODEL=BAAI/bge-m3")
        self.model.to(config.device).eval()
        self.model.requires_grad_(False)
        # XLM-R position IDs start at padding_idx + 1, rather than zero.
        position_capacity = self.model.config.max_position_embeddings - self.model.config.pad_token_id - 1
        self.window = min(8192, self.tokenizer.model_max_length, position_capacity)
        if self.window <= self.tokenizer.num_special_tokens_to_add(pair=False):
            raise ValueError("BGE window leaves no room for content tokens")
        self.metadata = {"resolved_revision": getattr(self.model.config, "_commit_hash", None),
                         "semantic_window": self.window, "dtype": "float32",
                         "semantic_dimension": self.model.config.hidden_size,
                         "transformers_version": __import__("transformers").__version__}

    def encode(self, texts):
        import torch
        from tqdm.auto import tqdm
        capacity = self.window - self.tokenizer.num_special_tokens_to_add(pair=False)
        sums, counts, token_counts, chunks = {}, Counter(), {}, []
        vectors = {}

        def flush():
            if not chunks:
                return
            owners, payloads, lengths = zip(*chunks)
            batch = self.tokenizer.pad(list(payloads), padding=True, return_tensors="pt")
            batch = {key: value.to(self.config.device) for key, value in batch.items()}
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                pooled = torch.nn.functional.normalize(hidden[:, 0].float(), p=2, dim=-1).cpu().numpy()
            for owner, value, length in zip(owners, pooled, lengths):
                sums[owner] = sums.get(owner, np.zeros_like(value)) + length * value
                counts[owner] += length
            chunks.clear()

        for text in tqdm(texts, desc="validation IDS BGE-M3 responses"):
            ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            token_counts[text] = len(ids)
            for start in range(0, len(ids), capacity):
                payload = self.tokenizer.prepare_for_model(
                    ids[start:start + capacity], add_special_tokens=True,
                    truncation=False)
                chunks.append((text, payload, min(capacity, len(ids) - start)))
                if len(chunks) >= self.config.batch_size:
                    flush()
        flush()
        for text in texts:
            if not counts[text]:
                raise ValueError("empty_bge_tokenization")
            value = sums[text] / counts[text]
            norm = np.linalg.norm(value)
            if not np.isfinite(value).all() or not np.isfinite(norm) or norm <= 0:
                raise ValueError("nonfinite_or_zero_bge_embedding")
            vectors[text] = value / norm
        self.metadata.update(unique_responses=len(texts),
                             chunked_responses=sum(n > capacity for n in token_counts.values()),
                             content_tokens=sum(token_counts.values()))
        return vectors

    def close(self):
        import torch
        device = next(self.model.parameters()).device if self.model is not None else None
        self.model = None
        # Release BGE's unused allocator blocks once at the evaluation boundary,
        # so a separate coach server can use the reserved GPU again.
        if device is not None and device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()


class InformationDiversityAudit:
    def __init__(self, config=None, encoder_factory=None):
        self.config = config or IDSConfig.from_env()
        self.encoder_factory = encoder_factory or BGEResponseEncoder
        self.graphs = []

    def add(self, rollout, subset):
        if not self.config.enabled:
            return
        for plan_index, plan in enumerate(rollout.decompositions):
            for repeat_index, selection in enumerate(plan.selections):
                graph = execution_graph(plan.decomposition, selection)
                graph.update(task_id=str(rollout.task.task_id), subset=subset,
                             plan_index=plan_index, execution_index=repeat_index,
                             decomposition_id=plan.decomposition.decomposition_id,
                             selection_id=selection.selection.selection_id,
                             final_correctness=selection.reward.final_answer_correctness)
                self.graphs.append(graph)

    def finish(self):
        metadata = self.config.metadata()
        if not self.config.enabled:
            return {"metadata": metadata, "overall": aggregate_ids([]), "subsets": {}, "rows": []}
        encoder = None
        failure = None
        vectors = {}
        # Do not load BGE for single-node/empty/malformed graphs.
        candidates = []
        for graph in self.graphs:
            if graph["error"]:
                continue
            try:
                weights = connection_weights(len(graph["texts"]), graph["edges"])
                if any(weights[i, j] and graph["texts"][i].strip() and graph["texts"][j].strip()
                       for i in range(len(weights)) for j in range(i + 1, len(weights))):
                    candidates.extend(text for text in graph["texts"] if text.strip())
            except ValueError as exc:
                graph["error"] = str(exc)
        try:
            if candidates:
                encoder = self.encoder_factory(self.config)
                vectors = encoder.encode(list(dict.fromkeys(candidates)))
                metadata.update(encoder.metadata)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            print(f"[hierarchical-rema][IDS] WARNING: unavailable: {failure}", flush=True)
        finally:
            if encoder is not None:
                encoder.close()
        rows = []
        for graph in self.graphs:
            row = {key: value for key, value in graph.items() if key not in ("texts", "error")}
            row["outputs"] = [
                {"node_id": node_id, "chars": len(text),
                 "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
                for node_id, text in zip(graph["node_ids"], graph["texts"])]
            try:
                exact = graph_ids_exact(graph)
            except Exception as exc:
                exact = {"ids_exact": None, "exact_status": "metric_error",
                         "exact_error": f"{type(exc).__name__}: {exc}"}
            try:
                if failure and exact["ids_exact"] is not None:
                    result = {"ids": None, "status": "encoder_unavailable", "error": failure}
                else:
                    result = graph_ids(graph, vectors)
            except Exception as exc:
                result = {"ids": None, "status": "metric_error", "error": f"{type(exc).__name__}: {exc}"}
            rows.append({**row, **exact, **result, "metric_version": VERSION,
                         "exact_metric_version": EXACT_VERSION})
        subsets = {name: aggregate_ids([row for row in rows if row["subset"] == name])
                   for name in sorted({row["subset"] for row in rows})}
        return {"metadata": metadata, "overall": aggregate_ids(rows), "subsets": subsets, "rows": rows}


def aggregate_ids(rows):
    semantic = _aggregate_scores(rows, "ids", "status")
    exact = _aggregate_scores(rows, "ids_exact", "exact_status")
    return {**semantic, "ids_exact": exact["ids_exact"],
            **{f"exact_{key}": exact[key] for key in (
                "eligible_tasks", "eligible_rollouts", "task_coverage", "rollout_coverage", "status_counts")}}


def _aggregate_scores(rows, score_key, status_key):
    tasks = defaultdict(list)
    for row in rows:
        if row.get(score_key) is not None and math.isfinite(row[score_key]):
            tasks[(row["subset"], row["task_id"])].append(row[score_key])
    means = [sum(scores) / len(scores) for scores in tasks.values()]
    total_tasks = len({(row["subset"], row["task_id"]) for row in rows})
    eligible = sum(map(len, tasks.values()))
    return {score_key: sum(means) / len(means) if means else None,
            "eligible_tasks": len(means), "total_tasks": total_tasks,
            "eligible_rollouts": eligible, "total_rollouts": len(rows),
            "task_coverage": len(means) / total_tasks if total_tasks else None,
            "rollout_coverage": eligible / len(rows) if rows else None,
            "status_counts": dict(Counter(row.get(status_key, "not_computed") for row in rows))}

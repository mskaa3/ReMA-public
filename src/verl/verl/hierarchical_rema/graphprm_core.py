"""Portable no-selector graph features and message passing for Graph+PRM."""
from __future__ import annotations

import math
import re
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from .graphprm_inputs import clean_result_text, delivered_result

GRAPH_LABELS = ('global_outcome', 'anti_collapse')
DECOMPOSER_LABELS = ('task_fulfillment', 'execution_discipline')
WORKER_LABELS = FINAL_LABELS = ('task_fulfillment', 'execution_discipline', 'causal_utility')
NODE_TYPES = ('question', 'decomposer', 'subtask', 'worker', 'final')
EDGE_TYPES_BASE = ('question_to_decomposer', 'decomposes_to', 'declared_dep', 'executes_subtask', 'uses_output', 'contributes_final')
EDGE_TYPES = EDGE_TYPES_BASE + tuple(x + '_rev' for x in EDGE_TYPES_BASE)
NODE_HEAD_CONTEXTS = ('center', 'ego')
GRAPH_ARCHITECTURES = ('relational', 'gat')
PRIMARY_FAILURE_STAGES = ('decomposition', 'worker', 'final', 'verification', 'none', 'unclear')


def normalize_text(value):
    return re.sub(r'\s+', ' ', str(value if value is not None else '')).strip()


def effective_final_node_id(record):
    return str(record.get('decomposition', {}).get('final_node_id') or record.get('trajectory', {}).get('final_node_id') or '')


def clamp01(value):
    return max(0., min(1., value))


def absence_strength_from_centered(value):
    return clamp01((1. - value) / 2.)


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.


def result_text(value):
    return clean_result_text(value)


def sort_node(item):
    node = str(item.get('node_id', ''))
    return (0, int(node)) if node.isdigit() else (1, node)


def build_structured_decomposition_text(record):
    plan = record.get('decomposition', {})
    parts = []
    for key in ('target_quantity', 'final_answer_format_hint'):
        if normalize_text(plan.get(key)):
            parts.append(f'{key}={normalize_text(plan[key])}')
    final_id = effective_final_node_id(record)
    if final_id:
        parts.append(f'final_node_id={final_id}')
    for subtask in sorted(plan.get('subtasks', []), key=sort_node):
        parts.extend((f"node_id={subtask.get('node_id')}",
                      f"instruction={normalize_text(subtask.get('instruction'))}",
                      'dependencies=' + (','.join(map(str, subtask.get('dependencies', []))) or 'none'),
                      'required_skills=' + (','.join(map(str, subtask.get('required_skills', []))) or 'none')))
        if normalize_text(subtask.get('required_skills_note')):
            parts.append('required_skills_note=' + normalize_text(subtask['required_skills_note']))
    return normalize_text(' '.join(parts))


def prepare_graph_blueprint(record):
    """Match the training builder's node order, metadata and reverse edges."""
    tasks = sorted(record['decomposition']['subtasks'], key=lambda x: str(x['node_id']))
    workers = sorted(record.get('workers', []), key=lambda x: str(x['node_id']))
    final_id = effective_final_node_id(record)
    texts, types, meta, task_indices, worker_indices, edges = [], [], [], {}, {}, []

    def add(kind, text, depth, indegree, outdegree, final=False, deps=0, downstream=0):
        index = len(texts)
        texts.append(normalize_text(text))
        types.append(NODE_TYPES.index(kind))
        meta.append([depth/8., indegree/8., outdegree/8., float(final), deps/8., downstream/8., float(kind == 'worker'), float(kind == 'subtask')])
        return index

    add('question', record['task']['prompt'], 0, 0, 1)
    add('decomposer', build_structured_decomposition_text(record), 1, 1, len(tasks), downstream=len(tasks))
    for task in tasks:
        node = str(task['node_id'])
        deps = task.get('dependencies', [])
        text = f"instruction={task.get('instruction', '')} required_skills={','.join(task.get('required_skills', []))} dependencies={','.join(map(str, deps))}"
        task_indices[node] = add('subtask', text, 2, 1+len(deps), 1, node == final_id, len(deps))
    for worker in workers:
        node = str(worker['node_id'])
        upstream = '; '.join(f"{item.get('node_id')}={result_text(item.get('used_value', item.get('output_text')))}" for item in worker.get('upstream_context', []))
        text = f"subtask={worker['subtask'].get('instruction', '')} result={delivered_result(worker)} upstream={upstream}"
        deps = len(worker.get('declared_dependencies', []))
        downstream = len(worker.get('downstream_used_by', []))
        worker_indices[node] = add('worker', text, 4, 1+deps, downstream+int(bool(worker.get('is_final_node'))), node == final_id, deps, downstream)
    final_index = add('final', f"final_answer={record['trajectory'].get('final_answer')}", 5, 1, 0, True, 1)

    def edge(src, dst, kind):
        relation = EDGE_TYPES_BASE.index(kind)
        edges.extend(((src, dst, relation), (dst, src, relation + len(EDGE_TYPES_BASE))))

    edge(0, 1, 'question_to_decomposer')
    for task in tasks:
        node = str(task['node_id'])
        edge(1, task_indices[node], 'decomposes_to')
        for dep in task.get('dependencies', []):
            if str(dep) in task_indices:
                edge(task_indices[str(dep)], task_indices[node], 'declared_dep')
    for node, index in worker_indices.items():
        if node in task_indices:
            edge(task_indices[node], index, 'executes_subtask')
    for item in record.get('graph', {}).get('used_dependency_edges', []):
        src, dst = str(item['from_node_id']), str(item['to_node_id'])
        if src in worker_indices and dst in worker_indices:
            edge(worker_indices[src], worker_indices[dst], 'uses_output')
    if final_id in worker_indices:
        edge(worker_indices[final_id], final_index, 'contributes_final')
    return SimpleNamespace(node_texts=texts, node_type_ids=types, metadata_rows=meta,
                           edge_rows=edges, worker_indices_by_node_id=worker_indices,
                           decomposer_index=1, final_index=final_index)


class RelationalMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim, relation_count, dropout):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.rel_linears = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(relation_count))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, edge_index, edge_type_ids):
        if not edge_index.numel():
            return hidden
        messages = torch.zeros_like(hidden)
        counts = hidden.new_zeros((hidden.shape[0], 1))
        for relation, linear in enumerate(self.rel_linears):
            src, dst = edge_index[:, edge_type_ids == relation]
            messages.index_add_(0, dst, linear(hidden[src]))
            counts.index_add_(0, dst, hidden.new_ones((dst.shape[0], 1)))
        return self.norm(hidden + self.dropout(F.gelu(self.self_linear(hidden) + messages/counts.clamp_min(1.))))


class RelationalGraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, relation_count, dropout, heads):
        super().__init__()
        if heads <= 0 or hidden_dim % heads:
            raise ValueError('GAT hidden dimension must be divisible by heads')
        self.hidden_dim, self.heads, self.head_dim = hidden_dim, heads, hidden_dim//heads
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rel_key = nn.Embedding(relation_count, hidden_dim)
        self.rel_value = nn.Embedding(relation_count, hidden_dim)
        self.rel_bias = nn.Embedding(relation_count, heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = 1. / math.sqrt(self.head_dim)

    def forward(self, hidden, edge_index, edge_type_ids):
        if not edge_index.numel():
            return hidden
        n = hidden.shape[0]
        src, dst = edge_index
        q = self.query_proj(hidden).view(n, self.heads, self.head_dim)
        k = self.key_proj(hidden).view(n, self.heads, self.head_dim)
        v = self.value_proj(hidden).view(n, self.heads, self.head_dim)
        rk = self.rel_key(edge_type_ids).view(-1, self.heads, self.head_dim)
        rv = self.rel_value(edge_type_ids).view(-1, self.heads, self.head_dim)
        scores = F.leaky_relu((q[dst]*(k[src]+rk)).sum(-1)*self.scale + self.rel_bias(edge_type_ids), negative_slope=.2)
        messages = v[src]+rv
        aggregated = hidden.new_zeros((n, self.heads, self.head_dim))
        for node in torch.unique(dst).detach().cpu().tolist():
            mask = dst == node
            attention = self.attn_dropout(torch.softmax(scores[mask], dim=0))
            aggregated[node] = (attention.unsqueeze(-1)*messages[mask]).sum(0)
        return self.norm(hidden + self.dropout(F.gelu(self.out_proj(aggregated.reshape(n, self.hidden_dim)))))

"""Binary, class-wise/source-wise reporting on fixed, aligned observations."""
from collections import defaultdict
import csv
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from . import graphprm_core as core
from .data import source, atomic_json, atomic_jsonl
from .rewards import compile_rewards

SCOPES = {'graph': core.GRAPH_LABELS, 'decomposer': core.DECOMPOSER_LABELS,
          'worker': core.WORKER_LABELS, 'final': core.FINAL_LABELS}


def targets(record):
    labels = record.get('labels_binary', {})
    result = {s: {n: labels.get(n) if s == 'graph' else labels.get(s, {}).get(n)
                  for n in names} for s, names in SCOPES.items() if s != 'worker'}
    result['workers'] = {str(k): {n: v.get(n) for n in core.WORKER_LABELS}
                         for k, v in labels.get('workers', {}).items()}
    for scope, values in result.items():
        groups = values.values() if scope == 'workers' else [values]
        for group in groups:
            if any(v is not None and v not in (0, 1) for v in group.values()):
                raise ValueError('Binary training accepts only 0, 1, or missing labels')
    return result


def logits_by_scope(outputs):
    return {'graph': [('', outputs['graph_label_logits'])],
            'decomposer': [('', outputs['decomposer_logits'])],
            'final': [('', outputs['final_logits'])], 'worker': list(outputs['worker_logits'].items())}


def predictions_from_outputs(outputs):
    result = {}
    for scope, nodes in logits_by_scope(outputs).items():
        for node, logits in nodes:
            probs = logits[:, [0, 2]].softmax(-1).detach().cpu()
            if not torch.isfinite(probs).all():
                raise ValueError('Non-finite model probabilities')
            values = {label: {'prediction': int(p[1] >= .5),
                       'probabilities': {'0': float(p[0]), '1': float(p[1])},
                       'centered_score': float(p[1] - p[0])} for label, p in zip(SCOPES[scope], probs)}
            if scope == 'worker':
                result.setdefault('workers', {})[node] = values
            else:
                result[scope] = values
    return result


def metric(values):
    y = np.array([v['target'] for v in values])
    p = np.array([v['p_good'] for v in values])
    pred = p >= .5
    cm = confusion_matrix(y, pred, labels=[0, 1])
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    return {'n': len(y), 'bad_support': n0, 'good_support': n1,
            'macro_f1': float(f1_score(y, pred, labels=[0, 1], average='macro', zero_division=0)),
            'accuracy': float((pred == y).mean()), 'confusion_matrix': cm.tolist(),
            'risky_recall': float(cm[0, 0] / n0) if n0 else None,
            'good_recall': float(cm[1, 1] / n1) if n1 else None,
            'false_positive_reward_rate': float((p[y == 0] > .5).mean()) if n0 else None,
            'mean_centered_bad': float((2*p[y == 0]-1).mean()) if n0 else None,
            'mean_centered_good': float((2*p[y == 1]-1).mean()) if n1 else None,
            'auc': float(roc_auc_score(y, p)) if n0 and n1 else None}


def mean_present(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return sum(values)/len(values) if values else 0.


def summarize(observations):
    grouped = defaultdict(list)
    for item in observations:
        for src in ('ALL', item['source']):
            grouped[(src, item['scope'], item['label'])].append(item)
    metrics = [{'source': src, 'scope': scope, 'label': label, **metric(values)}
               for (src, scope, label), values in sorted(grouped.items())]
    shared = [m for m in metrics if m['source'] == 'ALL' and m['scope'] in ('worker', 'final')]
    by_source = defaultdict(list)
    for m in metrics:
        if m['source'] != 'ALL' and m['scope'] in ('worker', 'final'):
            by_source[m['source']].append(m['macro_f1'])
    all_heads = [m for m in metrics if m['source'] == 'ALL']
    mean_f1 = mean_present(m['macro_f1'] for m in shared)
    risky = mean_present(m['risky_recall'] for m in shared)
    good = mean_present(m['good_recall'] for m in shared)
    worst = min((mean_present(v) for v in by_source.values()), default=0.)
    original = mean_present(by_source.get('original_examples', []))
    # All terms are maximized; subtracting recall/F1 would reward worse models.
    selection = .30*mean_f1 + .25*original + .20*worst + .15*risky + .10*good
    return {'selection_score': selection, 'mean_binary_macro_f1': mean_f1,
            'risky_recall': risky, 'good_recall': good, 'worst_source_macro_f1': worst,
            'original_examples_macro_f1': original,
            'all_head_macro_f1': mean_present(m['macro_f1'] for m in all_heads), 'metrics': metrics}


def evaluate(model, examples, device, bad_class_penalty=1.):
    observations, records = [], []
    model.eval()
    with torch.inference_mode():
        for index, example in enumerate(examples):
            row = example.source_record
            pred = predictions_from_outputs(model(example, device))
            truth = targets(row)
            compiled = compile_rewards(pred, row, bad_class_penalty)
            for scope, names in SCOPES.items():
                nodes = pred['workers'].items() if scope == 'worker' else [('', pred[scope])]
                for node, values in nodes:
                    gold = truth['workers'].get(node, {}) if scope == 'worker' else truth[scope]
                    for name in names:
                        if gold.get(name) is not None:
                            observations.append({'row_index': index, 'source': source(row), 'scope': scope,
                                'node_id': node, 'label': name, 'target': gold[name],
                                'p_good': values[name]['probabilities']['1']})
            records.append({'trajectory_id': row.get('trajectory_id', row.get('trajectory', {}).get('trajectory_id')),
                'task_id': row.get('task', {}).get('task_id'), 'source': source(row), 'predictions': pred,
                'compiled_rewards': compiled['node_rewards'], 'graph_summary': compiled['graph_summary'],
                'prm_step_scores': example.prm_scores})
    return summarize(observations), records, observations


def write_evaluation(directory, summary, records, observations):
    directory = Path(directory)
    atomic_json(directory / 'metrics.json', summary)
    atomic_jsonl(directory / 'predictions.jsonl', records)
    atomic_jsonl(directory / 'observations.jsonl', observations)
    metrics = summary['metrics']
    if metrics:
        with (directory / 'metrics_by_source_scope_label.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)

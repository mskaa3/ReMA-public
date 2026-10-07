"""Auditable migration and task-group splitting. Never calls a model/API."""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from .contract import canonical_record, model_record


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
    os.replace(tmp, path)


def group_keys(row):
    task = row.get('task', {})
    keys = {'prompt:' + ' '.join(str(task.get('prompt', '')).lower().split())}
    for name, value in [('id', task.get('task_id')), ('group', row.get('split_group_id'))]:
        if value:
            keys.add(name + ':' + str(value))
    return keys


def assert_disjoint(left, right):
    overlap = set().union(*(group_keys(r) for r in left)) & set().union(*(group_keys(r) for r in right))
    if overlap:
        raise ValueError(f'Task leakage: {len(overlap)} shared task/prompt/group keys, e.g. {sorted(overlap)[:3]}')


def quarantine_train_overlap(train, test):
    blocked = set().union(*(group_keys(r) for r in test))
    retained, excluded = list(train), []
    changed = True
    while changed:
        changed, next_rows = False, []
        for row in retained:
            keys = group_keys(row)
            if keys & blocked:
                excluded.append(row)
                blocked.update(keys)
                changed = True
            else:
                next_rows.append(row)
        retained = next_rows
    assert_disjoint(retained, test)
    return retained, excluded


def split_validation(rows, ratio, seed):
    # Union overlapping IDs AND prompts, not just the identifier in one dataset.
    import random
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in group_keys(row):
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(root(i), []).append(i)
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    selected, count = set(), 0
    for group in groups[:-1]:
        if count >= max(1, int(len(rows) * ratio)):
            break
        selected.update(group)
        count += len(group)
    train = [r for i, r in enumerate(rows) if i not in selected]
    val = [r for i, r in enumerate(rows) if i in selected]
    if not train or not val:
        raise ValueError('Need at least two independent task groups')
    assert_disjoint(train, val)
    return train, val


def source(row):
    p = row.get('provenance', {})
    return str(p.get('source_dataset') or p.get('source_bucket') or 'unknown')


def migration_report(rows):
    status, audit = Counter(), Counter()
    for row in rows:
        if row.get('dependency_audit', {}).get('invalid_declared_plan'):
            audit['invalid_declared_plan_kept_as_negative_input'] += 1
        for w in row['workers']:
            status[w['dependency_context_status']] += 1
            if not w['output_text'] and w.get('raw_output_text'):
                audit['empty_delivered_nonempty_raw'] += 1
            if w.get('invalid_reason'):
                audit['explicit_invalid_reason'] += 1
            if w.get('reference_answer_match'):
                audit['reference_match_audit_flag'] += 1
    return {'rows': len(rows), 'sources': dict(Counter(map(source, rows))),
            'dependency_context': dict(status), 'audit_flags': dict(audit)}


def migrate(input_path, output_path, policy):
    rows = [canonical_record(r, policy) for r in load_jsonl(input_path)]
    atomic_jsonl(output_path, rows)
    return rows


def feature_key(row, spec):
    return digest({'spec': spec, 'record': model_record(row)})

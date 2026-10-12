#!/usr/bin/env python3
"""Paired accuracy, frozen plan banks, and blinded human audits for EAC3."""

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

from eac3_experiments import question_id, plan_id, sha_file


def load_records(path):
    path = Path(path)
    if path.name.endswith('.partial'):
        raise ValueError('Partial evaluations are not experimental results')
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    metadata = json.loads(path.with_suffix('.metadata.json').read_text())
    if len(records) != metadata['rows'] or sha_file(path) != metadata['records_sha256']:
        raise ValueError('Evaluation completeness/checksum check failed')
    seen = set()
    for row in records:
        if row['question_id'] != question_id(row['question']) or row['plan_id'] != plan_id(row['decomposer_output']):
            raise ValueError('Question/plan identity mismatch')
        key = row.get('eval_item_id', row['question_id'])
        if key in seen:
            raise ValueError('Repeated evaluation items: use n=1 or explicitly aggregate samples first')
        seen.add(key)
    return records


def summarize(records):
    by_question = defaultdict(list)
    by_subset = defaultdict(list)
    for row in records:
        by_question[row['question_id']].append(row['raw_score'])
        by_subset[row['subset']].append(row)
    return {
        'rows': len(records), 'unique_questions': len(by_question),
        'accuracy_question_mean': statistics.mean(statistics.mean(v) for v in by_question.values()),
        'completion_tokens_per_item': statistics.mean(row['completion_tokens'] for row in records),
        'by_subset': {
            subset: {
                'rows': len(rows),
                'accuracy_question_mean': statistics.mean(
                    statistics.mean(row['raw_score'] for row in rows if row['question_id'] == q)
                    for q in sorted({row['question_id'] for row in rows})
                ),
            } for subset, rows in by_subset.items()
        },
    }


def compare(left, right, *, common_plans=False, seed=42, resamples=2000):
    if not left or not right:
        raise ValueError('Cannot compare empty evaluations')
    def indexed(rows):
        return {row.get('eval_item_id', row['question_id']): row for row in rows}
    a, b = indexed(left), indexed(right)
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError('Paired comparison requires exactly the same unique evaluation items')
    deltas = defaultdict(list)
    for key in a:
        if (a[key]['question_id'], a[key]['ground_truth'], a[key]['subset']) != (
            b[key]['question_id'], b[key]['ground_truth'], b[key]['subset']
        ):
            raise ValueError('Paired questions, ground truths, or dataset labels differ')
        if common_plans and (a[key]['plan_id'] != b[key]['plan_id'] or not a[key].get('answer_free_reviewed') or not b[key].get('answer_free_reviewed')):
            raise ValueError('Worker transfer requires identical, independently reviewed plans')
        deltas[a[key]['question_id']].append(b[key]['raw_score'] - a[key]['raw_score'])
    values = [statistics.mean(deltas[key]) for key in sorted(deltas)]
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(resamples))
    return {
        'left': summarize(left), 'right': summarize(right),
        'right_minus_left_accuracy': statistics.mean(values),
        'paired_question_bootstrap_95pct': [boot[int(.025 * resamples)], boot[min(int(.975 * resamples), resamples - 1)]],
        'interpretation': 'Question-clustered interval for this run pair, not uncertainty across training seeds.',
        'common_plans_checked': common_plans,
    }


def plan_candidates(paths):
    candidates = {}
    for source, path in paths:
        for row in load_records(path):
            key = row['question_id'] + ':' + row['plan_id']
            if key in candidates:
                candidates[key]['sources'].add(source)
            else:
                candidates[key] = {'row': row, 'sources': {source}}
    return candidates


def plan_review(paths, output):
    candidates = plan_candidates(paths)
    with Path(output).open('x', newline='') as stream:
        fields = ['eval_item_id', 'question', 'ground_truth', 'plan',
                  'supplies_computed_answers', 'executable', 'reviewer', 'notes']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, item in sorted(candidates.items()):
            row = item['row']
            writer.writerow(dict(eval_item_id=key, question=row['question'], ground_truth=row['ground_truth'], plan=row['decomposer_output']))


def make_plan_bank(paths, labels_path, output):
    import pandas as pd
    candidates = plan_candidates(paths)
    with Path(labels_path).open(newline='') as stream:
        labels = list(csv.DictReader(stream))
    indexed = {row['eval_item_id']: row for row in labels}
    if len(indexed) != len(labels) or set(indexed) != set(candidates):
        raise ValueError('Plan review must cover every candidate exactly once, including exclusions')
    rows = []
    rejected = Counter()
    for key, item in sorted(candidates.items()):
        label, row = indexed[key], item['row']
        if label['plan'] != row['decomposer_output'] or question_id(label['question']) != row['question_id']:
            raise ValueError('Reviewed text differs from the saved plan/question')
        for field in ('supplies_computed_answers', 'executable'):
            if label[field] not in ('yes', 'no', 'unknown') or not label['reviewer'].strip():
                raise ValueError('Complete every review with yes/no/unknown and a reviewer ID')
        if label['supplies_computed_answers'] != 'no' or label['executable'] != 'yes':
            rejected[label['supplies_computed_answers'] + '/' + label['executable']] += 1
            continue
        rows.append({
            'question': row['question'], 'subset': row['subset'], 'data_source': row['data_source'],
            'reward_model': {'style': 'rule', 'ground_truth': row['ground_truth']},
            'fixed_decomposer_output': row['decomposer_output'], 'eval_item_id': key,
            'plan_source': ','.join(sorted(item['sources'])), 'answer_free_reviewed': True,
        })
    if not rows:
        raise ValueError('No reviewed answer-free executable plans; do not substitute an unreviewed bank')
    output = Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(output)
    pd.DataFrame(rows).to_parquet(output, index=False)
    output.with_suffix('.json').write_text(json.dumps({
        'rows': len(rows), 'questions': len({question_id(row['question']) for row in rows}),
        'candidate_plans': len(candidates), 'excluded': dict(rejected),
        'review_sha256': sha_file(labels_path), 'bank_sha256': sha_file(output),
        'selection': 'Human review of plan only, without worker outcomes; selected execution suite, not full benchmark.',
        'sources': {source: sha_file(path) for source, path in paths},
    }, indent=2) + '\n')


AUDIT_FIELDS = ('plan_supplies_computed_answers', 'workers_execute_assigned_tasks',
                'relevant_previous_results_used', 'local_results_correct')


def audit_export(left, right, directory, count=100, seed=42):
    compare(left, right, resamples=10)  # Validate pairing before sampling.
    a = {row.get('eval_item_id', row['question_id']): row for row in left}
    b = {row.get('eval_item_id', row['question_id']): row for row in right}
    rng = random.Random(seed)
    # Sample questions, not successes or eligible trajectories. Include both arms.
    qs = sorted({row['question_id'] for row in left})
    chosen = set(rng.sample(qs, min(count, len(qs))))
    entries = [(arm, row) for arm, rows in [('left', a), ('right', b)]
               for row in rows.values() if row['question_id'] in chosen]
    rng.shuffle(entries)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    key, public = {}, []
    with (directory / 'labels.csv').open('w', newline='') as stream:
        fields = ['audit_id', *AUDIT_FIELDS, 'reviewer', 'notes']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, (arm, row) in enumerate(entries):
            audit_id = f'example-{i:05d}'
            public.append({
                'audit_id': audit_id, 'question': row['question'], 'ground_truth': row['ground_truth'],
                'terminal_role': row['terminal_role'],
                'history': [{k: item[k] for k in ('role', 'content', 'assigned_subtasks') if k in item}
                            for item in row['history']],
            })
            key[audit_id] = {'arm': arm, 'question_id': row['question_id'], 'raw_score': row['raw_score']}
            writer.writerow({'audit_id': audit_id})
    (directory / 'examples.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in public))
    (directory / 'PRIVATE-key.json').write_text(json.dumps(key, indent=2) + '\n')


def audit_summary(labels_path, key_path):
    key = json.loads(Path(key_path).read_text())
    with Path(labels_path).open(newline='') as stream:
        labels = list(csv.DictReader(stream))
    if len({r['audit_id'] for r in labels}) != len(labels) or {r['audit_id'] for r in labels} != set(key):
        raise ValueError('Audit labels must cover every blinded example exactly once')
    counters = defaultdict(Counter)
    for row in labels:
        if not row['reviewer'].strip():
            raise ValueError('Reviewer ID required; unreviewed examples are not negatives')
        for field in AUDIT_FIELDS:
            if row[field] not in ('yes', 'no', 'unknown', 'not_applicable'):
                raise ValueError(f'Invalid/missing audit label for {field}')
            counters[key[row['audit_id']]['arm'] + '/' + field][row[field]] += 1
    return {name: dict(counts) for name, counts in sorted(counters.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    paired = sub.add_parser('compare')
    audit = sub.add_parser('audit-export')
    for cmd in (paired, audit):
        cmd.add_argument('--left', required=True)
        cmd.add_argument('--right', required=True)
        cmd.add_argument('--seed', type=int, default=42)
    paired.add_argument('--common-plans', action='store_true')
    audit.add_argument('--output-dir', required=True)
    audit.add_argument('--questions', type=int, default=100)
    review = sub.add_parser('plan-review')
    bank = sub.add_parser('plan-bank')
    for cmd in (review, bank):
        cmd.add_argument('--source', nargs=2, action='append', metavar=('NAME', 'RECORDS'), required=True)
        cmd.add_argument('--output', required=True)
    bank.add_argument('--labels', required=True)
    summary = sub.add_parser('audit-summary')
    summary.add_argument('--labels', required=True)
    summary.add_argument('--key', required=True)
    args = parser.parse_args()
    if args.command == 'compare':
        print(json.dumps(compare(load_records(args.left), load_records(args.right), common_plans=args.common_plans, seed=args.seed), indent=2))
    elif args.command == 'audit-export':
        audit_export(load_records(args.left), load_records(args.right), args.output_dir, args.questions, args.seed)
    elif args.command == 'plan-review':
        plan_review(args.source, args.output)
    elif args.command == 'plan-bank':
        make_plan_bank(args.source, args.labels, args.output)
    else:
        print(json.dumps(audit_summary(args.labels, args.key), indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Freeze full plan caches, then report paired before/after worker evaluation."""

import argparse
import copy
import csv
import json
from pathlib import Path
import shlex
import statistics

import eac3_experiments as suite
from analyze_eac3_experiments import compare, load_records
from local_results import extract_local_result

LABELS = ('u_before', 'u_after', 'g_before', 'g_after')


def checked_evaluation(path, expected):
    path = Path(path)
    manifest = suite.read_json(path.parent / 'run.json')
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{path}: run.json does not match the requested checkpoints/design')
    rows = load_records(path)
    config = suite.read_json(path.with_suffix('.metadata.json'))['config']
    sampling = config['actor_rollout_ref']['rollout']['val_kwargs']
    if (not config['trainer']['val_only'] or config['data'].get('teacher_assisted_validation')
            or sampling.get('n') != 1 or sampling.get('do_sample') is not False):
        raise ValueError('Expected question-only, greedy, n=1, inference-only evaluation')
    if not rows or any(row.get('actor_update_step') != 0 or row.get('context') != 'val' for row in rows):
        raise ValueError('Use a complete initial evaluation, not a training checkpoint or teacher-assisted run')
    return rows, manifest


def build_caches(cfg, sources, directory):
    import pandas as pd

    expected_runs = {run['plan_source']: run for run in suite.make_runs(cfg, 'cache-plans')}
    if set(sources) != set(expected_runs):
        raise ValueError('Provide exactly one complete evaluation for each planner source: du and dg')
    _, held_out = suite.data_frames(suite.ROOT / cfg['train_file'], suite.ROOT / cfg['eval_file'], deduplicate_train=True)
    questions = {suite.question_id(row.question): row for row in held_out.itertuples()}
    prepared, fingerprints = {}, []
    for source, path in sources.items():
        rows, manifest = checked_evaluation(path, expected_runs[source])
        fingerprints.append(manifest.get('code_sha256'))
        indexed = {row['question_id']: row for row in rows}
        if len(indexed) != len(rows) or set(indexed) != set(questions):
            raise ValueError(f'{source}: missing/extra questions; do not filter failed, leaked or empty plans')
        cached = []
        for qid, original in questions.items():
            row = indexed[qid]
            if (row['ground_truth'] != str(original.reward_model['ground_truth'])
                    or row['data_source'] != str(original.data_source)
                    or row['subset'] != str(getattr(original, 'subset', original.data_source))):
                raise ValueError(f'{source}: benchmark labels changed')
            plan = row['decomposer_output']
            if not isinstance(plan, str):
                raise ValueError('A plan must be a string; an empty generated string is retained')
            cached.append({
                'question': row['question'], 'data_source': row['data_source'], 'subset': row['subset'],
                'reward_model': original.reward_model,
                'extra_info': {'index': len(cached), **(getattr(original, 'extra_info', {}) or {})},
                'fixed_decomposer_output': plan, 'eval_item_id': qid + ':' + suite.plan_id(plan),
                'plan_source': source, 'answer_free_reviewed': False, 'plan_cache_unfiltered': True,
            })
        prepared[source] = cached
    if not fingerprints[0] or any(value != fingerprints[0] for value in fingerprints):
        raise ValueError('Cache sources must use the same recorded code/prompt fingerprints')

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    replay_cfg = copy.deepcopy(cfg)
    replay_cfg['plan_caches'] = {}
    commands = ['#!/bin/bash', 'set -euo pipefail', 'cd "$(dirname "$0")"']
    for source, rows in prepared.items():
        target = directory / f'{source}.parquet'
        pd.DataFrame(rows).to_parquet(target, index=False)
        remote = f"{cfg['remote_root'].rstrip('/')}/{cfg['study']}/plan-cache/{source}.parquet"
        digest = suite.sha_file(target)
        replay_cfg['plan_caches'][source] = {'remote': remote, 'sha256': digest, 'code_sha256': fingerprints[0],
                                            'rows': len(rows)}
        target.with_suffix('.json').write_text(json.dumps({
            'rows': len(rows), 'sha256': digest,
            'empty_plans': sum(not row['fixed_decomposer_output'].strip() for row in rows),
            'planner': cfg['planners'][source], 'source_records_sha256': suite.sha_file(sources[source]),
            'eval_sha256': suite.sha_file(suite.ROOT / cfg['eval_file']),
            'code_sha256': fingerprints[0],
            'selection': 'All benchmark questions, one plan each; no selection using outcomes or probes.',
        }, indent=2) + '\n')
        commands += [shlex.join(['rclone', 'copyto', target.name, remote, '--s3-no-check-bucket']),
                     shlex.join(['rclone', 'copyto', target.with_suffix('.json').name,
                                 remote[:-len('.parquet')] + '.json', '--s3-no-check-bucket'])]
    (directory / 'upload.sh').write_text('\n'.join(commands) + '\n')
    (directory / 'replay-config.json').write_text(json.dumps(replay_cfg, indent=2) + '\n')
    print(f'Wrote unfiltered du/dg caches and replay-config.json to {directory}; nothing uploaded.')
    return replay_cfg


def result_file(root, run):
    folder = Path(root) / run['run_name']
    paths = list(folder.glob('val-step*-update0.jsonl'))
    if len(paths) != 1:
        raise ValueError(f'{folder}: expected exactly one complete validation JSONL, found {len(paths)}')
    return paths[0]


def validate_pairs(groups):
    question_sets = []
    for source, workers in groups.items():
        reference = {row['eval_item_id']: row for row in workers['u_before']}
        question_sets.append({row['question_id'] for row in reference.values()})
        for label, rows in workers.items():
            indexed = {row.get('eval_item_id'): row for row in rows}
            if len(indexed) != len(rows) or set(indexed) != set(reference):
                raise ValueError(f'{source}/{label}: incomplete or unmatched plan/question pairs')
            for key, row in indexed.items():
                fields = ('question_id', 'plan_id', 'decomposer_output', 'ground_truth', 'subset', 'data_source')
                if any(row[field] != reference[key][field] for field in fields) or row.get('plan_source') != source:
                    raise ValueError(f'{source}/{label}: workers did not receive the identical cached plans')
    if any(values != question_sets[0] for values in question_sets):
        raise ValueError('Both planner sources must cover the same full question set')


def reviewed_ids(groups, labels_path):
    candidates = {row['eval_item_id']: row for workers in groups.values() for row in workers['u_before']}
    with Path(labels_path).open(newline='') as stream:
        labels = list(csv.DictReader(stream))
    if len(labels) != len(candidates) or {row['eval_item_id'] for row in labels} != set(candidates):
        raise ValueError('Plan-only review must cover every unique cached plan, including unknowns/exclusions')
    accepted = set()
    for label in labels:
        row = candidates[label['eval_item_id']]
        if (label['plan'] != row['decomposer_output'] or suite.question_id(label['question']) != row['question_id']
                or label['ground_truth'] != row['ground_truth']):
            raise ValueError('Review does not match the saved plan/question/target')
        if (not label['reviewer'].strip()
                or any(label[field] not in ('yes', 'no', 'unknown') for field in ('executable', 'supplies_computed_answers'))):
            raise ValueError('Every plan needs yes/no/unknown labels and a reviewer ID')
        if label['executable'] == 'yes' and label['supplies_computed_answers'] == 'no':
            accepted.add(row['eval_item_id'])
    return accepted


def worker_stats(rows):
    workers = [item for row in rows for item in row['history']
               if item['role'].startswith('worker_stage_') and item.get('executed', True)]
    return {
        'accuracy': statistics.mean(row['raw_score'] for row in rows),
        'worker_tokens_per_item': sum(int(item.get('num_gen_tokens', 0)) for item in workers) / len(rows),
        'executed_workers_per_item': len(workers) / len(rows),
        'local_result_parse_rate': (sum(bool(extract_local_result(item['content'])) for item in workers) / len(workers)
                                    if workers else None),
        'worker_length_stop_rate': (sum(item.get('stop_reason') == 'length' for item in workers) / len(workers)
                                    if workers else None),
    }


def report(cfg, root, output, *, labels=None, resamples=1000):
    if resamples < 40:
        raise ValueError('Use at least 40 bootstrap resamples')
    groups, record_checksums, diagnostic_metrics = {}, {}, {}
    for run in suite.make_runs(cfg, 'replay-workers'):
        path = result_file(root, run)
        rows, manifest = checked_evaluation(path, run)
        inputs = suite.read_json(path.parent / 'input-data.json')
        if inputs['eval_sha256'] != run['evaluation_sha256']:
            raise ValueError('Runtime cache checksum differs from the pinned plan bank')
        cache = cfg['plan_caches'][run['plan_source']]
        if len(rows) != cache['rows'] or manifest.get('code_sha256') != cache['code_sha256']:
            raise ValueError('Replay changed the cached question count or code/prompt fingerprint')
        groups.setdefault(run['plan_source'], {})[run['worker_label']] = rows
        record_checksums[run['run_name']] = suite.sha_file(path)
        diagnostic_metrics[run['run_name']] = {
            key: value for key, value in suite.read_json(path.with_suffix('.metadata.json')).get('metrics', {}).items()
            if key.startswith(('val_diagnostics/leakage/', 'val/leakage/'))
        }
    validate_pairs(groups)
    selected = reviewed_ids(groups, labels) if labels else None
    scopes = [('full', None)] + ([('reviewed', selected)] if selected is not None else [])
    summaries, details = [], []
    for scope, allowed in scopes:
        for source, all_workers in groups.items():
            workers = {label: [row for row in rows if allowed is None or row['eval_item_id'] in allowed]
                       for label, rows in all_workers.items()}
            datasets = sorted({row['subset'] for row in workers['u_before']})
            for dataset in [None, *datasets]:
                by_label = {label: [row for row in rows if dataset is None or row['subset'] == dataset]
                            for label, rows in workers.items()}
                line = dict(scope=scope, plan_source=source, dataset=dataset or '__all__',
                            questions=len(by_label['u_before']))
                if not line['questions']:
                    summaries.append(line)
                    continue
                for label, rows in by_label.items():
                    line.update({f'{label}_{key}': value for key, value in worker_stats(rows).items()})
                for branch in ('u', 'g'):
                    paired = compare(by_label[f'{branch}_before'], by_label[f'{branch}_after'], resamples=resamples)
                    delta = paired['right_minus_left_accuracy']
                    line[f'{branch}_delta_accuracy'] = delta
                    line[f'{branch}_delta_ci_low'], line[f'{branch}_delta_ci_high'] = paired['paired_question_bootstrap_95pct']
                    details.append(dict(scope=scope, plan_source=source, dataset=dataset or '__all__', branch=branch, paired=paired))
                summaries.append(line)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fields = list(dict.fromkeys(key for line in summaries for key in line))
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    (output / 'report.json').write_text(json.dumps({
        'summaries': summaries, 'paired_comparisons': details, 'records_sha256': record_checksums,
        'diagnostic_metrics': diagnostic_metrics,
        'review_sha256': suite.sha_file(labels) if labels else None,
        'notes': [
            'Full metrics retain all plans, including empty, single-subtask and answer-supplying plans.',
            'Accuracy is final-answer correctness, not correctness of intermediate reasoning.',
            'Parse rate measures communication format, not correctness or task compliance; use blinded human audits.',
            'Leakage diagnostics use the same frozen du probe on up to 128 items (console plus stratified selection); no filtering.',
            'Compare within-branch before/after: branches have different training histories.',
            'Bootstrap intervals reflect paired evaluation questions, not variation across training seeds.',
        ],
    }, indent=2) + '\n')
    print(f'Wrote {output}/summary.csv and report.json; no examples removed from full metrics.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cache = sub.add_parser('cache')
    cache.add_argument('--source', nargs=2, action='append', required=True, metavar=('NAME', 'JSONL'))
    summary = sub.add_parser('report')
    summary.add_argument('--results-root', required=True)
    summary.add_argument('--labels', help='Optional independent plan-only review; adds a subset without new inference')
    summary.add_argument('--resamples', type=int, default=1000)
    for command in (cache, summary):
        command.add_argument('--config', required=True)
        command.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    cfg = suite.read_json(args.config)
    if args.command == 'cache':
        sources = dict(args.source)
        if len(sources) != len(args.source):
            parser.error('Repeated plan source')
        build_caches(cfg, sources, args.output_dir)
    else:
        report(cfg, args.results_root, args.output_dir, labels=args.labels, resamples=args.resamples)


if __name__ == '__main__':
    main()

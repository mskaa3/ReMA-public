#!/usr/bin/env python3
"""Sample, review and freeze a worker study; no training or model invocation."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import random

import eac3_fixed_prefix_eval as evaluation
from local_results import extract_complete_boxed_answer, extract_local_result

ROOT = evaluation.ROOT
CHECKS = (
    'instructions_do_not_supply_assigned_answers',
    'each_task_sufficient_without_question',
    'units_domains_indexing_and_exactness_explicit',
    'prefix_results_and_local_references_correct',
    'terminal_reference_matches_original_question',
    'plan_is_coherent',
)
IDENTITY_FIELDS = ('question_id', 'question', 'final_answer', 'data_source')


def question_id(question):
    return hashlib.sha256(' '.join(question.split()).encode()).hexdigest()


def stratum(row):
    # Existing frozen samples used data_source; preserve their interpretation.
    return row.get('stratum', row['data_source'])


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve(path):
    return (ROOT / path).resolve()


def read_rows(path):
    path = Path(path)
    if path.suffix == '.parquet':
        try:
            import pandas as pd
            return pd.read_parquet(path).to_dict('records')
        except ImportError as exc:
            raise ValueError('Parquet sampling needs pandas and pyarrow/fastparquet. '
                             'Use the training container; no GPU is needed.') from exc
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    raise ValueError('Candidate/training files must be parquet or JSONL')


def proportional_quotas(groups, total):
    """Largest-remainder allocation with deterministic ties, proportional to questions."""
    size = sum(len(rows) for rows in groups.values())
    if not 0 <= total <= size:
        raise ValueError(f'Requested {total} questions, but only {size} available')
    if not total:
        return {key: 0 for key in groups}
    numerators = {key: total * len(rows) for key, rows in groups.items()}
    quotas = {key: value // size for key, value in numerators.items()}
    order = sorted(groups, key=lambda key: (-(numerators[key] % size), key))
    for key in order[:total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def validate_config(config):
    if (config.get('schema_version') != 1
            or config.get('evidence_status') != 'historical_validation_exploratory'):
        raise ValueError('This workflow is for historical-validation checkpoint comparisons, not causal proof')
    for field in ('development_questions', 'reserved_questions', 'bootstrap_samples'):
        if type(config.get(field)) is not int or config[field] < 1:
            raise ValueError(f'{field} must be a positive integer')
    for field in ('seed', 'bootstrap_seed'):
        if type(config.get(field)) is not int:
            raise ValueError(f'{field} must be an integer')
    for field in ('candidate_files', 'known_training_files', 'exclude_datasets'):
        if not isinstance(config.get(field), list) or not config[field]:
            raise ValueError(f'Explicitly configure nonempty {field}')
    if config.get('missing_reference_policy', 'error') not in {'error', 'exclude'}:
        raise ValueError('missing_reference_policy must be error or exclude')
    if config.get('stratify_by', 'data_source') not in {'data_source', 'subset'}:
        raise ValueError('stratify_by must be data_source or subset')
    models = evaluation.read_json(resolve(config['model_config']))['models']
    if [model['id'] for model in models] != ['w0', 'ungated', 'gated']:
        raise ValueError('Study expects w0, ungated, gated in that order')
    if (config.get('primary_comparison') != ['ungated', 'gated'] or
            config.get('primary_endpoint') != 'accepted_local_correct_compute_nonterminal_full_question'):
        raise ValueError('Unexpected primary endpoint/comparison; design a new protocol explicitly')
    return models


def sample(config_path, output):
    config = evaluation.read_json(config_path)
    models = validate_config(config)
    sources = {p: file_hash(resolve(p)) for field in
               ('candidate_files', 'known_training_files', 'exclude_datasets') for p in config[field]}
    training = {question_id(row['question']) for p in config['known_training_files']
                for row in read_rows(resolve(p))}
    pilot = {question_id(row['question']) for p in config['exclude_datasets']
             for row in evaluation.load_dataset(resolve(p))['examples']}
    candidates, counts, input_exclusions = {}, Counter(), []
    stratify_by = config.get('stratify_by', 'data_source')
    missing_policy = config.get('missing_reference_policy', 'error')
    for path in config['candidate_files']:
        for row_index, row in enumerate(read_rows(resolve(path))):
            counts['candidate_rows'] += 1
            question = row.get('question')
            reward = row.get('reward_model')
            answer = reward.get('ground_truth') if isinstance(reward, dict) else row.get('ground_truth')
            source = row.get('data_source')
            group = row.get(stratify_by)
            location = f'{path}: row {row_index} (zero-based)'
            for field, value in [('question', question), ('data_source', source), (stratify_by, group)]:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f'{location}: {field} must be a nonempty string; got {value!r}')
            qid = question_id(question)
            missing = (answer is None or isinstance(answer, str) and not answer.strip()
                       or isinstance(answer, Real) and not math.isfinite(answer))
            if missing:
                if missing_policy == 'error':
                    raise ValueError(f'{location}: missing/empty/nonfinite ground truth for {qid}; '
                                     'set missing_reference_policy=exclude to record its exclusion before sampling')
                input_exclusions.append(dict(path=path, row_index=row_index, question_id=qid,
                    question=question, data_source=source, stratum=group, reason='missing_reference'))
                counts['missing_reference_rows'] += 1
                continue
            if isinstance(answer, bool) or not isinstance(answer, (str, Real)):
                raise ValueError(f'{location}: unsupported ground truth type {type(answer).__name__}; '
                                 'expected text or a finite number; refusing to stringify a structured answer')
            record = dict(question_id=qid, question=question, final_answer=str(answer),
                          data_source=source, stratum=group)
            if qid in candidates:
                if any(candidates[qid][key] != record[key] for key in ('final_answer', 'data_source', 'stratum')):
                    raise ValueError('Duplicate question has conflicting reference or stratum')
                counts['duplicate_rows'] += 1
            else:
                candidates[qid] = record
    groups = defaultdict(list)
    for qid, row in sorted(candidates.items()):
        if qid in training:
            counts['known_training_overlap_questions'] += 1
        elif qid in pilot:
            counts['pilot_overlap_questions'] += 1
        else:
            groups[stratum(row)].append(row)
    rng = random.Random(config['seed'])
    for source in sorted(groups):
        rng.shuffle(groups[source])
    pool_counts = {key: len(rows) for key, rows in sorted(groups.items())}
    splits = {}
    for split, field in [('development', 'development_questions'), ('reserved', 'reserved_questions')]:
        quotas = proportional_quotas(groups, config[field])
        splits[split] = [row for source in sorted(groups) for row in groups[source][:quotas[source]]]
        groups = {source: rows[quotas[source]:] for source, rows in groups.items()}
    selection = dict(schema_version=1, config=config, config_sha256=evaluation.digest(config),
                     source_file_sha256=sources, models=models, counts=dict(counts),
                     input_exclusions=input_exclusions,
                     sampler_source_sha256=file_hash(__file__),
                     eligible_question_counts=pool_counts, splits=splits)
    selection['selection_sha256'] = evaluation.digest(selection)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    evaluation.write_json(out / 'selection.json', selection)
    evaluation.write_json(out / 'input-exclusions.json', input_exclusions)
    for split, rows in splits.items():
        template = dict(selection_sha256=selection['selection_sha256'], split=split, examples=[
            {**row, 'decision': 'pending', 'exclusion_reason': '', 'author': '', 'checker': '',
             'author_type': '', 'checker_type': '', 'checks': {key: False for key in CHECKS},
             'tasks': []} for row in rows])
        name = 'development-review.json' if split == 'development' else 'PRIVATE-reserved-review.json'
        evaluation.write_json(out / name, template)
    print(f'Sampled {len(splits["development"])} development and {len(splits["reserved"])} reserved questions.')
    print(f'Excluded {len(input_exclusions)} rows with missing references before sampling; see input-exclusions.json.')
    print(f'Eligible questions stratified by {stratify_by}: {json.dumps(pool_counts, sort_keys=True)}')
    print('No references or plans were authored or certified. Review templates must be completed before freeze.')


def validate_review(selection, review):
    original = {k: v for k, v in selection.items() if k != 'selection_sha256'}
    if (selection.get('selection_sha256') != evaluation.digest(original)
            or selection['config_sha256'] != evaluation.digest(selection['config'])):
        raise ValueError('Selection fingerprint mismatch')
    if review.get('selection_sha256') != selection['selection_sha256']:
        raise ValueError('Review does not belong to this selection')
    split = review['split']
    if split not in {'development', 'reserved'}:
        raise ValueError('Unknown split')
    expected = {r['question_id']: r for r in selection['splits'][split]}
    rows = review['examples']
    if len(rows) != len(expected) or {r['question_id'] for r in rows} != set(expected):
        raise ValueError('Review every sampled question; exclusions need explicit reasons, not deletions')
    for row in rows:
        if any(row[field] != expected[row['question_id']][field] for field in IDENTITY_FIELDS):
            raise ValueError('Sampled question, reference or stratum was changed')
        if stratum(row) != stratum(expected[row['question_id']]):
            raise ValueError('Sampled question stratum was changed')
        if (not row['author'].strip() or not row['checker'].strip()
                or row['author'].strip() == row['checker'].strip()
                or row['author_type'] not in {'human', 'model'}
                or row['checker_type'] not in {'human', 'model'}):
            raise ValueError('Record distinct author/checker IDs and honest human/model provenance')
        if split == 'reserved' and row['checker_type'] != 'human':
            raise ValueError('Reserved questions require an independent human check, including exclusions')
        if row['decision'] == 'exclude':
            if not row.get('exclusion_reason', '').strip():
                raise ValueError('Exclusion reason required; never exclude by model performance')
            continue
        if row['decision'] != 'include' or any(row['checks'].get(key) is not True for key in CHECKS):
            raise ValueError('Pending questions/checks cannot be evaluated')
        if not 1 <= len(row['tasks']) <= 4:
            raise ValueError('Use 1-4 coherent tasks; single-task plans are allowed and reported as terminal-only')
        for task in row['tasks']:
            accepted = extract_local_result(task['checked_result'])
            if not accepted or extract_complete_boxed_answer(accepted) not in task['reference_answers']:
                raise ValueError('Checked prefix must be routable and its box must match a declared local reference')
    return rows


def freeze(selection_path, review_path, output):
    selection, review = evaluation.read_json(selection_path), evaluation.read_json(review_path)
    rows = validate_review(selection, review)
    config = selection['config']
    examples = []
    for row in rows:
        if row['decision'] == 'exclude':
            continue
        tasks = row['tasks']
        for index, task in enumerate(tasks):
            terminal = index == len(tasks) - 1
            for mode in (['subtask_context'] if terminal else ['full_question', 'subtask_context']):
                case = f"{row['question_id']}:s{index + 1}"
                examples.append(dict(id=f'{case}:{mode}', case_id=case, question_id=row['question_id'],
                    question=row['question'], family=stratum(row), kind=task['kind'],
                    subtasks=[t['instruction'] for t in tasks], target=index + 1,
                    previous_results=[t['checked_result'] for t in tasks[:index]],
                    reference_answers=task['reference_answers'], final_answer=row['final_answer'],
                    reference_check=task['reference_check'], question_mode=mode))
    if not examples:
        raise ValueError('All questions excluded; no study can run')
    included = [row for row in rows if row['decision'] == 'include']
    study = dict(study_id=config['study_id'], evidence_status=config['evidence_status'], split=review['split'],
        selection_sha256=selection['selection_sha256'], review_sha256=evaluation.digest(review),
        sampled_questions=len(rows), included_questions=len(included),
        single_task_questions=sum(len(r['tasks']) == 1 for r in included),
        sampled_by_source=dict(Counter(r['data_source'] for r in rows)),
        included_by_source=dict(Counter(r['data_source'] for r in included)),
        stratify_by=config.get('stratify_by', 'data_source'),
        sampled_by_stratum=dict(Counter(stratum(r) for r in rows)),
        included_by_stratum=dict(Counter(stratum(r) for r in included)),
        input_exclusions=selection.get('input_exclusions', []),
        exclusions=[dict(question_id=r['question_id'], reason=r['exclusion_reason']) for r in rows
                    if r['decision'] == 'exclude'],
        bootstrap_samples=config['bootstrap_samples'], bootstrap_seed=config['bootstrap_seed'],
        primary_endpoint=config['primary_endpoint'], primary_comparison=config['primary_comparison'],
        model_ids=[m['id'] for m in selection['models']], models=selection['models'])
    data = dict(schema_version=1, provenance='Random questions from historical validation, with independently '
                'reviewed task instructions and correct fixed prefixes. See frozen selection and review.',
                study=study, examples=examples)
    out = Path(output).resolve()
    try:
        dataset_path = (out / 'dataset.json').relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError('Freeze inside the repository so the existing launcher can mount the dataset') from exc
    # Validate the complete structure before publishing any launchable configuration.
    evaluation.validate_dataset(data)
    from eac3_worker_study_stats import validate_pairs
    validate_pairs(examples)
    out.mkdir(parents=True, exist_ok=False)
    evaluation.write_json(out / 'selection.json', selection)
    evaluation.write_json(out / 'review.json', review)
    evaluation.write_json(out / 'dataset.json', data)
    evaluation.write_json(out / 'eval-config.json', dict(dataset=str(dataset_path), models=selection['models']))
    print(f'Frozen {len(included)} questions, {len(examples)} prompts (terminals occur once, without Q).')
    print(f'sbatch scripts/eac3_fixed_prefix_eval.sh {out.relative_to(ROOT.resolve())}/eval-config.json')


def audit(review_path, output, count, seed):
    """Blind by question, including every model/context, without filtering outcomes."""
    rows = [json.loads(line) for line in Path(review_path).read_text().splitlines() if line.strip()]
    if not rows or count < 1:
        raise ValueError('Need a nonempty study review and positive question count')
    from eac3_worker_study_stats import validate_pairs
    validate_pairs(rows)
    models = set(rows[0]['outputs'])
    if len(models) < 2 or any(set(row['outputs']) != models for row in rows):
        raise ValueError('Every case needs all model outputs')
    questions = sorted({row['question_id'] for row in rows})
    if count > len(questions):
        raise ValueError(f'Only {len(questions)} questions available; choose an explicit smaller audit')
    rng = random.Random(seed)
    chosen = set(rng.sample(questions, count))
    entries = [(row, model) for row in rows if row['question_id'] in chosen for model in sorted(models)]
    rng.shuffle(entries)
    public, key, labels = [], [], []
    for i, (row, model) in enumerate(entries, 1):
        ident = f'audit-{i:05d}'
        public.append(dict(audit_id=ident, question=row['question'],
            assigned_task=row['subtasks'][row['target'] - 1], messages=row['messages'],
            local_reference_answers=row['reference_answers'], reference_check=row['reference_check'],
            terminal=row['target'] == len(row['subtasks']), response=row['outputs'][model]['response']))
        key.append(dict(audit_id=ident, question_id=row['question_id'], example_id=row['id'], model=model))
        labels.append(dict(audit_id=ident, reviewer='', assigned_task_followed=None,
            local_math_correct=None, reasoning_sound=None, whole_question_solving=None,
            previous_results_used=None, notes=''))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for name, values in [('examples.jsonl', public), ('labels.jsonl', labels)]:
        with (out / name).open('x') as stream:
            for value in values:
                stream.write(json.dumps(value) + '\n')
    evaluation.write_json(out / 'PRIVATE-key.json', dict(seed=seed, question_count=count,
        review_file_sha256=file_hash(review_path), mapping=key))
    print(f'Blinded {len(public)} responses from {count} random questions; keep PRIVATE-key.json away from annotators.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sampling = sub.add_parser('sample')
    sampling.add_argument('--config', default='config/eac3-worker-specialization-study.json')
    sampling.add_argument('--output', required=True)
    freezing = sub.add_parser('freeze')
    freezing.add_argument('--selection', required=True)
    freezing.add_argument('--review', required=True)
    freezing.add_argument('--output', required=True)
    auditing = sub.add_parser('audit')
    auditing.add_argument('--review', required=True)
    auditing.add_argument('--output', required=True)
    auditing.add_argument('--questions', type=int, default=50)
    auditing.add_argument('--seed', type=int, default=20261101)
    args = parser.parse_args()
    try:
        if args.command == 'sample':
            sample(args.config, args.output)
        elif args.command == 'freeze':
            freeze(args.selection, args.review, args.output)
        else:
            audit(args.review, args.output, args.questions, args.seed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()

"""Question-clustered analysis for the reviewed fixed-prefix worker study."""

from collections import Counter, defaultdict
import hashlib


def validate_pairs(examples):
    cases, questions = defaultdict(list), defaultdict(list)
    ids = set()
    for row in examples:
        if row['id'] in ids:
            raise ValueError('Duplicate example ID')
        ids.add(row['id'])
        if row['question_id'] != hashlib.sha256(' '.join(row['question'].split()).encode()).hexdigest():
            raise ValueError('Question ID does not match normalized question')
        if row['case_id'] != f"{row['question_id']}:s{row['target']}":
            raise ValueError('Case ID does not identify question and stage')
        cases[row['case_id']].append(row)
        questions[row['question_id']].append(row)
    fields = ('question', 'question_id', 'family', 'kind', 'target', 'subtasks',
              'previous_results', 'reference_answers', 'final_answer', 'reference_check')
    for case, rows in cases.items():
        first = rows[0]
        terminal = first['target'] == len(first['subtasks'])
        modes = ['subtask_context'] if terminal else ['full_question', 'subtask_context']
        if sorted(r['question_mode'] for r in rows) != modes:
            raise ValueError(f'{case}: require paired nonterminals, and only one no-Q terminal')
        if any(any(row[field] != first[field] for field in fields) for row in rows):
            raise ValueError(f'{case}: context ablation changed something other than Q visibility')
    for rows in questions.values():
        if ({r['target'] for r in rows} != set(range(1, len(rows[0]['subtasks']) + 1))
                or any(r['subtasks'] != rows[0]['subtasks'] or r['family'] != rows[0]['family'] for r in rows)):
            raise ValueError('Evaluate every task of one common plan per question')
    return cases


def estimate(values, examples, indices, samples, seed):
    """Equal-question mean, averaging targets within Q; resample Q within source."""
    import numpy as np

    grouped, strata = defaultdict(list), {}
    for value, index in zip(values, indices):
        row = examples[index]
        grouped[row['question_id']].append(float(value))
        strata[row['question_id']] = row['family']
    if len(values) != len(indices):
        raise ValueError('One outcome is required per selected prompt')
    if not grouped:
        return dict(mean=None, ci95=None, n_questions=0, n_prompts=0)
    means = {qid: sum(values) / len(values) for qid, values in grouped.items()}
    mean = sum(means.values()) / len(means)
    groups = defaultdict(list)
    for qid in sorted(means):
        groups[strata[qid]].append(means[qid])
    ci = None
    # Singleton strata cannot estimate within-stratum variation.
    if all(len(group) >= 2 for group in groups.values()):
        rng = np.random.default_rng(seed)
        boot = np.zeros(samples)
        for key in sorted(groups):
            group = np.asarray(groups[key])
            boot += group[rng.integers(0, len(group), size=(samples, len(group)))].sum(axis=1)
        ci = np.quantile(boot / len(means), [0.025, 0.975]).tolist()
    return dict(mean=mean, ci95=ci, n_questions=len(means), n_prompts=len(indices),
                questions_by_source={key: len(group) for key, group in sorted(groups.items())},
                ci_unavailable_reason=None if ci is not None else 'fewer_than_two_questions_in_a_stratum')


def analyze(data, models):
    examples, protocol = data['examples'], data['study']
    cases = validate_pairs(examples)
    if set(models) != set(protocol['model_ids']):
        raise ValueError('Need every checkpoint specified in the study')
    samples, seed = protocol['bootstrap_samples'], protocol['bootstrap_seed']
    if type(samples) is not int or samples < 1:
        raise ValueError('Positive bootstrap sample count required')
    names = protocol['model_ids']
    for rows in models.values():
        if [r['id'] for r in rows] != [r['id'] for r in examples]:
            raise ValueError('Outcomes must be aligned to exactly the prepared inputs')
    def success(model, index):
        return models[model][index]['status'] == 'correct'

    partitions = {}
    for mode in ('full_question', 'subtask_context'):
        partitions['compute_nonterminal_' + mode] = [i for i, r in enumerate(examples)
            if r['kind'] == 'compute' and r['target'] < len(r['subtasks']) and r['question_mode'] == mode]
    partitions['terminal_no_question'] = [i for i, r in enumerate(examples) if r['target'] == len(r['subtasks'])]
    for kind in ('forward', 'synthesize'):
        for mode in ('full_question', 'subtask_context'):
            partitions[f'{kind}_nonterminal_{mode}'] = [i for i, r in enumerate(examples)
                if r['kind'] == kind and r['target'] < len(r['subtasks']) and r['question_mode'] == mode]
    endpoints = {}
    for endpoint, indices in partitions.items():
        results, pairs = {}, {}
        for name in names:
            results[name] = dict(accepted_local_correct=estimate(
                [success(name, i) for i in indices], examples, indices, samples, seed),
                status_counts=dict(Counter(models[name][i]['status'] for i in indices)))
        for j, left in enumerate(names):
            for right in names[j + 1:]:
                pairs[f'{right} - {left}'] = estimate([int(success(right, i)) - int(success(left, i))
                    for i in indices], examples, indices, samples, seed)
        endpoints[endpoint] = dict(models=results, paired_differences=pairs)
    left, right = protocol['primary_comparison']
    index_by_id = {r['id']: i for i, r in enumerate(examples)}
    values, indices = [], []
    for rows in cases.values():
        if rows[0]['kind'] != 'compute' or rows[0]['target'] == len(rows[0]['subtasks']):
            continue
        by_mode = {r['question_mode']: index_by_id[r['id']] for r in rows}
        full, task = by_mode['full_question'], by_mode['subtask_context']
        difference = lambda i: int(success(right, i)) - int(success(left, i))
        values.append(difference(full) - difference(task))
        indices.append(full)
    return dict(protocol=protocol, endpoints=endpoints,
        context_interaction=estimate(values, examples, indices, samples, seed),
        interaction_definition='(gated_fullQ - ungated_fullQ) - (gated_taskOnly - ungated_taskOnly)',
        caution='Historical-validation, fixed-prefix checkpoint comparison, not a causal training experiment. '
                'Means weight questions equally within each endpoint. Bootstrap resamples questions within '
                'data sources; it does not measure training-seed uncertainty. Unknown/missing answers count '
                'as unconfirmed delivery, not incorrect mathematics. Role fidelity and reasoning require '
                'independent audit. Reserved here means reserved for this protocol, not previously unseen data.')


def markdown(result):
    lines = ['', '## Question-Clustered Endpoints', '',
             'Primary: gated minus ungated, compute/nonterminal/full_question.',
             'CIs are paired, stratified question-bootstrap intervals, conditional on these checkpoints.', '',
             '| Endpoint | Comparison | Questions | Difference (pp) | 95% CI (pp) |',
             '|---|---|---:|---:|---|']
    def value(v):
        return 'N/A' if v is None else f'{100 * v:.2f}'
    for name, endpoint in result['endpoints'].items():
        for pair, stat in endpoint['paired_differences'].items():
            ci = stat['ci95']
            interval = 'N/A' if ci is None else f'[{value(ci[0])}, {value(ci[1])}]'
            lines.append(f"| {name} | {pair} | {stat['n_questions']} | {value(stat['mean'])} | {interval} |")
    lines += ['', 'Context interaction (a positive interaction alone does not imply better performance):',
              result['interaction_definition'], '', '```json']
    import json
    lines += [json.dumps(result['context_interaction'], indent=2), '```']
    return lines

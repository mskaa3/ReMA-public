#!/usr/bin/env python3
"""Evaluate one worker at a time on identical, checked prefixes. Never train."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PPO = ROOT / 'src/verl/verl/rema_separated_trainer/ppo'
sys.path.insert(0, str(ROOT / 'src/verl'))
sys.path.insert(0, str(PPO))
from local_results import extract_local_result
from eac3_answer_diagnostic import VERSION as ANSWER_DIAGNOSTIC_VERSION, reported_answer
import eac3_math_answers as math_answers


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True)
        stream.write('\n')


def source_hashes():
    paths = [PPO / 'multi_agent_rollout.py', PPO / 'local_results.py',
             PPO / 'prefix_probe.py', PPO / 'math_verification.py',
             ROOT / 'prompt/math/hierarchical_mamrp.py', Path(__file__),
             ROOT / 'scripts/eac3_answer_diagnostic.py',
             ROOT / 'scripts/eac3_math_answers.py',
             ROOT / 'scripts/eac3_worker_study_stats.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def load_dataset(path):
    return validate_dataset(read_json(path))


def validate_dataset(data):
    if data.get('schema_version') != 1 or not data.get('provenance') or not data.get('examples'):
        raise ValueError('Expected schema_version=1, provenance and a nonempty examples list')
    ids = set()
    for row in data['examples']:
        ident = row['id']
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError('Example IDs must be nonempty and unique')
        ids.add(ident)
        for field in ('question', 'family', 'kind', 'reference_check'):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f'{ident}: missing {field}')
        tasks, target = row['subtasks'], row['target']
        if (not isinstance(tasks, list) or not 1 <= len(tasks) <= 4
                or any(not isinstance(t, str) or not t.strip() or '\n' in t for t in tasks)
                or type(target) is not int or not 1 <= target <= len(tasks)):
            raise ValueError(f'{ident}: require 1-4 single-line tasks and a 1-based target')
        if row['kind'] not in {'compute', 'forward', 'synthesize'}:
            raise ValueError(f'{ident}: unknown task kind')
        if row.get('question_mode', 'full_question') not in {'full_question', 'subtask_context'}:
            raise ValueError(f'{ident}: unsupported question_mode')
        if len(row['previous_results']) != target - 1:
            raise ValueError(f'{ident}: supply exactly one checked result for every preceding task')
        for result in row['previous_results']:
            if not extract_local_result(result):
                raise ValueError(f'{ident}: previous result would be rejected by the production router')
        answers = row['reference_answers']
        if (not isinstance(answers, list) or not answers
                or any(not isinstance(a, str) or not a.strip() or '\\boxed' in a for a in answers)):
            raise ValueError(f'{ident}: reference_answers must be nonempty unboxed LaTeX expressions')
        if not isinstance(row.get('final_answer'), str) or not row['final_answer'].strip():
            raise ValueError(f'{ident}: supply final_answer for the answer-match diagnostic')
    return data


class PromptCaptured(Exception):
    def __init__(self, messages):
        self.messages = messages


def capture_prompt(row, rollout_class):
    """Run the real router with canned upstream outputs; stop BEFORE target generation.

    No copy of the training prompt text is maintained here. The class is passed
    explicitly so CPU tests can load the production definitions without Ray.
    """
    plan = 'PLAN:\n' + '\n'.join(f'- S{i}: {task}' for i, task in enumerate(row['subtasks'], 1))
    stages = [f'worker_stage_{i}' for i in range(1, 5)]
    target = stages[row['target'] - 1]
    rollout = object.__new__(rollout_class)
    rollout.config = SimpleNamespace(stop_when_truncated=True)
    seen = []

    def generate(role, chats, *args, **kwargs):
        if len(chats) != 1:
            raise ValueError('Fixed-prefix capture must stay single-example')
        if role == target:
            if seen != stages[:row['target'] - 1]:
                raise ValueError('Production routing did not execute the expected prefix')
            raise PromptCaptured(chats[0])
        if role not in stages[:row['target'] - 1] or role in seen:
            raise ValueError(f'Unexpected generation during capture: {role}')
        seen.append(role)
        output = row['previous_results'][stages.index(role)]
        return [output], [1], ['stop'], None, [[1]]

    rollout._generate_from_chat_list = generate
    prompts = SimpleNamespace(non_tensor_batch={
        'question': [row['question']], 'uid': [row['id']], 'fixed_decomposer_output': [plan],
    }, meta_info={'validate': True, 'replay_plans': True})
    hierarchy = dict(stage_roles=stages, worker_roles=['general_math_worker'],
                     routing_mode='sequential_plan', max_planned_subtasks=4,
                     terminal_worker_as_answer=True,
                     worker_context_mode=row.get('question_mode', 'full_question'))
    system = runpy.run_path(str(ROOT / 'prompt/math/hierarchical_mamrp.py'))[
        'build_hierarchical_system_prompts'](stages)
    histories, flags, reasons = rollout._initialize_conversation_state(1)
    try:
        rollout._run_hierarchical_conversation(
            prompts, {}, 1, ['decomposer', 'selector', *stages], system, hierarchy,
            histories, flags, reasons, 1024, None)
    except PromptCaptured as captured:
        return captured.messages
    raise ValueError(f"Target {target} was not reached for {row['id']}")


def verifier_versions():
    versions = {'python': sys.version.split()[0]}
    for package in ('math-verify', 'latex2sympy2-extended', 'antlr4-python3-runtime', 'sympy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    return versions


def verify_reference_notation(data, verification_map):
    # Catch missing scorer dependencies or unsupported gold notation BEFORE GPU loading.
    versions = verifier_versions()
    print('Reference verifier versions: ' + json.dumps(versions, sort_keys=True), flush=True)
    if versions['math-verify'] == 'not installed':
        raise ValueError('Reference verification requires math-verify; versions: ' + json.dumps(versions))
    references = []
    for row in data['examples']:
        references.extend((row['id'], f'reference_answers[{i}]', ref)
                          for i, ref in enumerate(row['reference_answers']))
        references.append((row['id'], 'final_answer', row['final_answer']))
    checked, stats = verification_map(check_equivalence,
        [('LOCAL_RESULT: \\boxed{' + ref + '}', [ref]) for _, _, ref in references],
        timeout_seconds=10, max_workers=2)
    if len(checked) != len(references):
        raise ValueError('Reference verifier returned an incomplete batch')
    invalid = [dict(id=ident, field=field, reference=ref,
                    result='unknown' if result is None else 'false')
               for (ident, field, ref), result in zip(references, checked) if result is not True]
    if invalid:
        raise ValueError('Reference notation failed verifier self-check:\n' + json.dumps(
            dict(invalid=invalid, stats=stats, versions=versions), indent=2, sort_keys=True))
    return dict(stats=stats, versions=versions)


def prepare(dataset, output):
    from verl.rema_separated_trainer.ppo.multi_agent_rollout import MultiAgentRollout
    from verl.rema_separated_trainer.ppo.math_verification import bounded_verification_map

    data = load_dataset(dataset)
    if data.get('study'):
        from eac3_worker_study_stats import validate_pairs
        validate_pairs(data['examples'])
    verification = verify_reference_notation(data, bounded_verification_map)
    rows = []
    for row in data['examples']:
        messages = capture_prompt(row, MultiAgentRollout)
        rows.append({**row, 'messages': messages, 'prompt_sha256': digest(messages)})
    write_json(output, {**data, 'dataset_sha256': digest(data),
                        'source_hashes': source_hashes(), 'reference_verification': verification,
                        'examples': rows})
    print(f'Prepared {len(rows)} fixed target prompts; no models were loaded.')


def load_prepared(path):
    data = load_dataset(path)
    for row in data['examples']:
        if row.get('prompt_sha256') != digest(row['messages']):
            raise ValueError('Prepared prompt fingerprint mismatch')
    return data


def run_model(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from validate_hf_model_dir import validate_model_dir

    data = load_prepared(args.prepared)
    if data.get('study'):
        expected = {m['id']: m['source'] for m in data['study']['models']}
        if expected.get(args.model_id) != args.model_source:
            raise ValueError('Checkpoint source does not match the frozen study configuration')
    if data['source_hashes'] != source_hashes():
        raise ValueError('Code changed since prepare; use a fresh evaluation directory')
    validate_model_dir(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False, local_files_only=True)
    token_ids = [tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True)
                 for row in data['examples']]
    if any(len(ids) + args.max_new_tokens > args.max_model_len for ids in token_ids):
        raise ValueError('Prompt plus response exceeds context. No input truncation is permitted.')
    settings = dict(temperature=0.0, top_p=1.0, top_k=-1, n=1, seed=0,
                    max_tokens=args.max_new_tokens)
    versions = {package: importlib.metadata.version(package)
                for package in ('vllm', 'transformers', 'torch')}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'manifest.json', dict(
        model_id=args.model_id, model_source=args.model_source,
        prepared_sha256=digest(data), settings=settings, versions=versions,
        max_model_len=args.max_model_len, batch_size=args.batch_size,
        model_config=read_json(Path(args.model_dir) / 'config.json'),
        source_hashes=source_hashes()))
    engine = LLM(model=args.model_dir, tokenizer=args.model_dir, tensor_parallel_size=1,
                 distributed_executor_backend='uni', dtype='bfloat16', trust_remote_code=False,
                 enforce_eager=True, max_model_len=args.max_model_len,
                 max_num_seqs=args.batch_size, max_num_batched_tokens=args.max_model_len,
                 gpu_memory_utilization=0.8, generation_config='vllm',
                 enable_prefix_caching=False, seed=0)
    partial = out / 'responses.jsonl.partial'
    with partial.open('x') as stream:
        for start in range(0, len(token_ids), args.batch_size):
            requests = [{'prompt_token_ids': ids} for ids in token_ids[start:start + args.batch_size]]
            generated = engine.generate(requests, SamplingParams(**settings), use_tqdm=False)
            if len(generated) != len(requests):
                raise ValueError('Incomplete vLLM batch')
            for offset, result in enumerate(generated):
                index = start + offset
                row = data['examples'][index]
                if len(result.outputs) != 1 or result.outputs[0].finish_reason not in {'stop', 'length'}:
                    raise ValueError(f"Unexpected generation result: {row['id']}")
                response = result.outputs[0]
                record = dict(id=row['id'], model_id=args.model_id,
                              prompt_sha256=row['prompt_sha256'],
                              prompt_token_sha256=digest(token_ids[index]),
                              prompt_tokens=len(token_ids[index]), response=response.text,
                              generated_tokens=len(response.token_ids), finish_reason=response.finish_reason)
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                print(f"[{args.model_id}] {row['id']}: {record['generated_tokens']} tokens", flush=True)
    partial.rename(out / 'responses.jsonl')


def indexed_records(path, expected_ids):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    indexed = {row['id']: row for row in rows}
    if len(rows) != len(indexed) or set(indexed) != set(expected_ids):
        raise ValueError(f'{path}: missing, duplicate or extra examples; incomplete runs cannot be compared')
    return indexed


def check_equivalence(item):
    from local_results import extract_complete_boxed_answer

    response, references, *context = item
    accepted = extract_local_result(response)
    answer = extract_complete_boxed_answer(accepted) if accepted else ''
    return math_answers.equivalent((answer, references, context[0] if context else ''))


def score(prepared, directory):
    from verl.rema_separated_trainer.ppo.math_verification import bounded_verification_map

    data = load_prepared(prepared)
    out = Path(directory)
    manifest = read_json(out / 'manifest.json')
    if manifest['prepared_sha256'] != digest(data) or manifest['source_hashes'] != source_hashes():
        raise ValueError('Cannot score a different prepared dataset or source version')
    responses = indexed_records(out / 'responses.jsonl', [r['id'] for r in data['examples']])
    scored = score_records(data, responses, bounded_verification_map)
    write_json(out / 'scored.json', scored)
    print(json.dumps(summarize(scored['records']), indent=2))


def score_records(data, responses, verification_map):
    """Shared live/offline scorer; offline callers must validate provenance first."""
    jobs = []
    diagnostics = []
    form_jobs, form_indices = [], []
    for row in data['examples']:
        response = responses[row['id']]['response']
        diagnostic = reported_answer(response)
        diagnostics.append(diagnostic)
        task = row['subtasks'][row['target'] - 1]
        candidate = ('LOCAL_RESULT: \\boxed{' + diagnostic['answer'] + '}'
                     if diagnostic['answer'] else '')
        jobs.extend([(response, row['reference_answers'], task),
                     (response, [row['final_answer']], row['question']),
                     (candidate, row['reference_answers'], task)])
        if math_answers.form_rule(task):
            form_indices.append(len(diagnostics) - 1)
            form_jobs.append((diagnostic['answer'], task))
    values, stats = verification_map(check_equivalence, jobs, timeout_seconds=10, max_workers=2)
    forms, form_stats = verification_map(math_answers.requested_form, form_jobs,
                                        timeout_seconds=10, max_workers=2)
    if len(values) != len(jobs) or len(forms) != len(form_jobs):
        raise ValueError('Incomplete answer verification batch')
    form_values = dict(zip(form_indices, forms))
    records = []
    for i, row in enumerate(data['examples']):
        record = responses[row['id']]
        accepted = extract_local_result(record['response'])
        correct, final_match, answer_correct = values[3 * i:3 * i + 3]
        status = ('no_accepted_result' if not accepted else 'unknown' if correct is None
                  else 'correct' if correct else 'wrong')
        value_status = status
        form_status = ('not_checked' if i not in form_values else 'unknown' if form_values[i] is None
                       else 'satisfied' if form_values[i] else 'failed')
        if status == 'correct' and form_status in {'failed', 'unknown'}:
            status = 'wrong' if form_status == 'failed' else 'unknown'
        diagnostic = diagnostics[i]
        answer_status = ('no_unambiguous_answer' if not diagnostic['answer'] else
                         'unknown' if answer_correct is None else 'correct' if answer_correct else 'wrong')
        records.append({**record, 'accepted_result': accepted, 'local_correct': correct,
                        'status': status, 'value_status': value_status,
                        'task_form_status': form_status, 'full_question_answer_match': final_match,
                        'reported_answer': diagnostic['answer'],
                        'reported_answer_extraction': diagnostic['reason'],
                        'reported_answer_status': answer_status,
                        'terminal': row['target'] == len(row['subtasks']),
                        'family': row['family'], 'kind': row['kind'],
                        'manual_role_compliance': None, 'manual_reasoning_correct': None})
    return dict(records=records, verification=stats, form_verification=form_stats,
                scorer_version=math_answers.VERSION, scoring_source_hashes=source_hashes(),
                verifier_versions=verifier_versions(),
                answer_diagnostic_version=ANSWER_DIAGNOSTIC_VERSION,
                math_verify_version=importlib.metadata.version('math-verify'))


def summarize(records):
    n = len(records)
    counts = Counter(row['status'] for row in records)
    valid = counts['correct'] + counts['wrong']
    result = dict(n=n, status_counts=dict(counts), accepted_correct_fraction=counts['correct'] / n if n else None,
                correctness_among_decidable=counts['correct'] / valid if valid else None,
                accepted_result_fraction=sum(bool(r['accepted_result']) for r in records) / n if n else None,
                length_stop_count=sum(r['finish_reason'] == 'length' for r in records),
                mean_generated_tokens=sum(r['generated_tokens'] for r in records) / n if n else None)
    if records and all('reported_answer_status' in r for r in records):
        result['answer_diagnostic'] = dict(
            status_counts=dict(Counter(r['reported_answer_status'] for r in records)),
            correct_but_not_routable=sum(r['reported_answer_status'] == 'correct'
                                         and not r['accepted_result'] for r in records))
    if records and all('task_form_status' in r for r in records):
        result['task_form_counts'] = dict(Counter(r['task_form_status'] for r in records))
    return result


def compare_records(left, right):
    pairs = list(zip(left, right))
    decidable = [(a, b) for a, b in pairs if a['local_correct'] is not None and b['local_correct'] is not None]
    return dict(n_pairs=len(pairs), jointly_decidable=len(decidable),
                wrong_to_correct=sum(a['local_correct'] is False and b['local_correct'] is True for a, b in decidable),
                correct_to_wrong=sum(a['local_correct'] is True and b['local_correct'] is False for a, b in decidable),
                accepted_correct_gain=sum(b['status'] == 'correct' for a, b in pairs)
                - sum(a['status'] == 'correct' for a, b in pairs))


def report(prepared, directories, output):
    data = load_prepared(prepared)
    ids = [row['id'] for row in data['examples']]
    models, settings = {}, None
    for directory in directories:
        directory = Path(directory)
        manifest = read_json(directory / 'manifest.json')
        scored = read_json(directory / 'scored.json')
        common = {key: manifest[key] for key in ('prepared_sha256', 'settings', 'versions',
                                                'source_hashes', 'max_model_len', 'batch_size')}
        common['math_verify_version'] = scored['math_verify_version']
        common['answer_diagnostic_version'] = scored.get('answer_diagnostic_version')
        common['scorer_version'] = scored.get('scorer_version', 'legacy')
        common['scoring_source_hashes'] = scored.get('scoring_source_hashes')
        common['verifier_versions'] = scored.get('verifier_versions')
        if manifest['prepared_sha256'] != digest(data) or (settings is not None and common != settings):
            raise ValueError('Runs do not share identical inputs, inference settings and code/verifier versions')
        settings = common
        model = manifest['model_id']
        if data.get('study'):
            expected_sources = {m['id']: m['source'] for m in data['study']['models']}
            if expected_sources.get(model) != manifest['model_source']:
                raise ValueError('Checkpoint source does not match the frozen study configuration')
        rows = scored['records']
        indexed = {r['id']: r for r in rows}
        if model in models or len(indexed) != len(rows) or set(indexed) != set(ids):
            raise ValueError('Duplicate models or incomplete scored records')
        models[model] = [indexed[ident] for ident in ids]
    if len(models) < 2:
        raise ValueError('Need at least two complete model runs')
    names = list(models)
    for i, row in enumerate(data['examples']):
        fingerprints = {models[name][i]['prompt_token_sha256'] for name in names}
        if len(fingerprints) != 1 or any(models[name][i]['prompt_sha256'] != row['prompt_sha256'] for name in names):
            raise ValueError(f"{row['id']}: tokenizers/prompts differed; do not call this identical-input evaluation")
    summaries = {}
    for name, rows in models.items():
        summaries[name] = {'all': summarize(rows)}
        for kind in ('compute', 'forward', 'synthesize'):
            summaries[name][kind] = summarize([r for r in rows if r['kind'] == kind])
        for terminal in (True, False):
            summaries[name]['terminal' if terminal else 'nonterminal'] = summarize(
                [r for r in rows if r['terminal'] == terminal])
    comparisons = {f'{a} -> {b}': compare_records(models[a], models[b])
                   for i, a in enumerate(names) for b in names[i + 1:]}
    study_results = None
    caution = 'Curated pilot, not a population estimate. Answer correctness is not proof of role compliance.'
    if data.get('study'):
        from eac3_worker_study_stats import analyze
        study_results = analyze(data, models)
        caution = study_results['caution']
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'report.json', dict(provenance=data['provenance'],
        scorer_version=settings['scorer_version'], summary=summaries, pairs=comparisons,
        study=study_results, caution=caution))
    with (out / 'review.jsonl').open('x') as stream:
        for i, row in enumerate(data['examples']):
            stream.write(json.dumps({**row, 'outputs': {name: models[name][i] for name in names}}) + '\n')
    lines = ['# Fixed-Prefix Worker Study' if study_results else '# Fixed-Prefix Worker Pilot', '',
             caution, '',
             'The pooled table below mixes task kinds/contexts; use the stratified study endpoints for conclusions.', '',
             '| Model | N | Correct accepted | Wrong | Unknown | No accepted result | Length stop |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, summary in summaries.items():
        s = summary['all']
        c = s['status_counts']
        lines.append(f"| {name} | {s['n']} | {c.get('correct', 0)} | {c.get('wrong', 0)} | "
                     f"{c.get('unknown', 0)} | {c.get('no_accepted_result', 0)} | {s['length_stop_count']} |")
    lines += ['', 'Greedy, one response per fixed input. No gates or training applied.',
              'Unknown and missing results are not evidence of mathematically incorrect reasoning.',
              'Source-labelled S<n> LOCAL_RESULT quotations are not new answer declarations.',
              'Scorer: ' + settings['scorer_version'] + '. Explicit supported factorization forms are checked separately.',
              'The stated-answer diagnostic is secondary and does not verify reasoning or role compliance.',
              'Inspect review.jsonl for reasoning and role compliance; those labels remain unset.',
              'Several examples share a question; responses are not independent observations.', '',
              '## Paired Counts', '', '```json', json.dumps(comparisons, indent=2), '```']
    if study_results:
        from eac3_worker_study_stats import markdown
        lines += markdown(study_results)
    with (out / 'report.md').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check')
    check.add_argument('--dataset', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--dataset', required=True)
    prep.add_argument('--output', required=True)
    run = commands.add_parser('run')
    for field in ('prepared', 'model-dir', 'model-id', 'model-source', 'output'):
        run.add_argument('--' + field, required=True)
    run.add_argument('--max-new-tokens', type=int, default=1024)
    run.add_argument('--max-model-len', type=int, default=8192)
    run.add_argument('--batch-size', type=int, default=8)
    scoring = commands.add_parser('score')
    scoring.add_argument('--prepared', required=True)
    scoring.add_argument('--directory', required=True)
    reporting = commands.add_parser('report')
    reporting.add_argument('--prepared', required=True)
    reporting.add_argument('--runs', nargs='+', required=True)
    reporting.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'check':
        print(f"Valid dataset: {len(load_dataset(args.dataset)['examples'])} examples")
    elif args.command == 'prepare':
        prepare(args.dataset, args.output)
    elif args.command == 'run':
        if min(args.max_new_tokens, args.max_model_len, args.batch_size) < 1:
            parser.error('Token limits and batch size must be positive')
        run_model(args)
    elif args.command == 'score':
        score(args.prepared, args.directory)
    else:
        report(args.prepared, args.runs, args.output)


if __name__ == '__main__':
    main()

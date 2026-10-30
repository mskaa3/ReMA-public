#!/usr/bin/env python3
"""Rescore a completed fixed-prefix ZIP on CPU into a NEW directory, without inference."""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import zipfile

import eac3_fixed_prefix_eval as evaluation
from math_verification import bounded_verification_map


def load_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        roots = [n[:-len('prepared.json')] for n in names if n.endswith('/prepared.json')]
        if len(roots) != 1 or len(names) != len(set(names)):
            raise ValueError('Require exactly one completed evaluation with unique archive members')
        root = roots[0]
        data_bytes = archive.read(root + 'prepared.json')
        data = evaluation.validate_dataset(json.loads(data_bytes))
        ids = [r['id'] for r in data['examples']]
        if any(r['prompt_sha256'] != evaluation.digest(r['messages']) for r in data['examples']):
            raise ValueError('Prepared prompt fingerprint mismatch')
        config_bytes = archive.read(root + 'config.json')
        config = json.loads(config_bytes)
        expected = {m['id']: m['source'] for m in data['study']['models']}
        if {m['id']: m['source'] for m in config['models']} != expected:
            raise ValueError('Configuration differs from frozen checkpoint selection')
        runs, common = {}, None
        for name, source in expected.items():
            if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
                raise ValueError('Unsafe model ID')
            manifest_bytes = archive.read(root + name + '/manifest.json')
            manifest = json.loads(manifest_bytes)
            settings = {k: manifest[k] for k in ('prepared_sha256', 'settings', 'versions',
                                               'source_hashes', 'max_model_len', 'batch_size')}
            if (manifest['model_id'] != name or manifest['model_source'] != source
                    or manifest['prepared_sha256'] != evaluation.digest(data)
                    or manifest['source_hashes'] != data['source_hashes']
                    or (common is not None and common != settings)):
                raise ValueError('Mismatched checkpoint, generation settings, or source fingerprints')
            common = settings
            raw_bytes = archive.read(root + name + '/responses.jsonl')
            records = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
            indexed = {r['id']: r for r in records}
            if len(records) != len(ids) or set(indexed) != set(ids):
                raise ValueError('Missing, duplicate or extra generated responses')
            old_bytes = archive.read(root + name + '/scored.json')
            old = {r['id']: r for r in json.loads(old_bytes)['records']}
            if set(old) != set(ids) or len(json.loads(old_bytes)['records']) != len(ids):
                raise ValueError('Incomplete original scoring')
            for row in data['examples']:
                record = indexed[row['id']]
                if (record['model_id'] != name or record['prompt_sha256'] != row['prompt_sha256']
                        or any(old[row['id']][key] != value for key, value in record.items())):
                    raise ValueError('Scored records differ from raw generation or prepared prompts')
            runs[name] = dict(manifest=manifest_bytes, responses=raw_bytes,
                              original=old_bytes, records=indexed, old=old)
        for ident in ids:
            if len({r['records'][ident]['prompt_token_sha256'] for r in runs.values()}) != 1:
                raise ValueError('Checkpoint inputs differ at the token level')
        return data_bytes, config_bytes, data, runs, archive.read(root + 'comparison/report.json')


def rescore(path, output):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    raw_data, raw_config, data, runs, original_report = load_archive(path)
    # Validate installed dependencies and known failure/negative controls before processing.
    from eac3_math_answers import equivalent
    controls = [((r'7\times289=17\times119', ['2123=11*193'], ''), False),
                (('140 rubies,280 emeralds', ['(140,280)'], 'Return rubies, emeralds.'), True),
                (('280 rubies,140 emeralds', ['(140,280)'], 'Return rubies, emeralds.'), False)]
    if any(equivalent(item) is not expected for item, expected in controls):
        raise ValueError('Scorer regression controls failed; check dependency versions')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'prepared.json').write_bytes(raw_data)
    (output / 'config.json').write_bytes(raw_config)
    (output / 'report.original.json').write_bytes(original_report)
    changes, directories = [], []
    for name, run in runs.items():
        directory = output / name
        directory.mkdir()
        (directory / 'manifest.json').write_bytes(run['manifest'])
        (directory / 'responses.jsonl').write_bytes(run['responses'])
        (directory / 'scored.original.json').write_bytes(run['original'])
        print(f'Rescoring {name}: {len(run["records"])} existing responses', flush=True)
        result = evaluation.score_records(data, run['records'], bounded_verification_map)
        evaluation.write_json(directory / 'scored.json', result)
        for row in result['records']:
            old = run['old'][row['id']]
            fields = ('status', 'accepted_result', 'local_correct', 'full_question_answer_match',
                      'reported_answer_status')
            changed = {k: {'original': old[k], 'revised': row[k]} for k in fields if old[k] != row[k]}
            if changed:
                changes.append(dict(model_id=name, id=row['id'], changes=changed,
                                    task_form_status=row['task_form_status']))
        directories.append(directory)
    with redirect_stdout(io.StringIO()):
        evaluation.report(output / 'prepared.json', directories, output / 'comparison')
    report_path = output / 'comparison/report.md'
    report_path.write_text(
        'POST-HOC CPU RESCORING: original generation is unchanged; this is a measurement revision, '
        'not a new experiment. Original scores and a per-record change log are retained.\n\n'
        + report_path.read_text())
    with (output / 'score-changes.jsonl').open('x') as stream:
        for row in changes:
            stream.write(json.dumps(row) + '\n')
    evaluation.write_json(output / 'rescore-provenance.json', dict(
        source_archive=str(path.resolve()), archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        prepared_sha256=evaluation.digest(data), dataset_sha256=data['dataset_sha256'],
        scorer_version=evaluation.math_answers.VERSION, source_hashes=evaluation.source_hashes(),
        rescore_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        verifier_versions=evaluation.verifier_versions(), changed_records=len(changes),
        model_inference_run=False, original_artifacts_modified=False,
        post_hoc_scoring_revision=True, independent_review=False))
    print(f'Complete: {output / "comparison/report.md"}. Changes: {len(changes)} records.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rescore(args.archive, args.output)


if __name__ == '__main__':
    main()

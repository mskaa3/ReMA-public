#!/usr/bin/env python3
"""Package an existing reserved sample for review, without resampling or certifying."""

import argparse
from collections import Counter
import json
from pathlib import Path

import eac3_fixed_prefix_eval as evaluation
import eac3_worker_study as study


def checked_selection(path):
    selection = evaluation.read_json(path)
    original = {k: v for k, v in selection.items() if k != 'selection_sha256'}
    if (selection.get('selection_sha256') != evaluation.digest(original)
            or selection['config_sha256'] != evaluation.digest(selection['config'])):
        raise ValueError('Selection fingerprint mismatch; do not resample after seeing development results')
    reserved = selection['splits']['reserved']
    ids = [row['question_id'] for row in reserved]
    development = {row['question_id'] for row in selection['splits']['development']}
    if (len(ids) != len(set(ids)) or set(ids) & development
            or len(ids) != selection['config']['reserved_questions']):
        raise ValueError('Reserved sample is duplicated, overlapping, or incomplete')
    return selection


def prepare(selection_path, output, batch_size=25):
    selection = checked_selection(selection_path)
    if batch_size < 1:
        raise ValueError('Positive batch size required')
    rows = [{**row, 'decision': 'pending', 'exclusion_reason': '', 'author': '', 'checker': '',
             'author_type': '', 'checker_type': '', 'checks': {key: False for key in study.CHECKS},
             'tasks': []} for row in selection['splits']['reserved']]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    evaluation.write_json(out / 'selection.json', selection)
    batches = []
    for start in range(0, len(rows), batch_size):
        name = f'review-{start // batch_size + 1:02d}.json'
        values = rows[start:start + batch_size]
        evaluation.write_json(out / name, dict(selection_sha256=selection['selection_sha256'],
            split='reserved', review_mode='independent', examples=values))
        batches.append(dict(file=name, question_ids=[row['question_id'] for row in values]))
    evaluation.write_json(out / 'manifest.json', dict(schema_version=1,
        selection_sha256=selection['selection_sha256'], sampled_questions=len(rows),
        counts_by_stratum=dict(Counter(study.stratum(row) for row in rows)), batches=batches))
    (out / 'README.txt').write_text(
        'RESERVED REVIEW PACKAGE: NOT READY FOR INFERENCE\n\n'
        'This is the original reserved sample, not a new draw. No model responses were read.\n'
        'The JSON files are pending authoring/review templates, NOT completed annotations.\n'
        'Draft 1-4 coherent tasks and local references for every question. A distinct human\n'
        'checker must verify them; record real provenance. Include or exclude every row,\n'
        'with a substantive reason for each exclusion. Never exclude by model performance.\n'
        'Do not mark checks true automatically. Do not invent a checker or human review.\n'
        'Do not place the final/local reference answer in a compute-task instruction.\n'
        'See docs/eac3_worker_specialization_study.md for the schema and checklist.\n\n'
        'After all batches have been checked:\n'
        f'python3 -B scripts/eac3_worker_study_pack.py assemble --package {out} '
        f'--output {out}/completed-review.json\n')
    print(f'Prepared {len(rows)} reserved questions in {len(batches)} pending review batches.')
    print('No tasks were authored or certified. Independent human review is still required.')


def assemble(package, output):
    package = Path(package)
    selection = checked_selection(package / 'selection.json')
    manifest = evaluation.read_json(package / 'manifest.json')
    if manifest['selection_sha256'] != selection['selection_sha256']:
        raise ValueError('Package does not match selection')
    rows = []
    for batch in manifest['batches']:
        name = batch['file']
        if Path(name).name != name:
            raise ValueError('Batch filenames must remain inside the review package')
        content = evaluation.read_json(package / name)
        if (content['selection_sha256'] != selection['selection_sha256']
                or content['split'] != 'reserved' or content.get('review_mode', 'independent') != 'independent'
                or [row['question_id'] for row in content['examples']] != batch['question_ids']):
            raise ValueError(f'{name}: changed split, review mode, selection, or question membership/order')
        rows.extend(content['examples'])
    if [r['question_id'] for r in rows] != [r['question_id'] for r in selection['splits']['reserved']]:
        raise ValueError('Review package dropped, duplicated, replaced, or reordered selected questions')
    review = dict(selection_sha256=selection['selection_sha256'], split='reserved',
                  review_mode='independent', examples=rows)
    study.validate_review(selection, review)
    evaluation.write_json(output, review)
    print(f'Assembled {len(rows)} reviewed decisions. Run the existing freeze command next.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--selection', required=True)
    prep.add_argument('--output', required=True)
    prep.add_argument('--batch-size', type=int, default=25)
    merge = sub.add_parser('assemble')
    merge.add_argument('--package', required=True)
    merge.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            prepare(args.selection, args.output, args.batch_size)
        else:
            assemble(args.package, args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()

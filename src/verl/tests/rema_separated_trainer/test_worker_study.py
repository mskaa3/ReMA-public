"""CPU-only study design, sampling, pairing, audit and inference-report contracts."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import eac3_fixed_prefix_eval as evaluation
import eac3_worker_study as study
import eac3_worker_study_stats as stats
from test_fixed_prefix_eval import production_rollout_class
import experiment_io


class StudyTests(unittest.TestCase):
    def fixture(self, tmp):
        config = evaluation.read_json(ROOT / 'config/eac3-worker-specialization-study.json')
        rows = [dict(question=f'Question {i}: compute {i} plus 2.', data_source='ReMA-math',
                     subset='a' if i < 8 else 'b',
                     reward_model={'ground_truth': str(i + 2)}) for i in range(16)]
        candidates = tmp / 'candidates.jsonl'
        candidates.write_text(''.join(json.dumps(r) + '\n' for r in [*rows, rows[3]]))
        train = tmp / 'train.jsonl'
        train.write_text(json.dumps(rows[0]) + '\n')
        config.update(candidate_files=[str(candidates)], known_training_files=[str(train)],
            exclude_datasets=[str(ROOT / 'experiments/fixed-prefix-pilot.json')],
            model_config=str(ROOT / 'config/eac3-fixed-prefix-pilot.json'),
            development_questions=4, reserved_questions=6, bootstrap_samples=200)
        path = tmp / 'config.json'
        evaluation.write_json(path, config)
        return path

    def sample(self, tmp):
        config = self.fixture(tmp)
        with redirect_stdout(io.StringIO()):
            study.sample(config, tmp / 'sample')
        selection = evaluation.read_json(tmp / 'sample/selection.json')
        review = evaluation.read_json(tmp / 'sample/development-review.json')
        return selection, review

    def reviewed(self, review):
        review = copy.deepcopy(review)
        for row in review['examples']:
            # Artificial references for software tests, NOT a reviewed mathematical dataset.
            row.update(decision='include', author='fixture-author', checker='fixture-checker',
                       author_type='model', checker_type='human', checks={k: True for k in study.CHECKS})
            row['tasks'] = [dict(instruction='Compute 2 + 3. Return the sum.', kind='compute',
                reference_answers=['5'], checked_result=r'LOCAL_RESULT: \boxed{5}', reference_check='2+3=5.'),
                dict(instruction='Return twice S1, using its result.', kind='synthesize',
                reference_answers=['10'], checked_result=r'LOCAL_RESULT: \boxed{10}', reference_check='2*5=10.')]
        return review

    def freeze(self, tmp):
        selection, review = self.sample(tmp)
        review = self.reviewed(review)
        path = tmp / 'review.json'
        evaluation.write_json(path, review)
        with patch.object(study, 'ROOT', tmp), redirect_stdout(io.StringIO()):
            study.freeze(tmp / 'sample/selection.json', path, tmp / 'frozen')
        return evaluation.load_dataset(tmp / 'frozen/dataset.json')

    def test_sampling_deterministic_disjoint_deduplicated_and_training_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            selection, _ = self.sample(tmp)
            with redirect_stdout(io.StringIO()):
                study.sample(tmp / 'config.json', tmp / 'sample2')
            self.assertEqual(selection, evaluation.read_json(tmp / 'sample2/selection.json'))
            dev, reserved = [{r['question_id'] for r in selection['splits'][key]}
                             for key in ('development', 'reserved')]
            self.assertFalse(dev & reserved)
            self.assertEqual((len(dev), len(reserved)), (4, 6))
            self.assertNotIn(study.question_id('Question 0: compute 0 plus 2.'), dev | reserved)
            self.assertEqual(selection['counts']['duplicate_rows'], 1)
            self.assertEqual(selection['counts']['known_training_overlap_questions'], 1)
            self.assertEqual(sum(selection['eligible_question_counts'].values()), 15)
            with self.assertRaises(FileExistsError):
                study.sample(tmp / 'config.json', tmp / 'sample')

    def test_quotas_no_silent_shortfall(self):
        groups = {'a': list(range(10)), 'b': [1]}
        self.assertEqual(study.proportional_quotas(groups, 11), {'a': 10, 'b': 1})
        with self.assertRaises(ValueError):
            study.proportional_quotas(groups, 12)

    def test_missing_references_are_logged_before_sampling_and_zero_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            path = self.fixture(tmp)
            with (tmp / 'candidates.jsonl').open('a') as stream:
                for i, answer in enumerate(('', '  ', None, float('nan'), float('inf'), 0)):
                    stream.write(json.dumps(dict(question=f'Extra {i}', data_source='ReMA-math', subset='a',
                                                 reward_model={'ground_truth': answer})) + '\n')
            config = evaluation.read_json(path)
            config.update(development_questions=8, reserved_questions=8)
            path.write_text(json.dumps(config))
            with redirect_stdout(io.StringIO()):
                study.sample(path, tmp / 'sample')
            selection = evaluation.read_json(tmp / 'sample/selection.json')
            exclusions = evaluation.read_json(tmp / 'sample/input-exclusions.json')
            self.assertEqual(selection['counts']['missing_reference_rows'], 5)
            self.assertEqual(exclusions, selection['input_exclusions'])
            self.assertEqual({r['reason'] for r in exclusions}, {'missing_reference'})
            chosen = selection['splits']['development'] + selection['splits']['reserved']
            self.assertFalse({r['question_id'] for r in exclusions} & {r['question_id'] for r in chosen})
            self.assertEqual(next(r for r in chosen if r['question'] == 'Extra 5')['final_answer'], '0')
            self.assertEqual(set(selection['eligible_question_counts']), {'a', 'b'})
            self.assertEqual({r['data_source'] for r in chosen}, {'ReMA-math'})
            config['missing_reference_policy'] = 'error'
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'row 17 .*missing/empty/nonfinite'):
                study.sample(path, tmp / 'strict')
            self.assertFalse((tmp / 'strict').exists())

    def test_structured_labels_and_bad_schema_still_fail_with_row_details(self):
        import numpy as np

        for bad in ([1, 2], {'answer': 5}, np.array([1, 2]), False):
            with self.subTest(answer=repr(bad)), tempfile.TemporaryDirectory() as directory:
                tmp = Path(directory)
                config = self.fixture(tmp)
                original_reader = study.read_rows

                def read(path):
                    rows = original_reader(path)
                    if path.name == 'candidates.jsonl':
                        rows[0]['reward_model']['ground_truth'] = bad
                    return rows

                with patch.object(study, 'read_rows', side_effect=read), self.assertRaisesRegex(
                        ValueError, 'row 0 .*unsupported ground truth type'):
                    study.sample(config, tmp / 'out')
                self.assertFalse((tmp / 'out').exists())
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            config = self.fixture(tmp)
            rows = study.read_rows(tmp / 'candidates.jsonl')
            rows[0]['subset'] = None
            (tmp / 'candidates.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'row 0 .*subset must be a nonempty string'):
                study.sample(config, tmp / 'out')

    def test_actual_validation_json_source_empty_answers_and_subsets(self):
        # Mirror preprocess.py without requiring the local Parquet engine.
        # This checks the repository's real source, not a claim to read its Parquet.
        source = study.read_rows(ROOT / 'data/overall_math/all_test_data.jsonl')
        converted = [dict(question=r['question'], data_source='ReMA-math', subset=r['dataset'],
                          reward_model={'ground_truth': r['answer']}) for r in source]
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            config_path = self.fixture(tmp)
            (tmp / 'candidates.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in converted))
            with redirect_stdout(io.StringIO()):
                study.sample(config_path, tmp / 'sample')
            selection = evaluation.read_json(tmp / 'sample/selection.json')
            self.assertEqual(selection['counts']['candidate_rows'], 3221)
            self.assertEqual(selection['counts']['missing_reference_rows'], 4)
            self.assertEqual({e['row_index'] for e in selection['input_exclusions']}, {97, 1972, 2056, 2081})
            self.assertEqual(set(selection['eligible_question_counts']),
                {'gsm8k', 'olympiadbench', 'MATH500', 'gaokao2023en', 'minerva_math', 'amc23', 'aime24'})

    def test_conflicting_duplicate_is_not_arbitrarily_picked(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            config = self.fixture(tmp)
            path = tmp / 'candidates.jsonl'
            row = json.loads(path.read_text().splitlines()[3])
            row['reward_model']['ground_truth'] = 'wrong'
            with path.open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                study.sample(config, tmp / 'out')

    def test_freeze_requires_review_provenance_no_dropped_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            selection, review = self.sample(Path(directory))
            with self.assertRaises(ValueError):
                study.validate_review(selection, review)
            checked = self.reviewed(review)
            self.assertEqual(len(study.validate_review(selection, checked)), 4)
            for change in (lambda r: r['examples'].pop(),
                           lambda r: r['examples'][0].update(question='tampered'),
                           lambda r: r['examples'][0].update(stratum='tampered'),
                           lambda r: r['examples'][0].update(checker='fixture-author'),
                           lambda r: r['examples'][0]['checks'].update(each_task_sufficient_without_question=False),
                           lambda r: r['examples'][0]['tasks'][0].update(checked_result='The answer is five.')):
                bad = copy.deepcopy(checked)
                change(bad)
                with self.assertRaises(ValueError):
                    study.validate_review(selection, bad)
            reserved = evaluation.read_json(Path(directory) / 'sample/PRIVATE-reserved-review.json')
            reserved = self.reviewed(reserved)
            reserved['examples'][0]['checker_type'] = 'model'
            with self.assertRaisesRegex(ValueError, 'human'):
                study.validate_review(selection, reserved)

    def test_freeze_all_targets_and_only_one_terminal_no_question(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.freeze(Path(directory))
            self.assertEqual(len(data['examples']), 12)
            self.assertEqual(data['study']['included_questions'], 4)
            self.assertEqual(data['study']['stratify_by'], 'subset')
            self.assertEqual(set(data['study']['included_by_source']), {'ReMA-math'})
            self.assertEqual(set(data['study']['included_by_stratum']), {'a', 'b'})
            self.assertEqual({row['family'] for row in data['examples']}, {'a', 'b'})
            stats.validate_pairs(data['examples'])
            rollout = production_rollout_class()
            for row in data['examples']:
                with redirect_stdout(io.StringIO()), patch.dict(sys.modules, {
                    'verl.rema_separated_trainer.ppo.experiment_io': experiment_io}):
                    messages = evaluation.capture_prompt(row, rollout)
                self.assertEqual('Reference problem:' in messages[1]['content'],
                                 row['question_mode'] == 'full_question')
                self.assertNotIn(row['reference_check'], messages[1]['content'])
                self.assertNotIn('fixture-checker', json.dumps(messages))

    def test_model_self_review_is_explicit_and_development_only(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            selection, review = self.sample(tmp)
            checked = self.reviewed(review)
            checked['review_mode'] = 'model_self_review'
            for row in checked['examples']:
                row.update(checker=row['author'], checker_type='model')
            with self.assertRaisesRegex(ValueError, 'explicit'):
                study.validate_review(selection, checked)
            self.assertEqual(len(study.validate_review(selection, checked,
                allow_development_self_review=True)), 4)
            for change in (lambda r: r.pop('review_mode'),
                           lambda r: r.update(review_mode='unknown'),
                           lambda r: r['examples'][0].update(checker='invented-second-reviewer'),
                           lambda r: r['examples'][0].update(checker_type='human'),
                           lambda r: r['examples'][0].update(decision='pending'),
                           lambda r: r['examples'][0]['checks'].update(plan_is_coherent=False),
                           lambda r: r['examples'][0]['tasks'][0].update(checked_result='unroutable')):
                bad = copy.deepcopy(checked)
                change(bad)
                with self.assertRaises(ValueError):
                    study.validate_review(selection, bad, allow_development_self_review=True)
            reserved = self.reviewed(evaluation.read_json(tmp / 'sample/PRIVATE-reserved-review.json'))
            study.validate_review(selection, reserved)
            with self.assertRaisesRegex(ValueError, 'independent human'):
                study.validate_review(selection, reserved, allow_development_self_review=True)
            reserved['review_mode'] = 'model_self_review'
            with self.assertRaisesRegex(ValueError, 'independent human'):
                study.validate_review(selection, reserved)

            path = tmp / 'self-review.json'
            evaluation.write_json(path, checked)
            with patch.object(study, 'ROOT', tmp), redirect_stdout(io.StringIO()) as output:
                study.freeze(tmp / 'sample/selection.json', path, tmp / 'self-reviewed',
                             allow_development_self_review=True)
            self.assertIn('WARNING', output.getvalue())
            data = evaluation.load_dataset(tmp / 'self-reviewed/dataset.json')
            self.assertEqual(data['study']['review_mode'], 'model_self_review')
            self.assertFalse(data['study']['independently_reviewed'])
            self.assertIn('NOT independently reviewed', data['provenance'])
            with patch.object(study, 'ROOT', tmp), self.assertRaises(ValueError):
                study.freeze(tmp / 'sample/selection.json', path, tmp / 'blocked')
            self.assertFalse((tmp / 'blocked').exists())

            models = {name: [dict(id=r['id'], status='unknown') for r in data['examples']]
                      for name in data['study']['model_ids']}
            report = stats.analyze(data, models)
            self.assertIn('NOT independently reviewed', report['caution'])
            self.assertIn('pipeline development', report['caution'])

    def test_context_pairing_rejects_missing_or_changed_counterpart(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.freeze(Path(directory))['examples']
            for change in (lambda x: x.pop(0), lambda x: x[1].update(previous_results=['different']),
                           lambda x: x[2].update(question_mode='full_question'),
                           lambda x: x[0].update(case_id='bad')):
                bad = copy.deepcopy(rows)
                change(bad)
                with self.assertRaises(ValueError):
                    stats.validate_pairs(bad)

    def test_estimate_weights_questions_not_subtasks(self):
        rows = [dict(question_id='q1', family='a')] * 3 + [dict(question_id='q2', family='a')]
        estimate = stats.estimate([1, 1, 1, 0], rows, list(range(4)), 500, 42)
        self.assertEqual(estimate['mean'], 0.5)
        self.assertEqual(estimate['n_questions'], 2)
        self.assertEqual(estimate['ci95'], [0.0, 1.0])
        self.assertIsNone(stats.estimate([1], rows, [0], 10, 42)['ci95'])
        self.assertIsNone(stats.estimate([], rows, [], 10, 42)['mean'])

    def test_analysis_report_and_blinding_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            data = self.freeze(tmp)
            for row in data['examples']:
                row['messages'] = [{'role': 'user', 'content': row['subtasks'][row['target'] - 1]}]
                row['prompt_sha256'] = evaluation.digest(row['messages'])
            prepared = tmp / 'prepared.json'
            evaluation.write_json(prepared, data)
            dirs = []
            for model in data['study']['models']:
                name = model['id']
                out = tmp / name
                out.mkdir()
                dirs.append(out)
                evaluation.write_json(out / 'manifest.json', dict(model_id=name, model_source=model['source'],
                    prepared_sha256=evaluation.digest(data), settings={}, versions={}, source_hashes={},
                    max_model_len=8192, batch_size=8))
                records = []
                for row in data['examples']:
                    success = name == 'gated' and row['question_mode'] == 'full_question'
                    records.append(dict(id=row['id'], status='correct' if success else 'unknown',
                        local_correct=True if success else None, response='MODEL_RESPONSE', accepted_result='5',
                        finish_reason='stop', generated_tokens=20, terminal=row['target'] == len(row['subtasks']),
                        kind=row['kind'], prompt_token_sha256='same-tokens', prompt_sha256=row['prompt_sha256']))
                evaluation.write_json(out / 'scored.json', dict(records=records, math_verify_version='fake'))
            with redirect_stdout(io.StringIO()):
                evaluation.report(prepared, dirs, tmp / 'comparison')
            report = evaluation.read_json(tmp / 'comparison/report.json')['study']
            primary = report['endpoints']['compute_nonterminal_full_question']
            self.assertEqual(primary['paired_differences']['gated - ungated']['mean'], 1)
            self.assertEqual(primary['models']['ungated']['status_counts']['unknown'], 4)
            self.assertEqual(report['context_interaction']['mean'], 1)
            review_path = tmp / 'comparison/review.jsonl'
            with redirect_stdout(io.StringIO()):
                study.audit(review_path, tmp / 'audit', 2, 5)
            public = [json.loads(line) for line in (tmp / 'audit/examples.jsonl').read_text().splitlines()]
            self.assertEqual(len(public), 18)  # two questions, three conditions/targets, three models
            for row in public:
                self.assertFalse({'model_id', 'model', 'status', 'raw_score', 'gated_score'} & set(row))
            labels = (tmp / 'audit/labels.jsonl').read_text()
            self.assertNotIn('gated', labels)
            self.assertIn('null', labels)
            with self.assertRaises(ValueError):
                study.audit(review_path, tmp / 'bad-audit', 5, 5)
            manifest = evaluation.read_json(dirs[-1] / 'manifest.json')
            manifest['model_source'] = 'unexpected:checkpoint'
            (dirs[-1] / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Checkpoint source'):
                evaluation.report(prepared, dirs, tmp / 'bad-report')


if __name__ == '__main__':
    unittest.main()

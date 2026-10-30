"""CPU-only contracts for the fixed-prefix evaluator; no GPU/Ray runtime needed."""

import ast
from contextlib import redirect_stdout
import copy
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import re
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unicodedata
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import eac3_fixed_prefix_eval as evaluation
PPO = evaluation.PPO
import experiment_io


def production_rollout_class():
    # Same dependency-free production-code harness as test_c3_role_selection.
    path = PPO / 'multi_agent_rollout.py'
    definitions = [node for node in ast.parse(path.read_text()).body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    tree = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), *definitions], type_ignores=[])
    ast.fix_missing_locations(tree)
    namespace = dict(np=np, re=re, hashlib=hashlib, unicodedata=unicodedata,
                     extract_local_result=evaluation.extract_local_result)
    probe = next(n for n in ast.parse((PPO / 'validation_probes.py').read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name == 'make_probe_inputs')
    exec(compile(ast.Module(body=[probe], type_ignores=[]), '<probe>', 'exec'), namespace)
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['MultiAgentRollout']


class FixedPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = evaluation.load_dataset(ROOT / 'experiments/fixed-prefix-pilot.json')
        cls.rollout = production_rollout_class()

    def capture(self, row):
        with redirect_stdout(io.StringIO()), patch.dict(sys.modules, {
                'verl.rema_separated_trainer.ppo.experiment_io': experiment_io}):
            return evaluation.capture_prompt(row, self.rollout)

    def prepared(self):
        data = copy.deepcopy(self.data)
        for row in data['examples']:
            row['messages'] = self.capture(row)
            row['prompt_sha256'] = evaluation.digest(row['messages'])
        data['source_hashes'] = evaluation.source_hashes()
        return data

    def test_all_pilot_rows_capture_the_production_prompt(self):
        for row in self.data['examples']:
            with self.subTest(row=row['id']):
                messages = self.capture(row)
                self.assertEqual(messages, self.capture(row))
                user = messages[1]['content']
                self.assertEqual(messages[0]['content'], runpy.run_path(
                    str(ROOT / 'prompt/math/hierarchical_mamrp.py'))['WORKER_SYSTEM_PROMPT'])
                self.assertIn(f"- S{row['target']}: {row['subtasks'][row['target'] - 1]}", user)
                self.assertNotIn('PLAN:', user)
                self.assertNotIn(row['reference_check'], user)
                self.assertEqual('Reference problem:' in user, row['target'] < len(row['subtasks']))
                for i, result in enumerate(row['previous_results'], 1):
                    self.assertIn(f'S{i} LOCAL_RESULT: {evaluation.extract_local_result(result)}', user)
                for i, task in enumerate(row['subtasks'], 1):
                    if i != row['target']:
                        self.assertNotIn(f'- S{i}: {task}', user)

    def test_reference_labels_never_affect_messages(self):
        row = copy.deepcopy(self.data['examples'][0])
        original = self.capture(row)
        row['reference_answers'] = ['SECRET_LOCAL']
        row['final_answer'] = 'SECRET_FINAL'
        row['reference_check'] = 'SECRET_NOTES'
        self.assertEqual(original, self.capture(row))

    def test_question_ablation_and_single_task_terminal(self):
        row = copy.deepcopy(self.data['examples'][0])
        row['question_mode'] = 'subtask_context'
        self.assertNotIn(row['question'], self.capture(row)[1]['content'])
        row['question_mode'] = 'full_question'
        row['subtasks'] = row['subtasks'][:1]
        self.assertNotIn(row['question'], self.capture(row)[1]['content'])

    def test_pilot_scalar_references_checked_independently(self):
        expected = {
            'stickers-s1': str(15 - 7), 'stickers-s2': str(15 - 7 + 5),
            'height-s1': str(36 * 2), 'height-s2': str(36 * 2 - 2), 'height-s3': str(36 * 2 - 2 - 3),
            'polar-s1': str(int((0**2 + 3**2)**0.5)), 'circle-s1': str(30 + 5**2 + 3**2),
            'digit-s1': str((330 - 1) % 4 + 1), 'digit-s2': '6238'[(330 - 1) % 4],
            'distance-s1': str(abs(2 * 6 - 0 - 2)), 'temperature-s1': '20',
            'temperature-s2': str(int(Fraction(9, 5) * 20 + 32)),
        }
        for row in self.data['examples']:
            if row['id'] in expected:
                self.assertEqual(row['reference_answers'][0], expected[row['id']])
        self.assertEqual(Fraction(1, 5) * Fraction(8, 7), Fraction(8, 35))
        self.assertEqual(Fraction(8, 35) / Fraction(12, 20), Fraction(8, 21))

    def test_dataset_rejects_missing_prefix_and_unroutable_result(self):
        for mutate in (lambda r: r.update(previous_results=[]),
                       lambda r: r.update(previous_results=['There are eight stickers.']),
                       lambda r: r.update(target=0),
                       lambda r: r.update(reference_answers=[])):
            data = copy.deepcopy(self.data)
            mutate(data['examples'][1])
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'data.json'
                evaluation.write_json(path, data)
                with self.assertRaises(ValueError):
                    evaluation.load_dataset(path)

    def test_communication_failure_is_not_wrong_math(self):
        self.assertEqual(evaluation.extract_local_result('I get 8.'), '')
        self.assertEqual(evaluation.extract_local_result(r'LOCAL_RESULT: \boxed{8}'), r'\boxed{8}')
        self.assertEqual(evaluation.extract_local_result(r'LOCAL_RESULT: \boxed{8'), '')
        self.assertEqual(evaluation.extract_local_result(r'\boxed{8} or \boxed{13}'), '')

    def test_comparator_alternatives_preserve_unknown(self):
        module = 'verl.rema_separated_trainer.ppo.prefix_probe'
        for outputs, expected in (([False, None], None), ([False, True], True), ([False, False], False)):
            values = iter(outputs)
            with patch.dict(sys.modules, {module: SimpleNamespace(compare_worker_final_answers=lambda *a: next(values))}):
                self.assertIs(evaluation.check_equivalence(('response', ['8', 'a=8'])), expected)

    def test_reference_failure_reports_expression_field_result_and_versions(self):
        row = copy.deepcopy(self.data['examples'][0])
        row.update(reference_answers=[r'\frac{1}{r}+\frac{1}{h}'], final_answer='6')

        def bounded(function, jobs, **kwargs):
            self.assertEqual(len(jobs), 2)
            return [None, False], dict(checks=2, unknown_count=1)

        with patch.object(evaluation.importlib.metadata, 'version', side_effect=lambda p: p + '-test'), \
                redirect_stdout(io.StringIO()), self.assertRaises(ValueError) as caught:
            evaluation.verify_reference_notation({'examples': [row]}, bounded)
        diagnostic = json.loads(str(caught.exception).split('\n', 1)[1])
        self.assertEqual(diagnostic['invalid'], [
            dict(id=row['id'], field='reference_answers[0]', reference=row['reference_answers'][0], result='unknown'),
            dict(id=row['id'], field='final_answer', reference='6', result='false')])
        self.assertEqual(diagnostic['versions']['latex2sympy2-extended'], 'latex2sympy2-extended-test')
        self.assertEqual(diagnostic['stats']['unknown_count'], 1)

    def test_prepare_retains_verifier_versions_and_stops_before_prompt_capture_on_failure(self):
        for passes in (True, False):
            with self.subTest(passes=passes), tempfile.TemporaryDirectory() as directory:
                tmp = Path(directory)
                path, output = tmp / 'data.json', tmp / 'prepared.json'
                evaluation.write_json(path, self.data)

                def bounded(function, jobs, **kwargs):
                    return [True if passes else None for _ in jobs], dict(checks=len(jobs))

                modules = {'verl.rema_separated_trainer.ppo.math_verification':
                    SimpleNamespace(bounded_verification_map=bounded),
                    'verl.rema_separated_trainer.ppo.multi_agent_rollout':
                    SimpleNamespace(MultiAgentRollout=self.rollout)}
                with patch.dict(sys.modules, modules), patch.object(evaluation.importlib.metadata,
                        'version', return_value='test'), redirect_stdout(io.StringIO()), patch.object(
                        evaluation, 'capture_prompt', return_value=[dict(role='user', content='fixture')]) as capture:
                    if passes:
                        evaluation.prepare(path, output)
                        prepared = evaluation.read_json(output)
                        self.assertEqual(prepared['reference_verification']['versions']['sympy'], 'test')
                        self.assertEqual(prepared['reference_verification']['stats']['checks'],
                                         sum(len(r['reference_answers'])+1 for r in self.data['examples']))
                    else:
                        with self.assertRaisesRegex(ValueError, 'Reference notation failed'):
                            evaluation.prepare(path, output)
                        capture.assert_not_called()
                        self.assertFalse(output.exists())

    def test_missing_verifier_is_diagnosed_before_any_checks(self):
        def absent(package):
            raise evaluation.importlib.metadata.PackageNotFoundError(package)

        def bounded(*args, **kwargs):
            self.fail('Verification must not run without the required package')

        with patch.object(evaluation.importlib.metadata, 'version', side_effect=absent), \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'requires math-verify'):
            evaluation.verify_reference_notation(self.data, bounded)

    def test_mock_inference_and_paired_report(self):
        data = self.prepared()
        class Tokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def apply_chat_template(self, messages, **kwargs):
                self.assertion = kwargs == {'tokenize': True, 'add_generation_prompt': True}
                return list(json.dumps(messages).encode())

        class Engine:
            def __init__(self, **kwargs):
                pass

            def generate(self, requests, sampling, **kwargs):
                assert sampling['temperature'] == 0 and sampling['n'] == 1
                return [SimpleNamespace(outputs=[SimpleNamespace(text=r'LOCAL_RESULT: \boxed{8}',
                        token_ids=[1, 2], finish_reason='stop')]) for _ in requests]

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            prepared = tmp / 'prepared.json'
            evaluation.write_json(prepared, data)
            model_dir = tmp / 'model'
            model_dir.mkdir()
            evaluation.write_json(model_dir / 'config.json', {'model_type': 'qwen2'})
            directories = []
            for name in ('w0', 'gated'):
                args = SimpleNamespace(prepared=prepared, model_dir=model_dir, model_source='test:fake',
                    model_id=name, output=tmp / name, max_new_tokens=1024, max_model_len=8192, batch_size=8)
                modules = {'transformers': SimpleNamespace(AutoTokenizer=Tokenizer),
                           'vllm': SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: kw),
                           'validate_hf_model_dir': SimpleNamespace(validate_model_dir=lambda _: None)}
                with patch.dict(sys.modules, modules), patch.object(evaluation.importlib.metadata, 'version', return_value='test'), redirect_stdout(io.StringIO()):
                    evaluation.run_model(args)
                self.assertFalse((args.output / 'responses.jsonl.partial').exists())
                # Exercise scoring/report wiring with determinate and unknown checks,
                # not a substitute for a real Math-Verify integration test.
                def bounded(function, jobs, **kwargs):
                    return [None if i == 0 else i % 4 == 0 for i in range(len(jobs))], {'checks': len(jobs)}
                with patch.dict(sys.modules, {'verl.rema_separated_trainer.ppo.math_verification':
                        SimpleNamespace(bounded_verification_map=bounded)}), patch.object(
                        evaluation.importlib.metadata, 'version', return_value='test'), redirect_stdout(io.StringIO()):
                    evaluation.score(prepared, args.output)
                directories.append(args.output)
            with redirect_stdout(io.StringIO()):
                evaluation.report(prepared, directories, tmp / 'comparison')
            report = evaluation.read_json(tmp / 'comparison/report.json')
            self.assertEqual(report['pairs']['w0 -> gated']['accepted_correct_gain'], 0)
            self.assertEqual(report['summary']['w0']['all']['status_counts']['unknown'], 1)
            self.assertTrue((tmp / 'comparison/review.jsonl').is_file())
            with self.assertRaises(FileExistsError):
                evaluation.report(prepared, directories, tmp / 'comparison')
            path = directories[1] / 'scored.json'
            bad = evaluation.read_json(path)
            bad['records'][0]['prompt_token_sha256'] = 'different-tokenizer'
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'tokenizers/prompts differed'):
                evaluation.report(prepared, directories, tmp / 'bad-comparison')

    def test_duplicate_and_missing_records_are_not_silently_paired(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'responses.jsonl'
            path.write_text('{"id":"a"}\n{"id":"a"}\n')
            with self.assertRaises(ValueError):
                evaluation.indexed_records(path, ['a', 'b'])


if __name__ == '__main__':
    unittest.main()

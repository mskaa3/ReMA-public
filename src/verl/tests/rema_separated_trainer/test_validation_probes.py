"""CPU/stdlib tests, without importing Ray, torch, or a running inference server."""

import ast
from dataclasses import replace
import importlib.util
import io
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import URLError


ROOT = Path(__file__).resolve().parents[4]
PPO = ROOT / 'src/verl/verl/rema_separated_trainer/ppo'
PACKAGE = '_validation_probe_test_package'
package = ModuleType(PACKAGE)
package.__path__ = [str(PPO)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + '.validation_probes', PPO / 'validation_probes.py')
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


def example(key='q', dataset='gsm8k', system_score=1.):
    return probe.ProbeExample(key, dataset, 'ReMA-math', '26', None,
        probe.make_probe_inputs('S2: Add 20 to S1.', 'S1: Multiply 2 by 3.',
                                r'PREVIOUS LOCAL RESULTS: S1 LOCAL_RESULT: \boxed{6}'), system_score)


def extract_method(name):
    tree = ast.parse((PPO / 'ray_trainer.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RayReMASeparatedTrainer')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {'validation_metric_namespace': probe.validation_metric_namespace,
                 'FrozenProbeClient': probe.FrozenProbeClient}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<trainer-method>', 'exec'), namespace)
    return namespace[name]


class ValidationProbeTests(unittest.TestCase):
    def test_launcher_defaults_to_qwen_only_when_endpoint_is_configured(self):
        launcher = (ROOT / 'agent12-curriculum-trainer-multinode.sh').read_text()
        start = launcher.index('export VALIDATION_PROBE_URL=')
        end = launcher.index('\nexport SIF_NAME=', start)
        shell = 'set -euo pipefail\n' + launcher[start:end]
        shell += '\nprintf "MODEL=%s\\n" "$VALIDATION_PROBE_MODEL"\n'
        clean_env = {key: value for key, value in os.environ.items()
                     if not key.startswith('VALIDATION_PROBE_')}
        cases = [
            ({}, '', 0),
            ({'VALIDATION_PROBE_URL': 'http://probe:8000/v1'}, 'Qwen/Qwen3-4B-Instruct-2507', 0),
            ({'VALIDATION_PROBE_URL': 'http://probe:8000/v1',
              'VALIDATION_PROBE_MODEL': 'custom-frozen-v1'}, 'custom-frozen-v1', 0),
            ({'VALIDATION_PROBE_MODEL': 'custom-frozen-v1'}, None, 1),
        ]
        for overrides, model, code in cases:
            with self.subTest(overrides=overrides):
                result = subprocess.run(['bash', '-c', shell], env={**clean_env, **overrides},
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stderr)
                if code == 0:
                    self.assertIn(f'MODEL={model}\n', result.stdout)
                else:
                    self.assertIn('Set VALIDATION_PROBE_URL', result.stderr)

    def test_configured_probe_model_matches_launcher_default(self):
        for path in (ROOT / 'config/rema-rl.yaml',
                     ROOT / 'src/verl/verl/rema_separated_trainer/config/ppo_trainer.yaml'):
            with self.subTest(path=path):
                config = path.read_text().split('  validation_probe:\n', 1)[1]
                self.assertTrue(config.startswith('    enable: False\n    base_url: null\n'
                                                 '    model: Qwen/Qwen3-4B-Instruct-2507\n'))

    def test_context_conditions_preserve_boundaries(self):
        inputs = example().inputs
        self.assertEqual(inputs['terminal_only'], 'CURRENT TASK:\nS2: Add 20 to S1.')
        self.assertIn('Multiply 2 by 3', inputs['with_instructions'])
        self.assertNotIn('\\boxed{6}', inputs['with_instructions'])
        self.assertNotIn('Multiply 2 by 3', inputs['with_results'])
        self.assertIn('\\boxed{6}', inputs['with_results'])
        self.assertNotIn('26', ''.join(inputs.values()))

    def test_single_task_does_not_invent_upstream_information(self):
        inputs = probe.make_probe_inputs('Compute 2+3.', '', '')
        self.assertEqual(len(set(inputs.values())), 1)

    def test_actual_leaked_constants_are_not_sanitized(self):
        inputs = probe.make_probe_inputs('Return the answer, 26.', 'S1: The value is 6.', '')
        self.assertIn('26', inputs['terminal_only'])
        self.assertIn('The value is 6', inputs['with_instructions'])

    def test_selection_is_stable_dataset_stratified_and_not_score_filtered(self):
        rows = [example(str(i), dataset, float(i % 2)) for dataset in ['math', 'gsm8k'] for i in range(10)]
        selected = probe.select_probe_examples(rows, 4)
        self.assertEqual(selected, probe.select_probe_examples(list(reversed(rows)), 4))
        self.assertEqual(len(selected), 8)
        self.assertEqual(sum(row.dataset == 'math' for row in selected), 4)
        self.assertIn(0., [row.system_score for row in selected])
        changed = [replace(row, system_score=1. - row.system_score) for row in rows]
        self.assertEqual([r.question_id for r in selected],
                         [r.question_id for r in probe.select_probe_examples(changed, 4)])
        self.assertEqual(len(probe.select_probe_examples(rows + rows, 20)), 20)

    def test_paired_scores_include_invalid_and_unknown_denominators(self):
        rows = [example(), example('other', system_score=0.), replace(example('missing'), inputs=None)]
        calls = []

        def generate(messages):
            calls.extend(messages)
            self.assertEqual(len(messages), 3)  # Same input is measured once, even across rows.
            return [probe.ProbeResponse(r'\boxed{UNKNOWN}', 'stop'),
                    probe.ProbeResponse(r'\boxed{26}', 'length'),
                    probe.ProbeResponse(r'\boxed{26}', 'stop')]

        def score(sources, responses, truths, extras, **kwargs):
            self.assertEqual(truths, ['26', '26'])
            return [1., 1.]

        metrics, records = probe.evaluate_probes(rows, generate, lambda xs: [True] * len(xs), score)
        prefix = 'val/probe/gsm8k'
        self.assertEqual(metrics[f'{prefix}/terminal_only_acc'], 0.)
        self.assertEqual(metrics[f'{prefix}/with_instructions_acc'], 0.)
        self.assertAlmostEqual(metrics[f'{prefix}/with_results_acc'], 2/3)
        self.assertAlmostEqual(metrics[f'{prefix}/information_gain'], 2/3)
        self.assertAlmostEqual(metrics[f'{prefix}/result_advantage'], 2/3)
        self.assertEqual(metrics['val_diagnostics/probe/gsm8k/sample_count'], 3)
        self.assertAlmostEqual(metrics['val_diagnostics/probe/gsm8k/with_instructions/truncated_rate'], 2/3)
        self.assertTrue(records[0]['conditions']['terminal_only']['valid'])
        self.assertTrue(records[2]['conditions']['with_results']['missing_context'])
        self.assertNotIn('ground_truth', json.dumps(calls))

    def test_negative_differences_are_not_clipped(self):
        responses = [probe.ProbeResponse(r'\boxed{26}', 'stop'),
                     probe.ProbeResponse(r'\boxed{26}', 'stop'),
                     probe.ProbeResponse(r'\boxed{27}', 'stop')]
        metrics, _ = probe.evaluate_probes([example()], lambda _: responses,
            lambda xs: [True] * len(xs), lambda *args, **kwargs: [1., 1., 0.])
        self.assertEqual(metrics['val/probe/gsm8k/information_gain'], -1.)
        self.assertEqual(metrics['val/probe/gsm8k/result_advantage'], -1.)

    def test_parse_and_verification_failures_are_invalid_not_dropped(self):
        responses = [probe.ProbeResponse(r'\boxed{26}', 'stop')] * 3
        metrics, records = probe.evaluate_probes([example()], lambda _: responses,
            lambda _: [False, None, True], lambda *args, **kwargs: [float('nan')])
        self.assertTrue(all(not item['valid'] for item in records[0]['conditions'].values()))
        self.assertEqual(metrics['val/probe/gsm8k/with_results_acc'], 0.)

    def test_incomplete_generation_or_scoring_raises(self):
        with self.assertRaisesRegex(RuntimeError, 'generation'):
            probe.evaluate_probes([example()], lambda _: [], lambda _: [], lambda *args: [])
        with self.assertRaisesRegex(RuntimeError, 'score batch'):
            probe.evaluate_probes([example()], lambda xs: [probe.ProbeResponse(r'\boxed{26}', 'stop')] * len(xs),
                                  lambda xs: [True] * len(xs), lambda *args, **kwargs: [])

    def test_endpoint_receives_only_frozen_context_and_fixed_budget(self):
        client = probe.FrozenProbeClient({'base_url': 'http://probe:8000/v1', 'model': 'frozen-v1'})
        requests = []

        def open_request(request, timeout):
            requests.append(json.loads(request.data))
            self.assertEqual(timeout, 180)
            self.assertEqual(request.full_url, 'http://probe:8000/v1/chat/completions')
            return io.BytesIO(json.dumps({'model': 'frozen-v1', 'choices': [
                {'message': {'content': r'\boxed{26}'}, 'finish_reason': 'stop'}]}).encode())

        with patch.object(probe, 'urlopen', open_request):
            responses = client.generate(['CURRENT TASK: use S1'])
        self.assertEqual(responses[0].text, r'\boxed{26}')
        payload = requests[0]
        self.assertEqual(payload['temperature'], 0)
        self.assertEqual(payload['max_tokens'], 256)
        self.assertEqual(payload['n'], 1)
        self.assertEqual(payload['messages'][1]['content'], 'CURRENT TASK: use S1')
        self.assertNotIn('ground_truth', payload)

    def test_service_failure_and_model_mismatch_do_not_fake_zero_accuracy(self):
        client = probe.FrozenProbeClient({'base_url': 'http://probe/v1', 'model': 'frozen-v1'})
        with patch.object(probe, 'urlopen', side_effect=URLError('offline')):
            with self.assertRaisesRegex(RuntimeError, 'endpoint failed'):
                client.generate(['task'])
        with patch.object(probe, 'urlopen', return_value=io.BytesIO(b'{"model": "different"}')):
            with self.assertRaisesRegex(RuntimeError, 'different model'):
                client.generate(['task'])

    def test_configuration_fails_early_or_stays_disabled(self):
        init = extract_method('_init_validation_probes')
        trainer = SimpleNamespace(config=SimpleNamespace(trainer={}))
        init(trainer)
        self.assertIsNone(trainer.validation_probe_client)
        trainer.config.trainer = {'validation_probe': {'enable': True}}
        trainer._get_hierarchy_config = lambda: {'terminal_worker_as_answer': True}
        with self.assertRaisesRegex(ValueError, 'base_url'):
            init(trainer)

    def test_namespace_keeps_only_five_metrics_and_dataset_accuracy(self):
        raw = {'val/acc/math': .5, 'val/leakage/gated_accuracy': .2,
               'val/round_1_acc': .4, 'val/completion_tokens/mean': 100,
               'val/math_verify/parse/checks': 20,
               'val_diagnostics/probe/math/sample_count': 32}
        raw.update({f'val/probe/math/{name}': .3 for name in probe.METRICS})
        metrics = probe.validation_metric_namespace(raw)
        self.assertEqual(sum(key.startswith('val/') for key in metrics), 6)
        self.assertEqual(metrics['val_diagnostics/leakage/gated_accuracy'], .2)
        self.assertEqual(probe.validation_metric_namespace(metrics), metrics)

    def test_real_validate_wrapper_separates_teacher_assisted_results(self):
        validate = extract_method('_validate')
        trainer = SimpleNamespace(config=SimpleNamespace(data={'teacher_assisted_validation': True}))
        trainer._validate_context = lambda teacher_assisted: {
            'val/acc/math': float(teacher_assisted), 'val/leakage/gated_accuracy': .25,
            'val/probe/math/with_results_acc': .75,
            'val_diagnostics/probe/math/sample_count': 32,
        }
        metrics = validate(trainer)
        self.assertEqual(metrics['val/acc/math'], 0.)
        self.assertEqual(metrics['val_diagnostics/teacher_assisted/acc/math'], 1.)
        self.assertEqual(metrics['val_diagnostics/teacher_assisted/probe/math/sample_count'], 32)
        self.assertFalse(any(key.startswith('val/teacher_assisted/') for key in metrics))

    def test_probe_evidence_is_saved_with_config_and_checksum(self):
        evidence = importlib.import_module(PACKAGE + '.experiment_io')
        with tempfile.TemporaryDirectory() as tmp:
            writer = evidence.ValidationRecorder(tmp, 'val', 1, 1, 0,
                {'trainer': {'validation_probe': {'model': 'frozen-v1'}}})
            records = [{'question_id': 'abc', 'conditions': {'terminal_only': {'score': 0.}}}]
            path = writer.finish({'val/acc/math': 1.}, probe_records=records)
            saved = path.with_suffix('.probes.jsonl')
            self.assertEqual(json.loads(saved.read_text()), records[0])
            metadata = json.loads(path.with_suffix('.metadata.json').read_text())
            self.assertEqual(metadata['probe_records_sha256'], hashlib.sha256(saved.read_bytes()).hexdigest())
            self.assertEqual(metadata['config']['trainer']['validation_probe']['model'], 'frozen-v1')


if __name__ == '__main__':
    unittest.main()

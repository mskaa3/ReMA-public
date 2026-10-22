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
from unittest.mock import Mock, patch
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
local_probe = importlib.import_module(PACKAGE + '.local_validation_probe')


def example(key='q', dataset='gsm8k', system_score=1.):
    return probe.ProbeExample(key, dataset, 'ReMA-math', '26', None,
        probe.make_probe_inputs('S2: Add 20 to S1.', 'S1: Multiply 2 by 3.',
                                r'PREVIOUS LOCAL RESULTS: S1 LOCAL_RESULT: \boxed{6}'), system_score)


def extract_method(name, **bindings):
    tree = ast.parse((PPO / 'ray_trainer.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RayReMASeparatedTrainer')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {'validation_metric_namespace': probe.validation_metric_namespace,
                 'make_validation_probe_client': probe.make_validation_probe_client, **bindings}
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
            ({'VALIDATION_PROBE_URL': 'http://probe:8000/v1'}, 'Qwen/Qwen2.5-7B-Instruct', 0),
            ({'VALIDATION_PROBE_URL': 'http://probe:8000/v1',
              'DECOMPOSER_MODEL_PATH': '/checkpoints/trained-decomposer'}, 'Qwen/Qwen2.5-7B-Instruct', 0),
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
                self.assertTrue(config.startswith('    enable: False\n    backend: http'))
                self.assertIn('    model: Qwen/Qwen2.5-7B-Instruct\n', config)

    def test_local_launcher_needs_revision_not_endpoint(self):
        launcher = (ROOT / 'agent12-curriculum-trainer-multinode.sh').read_text()
        start = launcher.index('export VALIDATION_PROBE_URL=')
        end = launcher.index('\nexport SIF_NAME=', start)
        shell = 'set -euo pipefail\n' + launcher[start:end]
        clean = {k: v for k, v in os.environ.items() if not k.startswith('VALIDATION_PROBE_')}
        env = {**clean, 'VALIDATION_PROBE_BACKEND': 'local', 'VALIDATION_PROBE_REVISION': 'a' * 40}
        result = subprocess.run(['bash', '-c', shell], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('backend=local, model=Qwen/Qwen2.5-7B-Instruct', result.stdout)
        for changes in ({'VALIDATION_PROBE_REVISION': ''}, {'VALIDATION_PROBE_REVISION': 'main'},
                        {'VALIDATION_PROBE_URL': 'http://old-probe/v1'}, {'VALIDATION_PROBE_BACKEND': 'typo'}):
            result = subprocess.run(['bash', '-c', shell], env={**env, **changes}, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, changes)

    def test_metrics_export_selects_local_backend_without_hf_revision(self):
        launcher = (ROOT / 'agent12-curriculum-trainer-multinode.sh').read_text()
        shell = 'set -euo pipefail\n' + launcher[launcher.index('export VALIDATION_PROBE_URL='):
                                               launcher.index('\nexport SIF_NAME=')]
        clean = {k: v for k, v in os.environ.items() if not k.startswith('VALIDATION_PROBE_')}
        remote = 's3:bucket/decomposer/global_step_110/huggingface'
        env = dict(clean, VALIDATION_PROBE_MODEL_REMOTE=remote)
        result = subprocess.run(['bash', '-c', shell], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('backend=local, model=' + remote, result.stdout)
        for changes in ({'VALIDATION_PROBE_BACKEND': 'off'},
                        {'VALIDATION_PROBE_URL': 'http://metrics/v1'},
                        {'VALIDATION_PROBE_REVISION': 'a' * 40}):
            with self.subTest(changes=changes):
                result = subprocess.run(['bash', '-c', shell], env=dict(env, **changes),
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)

    def test_local_launcher_rejects_full_training_allocation(self):
        launcher = (ROOT / 'agent12-curriculum-trainer-multinode.sh').read_text()
        start = launcher.index('if [[ ! "$TRAINING_NNODES"')
        end = launcher.index('\nif (( GPUS_PER_NODE', start)
        env = {**os.environ, 'TRAINING_NNODES': '4', 'SLURM_NNODES': '4',
               'GPUS_PER_NODE': '4', 'POOL_GPUS_PER_NODE': '2', 'VALIDATION_PROBE_BACKEND': 'local'}
        shell = 'set -euo pipefail\n' + launcher[start:end]
        result = subprocess.run(['bash', '-c', shell], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('1 exclusive GPUs outside', result.stderr)
        result = subprocess.run(['bash', '-c', shell], env={**env, 'SLURM_NNODES': '5'},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

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

    def test_http_gate_request_does_not_change_evaluation_prompt_or_budget(self):
        client = probe.FrozenProbeClient({'base_url': 'http://probe/v1', 'model': 'fixed'})
        payloads = []

        def respond(request, timeout):
            payloads.append(json.loads(request.data))
            return io.BytesIO(json.dumps({'model': 'fixed', 'usage': {'completion_tokens': 12},
                'choices': [{'message': {'content': r'\boxed{UNKNOWN}'}, 'finish_reason': 'stop'}]}).encode())

        with patch.object(probe, 'urlopen', respond):
            gate = client.generate(['gate input'], system_prompt='Gate system', max_new_tokens=512)
            client.generate(['eval input'])
        self.assertEqual(gate[0].num_gen_tokens, 12)
        self.assertEqual([p['max_tokens'] for p in payloads], [512, 256])
        self.assertEqual([p['messages'][0]['content'] for p in payloads], ['Gate system', probe.SYSTEM_PROMPT])
        self.assertEqual(client.max_new_tokens, 256)
        for budget in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                client.generate(['task'], max_new_tokens=budget)

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


class LocalValidationProbeTests(unittest.TestCase):
    def config(self, **kwargs):
        return {'backend': 'local', 'model': 'Qwen/Qwen2.5-7B-Instruct',
                'revision': 'a' * 40, **kwargs}

    def test_factory_keeps_local_independent_of_http(self):
        client = probe.make_validation_probe_client(self.config())
        self.assertIsInstance(client, local_probe.LocalFrozenProbeClient)
        self.assertIsNone(client.actor)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            probe.make_validation_probe_client({'backend': 'typo'})
        for config in (self.config(revision=None), self.config(revision='main'),
                       self.config(base_url='http://probe/v1'), self.config(batch_size=0),
                       self.config(max_model_len=256), self.config(gpu_memory_utilization=1.1)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                probe.make_validation_probe_client(config)

    def test_local_capacity_includes_both_training_pools_plus_probe(self):
        client = local_probe.LocalFrozenProbeClient(self.config())
        ray = SimpleNamespace(available_resources=lambda: {'GPU': 16})
        with patch.dict(sys.modules, {'ray': ray}), patch.object(local_probe, 'check_local_probe_dependencies'):
            with self.assertRaisesRegex(RuntimeError, '16 training GPUs plus 1'):
                client.check_capacity(16)
            ray.available_resources = lambda: {'GPU': 20}
            client.check_capacity(16)

    def test_dependencies_fail_clearly_without_automatic_upgrades(self):
        from importlib.metadata import PackageNotFoundError
        for versions in (['0.8.4'], ['0.8.5', '4.50.0'], PackageNotFoundError('vllm')):
            with patch('importlib.metadata.version', side_effect=versions):
                with self.assertRaisesRegex(RuntimeError, 'compatible container or the HTTP backend'):
                    local_probe.check_local_probe_dependencies()
        with patch('importlib.metadata.version', side_effect=['0.8.5', '4.51.0']):
            local_probe.check_local_probe_dependencies()

    def test_actor_lifecycle_reserves_one_gpu_and_is_closed_once(self):
        actor = Mock()
        actor_type = Mock()
        actor_type.options.return_value.remote.return_value = actor
        ray = Mock()
        ray.available_resources.return_value = {'GPU': 4}
        ray.remote.return_value.return_value = actor_type
        ray.get.return_value = {'model': 'fixed'}
        client = local_probe.LocalFrozenProbeClient(self.config())
        with patch.dict(sys.modules, {'ray': ray}):
            client.start()
            client.start()
            ray.remote.assert_called_once_with(num_cpus=1, num_gpus=1, max_restarts=0, max_task_retries=0)
            actor_type.options.assert_called_once_with(scheduling_strategy='DEFAULT')
            ray.get.assert_called_once_with(actor.ready.remote.return_value, timeout=900)
            client.close()
            client.close()
        ray.kill.assert_called_once_with(actor, no_restart=True)

    def test_startup_timeout_kills_actor(self):
        ray = Mock()
        ray.available_resources.return_value = {'GPU': 1}
        ray.get.side_effect = TimeoutError('startup')
        client = local_probe.LocalFrozenProbeClient(self.config())
        with patch.dict(sys.modules, {'ray': ray}):
            with self.assertRaises(TimeoutError):
                client.start()
        self.assertIsNone(client.actor)
        ray.kill.assert_called_once()

    def test_cleanup_failure_preserves_original_error(self):
        ray = Mock()
        ray.available_resources.return_value = {'GPU': 1}
        ray.get.side_effect = TimeoutError('startup')
        ray.kill.side_effect = RuntimeError('Ray disconnected')
        client = local_probe.LocalFrozenProbeClient(self.config())
        with patch.dict(sys.modules, {'ray': ray}), self.assertLogs(local_probe.__name__, level='WARNING'):
            with self.assertRaisesRegex(TimeoutError, 'startup'):
                client.start()
        self.assertIsNone(client.actor)

    def test_generation_batches_are_ordered_and_timeout_is_not_zero_accuracy(self):
        ray = Mock()
        client = local_probe.LocalFrozenProbeClient(self.config(batch_size=2))
        client.actor = Mock()
        outputs = [probe.ProbeResponse(str(i), 'stop') for i in range(5)]
        ray.get.side_effect = [outputs[:2], outputs[2:4], outputs[4:]]
        with patch.dict(sys.modules, {'ray': ray}):
            self.assertEqual(client.generate(list('abcde')), outputs)
            self.assertEqual([call.args[0] for call in client.actor.generate.remote.call_args_list],
                             [list('ab'), list('cd'), list('e')])
            ray.get.side_effect = TimeoutError('inference')
            with self.assertRaises(TimeoutError):
                client.generate(['another'])
            self.assertIsNone(client.actor)
            ray.kill.assert_called_once()

    def test_local_engine_is_frozen_pinned_and_uses_its_own_template(self):
        engine = Mock()
        tokenizer = engine.get_tokenizer.return_value
        tokenizer.apply_chat_template.return_value = [1, 2, 3]
        engine.generate.return_value = [SimpleNamespace(outputs=[
            SimpleNamespace(text=r'\boxed{7}', finish_reason='length', token_ids=[7, 8])])]
        vllm = SimpleNamespace(LLM=Mock(return_value=engine), SamplingParams=Mock())
        ray = SimpleNamespace(get_gpu_ids=lambda: [3])
        with patch.dict(sys.modules, {'vllm': vllm, 'ray': ray}), \
                patch.object(local_probe, 'check_local_probe_dependencies'):
            worker = local_probe.LocalProbeWorker(self.config())
            result = worker.generate(['CURRENT TASK: use S1'])
        kwargs = vllm.LLM.call_args.kwargs
        self.assertEqual(kwargs['revision'], 'a' * 40)
        self.assertEqual(kwargs['tokenizer_revision'], 'a' * 40)
        self.assertEqual(kwargs['distributed_executor_backend'], 'uni')
        self.assertEqual(kwargs['tensor_parallel_size'], 1)
        self.assertEqual(kwargs['generation_config'], 'vllm')
        self.assertEqual(vllm.SamplingParams.call_args.kwargs['temperature'], 0)
        self.assertEqual(vllm.SamplingParams.call_args.kwargs['max_tokens'], 256)
        messages = tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(messages, [{'role': 'system', 'content': probe.SYSTEM_PROMPT},
                                    {'role': 'user', 'content': 'CURRENT TASK: use S1'}])
        self.assertEqual(result, [probe.ProbeResponse(r'\boxed{7}', 'length', 2)])
        self.assertFalse(hasattr(worker, 'optimizer'))
        engine.generate.reset_mock()
        tokenizer.apply_chat_template.return_value = list(range(16384))
        with patch.dict(sys.modules, {'vllm': vllm}), self.assertRaisesRegex(ValueError, 'rather than truncate'):
            worker.generate(['too long'])
        engine.generate.assert_not_called()

    def test_local_gate_and_metrics_share_weights_not_request_settings(self):
        engine = Mock()
        tokenizer = engine.get_tokenizer.return_value
        tokenizer.apply_chat_template.return_value = [1, 2]
        engine.generate.return_value = [SimpleNamespace(outputs=[
            SimpleNamespace(text=r'\boxed{7}', finish_reason='stop', token_ids=[7])])]
        vllm = SimpleNamespace(LLM=Mock(return_value=engine), SamplingParams=Mock())
        ray = SimpleNamespace(get_gpu_ids=lambda: [0])
        with patch.dict(sys.modules, {'vllm': vllm, 'ray': ray}), \
                patch.object(local_probe, 'check_local_probe_dependencies'):
            worker = local_probe.LocalProbeWorker(self.config())
            worker.generate(['gate input'], system_prompt='Gate system', max_new_tokens=512)
            worker.generate(['eval input'])
            tokenizer.apply_chat_template.return_value = list(range(16000))
            with self.assertRaisesRegex(ValueError, 'rather than truncate'):
                worker.generate(['gate too long'], max_new_tokens=512)
        vllm.LLM.assert_called_once()
        self.assertEqual([call.kwargs['max_tokens'] for call in vllm.SamplingParams.call_args_list], [512, 256])
        self.assertEqual([call.args[0][0]['content'] for call in tokenizer.apply_chat_template.call_args_list[:2]],
                         ['Gate system', probe.SYSTEM_PROMPT])
        self.assertEqual(worker.config['max_new_tokens'], 256)

    def test_local_client_forwards_gate_options_and_keeps_metrics_defaults(self):
        client = local_probe.LocalFrozenProbeClient(self.config(batch_size=1))
        client.actor = Mock()
        actor = client.actor
        ray = Mock()
        ray.get.return_value = [probe.ProbeResponse(r'\boxed{UNKNOWN}', 'stop', 3)]
        with patch.dict(sys.modules, {'ray': ray}):
            client.generate(['a', 'b'], system_prompt='Gate system', max_new_tokens=512)
            client.generate(['eval'])
        calls = actor.generate.remote.call_args_list
        self.assertEqual([call.kwargs['max_new_tokens'] for call in calls], [512, 512, 256])
        self.assertEqual([call.kwargs['system_prompt'] for call in calls],
                         ['Gate system', 'Gate system', probe.SYSTEM_PROMPT])
        self.assertIs(client.actor, actor)

    def test_training_finally_closes_probe_even_on_failure(self):
        source = ast.parse((PPO.parent / 'main_ppo.py').read_text())
        runner = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'TaskRunner')
        run = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        lifecycle = next(node for node in run.body if isinstance(node, ast.Try))
        code = compile(ast.Module(body=[lifecycle], type_ignores=[]), '<probe-cleanup>', 'exec')
        for failure in ('init_workers', 'fit', None):
            trainer = Mock()
            if failure:
                getattr(trainer, failure).side_effect = RuntimeError('test failure')
                with self.assertRaises(RuntimeError):
                    exec(code, {'trainer': trainer})
            else:
                exec(code, {'trainer': trainer})
            trainer.close_validation_probes.assert_called_once()


class GateBackendTests(unittest.TestCase):
    def trainer(self, backend='frozen_probe'):
        return SimpleNamespace(
            config=SimpleNamespace(trainer={}),
            prefix_probe_config={'solver_backend': backend, 'max_new_tokens': 512, 'system_prompt': 'Gate system'},
            _get_hierarchy_config=lambda: {'decomposer_role': 'decomposer'},
            validation_probe_client=Mock(),
        )

    def test_separate_checkpoint_preserves_evaluation_and_never_uses_trained_copy(self):
        trainer = self.trainer()
        trainer.prefix_probe_enabled = True
        trainer._get_hierarchy_config = lambda: {'terminal_worker_as_answer': True}
        with tempfile.TemporaryDirectory() as checkpoint:
            trainer.config.trainer = {
                'frozen_gate_probe': {'enable': True, 'model': checkpoint},
                'validation_probe': {'enable': True, 'backend': 'http',
                                     'base_url': 'http://metrics/v1', 'model': 'fixed-base-qwen'},
            }
            extract_method('_init_validation_probes')(trainer)
            gate, metrics = trainer.frozen_gate_probe_client, trainer.validation_probe_client
            self.assertIsInstance(gate, local_probe.LocalFrozenProbeClient)
            self.assertEqual(gate.model, checkpoint)
            self.assertEqual(metrics.model, 'fixed-base-qwen')
            self.assertIsNone(gate.actor)
            gate.generate = Mock(return_value=[probe.ProbeResponse(r'\boxed{7}', 'stop', 2)])
            metrics.generate = Mock(side_effect=AssertionError('Must not use metrics model for gates'))
            trainer.actor_rollout_wg = {'decomposer': Mock(side_effect=AssertionError('Trainable copy used'))}
            generate = extract_method('_generate_prefix_probe_responses')
            self.assertEqual(generate(trainer, ['task']), ([r'\boxed{7}'], [2], ['stop']))
            # Replacing/updating the live role does not replace the frozen client.
            trainer.actor_rollout_wg['decomposer'] = Mock()
            generate(trainer, ['task'])
            self.assertIs(trainer.frozen_gate_probe_client, gate)
            metrics.generate.assert_not_called()
            gate.generate.side_effect = RuntimeError('snapshot failed')
            with self.assertRaisesRegex(RuntimeError, 'snapshot failed'):
                generate(trainer, ['task'])
            trainer.frozen_gate_probe_client = None
            with self.assertRaisesRegex(RuntimeError, 'no evaluator fallback'):
                generate(trainer, ['task'])
            metrics.generate.assert_not_called()

    def test_metrics_snapshot_can_also_share_frozen_gates_with_one_client(self):
        trainer = self.trainer()
        trainer.prefix_probe_enabled = True
        trainer._get_hierarchy_config = lambda: {'terminal_worker_as_answer': True}
        with tempfile.TemporaryDirectory() as checkpoint:
            trainer.config.trainer = {'validation_probe': {
                'enable': True, 'backend': 'local', 'model': checkpoint,
                'checkpoint_source': 's3:bucket/step110/decomposer/huggingface',
            }}
            extract_method('_init_validation_probes')(trainer)
            client = trainer.validation_probe_client
            self.assertEqual(client.model, checkpoint)
            self.assertIsNone(trainer.frozen_gate_probe_client)
            self.assertEqual(client.config['checkpoint_source'], 's3:bucket/step110/decomposer/huggingface')
            client.generate = Mock(return_value=[probe.ProbeResponse(r'\boxed{7}', 'stop', 2)])
            self.assertEqual(extract_method('_generate_prefix_probe_responses')(trainer, ['task']),
                             ([r'\boxed{7}'], [2], ['stop']))
            client.generate.assert_called_once_with(['task'], system_prompt='Gate system', max_new_tokens=512)
            self.assertEqual(client.max_new_tokens, 256)
            client.close = Mock()
            extract_method('close_validation_probes')(trainer)
            client.close.assert_called_once()

    def test_separate_gate_works_without_metrics_and_rejects_bad_config(self):
        trainer = self.trainer()
        trainer.prefix_probe_enabled = True
        init = extract_method('_init_validation_probes')
        with tempfile.TemporaryDirectory() as checkpoint:
            trainer.config.trainer = {'frozen_gate_probe': {'enable': True, 'model': checkpoint}}
            init(trainer)
            self.assertIsNotNone(trainer.frozen_gate_probe_client)
            self.assertIsNone(trainer.validation_probe_client)
            gate_config = trainer.config.trainer['frozen_gate_probe']
            for changes, error in (({'model': 'unpinned-model'}, 'fixed 40-character'),
                                   ({'max_model_len': 512}, 'output budget'),
                                   ({'backend': 'http'}, 'dedicated local')):
                trainer.config.trainer['frozen_gate_probe'] = dict(gate_config, **changes)
                with self.assertRaisesRegex(ValueError, error):
                    init(trainer)
            trainer.config.trainer['frozen_gate_probe'] = gate_config
            trainer.prefix_probe_config['solver_backend'] = 'role'
            with self.assertRaisesRegex(ValueError, 'solver_backend=frozen_probe'):
                init(trainer)
            trainer.prefix_probe_enabled = False
            init(trainer)
            self.assertIsNone(trainer.frozen_gate_probe_client)

    def test_two_probes_capacity_startup_and_cleanup(self):
        trainer = self.trainer()
        trainer.validation_probe_client = local_probe.LocalFrozenProbeClient(
            {'model': 'Qwen/base', 'revision': 'a' * 40})
        trainer.frozen_gate_probe_client = local_probe.LocalFrozenProbeClient(
            {'model': 'Qwen/snapshot', 'revision': 'b' * 40})
        trainer.resource_pool_manager = Mock()
        trainer.resource_pool_manager.get_n_gpus.return_value = 12
        trainer.close_validation_probes = lambda: extract_method('close_validation_probes')(trainer)
        tree = ast.parse((PPO / 'ray_trainer.py').read_text())
        init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'init_workers')
        # Exercise the real preallocation and startup blocks without constructing training actors.
        capacity = ast.Module(body=init.body[1:4], type_ignores=[])
        startup = ast.Module(body=[init.body[-1]], type_ignores=[])
        ray = Mock()
        with patch.dict(sys.modules, {'ray': ray}), patch.object(local_probe, 'check_local_probe_dependencies'):
            ray.available_resources.return_value = {'GPU': 13}
            with self.assertRaisesRegex(RuntimeError, 'plus 2'):
                exec(compile(capacity, '<capacity>', 'exec'), {'self': trainer})
            trainer.resource_pool_manager.create_resource_pool.assert_not_called()
            ray.available_resources.return_value = {'GPU': 14}
            namespace = {'self': trainer}
            exec(compile(capacity, '<capacity>', 'exec'), namespace)
            trainer.resource_pool_manager.create_resource_pool.assert_called_once()
            for client in namespace['local_probes']:
                client.start, client.close = Mock(), Mock()
            trainer.frozen_gate_probe_client.start.side_effect = RuntimeError('startup failed')
            with self.assertRaisesRegex(RuntimeError, 'startup failed'):
                exec(compile(startup, '<startup>', 'exec'), namespace)
            trainer.validation_probe_client.close.assert_called_once()
            trainer.frozen_gate_probe_client.close.assert_called_once()

    def test_frozen_gate_uses_external_client_without_training_roles(self):
        trainer = self.trainer()
        trainer.validation_probe_client.generate.return_value = [probe.ProbeResponse(r'\boxed{7}', 'stop', 10)]
        generate = extract_method('_generate_prefix_probe_responses')
        self.assertEqual(generate(trainer, ['input']), ([r'\boxed{7}'], [10], ['stop']))
        trainer.validation_probe_client.generate.assert_called_once_with(
            ['input'], system_prompt='Gate system', max_new_tokens=512)
        self.assertEqual(generate(trainer, []), ([], [], []))
        trainer.validation_probe_client.generate.return_value = []
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            generate(trainer, ['input'])
        trainer.validation_probe_client.generate.side_effect = RuntimeError('endpoint failed')
        with self.assertRaisesRegex(RuntimeError, 'endpoint failed'):
            generate(trainer, ['input'])
        trainer.validation_probe_client = None
        with self.assertRaisesRegex(RuntimeError, 'refusing to fall back'):
            generate(trainer, ['input'])

    def test_role_gate_still_uses_current_decomposer_and_releases_generation_context(self):
        generate = extract_method('_generate_prefix_probe_responses',
            torch=SimpleNamespace(arange=lambda n: list(range(n))),
            DataProto=SimpleNamespace(from_dict=lambda x: x),
            pad_dataproto_to_divisor=lambda batch, divisor: (batch, 0))
        trainer = self.trainer('role')
        tokenizer, rollout = Mock(), Mock(world_size=1)
        tokenizer.batch_decode.return_value = [r'\boxed{7}']
        tokenizer.apply_chat_template.return_value = [[1, 2]]
        trainer._tokenizer_for_role = lambda role: tokenizer
        trainer.actor_rollout_wg = {'decomposer': rollout}
        trainer.role_tokenizers = {'decomposer': tokenizer}
        trainer.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(response_length=1024, prompt_length=8192)))
        trainer.multi_agent_rollout = Mock()
        trainer.multi_agent_rollout._generate_from_chat_list.return_value = ([''], [10], ['stop'], None, [[7]])
        self.assertEqual(generate(trainer, ['input']), ([r'\boxed{7}'], [10], ['stop']))
        call = trainer.multi_agent_rollout._generate_from_chat_list.call_args
        self.assertEqual(call.args[0], 'decomposer')
        self.assertEqual(call.args[1][0][0]['content'], 'Gate system')
        self.assertFalse(call.args[3]['do_sample'])
        self.assertEqual(call.kwargs['max_new_tokens'], 512)
        rollout.enter_generate_context.assert_called_once()
        rollout.exit_generate_context.assert_called_once()
        trainer.validation_probe_client.generate.assert_not_called()
        trainer.multi_agent_rollout._generate_from_chat_list.side_effect = RuntimeError('generation failed')
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            generate(trainer, ['input'])
        self.assertEqual(rollout.exit_generate_context.call_count, 2)
        tokenizer.apply_chat_template.return_value = [list(range(8193))]
        with self.assertRaisesRegex(ValueError, 'rather than truncate'):
            generate(trainer, ['long input'])
        self.assertEqual(rollout.enter_generate_context.call_count, 2)

    def test_frozen_gate_requires_enabled_metrics_and_sufficient_context(self):
        trainer = self.trainer()
        trainer.prefix_probe_enabled = True
        trainer.config = SimpleNamespace(trainer={})
        init = extract_method('_init_validation_probes')
        with self.assertRaisesRegex(ValueError, 'requires trainer.validation_probe.enable'):
            init(trainer)
        trainer._get_hierarchy_config = lambda: {'terminal_worker_as_answer': True}
        trainer.config.trainer = {'validation_probe': {'enable': True, 'backend': 'local', 'max_model_len': 512}}
        with self.assertRaisesRegex(ValueError, 'prefix-probe output budget'):
            init(trainer)

    def test_backend_validation_and_default_compatibility(self):
        init = extract_method('_init_prefix_probe')
        trainer = self.trainer()
        trainer.scoped_c3_grpo_config = {'prefix_probe': {'enable': False, 'solver_backend': 'typo'}}
        with self.assertRaisesRegex(ValueError, 'role or frozen_probe'):
            init(trainer)
        trainer.scoped_c3_grpo_config = {'prefix_probe': {'enable': True, 'solver_backend': 'frozen_probe'}}
        trainer.scoped_c3_grpo_enabled = True
        trainer._get_hierarchy_config = lambda: {'terminal_worker_as_answer': True, 'agent_roles': []}
        trainer.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout={'max_num_turns': 1}))
        with patch.dict(sys.modules, {'math_verify': SimpleNamespace(parse=Mock(), verify=Mock())}):
            init(trainer)
            trainer.scoped_c3_grpo_config['prefix_probe'].pop('solver_backend')
            with self.assertRaisesRegex(ValueError, 'solver_role'):
                init(trainer)


if __name__ == '__main__':
    unittest.main()

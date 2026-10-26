"""CPU tests of scheduling and trainer integration without Ray/CUDA dependencies."""

import ast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from threading import Event, get_ident
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
PPO = ROOT / 'src/verl/verl/rema_separated_trainer/ppo'


def load(name):
    spec = importlib.util.spec_from_file_location('_pipeline_test_' + name, PPO / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pipeline = load('rollout_pipeline')
context = load('rollout_context')


def response(messages):
    return [f'answer:{m}' for m in messages], [3] * len(messages), ['stop'] * len(messages)


class PipelineTests(unittest.TestCase):
    def test_validates_options(self):
        self.assertEqual(pipeline.pipeline_config(None), {'enable': False, 'groups_per_chunk': 12})
        for bad in (0, -1, True, 1.2, '12'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pipeline.pipeline_config({'groups_per_chunk': bad})
        with self.assertRaises(ValueError):
            pipeline.pipeline_config({'enable': 'false'})

    def test_complete_8_by_4_groups_and_uneven_tail(self):
        ids = [f'q{i}' for i in range(25) for _ in range(32)]
        slices = pipeline.complete_group_slices(ids, 32, 12)
        self.assertEqual([(s.start, s.stop) for s in slices], [(0, 384), (384, 768), (768, 800)])
        self.assertEqual([value for s in slices for value in ids[s]], ids)
        for s in slices:
            chunk = ids[s]
            self.assertTrue(all(chunk.count(uid) == 32 for uid in set(chunk)))

    def test_rejects_partial_interleaved_and_repeated_groups(self):
        for ids in ([], ['a'] * 31, ['a', 'b'] * 32, ['a'] * 64,
                    ['a'] * 32 + ['b'] * 32 + ['a'] * 32):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                pipeline.complete_group_slices(ids, 32, 12)

    def test_real_overlap_single_pending_and_barrier(self):
        main = get_ident()
        second_started, first_probe_started, last_probe_finished = Event(), Event(), Event()
        calls = []

        def generate(chunk):
            self.assertEqual(get_ident(), main)
            if chunk == 1:
                self.assertTrue(first_probe_started.wait(2))
                second_started.set()
            return chunk

        def prepare(chunk, output):
            self.assertEqual(get_ident(), main)
            return {str(chunk): str(output)}

        def probe(messages):
            self.assertNotEqual(get_ident(), main)
            calls.extend(messages)
            if messages == ['0']:
                first_probe_started.set()
                self.assertTrue(second_started.wait(2), 'probe must overlap next generation')
            else:
                last_probe_finished.set()
            return response(messages)

        cancel = Mock()
        outputs, cache, stats = pipeline.run_probe_pipeline([0, 1], generate, prepare, probe, cancel)
        self.assertEqual(outputs, [0, 1])
        self.assertEqual(calls, ['0', '1'])
        self.assertTrue(last_probe_finished.is_set())
        self.assertEqual(list(cache), ['0', '1'])
        self.assertEqual(stats['chunks'], 2)
        self.assertGreater(stats['overlap_seconds'], 0)
        cancel.assert_not_called()

    def test_deduplicates_across_chunks_and_preserves_context(self):
        probe = Mock(side_effect=response)
        requests = [
            {('same', 'gt1'): 'same'},
            {('same', 'gt1'): 'same', ('same', 'gt2'): 'same'},
            {('same', 'gt2'): 'same'},
        ]
        outputs, cache, stats = pipeline.run_probe_pipeline(
            range(3), lambda i: i, lambda i, _: requests[i], probe, Mock(),
        )
        self.assertEqual(outputs, [0, 1, 2])
        self.assertEqual(probe.call_count, 2)
        self.assertEqual(len(cache), 2)
        self.assertEqual(stats['unique_probe_requests'], 2)

    def test_slow_probe_applies_backpressure_before_third_generation(self):
        first_probe, second_generation, release_probe, third_generation = (Event() for _ in range(4))

        def generate(chunk):
            if chunk == 1:
                self.assertTrue(first_probe.wait(2))
                second_generation.set()
            if chunk == 2:
                third_generation.set()
            return chunk

        def probe(messages):
            if messages == ['0']:
                first_probe.set()
                self.assertTrue(release_probe.wait(2))
            return response(messages)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(pipeline.run_probe_pipeline, range(3), generate,
                                     lambda i, _: {str(i): str(i)}, probe, release_probe.set)
            try:
                self.assertTrue(second_generation.wait(2))
                self.assertFalse(third_generation.is_set())
            finally:
                release_probe.set()
            outputs, _, _ = future.result(timeout=3)
        self.assertEqual(outputs, [0, 1, 2])
        self.assertTrue(third_generation.is_set())

    def test_no_cache_survives_next_logical_batch(self):
        probe = Mock(side_effect=response)
        for _ in range(2):
            pipeline.run_probe_pipeline([0], lambda i: i, lambda *_: {'a': 'a'}, probe, Mock())
        self.assertEqual(probe.call_count, 2)

    def test_no_requests_and_invalid_response_are_not_dropped(self):
        probe = Mock(return_value=(['broken'], [256], ['length']))
        outputs, cache, _ = pipeline.run_probe_pipeline(
            [0, 1], lambda i: i, lambda i, _: {} if i == 0 else {'k': 'task'}, probe, Mock(),
        )
        self.assertEqual(outputs, [0, 1])
        self.assertEqual(cache, {'k': ('broken', 256, 'length')})

    def test_incomplete_or_failed_probe_aborts_not_zero_scores(self):
        for probe in (Mock(return_value=([], [], [])), Mock(side_effect=TimeoutError('inference timeout'))):
            cancel = Mock()
            with self.assertRaises((RuntimeError, TimeoutError)):
                pipeline.run_probe_pipeline([0], lambda i: i, lambda *_: {'k': 'task'}, probe, cancel)
            cancel.assert_called_once()

    def test_generation_failure_cancels_and_drains_running_probe(self):
        started, cancelled, finished = Event(), Event(), Event()

        def probe(messages):
            started.set()
            try:
                self.assertTrue(cancelled.wait(2))
                raise RuntimeError('probe cancelled')
            finally:
                finished.set()

        def generate(i):
            if i:
                self.assertTrue(started.wait(2))
                raise ValueError('generation failed')
            return i

        with self.assertRaisesRegex(ValueError, 'generation failed'):
            pipeline.run_probe_pipeline([0, 1], generate, lambda *_: {'k': 'task'}, probe, cancelled.set)
        self.assertTrue(finished.is_set())

    def test_prepare_failure_and_cancel_failure_preserve_original(self):
        with self.assertLogs(pipeline.__name__, level='WARNING'), self.assertRaisesRegex(ValueError, 'prepare'):
            pipeline.run_probe_pipeline([0], lambda i: i, Mock(side_effect=ValueError('prepare')),
                                        Mock(), Mock(side_effect=RuntimeError('cleanup')))

    def test_cache_lookup_is_ordered_and_never_falls_back(self):
        items = [('b',), ('a',), ('b',)]
        cache = {'a': ('A', 1, 'stop'), 'b': ('B', 2, 'length')}
        self.assertEqual(pipeline.cached_probe_responses(items, cache, lambda x: x),
                         (['B', 'A', 'B'], [2, 1, 2], ['length', 'stop', 'length']))
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            pipeline.cached_probe_responses([('c',)], cache, lambda x: x)


class FakeBatch:
    def __init__(self, ids, meta=None, **columns):
        self.non_tensor_batch = {'uid': np.asarray(ids, dtype=object), **columns}
        self.meta_info = meta or {}
        self.batch = list(range(len(ids)))

    def __len__(self):
        return len(self.non_tensor_batch['uid'])

    def __getitem__(self, selection):
        cols = {key: value[selection] for key, value in self.non_tensor_batch.items() if key != 'uid'}
        return FakeBatch(self.non_tensor_batch['uid'][selection], self.meta_info, **cols)

    @staticmethod
    def from_dict(data):
        return data

    @staticmethod
    def concat(batches):
        result = FakeBatch([uid for b in batches for uid in b.non_tensor_batch['uid']], batches[0].meta_info)
        for key in batches[0].non_tensor_batch:
            result.non_tensor_batch[key] = np.concatenate([b.non_tensor_batch[key] for b in batches])
        return result


class LocalClient:
    def __init__(self):
        self.close = Mock()


def trainer_methods(names, **bindings):
    tree = ast.parse((PPO / 'ray_trainer.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayReMASeparatedTrainer')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(deepcopy=deepcopy, DataProto=FakeBatch,
              torch=SimpleNamespace(arange=lambda *args: list(range(*args))),
              LocalFrozenProbeClient=LocalClient, generation_context=context.generation_context,
              pipeline_config=pipeline.pipeline_config, complete_group_slices=pipeline.complete_group_slices,
              run_probe_pipeline=pipeline.run_probe_pipeline)
    ns.update(bindings)
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<trainer>', 'exec'), ns)
    return type('Trainer', (), {name: ns[name] for name in names})


class TrainerPipelineTests(unittest.TestCase):
    def trainer(self, enabled=True):
        cls = trainer_methods(['_generate_training_rollouts', '_adopt_c3_focal_role'])
        trainer = cls()
        trainer.config = SimpleNamespace(
            trainer={'rollout_probe_pipeline': {'enable': enabled, 'groups_per_chunk': 1}},
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=32)),
        )
        trainer.scoped_c3_grpo_enabled = trainer.prefix_probe_enabled = True
        trainer.prefix_probe_config = {'solver_backend': 'frozen_probe'}
        trainer.validation_probe_client = LocalClient()
        trainer.frozen_gate_probe_client = None
        d, w = Mock(), Mock()
        trainer.actor_rollout_wg = {'decomposer': d, 'w1': w, 'w2': w}
        trainer.multi_turn_generate_sequences = Mock(return_value='synchronous')
        trainer.multi_agent_rollout = SimpleNamespace(generate=Mock(side_effect=lambda batch: batch))
        trainer._prefix_probe_context_key = lambda *args: args
        trainer._collect_prefix_probe_items = lambda metadata: (
            [], [(metadata['uid'][0], 'dataset', 'target', None)], [])
        trainer._generate_prefix_probe_responses = Mock(side_effect=response)
        batch = FakeBatch(['a'] * 32 + ['b'] * 32,
                          {'agent_roles': ['decomposer', 'w1', 'w2'], 'c3_focal_role': 'w1'})
        return trainer, batch

    def test_selected_role_is_used_for_probes_and_pinned_across_chunks(self):
        trainer, batch = self.trainer()
        trainer._current_train_agent = 'w1'
        trainer._get_train_agent_roles = lambda: ['w1', 'w2']
        batch.meta_info.update(c3_select_available_worker=True, c3_candidate_worker_roles=['w1', 'w2'])
        flags = []

        def generate(chunk):
            flags.append(chunk.meta_info['c3_select_available_worker'])
            chunk.meta_info.update(c3_focal_role='w2', c3_select_available_worker=False)
            return chunk

        def prepare(metadata):
            self.assertEqual(trainer._current_train_agent, 'w2')
            return [], [(metadata['uid'][0], 'dataset', 'target', None)], []

        trainer.multi_agent_rollout.generate.side_effect = generate
        trainer._collect_prefix_probe_items = prepare
        output, _ = trainer._generate_training_rollouts(batch, batch, {}, {})
        self.assertEqual(flags, [True, False])
        self.assertEqual(trainer._current_train_agent_idx, 1)
        self.assertEqual(output.meta_info['c3_focal_role'], 'w2')

    def test_sync_selection_handles_output_aliasing_input_metadata(self):
        trainer, batch = self.trainer(enabled=False)
        trainer._current_train_agent = 'w1'
        trainer._get_train_agent_roles = lambda: ['w1', 'w2']
        batch.meta_info.update(c3_select_available_worker=True, c3_candidate_worker_roles=['w1', 'w2'])

        def generate(chunk):
            chunk.meta_info.update(c3_focal_role='w2', c3_select_available_worker=False)
            return chunk

        trainer.multi_turn_generate_sequences.side_effect = generate
        trainer._generate_training_rollouts(batch, batch, {}, {})
        self.assertEqual(trainer._current_train_agent, 'w2')
        batch.meta_info['c3_focal_role'] = 'w1'
        with self.assertRaisesRegex(RuntimeError, 'pinned optimizer'):
            trainer._adopt_c3_focal_role(batch, batch)

    def test_one_pool_lifecycle_full_batch_order_and_cache(self):
        trainer, batch = self.trainer()
        metrics, timing = {}, {}
        output, cache = trainer._generate_training_rollouts(batch, batch, metrics, timing)
        self.assertEqual(list(output.non_tensor_batch['uid']), list(batch.non_tensor_batch['uid']))
        self.assertEqual(list(cache), [('a', 'dataset', 'target', None), ('b', 'dataset', 'target', None)])
        for role in ('decomposer', 'w1'):
            trainer.actor_rollout_wg[role].enter_generate_context.assert_called_once()
            trainer.actor_rollout_wg[role].exit_generate_context.assert_called_once()
        trainer.multi_turn_generate_sequences.assert_not_called()
        self.assertEqual(metrics['rollout/pipeline/enabled'], 1)
        self.assertEqual(metrics['rollout/pipeline/chunks'], 2)
        self.assertIn('pipeline_probe_wait', timing)
        # Mutating per-chunk metadata must never change the source batch.
        first = trainer.multi_agent_rollout.generate.call_args_list[0].args[0]
        first.meta_info['c3_focal_role'] = 'changed'
        self.assertEqual(batch.meta_info['c3_focal_role'], 'w1')

    def test_synchronous_fallbacks_leave_batch_unsplit(self):
        for reason in ('disabled', 'role', 'validation', 'no_c3', 'no_prefix', 'http'):
            with self.subTest(reason=reason):
                trainer, batch = self.trainer(enabled=reason != 'disabled')
                if reason == 'role':
                    trainer.prefix_probe_config['solver_backend'] = 'role'
                if reason == 'validation':
                    batch.meta_info['validate'] = True
                if reason == 'no_c3':
                    trainer.scoped_c3_grpo_enabled = False
                if reason == 'no_prefix':
                    trainer.prefix_probe_enabled = False
                if reason == 'http':
                    trainer.validation_probe_client = object()
                metrics = {}
                self.assertEqual(trainer._generate_training_rollouts(batch, batch, metrics, {}),
                                 ('synchronous', None))
                self.assertEqual(metrics['rollout/pipeline/enabled'], 0)
                trainer.multi_turn_generate_sequences.assert_called_once_with(batch)
                trainer._generate_prefix_probe_responses.assert_not_called()

    def test_separate_gate_client_is_required_no_metrics_fallback(self):
        trainer, batch = self.trainer()
        trainer.config.trainer['frozen_gate_probe'] = {'enable': True}
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            trainer._generate_training_rollouts(batch, batch, {}, {})
        trainer.multi_agent_rollout.generate.assert_not_called()

    def test_malformed_group_is_rejected_before_entering_pools(self):
        trainer, batch = self.trainer()
        batch.non_tensor_batch['uid'][0] = 'unexpected'
        with self.assertRaisesRegex(ValueError, 'complete UID'):
            trainer._generate_training_rollouts(batch, batch, {}, {})
        trainer.actor_rollout_wg['w1'].enter_generate_context.assert_not_called()

    def test_truncated_generated_chunk_is_not_scored_or_used(self):
        trainer, batch = self.trainer()
        trainer.multi_agent_rollout.generate.side_effect = lambda b: b[:1]
        with self.assertRaisesRegex(RuntimeError, 'incomplete chunk'):
            trainer._generate_training_rollouts(batch, batch, {}, {})
        trainer._generate_prefix_probe_responses.assert_not_called()
        trainer.actor_rollout_wg['w1'].exit_generate_context.assert_called_once()
        trainer.validation_probe_client.close.assert_called_once()

    def test_fit_consumes_cache_before_group_filter_and_update(self):
        source = (PPO / 'ray_trainer.py').read_text().split('    def fit(self):', 1)[1]
        self.assertLess(source.index('self._generate_training_rollouts('), source.index('self.reward_fn(new_batch)'))
        self.assertLess(source.index('prefetched_responses=prefetched_probe_responses'),
                        source.index('self._attach_scoped_c3_grpo_signals('))
        self.assertLess(source.index('self._attach_scoped_c3_grpo_signals('),
                        source.index('kept_prompt_uids = []'))
        self.assertEqual(source.count('num_gen_batches += 1'), 1)
        self.assertEqual(source.count('self.train_dataloader'), 1)


class ArrayTensor(np.ndarray):
    """Minimal CPU tensor API for exercising the real gate-attachment method."""
    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self.astype(float)

    def bool(self):
        return self.astype(bool)

    def numpy(self):
        return np.asarray(self)


def tensor(values, dtype=None):
    return np.asarray(values, dtype=dtype).view(ArrayTensor)


class GateEquivalenceTests(unittest.TestCase):
    def test_prefetch_preserves_gates_invalid_masks_metrics_and_scoring_contexts(self):
        local_results = load('local_results')
        module = ModuleType('_pipeline_real_prefix')
        sys.modules[module.__name__] = module
        tree = ast.parse((PPO / 'prefix_probe.py').read_text())
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module.endswith('.local_results'))]
        module.extract_complete_boxed_answer = local_results.extract_complete_boxed_answer
        module.extract_local_result = local_results.extract_local_result
        exec(compile(tree, '<prefix-probe>', 'exec'), module.__dict__)
        bindings = {name: getattr(module, name) for name in (
            'collect_prefix_probe_requests', 'answer_round_records', 'count_planned_subtasks',
            'apply_prefix_probe_gate', 'extract_complete_boxed_answer', 'select_console_probe_indices',
        )}
        cls = trainer_methods([
            '_collect_prefix_probe_items', '_prefix_probe_context_key',
            '_attach_prefix_probe_signals', '_record_leakage_metrics',
        ], np=np, json=json, defaultdict=defaultdict, cached_probe_responses=pipeline.cached_probe_responses,
            torch=SimpleNamespace(tensor=tensor, from_numpy=tensor, float32=float, bool=bool, long=int),
            **bindings)
        trainer = cls()
        trainer.prefix_probe_enabled = True
        trainer.prefix_probe_config = {'diagnostic_samples': 0}
        trainer._current_train_agent = 'worker_stage_1'
        trainer._get_hierarchy_config = lambda: {
            'decomposer_role': 'decomposer', 'stage_roles': ['worker_stage_1', 'worker_stage_2'],
        }
        trainer._print_leakage_summary = Mock()
        trainer._print_validation_leakage_examples = Mock()
        trainer._run_prefix_math_checks = lambda op, inputs, metrics, **kwargs: (
            [True] * len(inputs) if op == 'parse' else [False] * len(inputs))
        trainer.reward_fn = SimpleNamespace(num_examine=0,
            score_responses=Mock(side_effect=lambda sources, responses, targets, infos, **kw: [1.] * len(responses)))

        def infer(messages):
            return [r'\boxed{26}' if m == 'leak' else 'invalid' if m == 'invalid'
                    else r'\boxed{UNKNOWN}' for m in messages], [3] * len(messages), ['stop'] * len(messages)

        def make_batch():
            batch = FakeBatch(['a', 'b', 'c', 'd'])
            batch.batch = {}
            histories = np.empty(4, dtype=object)
            for i, instruction in enumerate(('clean', 'leak', 'invalid', 'clean')):
                histories[i] = [
                    dict(role='decomposer', content='PLAN:\n- S1: calculate\n- S2: finish',
                         executed=True, planned_subtask_count=2),
                    dict(role='worker_stage_1', content=r'LOCAL_RESULT: \boxed{6}', executed=True),
                    dict(role='worker_stage_2', content=r'LOCAL_RESULT: \boxed{26}', executed=True,
                         terminal_probe_input=instruction),
                ]
            batch.non_tensor_batch.update(
                history=histories, terminal_stage_role=np.array(['worker_stage_2'] * 4, dtype=object),
                data_source=np.array(['math'] * 4, dtype=object),
                reward_model=np.array([{'ground_truth': '26'}] * 4, dtype=object),
                extra_info=np.array([{'variant': 0}] * 3 + [{'variant': 1}], dtype=object),
            )
            return batch

        synchronous, prefetched = make_batch(), make_batch()
        sync_metrics, prefetch_metrics = {}, {}
        trainer._generate_prefix_probe_responses = Mock(side_effect=infer)
        trainer._attach_prefix_probe_signals(synchronous, tensor([1., 1., 1., 0.]), sync_metrics)
        # Deduplicate over two chunks exactly as the trainer pipeline does.
        def prepare(selection, output):
            _, items, _ = trainer._collect_prefix_probe_items(output.non_tensor_batch)
            return {trainer._prefix_probe_context_key(*item): item[0] for item in items}
        _, cache, _ = pipeline.run_probe_pipeline(
            [slice(0, 2), slice(2, 4)], lambda s: prefetched[s], prepare, infer, Mock(),
        )
        trainer._generate_prefix_probe_responses = Mock(side_effect=AssertionError('must use cache'))
        trainer._attach_prefix_probe_signals(prefetched, tensor([1., 1., 1., 0.]), prefetch_metrics,
                                             prefetched_responses=cache)
        self.assertEqual(sync_metrics, prefetch_metrics)
        self.assertEqual(set(synchronous.batch), set(prefetched.batch))
        for key in synchronous.batch:
            np.testing.assert_equal(synchronous.batch[key], prefetched.batch[key], err_msg=key)
        for key in synchronous.non_tensor_batch:
            if key.startswith('prefix_probe_'):
                if key == 'prefix_probe_worker_comparisons':
                    self.assertEqual(synchronous.non_tensor_batch[key].tolist(),
                                     prefetched.non_tensor_batch[key].tolist())
                else:
                    a, b = synchronous.non_tensor_batch[key], prefetched.non_tensor_batch[key]
                    if key.endswith('_score'):
                        a, b = a.astype(float), b.astype(float)
                    np.testing.assert_equal(a, b, err_msg=key)
        self.assertEqual(prefetched.non_tensor_batch['prefix_probe_rejection_reason'].tolist(),
                         ['eligible', 'terminal_recoverable', 'terminal_probe_invalid', 'eligible'])
        trainer._generate_prefix_probe_responses.assert_not_called()
        self.assertEqual(sync_metrics['reward/leakage/all/trajectory_count'], 4)


if __name__ == '__main__':
    unittest.main()

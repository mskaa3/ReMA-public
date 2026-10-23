"""CPU lifecycle and replica-scheduling tests, without Ray, CUDA or vLLM."""

import ast
import importlib
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[4]
PPO = ROOT / 'src/verl/verl/rema_separated_trainer/ppo'
PACKAGE = '_efficiency_tests'
package = ModuleType(PACKAGE)
package.__path__ = [str(PPO)]
sys.modules[PACKAGE] = package
context = importlib.import_module(PACKAGE + '.rollout_context')
probe = importlib.import_module(PACKAGE + '.local_validation_probe')


def extracted_methods(path, class_name, names, **bindings):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for method in methods:
        method.decorator_list = []
    ns = {'DataProto': object, **bindings}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), ns)
    return type('Extracted', (), {name: ns[name] for name in names})


class GenerationContextTests(unittest.TestCase):
    def test_aliases_enter_exit_once_and_body_runs_once(self):
        decomposer, worker, body = Mock(), Mock(), Mock()
        with context.generation_context([decomposer, worker, worker, worker], 'batch'):
            body()
        for group in (decomposer, worker):
            group.enter_generate_context.assert_called_once_with('batch')
            group.exit_generate_context.assert_called_once_with('batch')
        body.assert_called_once()

    def test_partial_entry_failure_cleans_up_all_attempted_groups(self):
        decomposer, worker = Mock(), Mock()
        worker.enter_generate_context.side_effect = RuntimeError('entry failed')
        with self.assertRaisesRegex(RuntimeError, 'entry failed'):
            with context.generation_context([decomposer, worker], 'batch'):
                self.fail('must not generate')
        for group in (decomposer, worker):
            group.exit_generate_context.assert_called_once()

    def test_cleanup_continues_and_preserves_original_generation_error(self):
        decomposer, worker = Mock(), Mock()
        worker.exit_generate_context.side_effect = ValueError('cleanup failed')
        with self.assertLogs(context.__name__, level='ERROR'), self.assertRaisesRegex(RuntimeError, 'body'):
            with context.generation_context([decomposer, worker], 'batch'):
                raise RuntimeError('body')
        decomposer.exit_generate_context.assert_called_once()

    def test_cleanup_error_is_not_silenced_after_success(self):
        group = Mock()
        group.exit_generate_context.side_effect = RuntimeError('cleanup failed')
        with self.assertLogs(context.__name__, level='ERROR'), self.assertRaisesRegex(RuntimeError, 'cleanup'):
            with context.generation_context([group], 'batch'):
                pass

    def test_pool_freezing_considers_all_training_roles_not_current_role(self):
        roles = ['decomposer', 'selector', 'w1', 'w2']
        for trained, expected in [(['w1', 'w2'], (True, False)),
                                  (['decomposer'], (False, True)),
                                  (['decomposer', 'w2'], (False, False)),
                                  (['selector'], (True, False))]:
            with self.subTest(trained=trained):
                self.assertEqual(context.frozen_pool_flags(roles, trained, 'decomposer'), expected)
        self.assertEqual(context.frozen_pool_flags(roles, [], 'decomposer', enabled=False), (False, False))

    def test_trainer_really_uses_deduplicated_context(self):
        trainer_cls = extracted_methods(PPO / 'ray_trainer.py', 'RayReMASeparatedTrainer',
            ['multi_turn_generate_sequences'], generation_context=context.generation_context,
            DataProto=SimpleNamespace(from_dict=lambda x: x), torch=SimpleNamespace(arange=lambda *x: x))
        trainer = trainer_cls()
        d, w = Mock(), Mock()
        trainer.actor_rollout_wg = {'decomposer': d, 'w1': w, 'w2': w}
        trainer.multi_agent_rollout = Mock()
        batch = SimpleNamespace(batch=[1, 2], meta_info={'agent_roles': ['decomposer', 'w1', 'w2']})
        trainer.multi_turn_generate_sequences(batch)
        w.enter_generate_context.assert_called_once()
        w.exit_generate_context.assert_called_once()
        trainer.multi_agent_rollout.generate.assert_called_once_with(batch)


class ResidentPoolTests(unittest.TestCase):
    def manager(self, resident=True):
        cuda = Mock()
        cuda.state = 'actor-rng'
        cuda.get_rng_state.side_effect = lambda: cuda.state
        cuda.set_rng_state.side_effect = lambda value: setattr(cuda, 'state', value)
        manager_cls = extracted_methods(
            ROOT / 'src/verl/verl/workers/sharding_manager/fsdp_vllm.py', 'FSDPVLLMShardingManager',
            ['__enter__', '__exit__', 'release_resident'],
            torch=SimpleNamespace(cuda=cuda), vllm_version='0.8.5', logger=logging.getLogger(__name__))
        manager = manager_cls()
        manager.keep_resident = resident
        manager._context_active = manager._weights_loaded = False
        manager._sync_weights = Mock()
        manager.device_mesh = object()
        manager.gen_random_states = 'generation-rng'
        manager.module = Mock()
        manager.inference_engine = Mock()
        return manager, cuda

    def test_frozen_pool_syncs_once_but_keeps_rng_scoped(self):
        manager, cuda = self.manager()
        for index in range(3):
            with manager:
                self.assertNotEqual(cuda.state, 'actor-rng')
                cuda.state = f'generated-{index}'
            self.assertEqual(cuda.state, 'actor-rng')
            self.assertEqual(manager.gen_random_states, f'generated-{index}')
        manager._sync_weights.assert_called_once()
        manager.inference_engine.sleep.assert_not_called()

    def test_trainable_pool_sleeps_and_resyncs_every_time(self):
        manager, _ = self.manager(resident=False)
        for _ in range(2):
            with manager:
                pass
        self.assertEqual(manager._sync_weights.call_count, 2)
        self.assertEqual(manager.inference_engine.sleep.call_count, 2)
        self.assertFalse(manager._weights_loaded)

    def test_checkpoint_invalidation_forces_next_sync(self):
        manager, _ = self.manager()
        with manager:
            pass
        manager.release_resident()
        manager.release_resident()
        manager.inference_engine.sleep.assert_called_once_with(level=1)
        with manager:
            pass
        self.assertEqual(manager._sync_weights.call_count, 2)

    def test_failed_generation_releases_resident_weights_and_restores_rng(self):
        manager, cuda = self.manager()
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            with manager:
                raise RuntimeError('generation failed')
        self.assertFalse(manager._weights_loaded)
        self.assertFalse(manager._context_active)
        self.assertEqual(cuda.state, 'actor-rng')

    def test_failed_weight_sync_does_not_leave_resident_state(self):
        manager, _ = self.manager()
        manager._sync_weights.side_effect = RuntimeError('sync failed')
        with self.assertRaisesRegex(RuntimeError, 'sync failed'):
            manager.__enter__()
        manager.inference_engine.sleep.assert_called_once()
        self.assertFalse(manager._context_active)
        self.assertFalse(manager._weights_loaded)

    def test_failed_sleep_still_restores_rng_and_can_be_retried(self):
        manager, cuda = self.manager(resident=False)
        manager.inference_engine.sleep.side_effect = RuntimeError('sleep failed')
        with self.assertRaisesRegex(RuntimeError, 'sleep failed'):
            with manager:
                pass
        self.assertFalse(manager._context_active)
        self.assertTrue(manager._weights_loaded)
        self.assertEqual(cuda.state, 'actor-rng')
        manager.inference_engine.sleep.side_effect = None
        manager.release_resident()
        self.assertFalse(manager._weights_loaded)

    def test_nested_entry_is_rejected_and_double_exit_is_safe(self):
        manager, _ = self.manager()
        with manager:
            with self.assertRaisesRegex(RuntimeError, 'already active'):
                manager.__enter__()
        manager.__exit__(None, None, None)

    def test_update_of_resident_pool_is_blocked(self):
        worker_cls = extracted_methods(ROOT / 'src/verl/verl/workers/fsdp_rema_workers.py',
            'ActorRolloutRefWorker', ['update_actor'])
        worker = worker_cls()
        worker.config = SimpleNamespace(rollout={'keep_frozen_resident': True})
        with self.assertRaisesRegex(RuntimeError, 'permanently frozen'):
            worker.update_actor(None)

    def test_checkpoint_io_releases_cache_before_touching_weights(self):
        events = []
        manager, _ = self.manager()
        with manager:
            pass
        manager.inference_engine.sleep.side_effect = lambda **kw: events.append('release')
        fake_torch = SimpleNamespace(distributed=SimpleNamespace(barrier=lambda: None))
        worker_cls = extracted_methods(ROOT / 'src/verl/verl/workers/fsdp_rema_workers.py',
            'ActorRolloutRefWorker', ['save_checkpoint', 'load_checkpoint'],
            load_fsdp_model_to_gpu=lambda _: events.append('load-fsdp'),
            offload_fsdp_model_to_cpu=lambda _: None, offload_fsdp_optimizer=lambda _: None)
        worker = worker_cls()
        worker._is_rollout = worker._is_actor = worker._is_offload_param = True
        worker._is_offload_optimizer = False
        worker.actor_module_fsdp = object()
        worker.rollout_sharding_manager = manager
        worker.checkpoint_manager = Mock()
        worker.checkpoint_manager.load_checkpoint.side_effect = lambda **kw: events.append('checkpoint-load')
        worker.checkpoint_manager.save_checkpoint.side_effect = lambda **kw: events.append('checkpoint-save')
        with patch.dict(sys.modules, {'torch': fake_torch}):
            worker.load_checkpoint('/checkpoint')
            self.assertEqual(events, ['release', 'load-fsdp', 'checkpoint-load'])
            with manager:
                pass
            events.clear()
            worker.save_checkpoint('/checkpoint')
            self.assertEqual(events, ['release', 'load-fsdp', 'checkpoint-save'])


class ReplicaTests(unittest.TestCase):
    def client(self, replicas=4, **kwargs):
        return probe.LocalFrozenProbeClient(dict(model='Qwen/fixed', revision='a' * 40,
                                                 replicas=replicas, batch_size=2, **kwargs))

    def test_replica_counts_are_validated_and_capacity_accounts_for_all(self):
        for count in (0, -1, True, 1.2, '4'):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.client(count)
        client = self.client()
        ray = SimpleNamespace(available_resources=lambda: {'GPU': 16})
        with patch.dict(sys.modules, {'ray': ray}), patch.object(probe, 'check_local_probe_dependencies'):
            client.check_capacity(12)
            with self.assertRaisesRegex(RuntimeError, '13 training GPUs plus 4'):
                client.check_capacity(13)
            with self.assertRaises(RuntimeError):
                client.check_capacity(12, probe_gpus=5)

    def test_out_of_order_completion_preserves_results_and_one_request_per_input(self):
        client = self.client(3)
        ray = Mock()
        outputs, submitted, queue_sizes = {}, [], []
        actors = [Mock() for _ in range(3)]
        client.actors = actors

        def submit(messages, **kw):
            ref = object()
            outputs[ref] = [probe.ProbeResponse(message, 'stop') for message in messages]
            submitted.extend(messages)
            self.assertEqual(kw, {'system_prompt': 'gate', 'max_new_tokens': 128})
            return ref

        for actor in actors:
            actor.generate.remote.side_effect = submit

        def wait(refs, **kw):
            queue_sizes.append(len(refs))
            self.assertLessEqual(len(refs), len(actors))
            return refs[-1:], refs[:-1]  # Deliberately complete the last request first.

        ray.wait.side_effect = wait
        ray.get.side_effect = lambda ref, **kw: outputs[ref]
        messages = [str(i) for i in range(13)]
        with patch.dict(sys.modules, {'ray': ray}):
            result = client.generate(messages, system_prompt='gate', max_new_tokens=128)
        self.assertEqual([row.text for row in result], messages)
        self.assertEqual(submitted, messages)
        self.assertEqual(queue_sizes[0], 3)
        self.assertTrue(all(actor.generate.remote.called for actor in actors))
        ray.kill.assert_not_called()

    def test_timeout_or_incomplete_response_kills_all_replicas(self):
        for failure in ('timeout', 'incomplete', 'actor-crash'):
            with self.subTest(failure=failure):
                client, ray = self.client(2), Mock()
                client.actors = [Mock(), Mock()]
                ray.wait.side_effect = lambda refs, **kw: ([], refs) if failure == 'timeout' else (refs[:1], refs[1:])
                ray.get.return_value = []
                if failure == 'actor-crash':
                    ray.get.side_effect = RuntimeError('actor died')
                with patch.dict(sys.modules, {'ray': ray}), self.assertRaises((RuntimeError, TimeoutError)):
                    client.generate(list('abcdef'))
                self.assertEqual(ray.kill.call_count, 2)
                self.assertEqual(client.actors, [])

    def test_startup_failure_closes_every_replica(self):
        client, ray = self.client(4), Mock()
        ray.available_resources.return_value = {'GPU': 4}
        actors = [Mock() for _ in range(4)]
        ray.remote.return_value.return_value.options.return_value.remote.side_effect = actors
        ray.get.side_effect = TimeoutError('startup')
        with patch.dict(sys.modules, {'ray': ray}), self.assertRaises(TimeoutError):
            client.start()
        self.assertEqual(client.actors, [])
        self.assertEqual(ray.kill.call_count, 4)


if __name__ == '__main__':
    unittest.main()

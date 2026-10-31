"""CPU tests of the real scheduling, masking and audit methods without Ray/CUDA.

NumPy stands in for tensor storage here, not for the C3 estimator. Torch-level
advantage invariance and loss-mask checks live in test_c3_update_policy.py.
"""

import ast
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
PPO = ROOT / 'src/verl/verl/rema_separated_trainer/ppo'
HELPERS = runpy.run_path(str(PPO / 'c3_role_selection.py'))
CreditAudit = runpy.run_path(str(PPO / 'credit_audit.py'))['CreditAudit']


def trainer():
    names = ('_begin_c3_worker_attempt', '_attach_c3_worker_functions', '_progress_metrics',
             '_record_worker_function_metrics', '_save_training_progress', '_load_training_progress',
             '_init_scoped_c3_grpo')
    cls = next(n for n in ast.parse((PPO / 'ray_trainer.py').read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'RayReMASeparatedTrainer')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(HELPERS, json=json, Path=Path,
                     OmegaConf=SimpleNamespace(to_container=lambda value, resolve: value),
                     AdvantageEstimator=SimpleNamespace(GRPO='grpo'),
                     torch=SimpleNamespace(tensor=lambda v, dtype: np.asarray(v, dtype=dtype),
                                           long=np.int64, ones_like=np.ones_like))
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<trainer>', 'exec'), namespace)
    result = type('Trainer', (), {name: namespace[name] for name in names})()
    result.scoped_c3_grpo_config = {'balance_worker_functions': True}
    result._current_train_agent = 'w2'
    result._get_hierarchy_config = lambda: {'terminal_worker_as_answer': True,
                                          'stage_roles': ['w1', 'w2', 'w3']}
    result.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(
        rollout=SimpleNamespace(max_num_turns=1)))
    result.global_steps = 5
    result.actor_update_steps = 0
    result.actor_updates_by_role = {}
    result.actor_update_count_origin_step = 0
    return result


class Batch:
    def __init__(self, uids, terminal_roles):
        self.non_tensor_batch = {'uid': uids, 'terminal_stage_role': terminal_roles}
        self.batch = {}
        self.meta_info = {}

    def __len__(self):
        return len(self.non_tensor_batch['uid'])


class WorkerFunctionBalanceTests(unittest.TestCase):
    def test_configuration_rejects_incompatible_or_unbounded_modes(self):
        class Config(dict):
            __getattr__ = dict.__getitem__

        for setting, bad in (('valid', None), ('enable', False), ('select_available_worker', False),
                             ('terminal_worker_as_answer', False), ('routing_mode', 'derive_verify'),
                             ('max_num_turns', 2), ('max_num_gen_batches', 0),
                             ('skip_update_on_zero_trainable', False)):
            with self.subTest(setting=setting):
                t = trainer()
                c3 = dict(enable=True, balance_worker_functions=True, select_available_worker=True)
                hierarchy = dict(enable=True, scoped_c3_grpo=c3, stage_roles=['w1', 'w2'],
                                 terminal_worker_as_answer=True, routing_mode='sequential_plan')
                filtering = dict(enable=True, max_num_gen_batches=8, skip_update_on_zero_trainable=True)
                rollout = Config(n=32, max_num_turns=1)
                target = (c3 if setting in c3 else hierarchy if setting in hierarchy else
                          rollout if setting in rollout else filtering)
                if setting != 'valid':
                    target[setting] = bad
                t.config = SimpleNamespace(algorithm=Config(hierarchy=hierarchy, adv_estimator='grpo',
                                                           filter_groups=filtering),
                                           actor_rollout_ref=SimpleNamespace(rollout=rollout))
                if setting == 'valid':
                    t._init_scoped_c3_grpo()
                    self.assertTrue(t.scoped_c3_grpo_enabled)
                else:
                    with self.assertRaisesRegex(ValueError, 'balance_worker_functions'):
                        t._init_scoped_c3_grpo()

    def test_skips_still_advance_function_without_advancing_actor_counter(self):
        t = trainer()
        seen = []
        for _ in range(4):
            t._begin_c3_worker_attempt()
            seen.append(t._c3_worker_function)
        self.assertEqual(seen, ['nonterminal', 'terminal'] * 2)
        self.assertEqual(t.worker_function_attempts, 4)
        self.assertEqual(t.actor_update_steps, 0)

    def test_decomposer_and_disabled_mode_do_not_consume_worker_attempts(self):
        t = trainer()
        t._begin_c3_worker_attempt()
        t._current_train_agent = 'decomposer'
        t._begin_c3_worker_attempt()
        self.assertIsNone(t._c3_worker_function)
        t._current_train_agent = 'w2'
        t.scoped_c3_grpo_config['balance_worker_functions'] = False
        t._begin_c3_worker_attempt()
        self.assertIsNone(t._c3_worker_function)
        self.assertEqual(t.worker_function_attempts, 1)

    def test_resume_keeps_next_function_and_cumulative_exposure(self):
        t = trainer()
        t._begin_c3_worker_attempt()
        t._record_worker_function_metrics({'train/credit/functions/nonterminal/used_tokens': 12.,
                                          'unrelated_metric': 9.})
        with tempfile.TemporaryDirectory() as path:
            t._save_training_progress(path)
            resumed = trainer()
            resumed._load_training_progress(path)
        resumed._begin_c3_worker_attempt()
        self.assertEqual(resumed._c3_worker_function, 'terminal')
        self.assertEqual(resumed._progress_metrics()['train/worker_function/totals/nonterminal/used_tokens'], 12.)
        self.assertNotIn('unrelated_metric', resumed.worker_function_totals)

    def test_older_checkpoint_starts_new_function_counters(self):
        t = trainer()
        with tempfile.TemporaryDirectory() as path:
            t._save_training_progress(path)
            file = Path(path) / 'training_progress.json'
            data = json.loads(file.read_text())
            del data['worker_function_attempts'], data['worker_function_totals']
            file.write_text(json.dumps(data))
            t._load_training_progress(path)
        t._begin_c3_worker_attempt()
        self.assertEqual(t._c3_worker_function, 'nonterminal')
        self.assertEqual(t.worker_function_totals, {})

    def test_retries_use_same_function_and_mask_whole_uids(self):
        t = trainer()
        t._begin_c3_worker_attempt()
        for _ in range(3):
            batch = Batch(['a'] * 32 + ['b'] * 32, ['w2'] * 32 + ['w3'] * 32)
            batch.meta_info['c3_worker_function'] = t._c3_worker_function
            selected = t._attach_c3_worker_functions(batch, 'w2', np.ones(64, dtype=bool))
            self.assertEqual(selected.tolist(), [False] * 32 + [True] * 32)
            self.assertEqual(batch.batch['scoped_c3_worker_function_id'].tolist(), [1] * 32 + [0] * 32)
        self.assertEqual(t.worker_function_attempts, 1)

    def test_disabled_mode_only_observes_no_masks_changed(self):
        t = trainer()
        t.scoped_c3_grpo_config['balance_worker_functions'] = False
        batch = Batch(['a', 'b', 'c'], ['w2', 'w3', ''])
        selected = t._attach_c3_worker_functions(batch, 'w2', np.array([True, True, False]))
        self.assertEqual(selected.tolist(), [True] * 3)
        self.assertEqual(batch.batch['scoped_c3_worker_function_id'].tolist(), [1, 0, -1])

    def test_missing_or_changed_selection_fails_closed(self):
        t = trainer()
        t._begin_c3_worker_attempt()
        batch = Batch(['a'], ['w2'])
        for target in (None, 'terminal'):
            batch.meta_info['c3_worker_function'] = target
            with self.assertRaisesRegex(ValueError, 'optimizer attempt|preselected'):
                t._attach_c3_worker_functions(batch, 'w2', np.array([True]))

    def test_real_launcher_forwards_flag_without_changing_default(self):
        wrapper = (ROOT / 'agent2-only-rl-trainer-multinode.sh').read_text()
        launcher = (ROOT / 'agent12-curriculum-trainer-multinode.sh').read_text()
        wrapper_exports = wrapper[wrapper.index('export TRAIN_DECOMPOSER='):wrapper.index('exec bash')]
        exports = launcher[:launcher.index('\nif (( GPUS_PER_NODE')].replace('source ./env.sh', ':')
        training = launcher[launcher.index('run_agent12_training() {'):
                            launcher.index('\nif [[ "$ONLINE_TEACHER_GENERATION"')]
        start = launcher.rindex('\nif [[ "$AGENT2_PILOT"')
        pilot = launcher[start:launcher.index('\nstart_ray', start)]
        shell = '\n'.join([
            'set -euo pipefail', wrapper_exports, exports,
            'srun() { printf "%s\\n" "${!#}" > "$CAPTURE"; }',
            'HEAD_NODE=head; IP_HEAD=head:6379; COMMON_MOUNTS=(--nv)', training, pilot,
            'run_agent12_training /tmp/test.train.parquet 48 True False',
        ])
        with tempfile.TemporaryDirectory() as path:
            capture = Path(path) / 'command.txt'
            env = dict(PATH=os.environ['PATH'], SLURM_JOB_ID='123', SLURM_NNODES='6',
                       CAPTURE=str(capture), AGENT2_PILOT='false')
            for enabled in (None, 'true'):
                if enabled:
                    env['C3_BALANCE_WORKER_FUNCTIONS'] = enabled
                subprocess.run(['bash', '-c', shell], env=env, check=True, capture_output=True, text=True)
                args = shlex.split(capture.read_text().rsplit(';', 1)[-1])
                self.assertIn('algorithm.hierarchy.scoped_c3_grpo.balance_worker_functions='
                              + (enabled or 'false'), args)
                self.assertIn('algorithm.hierarchy.scoped_c3_grpo.continuations_per_action=4', args)
                self.assertIn('actor_rollout_ref.rollout.n=32', args)


class Tensor(np.ndarray):
    """Only the .ne method used by the audit needs a NumPy adapter."""
    def ne(self, value):
        return np.asarray(self) != value


class WorkerFunctionAuditTests(unittest.TestCase):
    def sample(self, uid, function, selected=True):
        batch = Batch([uid] * 8, ['w2'] * 8)
        batch.non_tensor_batch.update(c3_action_index=[0] * 4 + [1] * 4,
                                      c3_suffix_index=list(range(4)) * 2,
                                      w2_action_token_ids=[[1, 2, 3]] * 8)
        representative = np.array([True, False, False, False] * 2)
        advantage = np.array([1., 0., 0., 0., -1., 0., 0., 0.]) * selected
        effective = representative & selected
        if function == 0:
            effective &= advantage > 0
        labels = np.ones((8, 3), dtype=int).view(Tensor)
        labels[~effective] = -100
        batch.batch.update({
            'scoped_c3_raw_outcome_score': np.array([1.] * 4 + [0.] * 4),
            'scoped_c3_outcome_score': np.array([1.] * 4 + [0.] * 4),
            'scoped_c3_positive_only_mask': np.full(8, function == 0),
            'scoped_c3_worker_function_id': np.full(8, function),
            'scoped_c3_function_selected': np.full(8, selected),
            'scoped_c3_action_representative': representative,
            'labels': labels, 'step_ids': labels.copy(),
            'advantages': np.repeat(advantage[:, None], 3, axis=1),
        })
        estimate = SimpleNamespace(advantage=advantage, effective_mask=effective,
                                   eligible_success_group_mask=np.ones(8, dtype=bool))
        return batch, estimate

    def test_counts_distinct_actions_and_real_updates_not_replica_rows(self):
        with tempfile.TemporaryDirectory() as path:
            audit = CreditAudit(path, console_examples=0)
            first, estimate = self.sample('local', 0)
            audit.capture(first, estimate, step=1, role='w2')
            second, estimate = self.sample('terminal', 1)
            audit.capture(second, estimate, step=1, role='w2')
            # Only second reaches the optimizer; first must not get update/token credit.
            metrics = audit.finish(second, actor_updated=True, actor_update_step=1)
        for function in ('nonterminal', 'terminal'):
            prefix = f'train/credit/functions/{function}'
            self.assertEqual(metrics[f'{prefix}/generated_actions'], 2)
            self.assertEqual(metrics[f'{prefix}/generated_tokens'], 6)
            self.assertEqual(metrics[f'{prefix}/generated_groups'], 1)
        prefix = 'train/credit/functions/terminal'
        self.assertEqual(metrics[f'{prefix}/used_positive_actions'], 1)
        self.assertEqual(metrics[f'{prefix}/used_negative_actions'], 1)
        self.assertEqual(metrics[f'{prefix}/used_tokens'], 6)
        self.assertEqual(metrics[f'{prefix}/actor_update_count'], 1)
        self.assertEqual(metrics['train/credit/functions/nonterminal/used_tokens'], 0)

    def test_wrong_function_is_not_logged_as_gate_failure_or_zero_contrast(self):
        with tempfile.TemporaryDirectory() as path:
            audit = CreditAudit(path, console_examples=0)
            batch, estimate = self.sample('q', 0, selected=False)
            batch.meta_info['c3_worker_function'] = 'terminal'
            audit.capture(batch, estimate, step=1, role='w2')
            metrics = audit.finish(batch, actor_updated=False, actor_update_step=0)
        self.assertEqual(metrics['train/credit/status/outside_selected_worker_function'], 2)
        self.assertEqual(metrics['train/credit/functions/nonterminal/sampled_actions'], 0)
        self.assertEqual(metrics['train/credit/functions/nonterminal/zero_advantage_actions'], 0)
        self.assertEqual(metrics['train/credit/functions/terminal/attempt_count'], 1)
        self.assertEqual(metrics['train/credit/functions/terminal/actor_update_count'], 0)

    def test_gate_and_unknown_counters_are_separate_from_actual_update(self):
        with tempfile.TemporaryDirectory() as path:
            audit = CreditAudit(path, console_examples=0)
            batch, estimate = self.sample('q', 1)
            batch.batch['prefix_probe_gate_valid'] = np.array([True, False, True, True] * 2)
            audit.capture(batch, estimate, step=1, role='w2')
            metrics = audit.finish(batch, actor_updated=False, actor_update_step=0)
        prefix = 'train/credit/functions/terminal'
        self.assertEqual(metrics[f'{prefix}/probe_unknown_actions'], 2)
        self.assertEqual(metrics[f'{prefix}/gate_rejected_actions'], 2)
        self.assertEqual(metrics[f'{prefix}/used_tokens'], 0)
        self.assertEqual(metrics[f'{prefix}/actor_update_count'], 0)


if __name__ == '__main__':
    unittest.main()

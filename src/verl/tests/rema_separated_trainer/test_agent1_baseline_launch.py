"""CPU-only launch contract tests; no cluster credentials, Ray, or torch needed."""

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
BASELINE = ROOT / 'agent1-baseline-rl-trainer-multinode.sh'
AGENT1 = ROOT / 'agent1-only-rl-trainer-multinode.sh'
COMMON = (ROOT / 'agent12-curriculum-trainer-multinode.sh').read_text()


class BaselineLaunchTests(unittest.TestCase):
    def launch(self, baseline=True, extra=None, worker=True):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / AGENT1.name).write_text(AGENT1.read_text())
            defaults = COMMON[:COMMON.index('\nif (( GPUS_PER_NODE')].replace('source ./env.sh', ':')
            function = COMMON[COMMON.index('run_agent12_training() {'):COMMON.index('\nif [[ "$ONLINE_TEACHER_GENERATION"')]
            (directory / 'agent12-curriculum-trainer-multinode.sh').write_text(
                defaults + '\nsrun() { printf "%s\\n" "${!#}" > "$CAPTURE"; }\n'
                'HEAD_NODE=head; IP_HEAD=head:6379; COMMON_MOUNTS=(--nv)\n'
                + function + '\nrun_agent12_training /tmp/train.parquet "$TOTAL_STEPS" True True\n'
            )
            capture = directory / 'command.txt'
            env = {'PATH': os.environ['PATH'], 'SLURM_JOB_ID': '123', 'SLURM_NNODES': '6',
                   'SLURM_SUBMIT_DIR': str(directory), 'CAPTURE': str(capture)}
            if worker:
                env['WORKER_MODEL_REMOTE'] = 's3:bucket/fixed-workers/huggingface'
            env.update(extra or {})
            result = subprocess.run(['bash', str(BASELINE if baseline else AGENT1)],
                                    env=env, capture_output=True, text=True)
            overrides = {}
            if capture.exists():
                command = shlex.split(capture.read_text().rsplit(';', 1)[-1])
                overrides = dict(arg.split('=', 1) for arg in command if '=' in arg and not arg.startswith('--'))
            return result, overrides

    def test_only_training_eligibility_and_run_identity_differ(self):
        baseline, left = self.launch()
        eac3, right = self.launch(baseline=False)
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(eac3.returncode, 0, eac3.stderr)
        differing = {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}
        self.assertEqual(differing, {
            'algorithm.hierarchy.scoped_c3_grpo.prefix_probe.apply_to_training',
            'algorithm.hierarchy.scoped_c3_grpo.require_eligible_success',
            'trainer.experiment_name', 'trainer.wandb_run_id',
        })
        self.assertEqual(left['algorithm.hierarchy.scoped_c3_grpo.prefix_probe.enable'], 'true')
        self.assertEqual(left['algorithm.hierarchy.scoped_c3_grpo.prefix_probe.apply_to_training'], 'false')
        self.assertEqual(right['algorithm.hierarchy.scoped_c3_grpo.prefix_probe.apply_to_training'], 'true')
        self.assertEqual(left['algorithm.hierarchy.train_agent_roles'], '[decomposer]')
        self.assertEqual(left['algorithm.hierarchy.scoped_c3_grpo.continuations_per_action'], '4')
        self.assertEqual(left['actor_rollout_ref.rollout.n'], '32')
        self.assertEqual(left['trainer.save_freq'], '50')
        self.assertEqual(left['trainer.test_freq'], '10')
        self.assertEqual(left['trainer.default_local_dir'], '/mnt/lscratch/slurm/123/agent12/checkpoints/agent12')

    def test_baseline_overrides_stale_mode_and_wandb_identity(self):
        result, args = self.launch(extra={
            'AGENT1_ELIGIBILITY_MODE': 'eac3', 'PREFIX_PROBE_APPLY_TO_TRAINING': 'true',
            'C3_REQUIRE_ELIGIBLE_SUCCESS': 'true', 'AGENT12_RUN_NAME': 'old-eac3',
            'AGENT12_WANDB_RUN_ID': 'old-eac3', 'TRAIN_AGENT_ROLES': '[worker_stage_1]',
        })
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args['trainer.wandb_run_id'], 'agent1-baseline-123')
        self.assertEqual(args['algorithm.hierarchy.scoped_c3_grpo.prefix_probe.apply_to_training'], 'false')
        self.assertEqual(args['algorithm.hierarchy.scoped_c3_grpo.require_eligible_success'], 'false')
        self.assertIn('Agent 1 objective: baseline', result.stdout)

    def test_missing_worker_and_foreign_experiment_overrides_fail(self):
        result, args = self.launch(worker=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Set WORKER_MODEL_REMOTE', result.stderr)
        self.assertFalse(args)
        for variable in ('AGENT12_EXPERIMENT_OVERRIDES', 'EAC3_RUN_FILE'):
            result, args = self.launch(extra={variable: 'foreign-experiment'})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Unset AGENT12_EXPERIMENT_OVERRIDES', result.stderr)
            self.assertFalse(args)

    def test_custom_baseline_name_and_invalid_mode(self):
        result, args = self.launch(extra={'BASELINE_RUN_NAME': 'baseline-seed11'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args['trainer.wandb_run_id'], 'baseline-seed11')
        result, args = self.launch(baseline=False, extra={'AGENT1_ELIGIBILITY_MODE': 'invalid'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must be eac3 or baseline', result.stderr)
        self.assertFalse(args)

    def test_gate_switch_defaults_and_reward_wiring(self):
        path = ROOT / 'src/verl/verl/rema_separated_trainer/ppo/ray_trainer.py'
        tree = ast.parse(path.read_text())
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                       and node.name == 'RayReMASeparatedTrainer')
        methods = {node.name: node for node in trainer.body if isinstance(node, ast.FunctionDef)}
        namespace = {}
        exec(compile(ast.Module(body=[methods['_prefix_probe_gates_training']], type_ignores=[]),
                     str(path), 'exec'), namespace)
        for enabled, config, expected in [
            (True, {}, True), (False, {}, False),
            (True, {'prefix_probe': {'apply_to_training': False}}, False),
            (True, {'prefix_probe': {'apply_to_training': True}}, True),
        ]:
            fake = SimpleNamespace(prefix_probe_enabled=enabled, scoped_c3_grpo_config=config)
            self.assertEqual(namespace['_prefix_probe_gates_training'](fake), expected)
        attach = ast.unparse(methods['_attach_scoped_c3_grpo_signals'])
        self.assertIn('apply_probe_gate = self._prefix_probe_gates_training()', attach)
        self.assertEqual(attach.count('if apply_probe_gate'), 2)
        self.assertNotIn('if self.prefix_probe_enabled', attach)
        # Measurement itself is still switched by enable, not apply_to_training.
        probes = ast.unparse(methods['_attach_prefix_probe_signals'])
        self.assertIn('if not self.prefix_probe_enabled', probes)
        self.assertNotIn('if not self._prefix_probe_gates_training()', probes)

    def test_diagnostic_gate_failure_is_not_reported_as_training_rejection(self):
        audit_type = runpy.run_path(str(
            ROOT / 'src/verl/verl/rema_separated_trainer/ppo/credit_audit.py'
        ))['CreditAudit']

        class Batch:
            def __len__(self):
                return len(self.non_tensor_batch['uid'])

        batch = Batch()
        batch.non_tensor_batch = {'uid': np.array(['q'], dtype=object)}
        batch.batch = {
            'scoped_c3_raw_outcome_score': np.array([1.]),
            'scoped_c3_outcome_score': np.array([1.]),
            'scoped_c3_positive_only_mask': np.array([False]),
            'scoped_c3_eligibility_applied': np.array([False]),
            'prefix_probe_gate_valid': np.array([False]),
            'prefix_probe_collaboration_eligible': np.array([False]),
        }
        estimate = SimpleNamespace(advantage=np.array([0.]), effective_mask=np.array([False]),
                                   eligible_success_group_mask=np.array([False]))
        with tempfile.TemporaryDirectory() as tmp:
            audit = audit_type(tmp, console_examples=0)
            audit.capture(batch, estimate, step=1, role='decomposer')
            # Nothing was selected for optimization because C3 had no contrast.
            audit.finish(SimpleNamespace(non_tensor_batch={}), actor_updated=False, actor_update_step=0)
            events = [json.loads(line) for line in (Path(tmp) / 'train_step_1.jsonl').read_text().splitlines()]
            event = events[-1]
            self.assertFalse(event['eligibility_applied_to_training'])
            self.assertFalse(event['all_suffix_gates_pass'])
            self.assertEqual(event['action_reward'], 1.)
            self.assertEqual(event['status'], 'no_effective_c3_contrast')


if __name__ == '__main__':
    unittest.main()

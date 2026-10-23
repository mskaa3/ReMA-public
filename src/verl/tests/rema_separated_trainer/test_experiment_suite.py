"""CPU-only experiment contract tests; runnable with unittest, without Ray/CUDA."""

import ast
from contextlib import nullcontext
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unicodedata
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import eac3_experiments as suite
import analyze_eac3_experiments as analysis
import experiment_io as evidence
from local_results import extract_local_result


def config():
    result = suite.read_json(ROOT / 'config/eac3-experiments.json')
    result.update(planners={'baseline': 's3:bucket/baseline/huggingface', 'eac3': 's3:bucket/eac3/huggingface'},
                  train_remote='s3:bucket/train.parquet', eval_remote='s3:bucket/eval.parquet',
                  common_plan_remote='s3:bucket/plans.parquet')
    return result


def frame(question):
    return pd.DataFrame([dict(question=question, data_source='ReMA-math', subset='math',
                             reward_model={'style': 'rule', 'ground_truth': '7'})])


def repository_config():
    result = config()
    result.pop('train_remote')
    result.pop('eval_remote')
    result.update(data_source='repository', train_file='data/MATH/train_lv3to5_8k.parquet',
                  eval_file='data/overall_math/test.parquet')
    return result


def record(question='q', plan='PLAN:\n- S1: Compute x.\n- S2: Use S1.', score=1., item=None):
    row = dict(question=question, question_id=evidence.question_id(question),
               plan_id=evidence.plan_id(plan), decomposer_output=plan, raw_score=score,
               subset='math', data_source='ReMA-math', ground_truth='7', completion_tokens=20,
               terminal_role='worker_stage_2', history=[{'role': 'decomposer', 'content': plan}])
    if item:
        row.update(eval_item_id=item, answer_free_reviewed=True)
    return row


class ExperimentTests(unittest.TestCase):
    def test_probe_replicas_reserve_enough_nodes_and_export_batch_settings(self):
        cfg = config()
        cfg['nodes'] = 3
        cfg['gate_solver_backend'] = 'frozen_probe'
        cfg['local_validation_probe'] = {'model': 's3:bucket/frozen', 'replicas': 4, 'batch_size': 8}
        run = suite.make_runs(cfg, 'workers')[0]
        env = suite.run_environment(run, check_allocation=False)
        self.assertEqual(suite.allocated_nodes(cfg), 4)
        self.assertEqual(env['VALIDATION_PROBE_REPLICAS'], 4)
        self.assertEqual(env['VALIDATION_PROBE_BATCH_SIZE'], 8)
        self.assertEqual(env['KEEP_FROZEN_MODELS_RESIDENT'], 'true')
        self.assertEqual(env['ROLLOUT_N'], cfg['rollout_n'])
        self.assertEqual(env['C3_CONTINUATIONS_PER_ACTION'], cfg['continuations_per_action'])
        cfg['frozen_gate_probe'] = {'model': 's3:bucket/other-frozen', 'replicas': 2}
        run = suite.make_runs(cfg, 'workers')[0]
        self.assertEqual(suite.allocated_nodes(cfg), 5)
        self.assertEqual(suite.run_environment(run, check_allocation=False)['FROZEN_GATE_REPLICAS'], 2)
        for field in ('local_validation_probe', 'frozen_gate_probe'):
            for count in (0, -1, True, 1.5):
                broken = dict(cfg, **{field: dict(cfg[field], replicas=count)})
                with self.subTest(field=field, count=count), self.assertRaisesRegex(ValueError, 'replicas'):
                    suite.make_runs(broken, 'workers')

    def test_metrics_can_use_snapshot_with_shared_fixed_or_live_role_gates(self):
        cfg = suite.read_json(ROOT / 'config/eac3-round2-gated.json')
        snapshot = cfg['planners']['eac3']
        cfg['local_validation_probe'] = {'model': snapshot}
        for backend in ('role', 'frozen_probe'):
            run = suite.make_runs(dict(cfg, gate_solver_backend=backend), 'adapt-planners')[0]
            env = suite.run_environment(run, check_allocation=False)
            self.assertEqual(env['VALIDATION_PROBE_MODEL_REMOTE'], snapshot)
            self.assertEqual(env['VALIDATION_PROBE_MODEL'], '')
            self.assertEqual(env['VALIDATION_PROBE_REVISION'], '')
            self.assertEqual(env['FROZEN_GATE_MODEL_REMOTE'], '')
            self.assertEqual(env['FROZEN_GATE_MODEL_PATH'], '')
            self.assertEqual(env['PREFIX_PROBE_SOLVER_BACKEND'], backend)
            self.assertEqual(suite.allocated_nodes(run['study_config']), cfg['nodes'] + 1)
        for model in ('Qwen/unpinned', 'SET_ME', '', 'bad checkpoint'):
            with self.subTest(model=model), self.assertRaises(ValueError):
                suite.make_runs(dict(cfg, local_validation_probe={'model': model}), 'adapt-planners')
        with self.assertRaisesRegex(ValueError, 'immutable path'):
            suite.make_runs(dict(cfg, local_validation_probe={'model': snapshot, 'revision': 'a' * 40}), 'workers')
        suite.make_runs(dict(cfg, local_validation_probe={'model': '/shared/immutable-decomposer'}), 'workers')

    def test_metrics_checkpoint_cli_records_original_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'runs'
            source = 's3:bucket/step110/decomposer/huggingface'
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/eac3_experiments.py'),
                'plan', '--config', str(ROOT / 'config/eac3-round2-gated.json'),
                '--phase', 'adapt-planners', '--output-dir', str(destination),
                '--local-probe-checkpoint', source, '--gate-solver-backend', 'frozen_probe'],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = suite.read_json(next(destination.glob('*.json')))
            self.assertEqual(manifest['study_config']['local_validation_probe']['model'], source)
            env = suite.run_environment(manifest, check_allocation=False)
            self.assertEqual(env['VALIDATION_PROBE_MODEL_REMOTE'], source)
            self.assertEqual(env['FROZEN_GATE_MODEL_REMOTE'], '')

    def test_metrics_snapshot_staging_and_effective_training_command(self):
        launcher = (ROOT / 'agent12-curriculum-trainer-multinode.sh').read_text()
        defaults = launcher[:launcher.index('\nif (( GPUS_PER_NODE')].replace('source ./env.sh', ':')
        function = launcher[launcher.index('run_agent12_training() {'):launcher.index('\nif [[ "$ONLINE_TEACHER_GENERATION"')]
        start = launcher.index('if [[ "$VALIDATION_PROBE_BACKEND" == local && -n "$VALIDATION_PROBE_MODEL_REMOTE" ]]; then')
        stage = launcher[start:launcher.index('\nif [[ "$PREFIX_PROBE_ENABLE" == true && "$FROZEN_GATE_ENABLE" == true ]]; then', start)]
        cfg = suite.read_json(ROOT / 'config/eac3-round2-gated.json')
        source = 's3:bucket/step110/decomposer/huggingface'
        cfg.update(gate_solver_backend='frozen_probe', local_validation_probe={'model': source})
        run = suite.make_runs(cfg, 'adapt-planners')[0]
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / 'command'
            env = {**os.environ, **{k: str(v) for k, v in suite.run_environment(run, check_allocation=False).items()},
                   'SLURM_JOB_ID': '123', 'SLURM_NNODES': '5', 'CAPTURE': str(capture), 'JOB_TMP': tmp}
            shell = defaults + '\nsrun() { printf "%s\\n" "${!#}" >> "$CAPTURE"; }\n'
            shell += 'HEAD_NODE=head; IP_HEAD=head:6379; COMMON_MOUNTS=(--nv)\n' + stage + '\n' + function
            shell += '\nrun_agent12_training /tmp/train.parquet "$TOTAL_STEPS" True True\n'
            result = subprocess.run(['bash', '-c', shell], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            captured = capture.read_text()
            self.assertIn('rclone copy "$VALIDATION_PROBE_MODEL_REMOTE" "$VALIDATION_PROBE_MODEL.partial"', captured)
            command = shlex.split(captured.rsplit(';', 1)[-1])
            opts = dict(arg.split('=', 1) for arg in command if '=' in arg and not arg.startswith('--'))
            self.assertEqual(opts['trainer.validation_probe.model'], tmp + '/validation_probe_model')
            self.assertEqual(opts['trainer.validation_probe.checkpoint_source'], source)
            self.assertEqual(opts['trainer.validation_probe.revision'], 'null')
            self.assertEqual(opts['trainer.frozen_gate_probe.enable'], 'false')
            result = subprocess.run(['bash', '-c', defaults], env=dict(env, SLURM_NNODES='4'),
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('1 exclusive GPUs', result.stderr)

    def test_frozen_decomposer_snapshot_independent_of_trainable_copy_and_metrics(self):
        cfg = suite.read_json(ROOT / 'config/eac3-round2-gated.json')
        cfg['gate_solver_backend'] = 'frozen_probe'
        cfg['local_validation_probe'] = {'revision': 'a' * 40}
        cfg['frozen_gate_probe'] = {'model': cfg['planners']['eac3']}
        run = suite.make_runs(cfg, 'adapt-planners')[0]
        env = suite.run_environment(run, check_allocation=False)
        self.assertEqual(env['TRAIN_DECOMPOSER'], 'true')
        self.assertEqual(env['FROZEN_GATE_MODEL_REMOTE'], env['DECOMPOSER_MODEL_REMOTE'])
        self.assertEqual(env['VALIDATION_PROBE_MODEL'], 'Qwen/Qwen2.5-7B-Instruct')
        self.assertEqual(suite.allocated_nodes(cfg), cfg['nodes'] + 1)
        # Fixed gates alone are supported, without silently allocating a metrics model.
        del cfg['local_validation_probe']
        env = suite.run_environment(suite.make_runs(cfg, 'adapt-planners')[0], check_allocation=False)
        self.assertNotIn('VALIDATION_PROBE_MODEL', env)
        self.assertEqual(suite.allocated_nodes(cfg), cfg['nodes'] + 1)
        for gate in ({}, {'model': 'Qwen/unpinned'}, {'model': 'SET_ME'},
                     {'model': 's3:bucket/snapshot', 'max_model_len': 256}):
            with self.subTest(gate=gate), self.assertRaises(ValueError):
                suite.make_runs(dict(cfg, frozen_gate_probe=gate), 'adapt-planners')
        with self.assertRaisesRegex(ValueError, 'gate_solver_backend=frozen_probe'):
            suite.make_runs(dict(cfg, gate_solver_backend='role'), 'adapt-planners')

    def test_snapshot_cli_pins_separate_gate_and_retains_metrics_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'runs'
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/eac3_experiments.py'),
                'plan', '--config', str(ROOT / 'config/eac3-round2-gated.json'),
                '--phase', 'adapt-planners', '--output-dir', str(destination),
                '--local-probe-revision', 'a' * 40,
                '--frozen-gate-checkpoint', 's3:bucket/decomposer/step110/huggingface'],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = suite.read_json(next(destination.glob('*.json')))
            env = suite.run_environment(manifest, check_allocation=False)
            self.assertEqual(env['PREFIX_PROBE_SOLVER_BACKEND'], 'frozen_probe')
            self.assertEqual(env['FROZEN_GATE_MODEL_REMOTE'], 's3:bucket/decomposer/step110/huggingface')
            self.assertEqual(env['VALIDATION_PROBE_MODEL'], 'Qwen/Qwen2.5-7B-Instruct')

    def test_snapshot_launcher_reserves_gpu_and_uses_dedicated_path(self):
        launcher = (ROOT / 'agent12-curriculum-trainer-multinode.sh').read_text()
        defaults = launcher[:launcher.index('\nif (( GPUS_PER_NODE')].replace('source ./env.sh', ':')
        function = launcher[launcher.index('run_agent12_training() {'):launcher.index('\nif [[ "$ONLINE_TEACHER_GENERATION"')]
        stage = launcher[launcher.index('if [[ "$PREFIX_PROBE_ENABLE" == true && "$FROZEN_GATE_ENABLE" == true ]]; then\n',
                                         launcher.index('COMMON_MOUNTS=(')):launcher.index('\nRAY_JOB_PIDS=()')]
        cfg = suite.read_json(ROOT / 'config/eac3-round2-gated.json')
        cfg.update(gate_solver_backend='frozen_probe', local_validation_probe={'revision': 'a' * 40},
                   frozen_gate_probe={'model': cfg['planners']['eac3']})
        run = suite.make_runs(cfg, 'adapt-planners')[0]
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / 'command'
            env = {**os.environ, **{k: str(v) for k, v in suite.run_environment(run, check_allocation=False).items()},
                   'SLURM_JOB_ID': '123', 'SLURM_NNODES': '5', 'CAPTURE': str(capture), 'JOB_TMP': tmp}
            shell = defaults + '\nsrun() { printf "%s\\n" "${!#}" >> "$CAPTURE"; }\n'
            shell += 'HEAD_NODE=head; IP_HEAD=head:6379; COMMON_MOUNTS=(--nv)\n' + stage + '\n' + function
            shell += '\nrun_agent12_training /tmp/train.parquet "$TOTAL_STEPS" True True\n'
            result = subprocess.run(['bash', '-c', shell], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            captured = capture.read_text()
            self.assertIn('rclone copy "$FROZEN_GATE_MODEL_REMOTE" "$FROZEN_GATE_MODEL_PATH.partial"', captured)
            command = shlex.split(captured.rsplit(';', 1)[-1])
            opts = dict(arg.split('=', 1) for arg in command if '=' in arg and not arg.startswith('--'))
            self.assertEqual(opts['trainer.frozen_gate_probe.model'], tmp + '/frozen_gate_model')
            self.assertEqual(opts['trainer.frozen_gate_probe.enable'], 'true')
            self.assertEqual(opts['trainer.validation_probe.model'], 'Qwen/Qwen2.5-7B-Instruct')
            self.assertNotEqual(opts['trainer.frozen_gate_probe.model'], opts['actor_rollout_ref.model.path'])
            result = subprocess.run(['bash', '-c', defaults], env=dict(env, SLURM_NNODES='4'),
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('2 exclusive GPUs', result.stderr)

    def test_gate_solver_backend_is_explicit_and_requires_pinned_probe(self):
        cfg = suite.read_json(ROOT / 'config/eac3-round2-gated.json')
        with self.assertRaisesRegex(ValueError, 'gate_solver_backend must'):
            suite.make_runs(dict(cfg, gate_solver_backend='typo'), 'workers')
        with self.assertRaisesRegex(ValueError, 'pinned local_validation_probe'):
            suite.make_runs(dict(cfg, gate_solver_backend='frozen_probe'), 'workers')
        cfg['local_validation_probe'] = {'revision': 'a' * 40}
        for backend in ('role', 'frozen_probe'):
            for phase in ('workers', 'adapt-planners'):
                run = suite.make_runs(dict(cfg, gate_solver_backend=backend), phase)[0]
                env = suite.run_environment(run, check_allocation=False)
                self.assertEqual(env['PREFIX_PROBE_SOLVER_BACKEND'], backend)
                self.assertEqual(env['PREFIX_PROBE_APPLY_TO_TRAINING'], 'true')
                self.assertEqual(env['VALIDATION_PROBE_MODEL'], 'Qwen/Qwen2.5-7B-Instruct')
                self.assertEqual(env['VALIDATION_PROBE_REVISION'], 'a' * 40)
                self.assertIn('prefix_probe.solver_backend=' + backend, env['AGENT12_EXPERIMENT_OVERRIDES'])

    def test_documented_gate_pair_preserves_identical_starting_partners(self):
        recipe = (ROOT / 'docs/eac3_round2.md').read_text().split('## Compare fixed and adaptive gates', 1)[1]
        code = recipe.split("python3 - <<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'PROBE_REVISION': 'a' * 40,
                    'GATE_COMPARISON_STAMP': 'test', 'GATE_COMPARISON_DIR': tmp}):
                exec(compile(code, '<documented-gate-pair>', 'exec'), {})
            runs = [suite.read_json(next((Path(tmp) / backend).glob('*.json')))
                    for backend in ('role', 'frozen_probe')]
            self.assertEqual(runs[0]['planner'], runs[1]['planner'])
            self.assertEqual(runs[0]['worker'], runs[1]['worker'])
            self.assertNotEqual(runs[0]['run_name'], runs[1]['run_name'])
            configs = [dict(run['study_config']) for run in runs]
            for cfg in configs:
                self.assertEqual(cfg['worker_gate_mode'], 'gated')
                self.assertEqual(suite.allocated_nodes(cfg), 4)
                cfg.pop('study')
                cfg.pop('gate_solver_backend')
            self.assertEqual(*configs)

    def test_gate_solver_selection_reaches_actual_training_command(self):
        launcher = (ROOT / 'agent12-curriculum-trainer-multinode.sh').read_text()
        defaults = launcher[:launcher.index('\nif (( GPUS_PER_NODE')].replace('source ./env.sh', ':')
        function = launcher[launcher.index('run_agent12_training() {'):launcher.index('\nif [[ "$ONLINE_TEACHER_GENERATION"')]
        cfg = suite.read_json(ROOT / 'config/eac3-round2-gated.json')
        cfg['local_validation_probe'] = {'revision': 'a' * 40}
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / 'command'
            for backend in ('role', 'frozen_probe'):
                for phase in ('workers', 'adapt-planners'):
                    run = suite.make_runs(dict(cfg, gate_solver_backend=backend), phase)[0]
                    env = {**os.environ, 'PREFIX_PROBE_SOLVER_BACKEND': 'bad-inherited-value',
                           **{k: str(v) for k, v in suite.run_environment(run, check_allocation=False).items()},
                           'SLURM_JOB_ID': '123', 'SLURM_NNODES': '5', 'CAPTURE': str(capture)}
                    shell = defaults + '\nsrun() { printf "%s\\n" "${!#}" > "$CAPTURE"; }\n'
                    shell += 'HEAD_NODE=head; IP_HEAD=head:6379; COMMON_MOUNTS=(--nv)\n' + function
                    shell += '\nrun_agent12_training /tmp/train.parquet "$TOTAL_STEPS" True True\n'
                    result = subprocess.run(['bash', '-c', shell], env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    command = shlex.split(capture.read_text().rsplit(';', 1)[-1])
                    opts = dict(arg.split('=', 1) for arg in command if '=' in arg and not arg.startswith('--'))
                    self.assertEqual(opts['algorithm.hierarchy.scoped_c3_grpo.prefix_probe.solver_backend'], backend)
                    self.assertEqual(opts['algorithm.hierarchy.scoped_c3_grpo.prefix_probe.apply_to_training'], 'true')
                    self.assertEqual(opts['trainer.validation_probe.model'], 'Qwen/Qwen2.5-7B-Instruct')
                    self.assertEqual(opts['trainer.validation_probe.enable'], 'True')

    def test_matrix_size_and_freezes(self):
        cfg = config()
        for phase, count in [('planners', 2), ('workers', 6), ('transfer', 7), ('cross', 12)]:
            runs = suite.make_runs(cfg, phase)
            self.assertEqual(len(runs), count)
            for run in runs:
                env = suite.run_environment(run)
                self.assertEqual(env['TRAIN_DECOMPOSER'], 'false')
                self.assertNotIn('decomposer', env['TRAIN_AGENT_ROLES'])
                self.assertEqual(env['PREFIX_PROBE_ENABLE'], 'false')
                self.assertIn('algorithm.hierarchy.agent12_curriculum.enable=false', env['AGENT12_EXPERIMENT_OVERRIDES'])
                self.assertIn('trainer.resume_mode=disable', env['AGENT12_EXPERIMENT_OVERRIDES'])
                self.assertIn('algorithm.hierarchy.start_agent=worker_stage_1', env['AGENT12_EXPERIMENT_OVERRIDES'])

    def test_workers_differ_only_in_planner_and_run_identity(self):
        a, b = suite.make_runs(config(), 'workers')[:2]
        self.assertEqual(a['worker'], b['worker'])
        self.assertEqual(a['seed'], b['seed'])
        x, y = suite.run_environment(a), suite.run_environment(b)
        for key in ('AGENT12_RUN_NAME', 'AGENT12_WANDB_RUN_ID', 'DECOMPOSER_MODEL_REMOTE'):
            x.pop(key); y.pop(key)
        self.assertEqual(x, y)

    def test_worker_gate_modes_are_explicit_and_training_only(self):
        for mode in ('off', 'diagnostic', 'gated'):
            cfg = repository_config()
            cfg['worker_gate_mode'] = mode
            for phase in ('planners', 'workers', 'transfer', 'cross'):
                with self.subTest(mode=mode, phase=phase):
                    run = suite.make_runs(cfg, phase)[0]
                    env = suite.run_environment(run)
                    self.assertEqual(env['PREFIX_PROBE_ENABLE'], str(mode != 'off').lower())
                    self.assertEqual(env['PREFIX_PROBE_APPLY_TO_TRAINING'],
                                     str(mode == 'gated' and phase == 'workers').lower())
                    self.assertEqual(env['C3_REQUIRE_ELIGIBLE_SUCCESS'], 'true')
                    self.assertEqual(env['PREFIX_PROBE_MAX_NEW_TOKENS'], 256)
                    self.assertEqual(env['PREFIX_PROBE_VALIDATION_MAX_SAMPLES'], 128)
                    self.assertEqual(env['TRAIN_DECOMPOSER'], 'false')
                    self.assertIn('prefix_probe.solver_role=decomposer', env['AGENT12_EXPERIMENT_OVERRIDES'])
        for value in ('enabled', True, None):
            cfg['worker_gate_mode'] = value
            with self.assertRaisesRegex(ValueError, 'worker_gate_mode'):
                suite.make_runs(cfg, 'workers')

    def test_single_planner_runs_do_not_create_baseline_jobs(self):
        cfg = repository_config()
        cfg['planners'].pop('baseline')
        cfg['seeds'] = [11]
        for phase, count in [('planners', 1), ('workers', 1), ('transfer', 2), ('cross', 1)]:
            runs = suite.make_runs(cfg, phase)
            self.assertEqual(len(runs), count)
            for run in runs:
                self.assertEqual(run['planner'], cfg['planners']['eac3'])
                self.assertNotIn('baseline', run['run_name'])
                suite.run_environment(run)
        for planners in ({}, {'unknown': 'model'}, {'baseline': 'model', 'eac3': 'model'}):
            cfg['planners'] = planners
            with self.assertRaises(ValueError):
                suite.validate_study(cfg)

    def test_eac3_worker_gate_pair_changes_only_training_gate_and_identity(self):
        configs = [suite.read_json(ROOT / f'config/eac3-worker-gates-qwen7-step110-{arm}.json')
                   for arm in ('ungated', 'gated')]
        self.assertEqual([cfg['worker_gate_mode'] for cfg in configs], ['diagnostic', 'gated'])
        self.assertEqual({k: v for k, v in configs[0].items() if k not in ('study', 'worker_gate_mode')},
                         {k: v for k, v in configs[1].items() if k not in ('study', 'worker_gate_mode')})
        envs, exports = [], []
        with tempfile.TemporaryDirectory() as tmp:
            for cfg in configs:
                self.assertEqual(set(cfg['planners']), {'eac3'})
                self.assertEqual(cfg['test_update_steps'], [1, 2, 5])
                self.assertEqual(cfg['test_freq'], 10)
                self.assertEqual(cfg['save_freq'], 10)
                self.assertEqual(cfg['worker_updates'], 100)
                directory = Path(tmp) / cfg['study']
                suite.write_runs(cfg, 'workers', directory)
                self.assertEqual(len(list(directory.glob('*.json'))), 1)
                self.assertEqual(len((directory / 'commands.txt').read_text().splitlines()), 1)
                run = suite.read_json(next(directory.glob('*.json')))
                env = suite.run_environment(run)
                self.assertEqual(env['PREFIX_PROBE_ENABLE'], 'true')
                self.assertEqual(env['TRAIN_DECOMPOSER'], 'false')
                self.assertEqual(env['SAVE_FREQ'], 10)
                self.assertIn('trainer.test_update_steps=[1, 2, 5]', env['AGENT12_EXPERIMENT_OVERRIDES'])
                self.assertIn('trainer.resume_mode=disable', env['AGENT12_EXPERIMENT_OVERRIDES'])
                self.assertIn('trainer.val_before_train=true', env['AGENT12_EXPERIMENT_OVERRIDES'])
                envs.append(env)
                exports.append(suite.worker_export(cfg, run['arm'], run['seed']))
        self.assertEqual({k for k in envs[0] if envs[0][k] != envs[1][k]}, {
            'AGENT12_RUN_NAME', 'AGENT12_WANDB_RUN_ID', 'PREFIX_PROBE_APPLY_TO_TRAINING',
        })
        self.assertEqual([env['PREFIX_PROBE_APPLY_TO_TRAINING'] for env in envs], ['false', 'true'])
        self.assertNotEqual(*exports)
        self.assertTrue(all(export != configs[0]['initial_worker'] for export in exports))

    def test_early_evaluation_uses_completed_updates_without_repeating_skips(self):
        path = ROOT / 'src/verl/verl/rema_separated_trainer/ppo/ray_trainer.py'
        trainer = next(node for node in ast.parse(path.read_text()).body
                       if isinstance(node, ast.ClassDef) and node.name == 'RayReMASeparatedTrainer')
        method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_update_frequency_due')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        due = namespace['_update_frequency_due']
        state = SimpleNamespace(actor_update_steps=0)
        seen = []
        for step in range(21):
            state.actor_update_steps = step
            self.assertFalse(due(state, 10, False, [1, 2, 5]))
            if due(state, 10, True, [1, 2, 5]):
                seen.append(step)
        self.assertEqual(seen, [1, 2, 5, 10, 20])
        state.actor_update_steps = 1
        self.assertFalse(due(state, 50, True))  # Early validation does not force a checkpoint.
        state.actor_update_steps = 50
        self.assertTrue(due(state, 50, True))
        self.assertFalse(due(state, 0, True))
        fit = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == 'fit')
        self.assertIn("self.config.trainer.get('test_update_steps', [])", ast.unparse(fit))
        cfg = config()
        for bad in (None, '1,2', [0], [-1], [True], [1, 1], [1.5], [{}]):
            cfg['test_update_steps'] = bad
            with self.assertRaisesRegex(ValueError, 'test_update_steps'):
                suite.validate_study(cfg)

    def test_checkpoint_schedule_runs_before_validation_even_when_it_fails(self):
        path = ROOT / 'src/verl/verl/rema_separated_trainer/ppo/ray_trainer.py'
        trainer = next(node for node in ast.parse(path.read_text()).body
                       if isinstance(node, ast.ClassDef) and node.name == 'RayReMASeparatedTrainer')
        method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_update_frequency_due')
        fit = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == 'fit')
        blocks = sorted((node for node in ast.walk(fit) if isinstance(node, ast.If) and any(
            isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == '_update_frequency_due' for call in ast.walk(node.test)
        )), key=lambda node: node.lineno)
        self.assertEqual(len(blocks), 2)
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        due = namespace['_update_frequency_due']
        scheduled = compile(ast.Module(body=blocks, type_ignores=[]), str(path), 'exec')
        cases = [
            (10, True, False, True, ['save', 'validate']),
            (10, True, False, False, ['save', 'validate']),
            (9, True, False, False, []),
            (10, False, False, False, []),
            (9, False, True, False, ['save', 'validate']),
            (5, True, False, False, ['validate']),
        ]
        for updates, updated, ending, fail_validation, expected in cases:
            with self.subTest(updates=updates, updated=updated, ending=ending, failure=fail_validation):
                events = []

                def validate():
                    events.append('validate')
                    if fail_validation:
                        raise RuntimeError('validation failed')
                    return {'val/acc': 0.5}

                options = SimpleNamespace(save_freq=10, test_freq=10,
                                          get={'test_update_steps': [1, 2, 5]}.get)
                state = SimpleNamespace(actor_update_steps=updates, config=SimpleNamespace(trainer=options),
                                        val_reward_fn=True, _validate=validate,
                                        _save_checkpoint=lambda: events.append('save'))
                state._update_frequency_due = lambda frequency, updated, extra_steps=(): due(
                    state, frequency, updated, extra_steps)
                env = dict(self=state, actor_updated=updated, is_session_last_step=ending,
                           is_training_last_step=ending, timing_raw={}, metrics={},
                           _timer=lambda *args: nullcontext())
                if fail_validation:
                    with self.assertRaisesRegex(RuntimeError, 'validation failed'):
                        exec(scheduled, env)
                else:
                    exec(scheduled, env)
                self.assertEqual(events, expected)

    def test_gated_pilot_preserves_controls_and_has_separate_outputs(self):
        off = suite.read_json(ROOT / 'config/eac3-worker-learning-qwen7-step110.json')
        gated = suite.read_json(ROOT / 'config/eac3-worker-learning-qwen7-step110-gated.json')
        self.assertEqual(gated['worker_gate_mode'], 'gated')
        self.assertNotEqual(off['study'], gated['study'])
        self.assertEqual({k: v for k, v in off.items() if k not in ('study', 'worker_gate_mode')},
                         {k: v for k, v in gated.items() if k not in ('study', 'worker_gate_mode')})
        for a, b in zip(suite.make_runs(off, 'workers'), suite.make_runs(gated, 'workers')):
            x, y = suite.run_environment(a), suite.run_environment(b)
            self.assertEqual({k for k in x if x[k] != y[k]}, {
                'AGENT12_RUN_NAME', 'AGENT12_WANDB_RUN_ID',
                'PREFIX_PROBE_ENABLE', 'PREFIX_PROBE_APPLY_TO_TRAINING',
            })
            self.assertNotEqual(suite.worker_export(off, a['arm'], a['seed']),
                                suite.worker_export(gated, b['arm'], b['seed']))

    def test_gate_mode_reaches_training_command_despite_inherited_flags(self):
        launcher = (ROOT / 'agent12-curriculum-trainer-multinode.sh').read_text()
        defaults = launcher[:launcher.index('\nif (( GPUS_PER_NODE')].replace('source ./env.sh', ':')
        function = launcher[launcher.index('run_agent12_training() {'):launcher.index('\nif [[ "$ONLINE_TEACHER_GENERATION"')]
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / 'command'
            for mode in ('off', 'diagnostic', 'gated'):
                with self.subTest(mode=mode):
                    cfg = repository_config()
                    cfg['worker_gate_mode'] = mode
                    cfg['test_update_steps'] = [1, 2, 5]
                    run = suite.make_runs(cfg, 'workers')[0]
                    env = {**os.environ, 'PREFIX_PROBE_ENABLE': 'inherited',
                           'PREFIX_PROBE_APPLY_TO_TRAINING': 'inherited',
                           'C3_REQUIRE_ELIGIBLE_SUCCESS': 'false',
                           **{k: str(v) for k, v in suite.run_environment(run).items()},
                           'SLURM_JOB_ID': '123', 'SLURM_NNODES': '6', 'CAPTURE': str(capture)}
                    shell = defaults + '\nsrun() { printf "%s\\n" "${!#}" > "$CAPTURE"; }\n'
                    shell += 'HEAD_NODE=head; IP_HEAD=head:6379; COMMON_MOUNTS=(--nv)\n' + function
                    shell += '\nrun_agent12_training /tmp/train.parquet "$TOTAL_STEPS" True True\n'
                    result = subprocess.run(['bash', '-c', shell], env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    command = shlex.split(capture.read_text().rsplit(';', 1)[-1])
                    overrides = dict(arg.split('=', 1) for arg in command if '=' in arg and not arg.startswith('--'))
                    prefix = 'algorithm.hierarchy.scoped_c3_grpo.'
                    self.assertEqual(overrides[prefix + 'prefix_probe.enable'], str(mode != 'off').lower())
                    self.assertEqual(overrides[prefix + 'prefix_probe.apply_to_training'], str(mode == 'gated').lower())
                    self.assertEqual(overrides[prefix + 'require_eligible_success'], 'true')
                    self.assertEqual(overrides['trainer.val_only'], 'false')
                    self.assertEqual(json.loads(overrides['trainer.test_update_steps']), [1, 2, 5])
                    self.assertEqual(overrides['algorithm.hierarchy.train_agent_roles'], suite.WORKER_ROLES)

    def test_repository_data_needs_no_train_or_eval_remote(self):
        cfg = repository_config()
        for phase in ('planners', 'workers', 'transfer', 'cross'):
            for run in suite.make_runs(cfg, phase):
                env = suite.run_environment(run)
                self.assertEqual(env['EXPERIMENT_DATA_SOURCE'], 'repository')
                self.assertEqual(env['TEACHER_TRAIN_REMOTE'], '')
                self.assertEqual(env['GENERATE_TEACHER_DATA'], 0)
                self.assertEqual(env['EXPERIMENT_TRAIN_RELATIVE'], 'MATH/train_lv3to5_8k.parquet')
                self.assertEqual(env['EXPERIMENT_EVAL_RELATIVE'], 'overall_math/test.parquet')
                if phase == 'transfer':
                    self.assertEqual(env['EXPERIMENT_VAL_REMOTE'], cfg['common_plan_remote'])
                else:
                    self.assertEqual(env['EXPERIMENT_VAL_REMOTE'], cfg['eval_file'])
        a, b = [suite.run_environment(run) for run in suite.make_runs(cfg, 'workers')[:2]]
        for key in ('AGENT12_RUN_NAME', 'AGENT12_WANDB_RUN_ID', 'DECOMPOSER_MODEL_REMOTE'):
            a.pop(key); b.pop(key)
        self.assertEqual(a, b)

    def test_repository_paths_stay_in_staged_directories(self):
        for path in ('/tmp/train.parquet', 'data/MATH/../../train.parquet',
                     'data/unknown/train.parquet', 'data/MATH/train.csv', 'SET_ME'):
            with self.subTest(path=path):
                cfg = repository_config()
                cfg['train_file'] = path
                with self.assertRaisesRegex(ValueError, 'train_file'):
                    suite.make_runs(cfg, 'workers')
        cfg = repository_config()
        cfg['data_source'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'data_source'):
            suite.validate_study(cfg)

    def test_round2_uses_two_frozen_worker_partners_and_eac3_for_both_planners(self):
        configs = [suite.read_json(ROOT / f'config/eac3-round2-{arm}.json')
                   for arm in ('ungated', 'gated')]
        self.assertEqual({k: v for k, v in configs[0].items()
                          if k not in ('study', 'initial_worker', 'worker_gate_mode')},
                         {k: v for k, v in configs[1].items()
                          if k not in ('study', 'initial_worker', 'worker_gate_mode')})
        self.assertIn('recovered-6106013/snapshots/global_step_63/', configs[0]['initial_worker'])
        self.assertIn('recovered-6106014/snapshots/global_step_65/', configs[1]['initial_worker'])
        planner_envs = []
        exports = []
        for cfg in configs:
            planner_runs = suite.make_runs(cfg, 'adapt-planners')
            self.assertEqual(len(planner_runs), 1)
            run = planner_runs[0]
            env = suite.run_environment(run)
            planner_envs.append(env)
            self.assertEqual(run['mode'], 'planner_train')
            self.assertEqual(env['TRAIN_DECOMPOSER'], 'true')
            self.assertEqual(env['TRAIN_AGENT_ROLES'], '[decomposer]')
            self.assertEqual(env['WORKER_MODEL_REMOTE'], cfg['initial_worker'])
            self.assertEqual(env['DECOMPOSER_MODEL_REMOTE'], cfg['planners']['eac3'])
            self.assertEqual(env['PREFIX_PROBE_ENABLE'], 'true')
            self.assertEqual(env['PREFIX_PROBE_APPLY_TO_TRAINING'], 'true')
            self.assertEqual(env['C3_REQUIRE_ELIGIBLE_SUCCESS'], 'true')
            self.assertIn('algorithm.hierarchy.start_agent=decomposer', env['AGENT12_EXPERIMENT_OVERRIDES'])
            self.assertIn('trainer.max_actor_updates=40', env['AGENT12_EXPERIMENT_OVERRIDES'])
            self.assertIn('agent12_curriculum.enable=false', env['AGENT12_EXPERIMENT_OVERRIDES'])
            self.assertIn('trainer.resume_mode=disable', env['AGENT12_EXPERIMENT_OVERRIDES'])
            self.assertEqual(env['TEACHER_TRAIN_REMOTE'], '')
            self.assertEqual(env['EXPERIMENT_DATA_SOURCE'], 'repository')

            workers = suite.make_runs(cfg, 'adapt-workers')
            self.assertEqual(len(workers), 1)
            worker_run = workers[0]
            worker_env = suite.run_environment(worker_run)
            self.assertEqual(worker_run['worker'], run['worker'])
            self.assertEqual(worker_run['planner'], suite.adapted_planner_export(cfg, 'eac3', 11))
            self.assertNotEqual(worker_run['planner'], run['planner'])
            self.assertEqual(worker_env['TRAIN_DECOMPOSER'], 'false')
            self.assertEqual(worker_env['TRAIN_AGENT_ROLES'], suite.WORKER_ROLES)
            self.assertEqual(worker_env['PREFIX_PROBE_ENABLE'], 'true')
            self.assertEqual(worker_env['PREFIX_PROBE_APPLY_TO_TRAINING'],
                             str(cfg['worker_gate_mode'] == 'gated').lower())
            self.assertIn('trainer.max_actor_updates=40', worker_env['AGENT12_EXPERIMENT_OVERRIDES'])
            exports.append(suite.worker_export(cfg, 'eac3', 11, phase='adapt-workers'))
        self.assertEqual({key for key in planner_envs[0] if planner_envs[0][key] != planner_envs[1][key]},
                         {'AGENT12_RUN_NAME', 'AGENT12_WANDB_RUN_ID', 'WORKER_MODEL_REMOTE'})
        self.assertNotEqual(*exports)

    def test_round2_configuration_rejects_ambiguous_or_invalid_adaptation(self):
        good = suite.read_json(ROOT / 'config/eac3-round2-gated.json')
        for value in (0, -1, True, None, '40', 1201):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'planner_updates'):
                suite.make_runs(dict(good, planner_updates=value), 'adapt-planners')
        for phase in ('adapt-planners', 'adapt-workers'):
            cfg = dict(good)
            cfg.pop('planner_updates')
            with self.assertRaisesRegex(ValueError, 'planner_updates'):
                suite.make_runs(cfg, phase)
            cfg = dict(good, planners={'baseline': good['planners']['eac3']})
            with self.assertRaisesRegex(ValueError, 'one eac3 planner'):
                suite.make_runs(cfg, phase)
            with self.assertRaisesRegex(ValueError, 'question-only'):
                suite.make_runs(dict(good, data_source='remote'), phase)

    def test_round2_effective_shell_command_and_model_export(self):
        launcher = (ROOT / 'agent12-curriculum-trainer-multinode.sh').read_text()
        defaults = launcher[:launcher.index('\nif (( GPUS_PER_NODE')].replace('source ./env.sh', ':')
        function = launcher[launcher.index('run_agent12_training() {'):launcher.index('\nif [[ "$ONLINE_TEACHER_GENERATION"')]
        export_guard = launcher[launcher.index('    if [[ "$EXPERIMENT_MODE" != "worker_train"'):
                                launcher.index('\nLATEST_STEP=')]
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / 'command'
            for arm in ('ungated', 'gated'):
                cfg = suite.read_json(ROOT / f'config/eac3-round2-{arm}.json')
                for phase in ('adapt-planners', 'adapt-workers'):
                    with self.subTest(arm=arm, phase=phase):
                        run = suite.make_runs(cfg, phase)[0]
                        env = {**os.environ, **{k: str(v) for k, v in suite.run_environment(run).items()},
                               'SLURM_JOB_ID': '123', 'SLURM_NNODES': '4', 'CAPTURE': str(capture),
                               'VALIDATION_PROBE_URL': 'http://probe-node:8000/v1',
                               'VALIDATION_PROBE_MODEL': 'frozen-probe-v1',
                               'VALIDATION_PROBE_SAMPLES_PER_DATASET': '12',
                               'VALIDATION_PROBE_MAX_NEW_TOKENS': '256'}
                        shell = defaults + '\nsrun() { printf "%s\\n" "${!#}" > "$CAPTURE"; }\n'
                        shell += 'HEAD_NODE=head; IP_HEAD=head:6379; COMMON_MOUNTS=(--nv)\n' + function
                        shell += '\nrun_agent12_training /tmp/train.parquet "$TOTAL_STEPS" True True\n'
                        result = subprocess.run(['bash', '-c', shell], env=env, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        command = shlex.split(capture.read_text().rsplit(';', 1)[-1])
                        opts = dict(arg.split('=', 1) for arg in command if '=' in arg and not arg.startswith('--'))
                        planner = phase == 'adapt-planners'
                        self.assertEqual(opts['algorithm.hierarchy.train_agent_roles'],
                                         '[decomposer]' if planner else suite.WORKER_ROLES)
                        self.assertEqual(opts['algorithm.hierarchy.start_agent'],
                                         'decomposer' if planner else 'worker_stage_1')
                        self.assertEqual(opts['algorithm.hierarchy.agent12_curriculum.enable'], 'false')
                        self.assertEqual(opts['trainer.max_actor_updates'], '40')
                        self.assertEqual(opts['trainer.val_only'], 'false')
                        self.assertEqual(opts['trainer.save_freq'], '10')
                        self.assertEqual(opts['trainer.validation_probe.enable'], 'True')
                        self.assertEqual(opts['trainer.validation_probe.model'], 'frozen-probe-v1')
                        self.assertEqual(opts['trainer.validation_probe.base_url'], 'http://probe-node:8000/v1')
                        self.assertEqual(opts['trainer.validation_probe.samples_per_dataset'], '12')
                        self.assertEqual(opts['algorithm.hierarchy.scoped_c3_grpo.prefix_probe.apply_to_training'],
                                         str(planner or arm == 'gated').lower())
                        # Both training modes must continue past evaluation upload to HF export.
                        shell = 'if true; then\n' + export_guard + '\necho MODEL_EXPORT\n'
                        result = subprocess.run(['bash', '-c', shell], env=env, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(result.stdout.strip(), 'MODEL_EXPORT')
            env['EXPERIMENT_MODE'] = 'planner_eval'
            env['REMOTE_RUN'] = 's3:bucket/eval'
            result = subprocess.run(['bash', '-c', shell], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn('MODEL_EXPORT', result.stdout)

    def test_round2_manifests_and_unresolved_inputs(self):
        cfg = suite.read_json(ROOT / 'config/eac3-round2-gated.json')
        with tempfile.TemporaryDirectory() as tmp:
            for phase in ('adapt-planners', 'adapt-workers'):
                directory = Path(tmp) / phase
                # Planning may happen inside a one-node recovery allocation.
                with patch.dict(os.environ, {'SLURM_NNODES': '1'}):
                    suite.write_runs(cfg, phase, directory)
                run = next(directory.glob('*.json'))
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/eac3_experiments.py'),
                                         'environment', '--run', str(run)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('trainer.resume_mode=disable', result.stdout)
                self.assertIn('unset JOB_TMP', result.stdout)
                self.assertIn('sbatch --export=ALL --nodes=4 --gres=gpu:hopper:4,storage:local:200G',
                              (directory / 'commands.txt').read_text())
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/eac3_experiments.py'),
                                         'environment', '--run', str(run)], capture_output=True, text=True,
                                        env={**os.environ, 'SLURM_NNODES': '1'})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Slurm node allocation differs', result.stderr)
            cfg['initial_worker'] = 'SET_ME_WORKERS'
            with self.assertRaisesRegex(ValueError, 'unresolved'):
                suite.write_runs(cfg, 'adapt-planners', Path(tmp) / 'invalid')
            self.assertFalse((Path(tmp) / 'invalid').exists())

    def test_local_probe_reserves_capacity_without_changing_training_pools(self):
        launcher = (ROOT / 'agent12-curriculum-trainer-multinode.sh').read_text()
        defaults = launcher[:launcher.index('\nif (( GPUS_PER_NODE')].replace('source ./env.sh', ':')
        function = launcher[launcher.index('run_agent12_training() {'):launcher.index('\nif [[ "$ONLINE_TEACHER_GENERATION"')]
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ('ungated', 'gated'):
                cfg = suite.read_json(ROOT / f'config/eac3-round2-{arm}.json')
                for phase in ('adapt-workers', 'adapt-planners'):
                    with self.subTest(arm=arm, phase=phase):
                        original = suite.run_environment(suite.make_runs(cfg, phase)[0], check_allocation=False)
                        local_cfg = dict(cfg, local_validation_probe={'revision': 'a' * 40})
                        run = suite.make_runs(local_cfg, phase)[0]
                        env = suite.run_environment(run, check_allocation=False)
                        self.assertEqual(suite.allocated_nodes(local_cfg), 5)
                        self.assertEqual({k: v for k, v in env.items() if not k.startswith('VALIDATION_PROBE_')},
                                         {k: v for k, v in original.items() if not k.startswith('VALIDATION_PROBE_')})
                        with patch.dict(os.environ, {'SLURM_NNODES': '4'}):
                            with self.assertRaisesRegex(ValueError, 'allocation differs'):
                                suite.run_environment(run)
                        with patch.dict(os.environ, {'SLURM_NNODES': '5'}):
                            suite.run_environment(run)
                        capture = Path(tmp) / 'command'
                        env = {**os.environ, **{k: str(v) for k, v in env.items()},
                               'SLURM_JOB_ID': '123', 'SLURM_NNODES': '5', 'CAPTURE': str(capture)}
                        shell = defaults + '\nsrun() { printf "%s\\n" "${!#}" > "$CAPTURE"; }\n'
                        shell += 'HEAD_NODE=head; IP_HEAD=head:6379; COMMON_MOUNTS=(--nv)\n' + function
                        shell += '\nrun_agent12_training /tmp/train.parquet "$TOTAL_STEPS" True True\n'
                        result = subprocess.run(['bash', '-c', shell], env=env, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        command = shlex.split(capture.read_text().rsplit(';', 1)[-1])
                        opts = dict(arg.split('=', 1) for arg in command if '=' in arg and not arg.startswith('--'))
                        self.assertEqual(opts['trainer.nnodes'], '4')
                        self.assertEqual(opts['trainer.n_gpus_per_node'], '2')
                        self.assertEqual(opts['trainer.validation_probe.backend'], 'local')
                        self.assertEqual(opts['trainer.validation_probe.base_url'], 'null')
                        self.assertEqual(opts['trainer.validation_probe.revision'], 'a' * 40)
                        self.assertEqual(opts['trainer.validation_probe.max_new_tokens'], '256')
                        self.assertEqual(opts['algorithm.hierarchy.scoped_c3_grpo.prefix_probe.apply_to_training'],
                                         str(phase == 'adapt-planners' or arm == 'gated').lower())

    def test_round2_local_probe_rerun_recipe_allocates_four_nodes(self):
        documentation = (ROOT / 'docs/eac3_round2.md').read_text()
        recipe = documentation.split('## Rerun workers with local frozen probes', 1)[1]
        code = recipe.split("python3 - <<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        real_run = subprocess.run
        remote_checks = []

        def checked_run(command, *args, **kwargs):
            if command[0] == 'rclone':
                self.assertEqual(command[1], 'cat')
                self.assertTrue(command[2].endswith('/config.json'))
                remote_checks.append(command[2])
                return subprocess.CompletedProcess(command, 0)
            return real_run(command, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            environment = {'PROBE_REVISION': 'a' * 40, 'RUN_STAMP': 'test', 'RUN_DIR': tmp}
            with patch.dict(os.environ, environment), patch.object(subprocess, 'run', side_effect=checked_run):
                exec(compile(code, '<documented-round2-rerun>', 'exec'), {})
            for arm in ('ungated', 'gated'):
                cfg = suite.read_json(ROOT / f'config/eac3-round2-{arm}.json')
                original = suite.make_runs(cfg, 'adapt-workers')[0]
                directory = Path(tmp) / arm
                manifest = suite.read_json(next(directory.glob('*.json')))
                self.assertEqual(manifest['mode'], 'worker_train')
                self.assertEqual(manifest['planner'], original['planner'])
                self.assertEqual(manifest['worker'], original['worker'])
                self.assertNotEqual(manifest['run_name'], original['run_name'])
                self.assertEqual(manifest['study_config'], dict(
                    cfg, nodes=3, study=cfg['study'] + '-local-probe-test',
                    planners={'eac3': original['planner']},
                    local_validation_probe={'model': 'Qwen/Qwen2.5-7B-Instruct',
                                            'revision': 'a' * 40, 'extra_nodes': 1,
                                            'batch_size': 8, 'replicas': 4},
                ))
                self.assertEqual(suite.allocated_nodes(manifest['study_config']), 4)
                self.assertIn('sbatch --export=ALL --nodes=4 --gres=gpu:hopper:4,storage:local:200G',
                              (directory / 'commands.txt').read_text())
                with patch.dict(os.environ, {'SLURM_NNODES': '4'}):
                    env = suite.run_environment(manifest)
                self.assertEqual(env['TRAINING_NNODES'], 3)
                self.assertEqual(env['POOL_GPUS_PER_NODE'], 2)
                self.assertEqual(env['TRAIN_DECOMPOSER'], 'false')
                self.assertEqual(env['PREFIX_PROBE_APPLY_TO_TRAINING'], str(arm == 'gated').lower())
                self.assertEqual(env['VALIDATION_PROBE_BACKEND'], 'local')
                self.assertEqual(env['VALIDATION_PROBE_MODEL'], 'Qwen/Qwen2.5-7B-Instruct')
                self.assertEqual(env['BASE_QUESTION_BATCH_SIZE'], 48)
                self.assertEqual(env['ROLLOUT_N'], 32)
            self.assertEqual(len(remote_checks), 4)

    def test_local_probe_cli_pins_revision_in_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'runs'
            result = subprocess.run([
                sys.executable, str(ROOT / 'scripts/eac3_experiments.py'), 'plan',
                '--config', str(ROOT / 'config/eac3-round2-gated.json'),
                '--phase', 'adapt-workers', '--output-dir', str(directory),
                '--local-probe-revision', 'a' * 40,
                '--gate-solver-backend', 'frozen_probe',
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('sbatch --export=ALL --nodes=5', (directory / 'commands.txt').read_text())
            manifest = suite.read_json(next(directory.glob('*.json')))
            self.assertEqual(manifest['study_config']['local_validation_probe']['revision'], 'a' * 40)
            self.assertEqual(manifest['study_config']['gate_solver_backend'], 'frozen_probe')
            self.assertIn('src/verl/verl/rema_separated_trainer/ppo/local_validation_probe.py', manifest['code_sha256'])
        for probe_cfg in ({'revision': 'main'}, {'revision': 'a' * 40, 'extra_nodes': 0},
                          {'revision': 'a' * 40, 'max_new_tokens': 16384}):
            cfg = suite.read_json(ROOT / 'config/eac3-round2-gated.json')
            cfg['local_validation_probe'] = probe_cfg
            with self.assertRaises(ValueError):
                suite.make_runs(cfg, 'adapt-workers')

    def test_repository_manifests_pin_dataset_contents(self):
        cfg = repository_config()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'runs'
            suite.write_runs(cfg, 'workers', directory)
            path = next(directory.glob('*.json'))
            run = suite.read_json(path)
            self.assertEqual(run['data_sha256'], {
                name: suite.sha_file(ROOT / name) for name in (cfg['train_file'], cfg['eval_file'])})
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/eac3_experiments.py'),
                                     'environment', '--run', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('export EXPERIMENT_DATA_SOURCE=repository', result.stdout)
            run['data_sha256'][cfg['train_file']] = '0' * 64
            path.write_text(json.dumps(run))
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/eac3_experiments.py'),
                                     'environment', '--run', str(path)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Data changed after manifest creation', result.stderr)

    def test_launcher_prepares_repository_data_without_remote_io(self):
        launcher = (ROOT / 'agent12-curriculum-trainer-multinode.sh').read_text()
        start = launcher.index('if [[ -n "${EAC3_RUN_FILE:-}" && "${EXPERIMENT_DATA_SOURCE:-remote}" == "repository" ]]; then')
        block = launcher[start:launcher.index('\nif [[ -n "$WORKER_MODEL_REMOTE"', start)]
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            (job / 'MATH').mkdir()
            (job / 'overall_math').mkdir()
            train = pd.concat([frame('train'), frame(' train ')], ignore_index=True)
            train['teacher_attempts'] = [['hidden solution']] * 2
            train.to_parquet(job / 'MATH/train.parquet')
            frame('held out').to_parquet(job / 'overall_math/test.parquet')
            shell = r'''
set -euo pipefail
COMMON_MOUNTS=(--bind /tmp:/tmp)
srun() {
    printf '%s\n' "$*" >> "$CALLS"
    while [[ "$1" != apptainer ]]; do shift; done
    "$@"
}
apptainer() {
    while [[ "$1" != python3 ]]; do shift; done
    shift
    [[ "$1" == /root/ReMA-public/scripts/eac3_experiments.py ]]
    shift
    "$PYTHON" "$SUITE" "$@"
}
rclone() { echo 'unexpected S3 transfer' >&2; return 99; }
'''
            env = {**os.environ, 'JOB_TMP': str(job), 'SLURM_NNODES': '4',
                   'EXPERIMENT_DATA_SOURCE': 'repository', 'EAC3_RUN_FILE': 'run.json',
                   'EXPERIMENT_TRAIN_RELATIVE': 'MATH/train.parquet',
                   'EXPERIMENT_EVAL_RELATIVE': 'overall_math/test.parquet', 'SIF_NAME': 'test.sif',
                   'CALLS': str(job / 'calls.txt'), 'PYTHON': sys.executable,
                   'SUITE': str(ROOT / 'scripts/eac3_experiments.py')}
            result = subprocess.run(['bash', '-c', shell + block + '\nprintf "%s\\n" "$TEACHER_TRAIN_FILE" "$AGENT12_VAL_FILE"'],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            prepared = job / 'experiment_data'
            self.assertEqual(result.stdout.splitlines()[-2:], [str(prepared / 'train.parquet'), str(prepared / 'eval.parquet')])
            self.assertIn('--nodes=4 --ntasks=4', (job / 'calls.txt').read_text())
            self.assertEqual(len(pd.read_parquet(prepared / 'train.parquet')), 1)
            self.assertNotIn('teacher_attempts', pd.read_parquet(prepared / 'train.parquet'))
            self.assertEqual(len(pd.read_parquet(prepared / 'eval.parquet')), 1)
            self.assertTrue((prepared / 'data.json').is_file())
            # The prepared files must bypass the legacy teacher-cache loader.
            self.assertIn('elif [[ -n "${EAC3_RUN_FILE:-}" && "${EXPERIMENT_DATA_SOURCE:-remote}" == "repository" ]]; then\n'
                          '    echo "Using locally prepared experiment questions: $TEACHER_TRAIN_FILE"\n'
                          'else\n    reuse_teacher_data', launcher)
            self.assertIn('"${EXPERIMENT_DATA_SOURCE:-remote}" != "repository" || "$EXPERIMENT_MODE" == "replay_eval"', launcher)

    def test_invalid_manifest_fails_before_launch(self):
        cfg = config()
        cfg['rollout_n'] = 31
        with self.assertRaises(ValueError):
            suite.make_runs(cfg, 'workers')
        cfg = config()
        cfg['planners']['eac3'] = 'SET_ME_MODEL'
        with self.assertRaises(ValueError):
            suite.run_environment(suite.make_runs(cfg, 'planners')[1])
        run = suite.make_runs(config(), 'workers')[0]
        run['worker'] = 'different/initial-model'
        with self.assertRaises(ValueError):
            suite.run_environment(run)

    def test_launcher_applies_final_overrides_safely(self):
        launcher = (ROOT / 'agent12-curriculum-trainer-multinode.sh').read_text()
        defaults = launcher[:launcher.index('\nif (( GPUS_PER_NODE')].replace('source ./env.sh', ':')
        function = launcher[launcher.index('run_agent12_training() {'):launcher.index('\nif [[ "$ONLINE_TEACHER_GENERATION"')]
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / 'command'
            run = suite.make_runs(config(), 'transfer')[0]
            env = {**os.environ, **{k: str(v) for k, v in suite.run_environment(run).items()},
                   'SLURM_JOB_ID': '123', 'SLURM_NNODES': '6', 'CAPTURE': str(capture)}
            shell = defaults + '\nsrun() { printf "%s\\n" "${!#}" > "$CAPTURE"; }\n'
            shell += 'HEAD_NODE=head; IP_HEAD=head:6379; COMMON_MOUNTS=(--nv)\n' + function
            shell += '\nrun_agent12_training /tmp/train.parquet "$TOTAL_STEPS" True True\n'
            result = subprocess.run(['bash', '-c', shell], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            command = shlex.split(capture.read_text().rsplit(';', 1)[-1])
            overrides = dict(arg.split('=', 1) for arg in command if '=' in arg and not arg.startswith('--'))
            self.assertEqual(overrides['trainer.val_only'], 'true')
            self.assertEqual(overrides['data.replay_plans'], 'true')
            self.assertEqual(overrides['trainer.validation_output_dir'], '${trainer.default_local_dir}/evaluations')
            self.assertEqual(overrides['algorithm.hierarchy.agent12_curriculum.enable'], 'false')

    def test_disjoint_data_and_no_teacher_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            train, val = tmp / 'train.parquet', tmp / 'val.parquet'
            f = frame('training question')
            f['teacher_attempts'] = [['private answer']]
            f.to_parquet(train)
            frame('held-out question').to_parquet(val)
            suite.prepare_data(train, val, tmp / 'prepared')
            self.assertNotIn('teacher_attempts', pd.read_parquet(tmp / 'prepared/train.parquet'))
            frame(' training   question ').to_parquet(val)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                suite.data_frames(train, val)

    def test_replay_cannot_leak_into_training_or_fallback(self):
        values = {'question': ['q'], 'fixed_decomposer_output': ['PLAN: S1: Compute.']}
        self.assertEqual(evidence.validation_plans(values, enabled=True, validation=True, max_turns=1), values['fixed_decomposer_output'])
        for validation, turns in [(False, 1), (True, 2)]:
            with self.assertRaises(ValueError):
                evidence.validation_plans(values, enabled=True, validation=validation, max_turns=turns)
        for bad in [None, [''], []]:
            values['fixed_decomposer_output'] = bad
            with self.assertRaises(ValueError):
                evidence.validation_plans(values, enabled=True, validation=True, max_turns=1)

    def test_deduplication_rejects_conflicting_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            train, val = Path(tmp) / 'train.parquet', Path(tmp) / 'eval.parquet'
            duplicate = pd.concat([frame('q'), frame(' q ')], ignore_index=True)
            duplicate.to_parquet(train)
            frame('new').to_parquet(val)
            cleaned, _ = suite.data_frames(train, val, deduplicate_train=True)
            self.assertEqual(len(cleaned), 1)
            duplicate.at[1, 'reward_model'] = {'style': 'rule', 'ground_truth': '8'}
            duplicate.to_parquet(train)
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                suite.data_frames(train, val, deduplicate_train=True)

    def test_plan_bank_requires_outcome_blind_review_and_tracks_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = []
            for name, question in [('baseline', 'q1'), ('eac3', 'q2')]:
                writer = evidence.ValidationRecorder(tmp / name, 'val', 0, 0, 1, {})
                batch = {key: frame(question)[key].to_numpy() for key in frame(question)}
                writer.add_batch(batch, [record(question)['history']], ['answer'], [1.], ['worker_stage_2'])
                paths.append((name, writer.finish({})))
            labels = tmp / 'review.csv'
            analysis.plan_review(paths, labels)
            with self.assertRaises(ValueError):
                analysis.make_plan_bank(paths, labels, tmp / 'bank.parquet')
            with labels.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            for index, row in enumerate(rows):
                row.update(supplies_computed_answers='unknown' if index else 'no', executable='yes', reviewer='reviewer1')
            with labels.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader(); writer.writerows(rows)
            analysis.make_plan_bank(paths, labels, tmp / 'bank.parquet')
            bank = pd.read_parquet(tmp / 'bank.parquet')
            self.assertEqual(len(bank), 1)
            self.assertTrue(bank.answer_free_reviewed.all())
            self.assertEqual(json.loads((tmp / 'bank.json').read_text())['excluded'], {'unknown/yes': 1})
            frame('training').to_parquet(tmp / 'train.parquet')
            suite.data_frames(tmp / 'train.parquet', tmp / 'bank.parquet', replay=True)
            bank['answer_free_reviewed'] = False
            bank.to_parquet(tmp / 'bad.parquet')
            with self.assertRaisesRegex(ValueError, 'reviewed'):
                suite.data_frames(tmp / 'train.parquet', tmp / 'bad.parquet', replay=True)

    def test_plan_writes_no_submissions_and_wont_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'runs'
            suite.write_runs(config(), 'planners', directory)
            self.assertEqual(len(list(directory.glob('*.json'))), 2)
            self.assertEqual(len((directory / 'commands.txt').read_text().splitlines()), 2)
            with self.assertRaises(FileExistsError):
                suite.write_runs(config(), 'planners', directory)

    def test_complete_evidence_and_mismatched_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = evidence.ValidationRecorder(tmp, 'val', 1, 0, 1, {})
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                writer.finish({})
            batch = {key: frame('q')[key].to_numpy() for key in frame('q')}
            writer.add_batch(batch, [record()['history']], ['answer'], [np.float32(1.)], ['worker_stage_2'])
            path = writer.finish({'val/acc/math': 1.})
            self.assertEqual(len(analysis.load_records(path)), 1)
            with self.assertRaises(FileExistsError):
                evidence.ValidationRecorder(tmp, 'val', 1, 0, 1, {})
            path.write_text(path.read_text() + '{}\n')
            with self.assertRaises(ValueError):
                analysis.load_records(path)

    def test_paired_accuracy_and_question_clustered_weight(self):
        a = [record('q1', score=0, item='a'), record('q1', score=0, item='b'), record('q2', score=1, item='c')]
        b = [record('q1', score=1, item='a'), record('q1', score=1, item='b'), record('q2', score=1, item='c')]
        result = analysis.compare(a, b, common_plans=True, resamples=100)
        self.assertEqual(result['right_minus_left_accuracy'], .5)
        b[0]['plan_id'] = 'different'
        with self.assertRaises(ValueError):
            analysis.compare(a, b, common_plans=True)
        with self.assertRaises(ValueError):
            analysis.compare(a, b[:1])

    def test_blinded_audit_does_not_filter_wrong_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'audit'
            a = [record('q1', score=0), record('q2', score=1)]
            b = [record('q1', score=1), record('q2', score=0)]
            analysis.audit_export(a, b, directory, count=2)
            rows = [json.loads(line) for line in (directory / 'examples.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all('raw_score' not in row and 'arm' not in row for row in rows))
            with self.assertRaises(ValueError):
                analysis.audit_summary(directory / 'labels.csv', directory / 'PRIVATE-key.json')

    def test_replay_uses_actual_rollout_routing_without_decomposer_generation(self):
        # Execute the real routing class with a fake generation boundary. Avoid
        # importing Ray/FSDP/torch in this CPU-only contract test.
        path = ROOT / 'src/verl/verl/rema_separated_trainer/ppo/multi_agent_rollout.py'
        parsed = ast.parse(path.read_text())
        definitions = [node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
        tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *definitions], type_ignores=[])
        ast.fix_missing_locations(tree)
        namespace = dict(np=np, re=re, hashlib=hashlib, unicodedata=unicodedata, extract_local_result=extract_local_result)
        probe_path = path.with_name('validation_probes.py')
        helper = next(node for node in ast.parse(probe_path.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == 'make_probe_inputs')
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(probe_path), 'exec'), namespace)
        exec(compile(tree, str(path), 'exec'), namespace)
        previous = sys.modules.get('verl.rema_separated_trainer.ppo.experiment_io')
        sys.modules['verl.rema_separated_trainer.ppo.experiment_io'] = evidence
        self.addCleanup(lambda: sys.modules.pop('verl.rema_separated_trainer.ppo.experiment_io', None) if previous is None else sys.modules.update({'verl.rema_separated_trainer.ppo.experiment_io': previous}))
        rollout = object.__new__(namespace['MultiAgentRollout'])
        rollout.config = SimpleNamespace(stop_when_truncated=True)
        stages = ['worker_stage_1', 'worker_stage_2', 'worker_stage_3', 'worker_stage_4']
        roles = ['decomposer', 'selector', *stages]
        hierarchy = dict(stage_roles=stages, worker_roles=['general_math_worker'], routing_mode='sequential_plan',
                         max_planned_subtasks=4, terminal_worker_as_answer=True, worker_context_mode='full_question')
        chats_seen = {}
        def generate(role, chats, *args, **kwargs):
            self.assertNotIn(role, ['decomposer', 'selector'])
            chats_seen[role] = chats
            outputs = ['REASONING: Compute.\nLOCAL_RESULT: \\boxed{7}'] * len(chats)
            return outputs, [12] * len(chats), ['stop'] * len(chats), None, [[7]] * len(chats)
        rollout._generate_from_chat_list = generate
        plan = 'PLAN:\n- S1: Compute x.\n- S2: Use the result of S1.'
        prompts = SimpleNamespace(non_tensor_batch={'question': np.array(['QUESTION_SENTINEL'], dtype=object),
                                                   'fixed_decomposer_output': [plan]},
                                  meta_info={'validate': True, 'replay_plans': True})
        system = runpy.run_path(str(ROOT / 'prompt/math/hierarchical_mamrp.py'))['build_hierarchical_system_prompts'](stages)
        histories, flags, reasons = rollout._initialize_conversation_state(1)
        outputs, _, _, state = rollout._run_hierarchical_conversation(prompts, {}, 1, roles, system, hierarchy,
                                                                     histories, flags, reasons, 1024, None)
        self.assertEqual(state['terminal_stage_roles'], ['worker_stage_2'])
        self.assertEqual(histories[0][0]['content'], plan)
        self.assertFalse(histories[0][0]['executed'])
        self.assertTrue(histories[0][0]['provided_plan'])
        self.assertIn('QUESTION_SENTINEL', str(chats_seen['worker_stage_1']))
        self.assertNotIn('QUESTION_SENTINEL', str(chats_seen['worker_stage_2']))
        self.assertIn('\\boxed{7}', str(chats_seen['worker_stage_2']))
        self.assertIn('\\boxed{7}', outputs[0])
        terminal = next(record for record in histories[0] if record['role'] == 'worker_stage_2')
        inputs = terminal['validation_probe_inputs']
        self.assertEqual(inputs['terminal_only'], 'CURRENT TASK:\n- S2: Use the result of S1.')
        self.assertIn('S1: Compute x.', inputs['with_instructions'])
        self.assertNotIn('\\boxed{7}', inputs['with_instructions'])
        self.assertIn('S1 LOCAL_RESULT: \\boxed{7}', inputs['with_results'])
        self.assertNotIn('QUESTION_SENTINEL', str(inputs))
        self.assertNotIn('REASONING:', str(inputs))


if __name__ == '__main__':
    unittest.main()

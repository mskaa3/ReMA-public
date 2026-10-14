"""CPU-only experiment contract tests; runnable with unittest, without Ray/CUDA."""

import ast
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
            for phase in suite.PHASES:
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
                self.assertEqual(cfg['worker_updates'], 100)
                directory = Path(tmp) / cfg['study']
                suite.write_runs(cfg, 'workers', directory)
                self.assertEqual(len(list(directory.glob('*.json'))), 1)
                self.assertEqual(len((directory / 'commands.txt').read_text().splitlines()), 1)
                run = suite.read_json(next(directory.glob('*.json')))
                env = suite.run_environment(run)
                self.assertEqual(env['PREFIX_PROBE_ENABLE'], 'true')
                self.assertEqual(env['TRAIN_DECOMPOSER'], 'false')
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
        self.assertFalse(due(state, 50, True))  # Checkpoint cadence stays unchanged.
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
        for phase in suite.PHASES:
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


if __name__ == '__main__':
    unittest.main()

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

"""CPU-only tests for the explicit round-2 fixed-plan comparison."""

import copy
import ast
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import eac3_experiments as suite
import eac3_fixed_plan_evaluation as fixed
import experiment_io


def config():
    return suite.read_json(ROOT / 'config/eac3-round2-fixed-plans.json')


def record(question, plan, score=0):
    return dict(question=question, question_id=suite.question_id(question), plan_id=suite.plan_id(plan),
                decomposer_output=plan, ground_truth='7', subset='math', data_source='ReMA-math',
                raw_score=score, completion_tokens=10, context='val', actor_update_step=0,
                terminal_role='worker_stage_2', history=[
                    {'role': 'decomposer', 'content': plan, 'num_gen_tokens': 0, 'executed': False},
                    {'role': 'worker_stage_1', 'content': 'LOCAL_RESULT: \\boxed{3}', 'executed': True,
                     'num_gen_tokens': 6, 'stop_reason': 'stop'},
                    {'role': 'worker_stage_2', 'content': '\\boxed{7}', 'executed': True,
                     'num_gen_tokens': 4, 'stop_reason': 'stop'},
                ])


def write_evaluation(directory, run, rows):
    directory.mkdir(parents=True)
    path = directory / 'val-step0-update0.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    path.with_suffix('.metadata.json').write_text(json.dumps({
        'rows': len(rows), 'records_sha256': suite.sha_file(path),
        'config': {'trainer': {'val_only': True}, 'data': {'teacher_assisted_validation': False},
                   'actor_rollout_ref': {'rollout': {'val_kwargs': {'n': 1, 'do_sample': False}}}},
    }))
    (directory / 'run.json').write_text(json.dumps(dict(run, code_sha256={'test': 'fingerprint'})))
    (directory / 'input-data.json').write_text(json.dumps({'eval_sha256': run.get('evaluation_sha256', '')}))
    return path


class FixedPlanTests(unittest.TestCase):
    def fixture_cache(self, tmp):
        cfg = config()
        plans = {'du': ['', 'PLAN:\n- S1: Return 7.'],
                 'dg': ['PLAN:\n- S1: Compute a.\n- S2: Use a.', 'PLAN:\n- S1: Compute b.']}
        sources = {}
        for run in suite.make_runs(cfg, 'cache-plans'):
            source = run['plan_source']
            rows = [record(f'q{i}', plan, i) for i, plan in enumerate(plans[source])]
            sources[source] = write_evaluation(tmp / source, run, rows)
        held_out = pd.DataFrame([dict(question=f'q{i}', subset='math', data_source='ReMA-math',
                                      reward_model={'style': 'rule', 'ground_truth': '7'}) for i in range(2)])
        with patch.object(suite, 'data_frames', return_value=(held_out, held_out)):
            replay = fixed.build_caches(cfg, sources, tmp / 'cache')
        return replay, sources

    def fixture_results(self, tmp, cfg):
        for run in suite.make_runs(cfg, 'replay-workers'):
            bank = pd.read_parquet(tmp / 'cache' / (run['plan_source'] + '.parquet'))
            rows = []
            for i, item in enumerate(bank.itertuples()):
                score = 1 if run['worker_label'].endswith('after') or i == 0 else 0
                row = record(item.question, item.fixed_decomposer_output, score)
                row.update(eval_item_id=item.eval_item_id, plan_source=item.plan_source,
                           answer_free_reviewed=False, plan_cache_unfiltered=True)
                rows.append(row)
            write_evaluation(tmp / 'results' / run['run_name'], run, rows)

    def test_explicit_checkpoints_and_no_legacy_phase(self):
        cfg = config()
        plans = suite.make_runs(cfg, 'cache-plans')
        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0]['worker'], cfg['workers']['u_before'])
        for phase in ('workers', 'cross', 'transfer', 'adapt-workers'):
            with self.assertRaisesRegex(ValueError, 'legacy phases'):
                suite.make_runs(cfg, phase)
        with self.assertRaisesRegex(ValueError, 'caches first'):
            suite.make_runs(cfg, 'replay-workers')
        for label, job in (('u_after', '6154441'), ('g_after', '6154442')):
            self.assertIn(f'recovered-{job}/snapshots/global_step_45/', cfg['workers'][label])
        self.assertIn('global_step_63', cfg['workers']['u_before'])
        self.assertIn('global_step_65', cfg['workers']['g_before'])

    def test_cache_keeps_every_question_and_pins_eight_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            cfg, _ = self.fixture_cache(tmp)
            bank = pd.read_parquet(tmp / 'cache/du.parquet')
            self.assertEqual(list(bank.fixed_decomposer_output)[0], '')
            self.assertEqual(len(bank), 2)
            self.assertFalse(bank.answer_free_reviewed.any())
            self.assertNotIn('raw_score', bank)
            jobs = suite.make_runs(cfg, 'replay-workers')
            self.assertEqual(len(jobs), 8)
            for run in jobs:
                self.assertEqual(run['worker'], cfg['workers'][run['worker_label']])
                self.assertEqual(run['planner'], cfg['planners']['du'])
                self.assertEqual(run['evaluation_sha256'], suite.sha_file(tmp / 'cache' / (run['plan_source'] + '.parquet')))
                env = suite.run_environment(run, check_allocation=False)
                options = dict(word.split('=', 1) for word in shlex.split(env['AGENT12_EXPERIMENT_OVERRIDES']))
                self.assertEqual(options['trainer.val_only'], 'true')
                self.assertEqual(options['actor_rollout_ref.rollout.val_kwargs.n'], '1')
                self.assertEqual(options['actor_rollout_ref.rollout.val_kwargs.do_sample'], 'false')
                self.assertEqual(options['data.replay_plans'], 'true')
                self.assertEqual(options['data.replay_allow_empty_plans'], 'true')
                self.assertEqual(options['algorithm.hierarchy.worker_context_mode'], 'full_question')
                self.assertEqual(env['PREFIX_PROBE_APPLY_TO_TRAINING'], 'false')
                self.assertEqual(env['PREFIX_PROBE_ENABLE'], 'true')
                self.assertEqual(env['VALIDATION_PROBE_BACKEND'], 'off')
            subprocess.run(['bash', '-n', str(tmp / 'cache/upload.sh')], check=True)

    def test_empty_cache_is_opt_in_and_not_replanned(self):
        values = {'question': ['q'], 'fixed_decomposer_output': ['']}
        self.assertEqual(experiment_io.validation_plans(values, enabled=True, validation=True,
                                                       max_turns=1, allow_empty=True), [''])
        with self.assertRaises(ValueError):
            experiment_io.validation_plans(values, enabled=True, validation=True, max_turns=1)
        with self.assertRaises(ValueError):
            experiment_io.validation_plans(values, enabled=True, validation=False, max_turns=1, allow_empty=True)
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            self.fixture_cache(tmp)
            pd.DataFrame([dict(question='training', data_source='ReMA-math', reward_model={'ground_truth': '8'})]).to_parquet(tmp / 'train.parquet')
            suite.data_frames(tmp / 'train.parquet', tmp / 'cache/du.parquet', replay=True, unreviewed_cache=True)
            with self.assertRaises(ValueError):
                suite.data_frames(tmp / 'train.parquet', tmp / 'cache/du.parquet', replay=True)
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/eac3_experiments.py'), 'validate-data',
                                     '--train', str(tmp / 'train.parquet'), '--eval', str(tmp / 'cache/du.parquet'),
                                     '--mode', 'cached_replay_eval', '--expected-eval-sha256', '0' * 64], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('parquet changed', result.stderr)

    def test_report_pairs_full_and_independent_review_without_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            cfg, sources = self.fixture_cache(tmp)
            self.fixture_results(tmp, cfg)
            fixed.report(cfg, tmp / 'results', tmp / 'report', resamples=40)
            report = suite.read_json(tmp / 'report/report.json')
            for row in report['summaries']:
                self.assertEqual(row['questions'], 2)
                self.assertEqual(row['u_before_accuracy'], 0.5)
                self.assertEqual(row['u_after_accuracy'], 1)
                self.assertEqual(row['u_delta_accuracy'], 0.5)
                self.assertEqual(row['g_delta_accuracy'], 0.5)
                self.assertEqual(row['g_after_worker_tokens_per_item'], 10)
            from analyze_eac3_experiments import plan_review
            labels = tmp / 'labels.csv'
            plan_review(list(sources.items()), labels)
            with labels.open() as stream:
                review = list(csv.DictReader(stream))
            for row in review:
                row.update(executable='yes' if row['plan'] else 'unknown', supplies_computed_answers='no', reviewer='independent')
            with labels.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(review[0]))
                writer.writeheader()
                writer.writerows(review)
            fixed.report(cfg, tmp / 'results', tmp / 'reviewed', labels=labels, resamples=40)
            summary = suite.read_json(tmp / 'reviewed/report.json')['summaries']
            self.assertEqual(next(row for row in summary if row['scope'] == 'reviewed' and row['plan_source'] == 'du')['questions'], 1)
            self.assertTrue(all(row['questions'] == 2 for row in summary if row['scope'] == 'full'))

    def test_report_rejects_worker_specific_plan_and_incomplete_run(self):
        rows = [record('q', 'PLAN: a')]
        rows[0].update(eval_item_id='item', plan_source='du')
        groups = {'du': {label: copy.deepcopy(rows) for label in fixed.LABELS}}
        groups['du']['g_after'][0]['decomposer_output'] = 'PLAN: changed'
        with self.assertRaisesRegex(ValueError, 'identical cached plans'):
            fixed.validate_pairs(groups)
        groups['du']['g_after'] = []
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            fixed.validate_pairs(groups)

    def test_report_retains_current_and_legacy_leakage_metrics(self):
        path = ROOT / 'src/verl/verl/rema_separated_trainer/ppo/validation_probes.py'
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'validation_metric_namespace')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        raw_metrics = {
            'val/leakage/raw_acc': 0.5,
            'val/leakage/gated_acc': 0.0,
            'val/leakage/plan_eligible_rate': 0.25,
            'val/acc/math': 0.5,
            'val_diagnostics/teacher_assisted/leakage/raw_acc': 1.0,
        }
        current_metrics = namespace['validation_metric_namespace'](raw_metrics)
        for prefix, metrics in (('val_diagnostics/leakage/', current_metrics),
                                ('val/leakage/', raw_metrics)):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as directory:
                tmp = Path(directory)
                cfg, _ = self.fixture_cache(tmp)
                self.fixture_results(tmp, cfg)
                for metadata_path in (tmp / 'results').glob('*/val-step0-update0.metadata.json'):
                    metadata = suite.read_json(metadata_path)
                    metadata['metrics'] = metrics
                    metadata_path.write_text(json.dumps(metadata))
                fixed.report(cfg, tmp / 'results', tmp / 'report', resamples=40)
                result = suite.read_json(tmp / 'report/report.json')
                expected = {prefix + 'raw_acc': 0.5, prefix + 'gated_acc': 0.0,
                            prefix + 'plan_eligible_rate': 0.25}
                self.assertEqual(len(result['diagnostic_metrics']), 8)
                for recorded in result['diagnostic_metrics'].values():
                    self.assertEqual(recorded, expected)

    def test_prefix_diagnostics_include_replayed_but_not_carried_plans(self):
        path = ROOT / 'src/verl/verl/rema_separated_trainer/ppo/prefix_probe.py'
        functions = [node for node in ast.parse(path.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name in ('answer_round_records', 'count_planned_subtasks')]
        tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *functions], type_ignores=[])
        ast.fix_missing_locations(tree)
        namespace = {}
        exec(compile(tree, str(path), 'exec'), namespace)
        records = [dict(role='decomposer', content='plan', executed=False, provided_plan=True, planned_subtask_count=2),
                   dict(role='worker_stage_2', executed=True, content='answer')]
        selected = namespace['answer_round_records'](records, 'worker_stage_2', 'decomposer')
        self.assertEqual(selected, records)
        self.assertEqual(namespace['count_planned_subtasks'](selected, 'decomposer'), 2)
        records[0]['provided_plan'] = False
        self.assertEqual(namespace['answer_round_records'](records, 'worker_stage_2', 'decomposer'), [])

    def test_cache_rejects_incomplete_or_wrong_checkpoint_source(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            _, sources = self.fixture_cache(tmp)
            manifest = sources['du'].parent / 'run.json'
            value = suite.read_json(manifest)
            value['planner'] = 's3:wrong-checkpoint'
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'checkpoints/design'):
                fixed.build_caches(config(), sources, tmp / 'bad')
            self.assertFalse((tmp / 'bad').exists())

    def test_cache_manifest_commands_and_fetch_script(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory) / 'runs'
            suite.write_runs(config(), 'cache-plans', tmp)
            self.assertEqual(len(list(tmp.glob('*.json'))), 2)
            subprocess.run(['bash', '-n', str(tmp / 'fetch-results.sh')], check=True)
            subprocess.run(['bash', '-n', str(tmp / 'commands.txt')], check=True)
            manifest = suite.read_json(next(tmp.glob('*.json')))
            self.assertIn('scripts/eac3_fixed_plan_evaluation.py', manifest['code_sha256'])

    def test_real_benchmark_cache_and_eight_replay_manifests(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            suite.write_runs(cfg, 'cache-plans', tmp / 'manifests')
            benchmark = pd.read_parquet(ROOT / cfg['eval_file'])
            sources = {}
            for manifest_path in (tmp / 'manifests').glob('*.json'):
                manifest = suite.read_json(manifest_path)
                rows = []
                for i, question in enumerate(benchmark.itertuples()):
                    row = record(question.question, '' if i == 0 else 'PLAN:\n- S1: Compute.\n- S2: Use S1.')
                    row.update(ground_truth=str(question.reward_model['ground_truth']),
                               data_source=str(question.data_source), subset=str(question.subset))
                    rows.append(row)
                source = manifest['plan_source']
                sources[source] = write_evaluation(tmp / source, manifest, rows)
                (tmp / source / 'run.json').write_text(json.dumps(manifest))
            replay = fixed.build_caches(cfg, sources, tmp / 'cache')
            suite.write_runs(replay, 'replay-workers', tmp / 'replay')
            self.assertEqual(len(list((tmp / 'replay').glob('*.json'))), 8)
            self.assertEqual(replay['plan_caches']['du']['rows'], len(benchmark))
            with self.assertRaisesRegex(ValueError, 'Code/prompts changed'):
                changed = copy.deepcopy(replay)
                changed['plan_caches']['du']['code_sha256'] = {}
                suite.write_runs(changed, 'replay-workers', tmp / 'changed')

    def test_launcher_effective_options_remain_inference_only(self):
        launcher = (ROOT / 'agent12-curriculum-trainer-multinode.sh').read_text()
        defaults = launcher[:launcher.index('\nif (( GPUS_PER_NODE')].replace('source ./env.sh', ':')
        function = launcher[launcher.index('run_agent12_training() {'):launcher.index('\nif [[ "$ONLINE_TEACHER_GENERATION"')]
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            cfg, _ = self.fixture_cache(tmp)
            run = suite.make_runs(cfg, 'replay-workers')[0]
            env = {**os.environ, **{k: str(v) for k, v in suite.run_environment(run, check_allocation=False).items()},
                   'SLURM_JOB_ID': '123', 'SLURM_NNODES': '4', 'CAPTURE': str(tmp / 'command')}
            shell = defaults + '\nsrun() { printf "%s\\n" "${!#}" > "$CAPTURE"; }\n'
            shell += 'HEAD_NODE=head; IP_HEAD=head:6379; COMMON_MOUNTS=(--nv)\n' + function
            shell += '\nrun_agent12_training /tmp/train.parquet "$TOTAL_STEPS" True True\n'
            result = subprocess.run(['bash', '-c', shell], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            words = shlex.split((tmp / 'command').read_text().rsplit(';', 1)[-1])
            opts = dict(word.split('=', 1) for word in words if '=' in word and not word.startswith('--'))
            self.assertEqual(opts['trainer.val_only'], 'true')
            self.assertEqual(opts['data.replay_plans'], 'true')
            self.assertEqual(opts['actor_rollout_ref.rollout.val_kwargs.n'], '1')
            self.assertEqual(opts['algorithm.hierarchy.agent12_curriculum.enable'], 'false')


if __name__ == '__main__':
    unittest.main()

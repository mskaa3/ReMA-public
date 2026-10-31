"""CPU-only contracts for the fresh curriculum comparison. No cluster required."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import eac3_fresh_pair as pair


def config():
    return pair.read_json(ROOT / 'config/eac3-fresh-pair.json')


def run(stage='workers', arm='gated'):
    return dict(config=config(), stage=stage, arm=arm,
                worker_remote='' if stage == 'workers' else f's3:bucket/{arm}/workers/huggingface')


class FreshPairTests(unittest.TestCase):
    def test_arms_differ_only_in_gate_and_identity(self):
        for stage in ('workers', 'decomposers'):
            a = pair.environment(run(stage, 'gated'), '123', 4)
            b = pair.environment(run(stage, 'ungated'), '123', 4)
            differences = {key for key in a if a[key] != b[key]}
            expected = {'AGENT12_RUN_NAME', 'AGENT12_WANDB_RUN_ID', 'PREFIX_PROBE_APPLY_TO_TRAINING',
                        'C3_REQUIRE_ELIGIBLE_SUCCESS', 'AGENT1_ELIGIBILITY_MODE'}
            if stage == 'decomposers':
                expected.add('WORKER_MODEL_REMOTE')
            self.assertEqual(differences, expected)
            self.assertEqual(a['PREFIX_PROBE_ENABLE'], b['PREFIX_PROBE_ENABLE'])
            self.assertEqual(a['PREFIX_PROBE_ENABLE'], 'true')
            self.assertEqual(a['C3_BALANCE_WORKER_FUNCTIONS'], 'true')
            self.assertEqual(a['ROLLOUT_N'], '32')
            self.assertEqual(a['C3_CONTINUATIONS_PER_ACTION'], '4')

    def test_worker_bootstrap_and_decomposer_fade_have_separate_budgets(self):
        w = pair.environment(run(), '123', 4)
        d = pair.environment(run('decomposers'), '124', 4)
        self.assertEqual(w['WORKER_BOOTSTRAP_STEPS'], w['TOTAL_STEPS'])
        self.assertEqual(w['DECOMPOSER_TRANSFER_STEPS'], '0')
        self.assertEqual(w['WORKER_MODEL_REMOTE'], '')
        self.assertEqual(d['WORKER_BOOTSTRAP_STEPS'], '0')
        self.assertEqual(d['DECOMPOSER_TRANSFER_STEPS'], '50')
        self.assertEqual(w['DECOMPOSER_MODEL_REMOTE'], d['DECOMPOSER_MODEL_REMOTE'])
        self.assertEqual(d['DECOMPOSER_MODEL_REMOTE'], '')
        for env in (w, d):
            self.assertIn('trainer.max_actor_updates=100', env['AGENT12_EXPERIMENT_OVERRIDES'])
            self.assertIn('trainer.resume_mode=disable', env['AGENT12_EXPERIMENT_OVERRIDES'])
            self.assertEqual(env['TEACHER_ASSISTED_VALIDATION'], 'false')
            self.assertEqual(env['GENERATE_TEACHER_DATA'], '0')
            self.assertEqual(env['ONLINE_TEACHER_GENERATION'], 'false')
            self.assertEqual(env['OPTIMIZER_PROMPT_BATCH_SIZE'], '48')

    def test_invalid_budgets_and_allocations_fail(self):
        for key, value in (('worker_updates', 0), ('rollout_step_limit', 2),
                           ('decomposer_transfer_updates', 100), ('gpus_per_node', 3),
                           ('continuations_per_action', 3), ('seed', True),
                           ('balance_worker_functions', 'true')):
            cfg = config()
            cfg[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                pair.validate(cfg)
        with self.assertRaisesRegex(ValueError, 'node count'):
            pair.environment(run(), '123', 6)

    def test_effective_training_command_resists_inherited_and_site_settings(self):
        common = pair.COMMON.read_text()
        defaults = common[:common.index('\nif (( GPUS_PER_NODE')]
        # Mimic env.sh containing an old experiment, after the stage wrapper ran.
        defaults = defaults.replace('source ./env.sh', '\n'.join([
            'export WORKER_MODEL_REMOTE=s3:bucket/old-workers',
            'export DECOMPOSER_MODEL_REMOTE=s3:bucket/old-planner',
            'export TRAIN_DECOMPOSER=true', 'export C3_BALANCE_WORKER_FUNCTIONS=false',
            'export TEACHER_ASSISTED_VALIDATION=true',
            'export EAC3_RUN_FILE=/tmp/old.json',
        ]))
        function = common[common.index('run_agent12_training() {'):
                          common.index('\nif [[ "$ONLINE_TEACHER_GENERATION"')]
        for stage in ('workers', 'decomposers'):
            for arm in pair.ARMS:
                with self.subTest(stage=stage, arm=arm), tempfile.TemporaryDirectory() as tmp:
                    pinned = Path(tmp) / 'settings.sh'
                    pinned.write_text(pair.shell_environment(run(stage, arm), '123', 4))
                    wrapper = (ROOT / ('agent2-only-rl-trainer-multinode.sh' if stage == 'workers'
                                       else 'agent1-only-rl-trainer-multinode.sh')).read_text()
                    wrapper = wrapper[:wrapper.index('exec bash')]
                    capture = Path(tmp) / 'command.txt'
                    state = Path(tmp) / 'environment.txt'
                    shell = '\n'.join([
                        'source "$AGENT12_PINNED_ENV_FILE"', wrapper, defaults,
                        'env > "$STATE"',
                        'srun() { printf "%s\\n" "${!#}" > "$CAPTURE"; }',
                        'HEAD_NODE=head; IP_HEAD=head:6379; COMMON_MOUNTS=(--nv)', function,
                        'run_agent12_training /tmp/train.parquet "$TOTAL_STEPS" True True',
                    ])
                    result = subprocess.run(['bash', '-c', shell], text=True, capture_output=True, env={
                        'PATH': os.environ['PATH'], 'SLURM_JOB_ID': '123', 'SLURM_NNODES': '4',
                        'SLURM_SUBMIT_DIR': str(ROOT), 'AGENT12_PINNED_ENV_FILE': str(pinned),
                        'CAPTURE': str(capture), 'STATE': str(state),
                        'FROZEN_GATE_MODEL_REMOTE': 's3:bucket/old-auditor',
                        'VALIDATION_PROBE_URL': 'http://old-server',
                        'JOB_TMP': '/old-job', 'TRAIN_AGENT_ROLES': '[decomposer]',
                        'AGENT12_EXPERIMENT_OVERRIDES': 'trainer.resume_mode=auto',
                    })
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    env = dict(line.split('=', 1) for line in state.read_text().splitlines() if '=' in line)
                    self.assertNotIn('EAC3_RUN_FILE', env)
                    self.assertEqual(env['FROZEN_GATE_MODEL_REMOTE'], '')
                    self.assertEqual(env['VALIDATION_PROBE_URL'], '')
                    self.assertEqual(env['DECOMPOSER_MODEL_REMOTE'], '')
                    self.assertEqual(env['WORKER_MODEL_REMOTE'], run(stage, arm)['worker_remote'])
                    tokens = shlex.split(capture.read_text().rsplit(';', 1)[-1])
                    opts = dict(token.split('=', 1) for token in tokens if '=' in token and not token.startswith('--'))
                    self.assertEqual(opts['trainer.max_actor_updates'], '100')
                    self.assertEqual(opts['trainer.total_training_steps'], '1200')
                    self.assertEqual(opts['trainer.session_stop_step'], '1200')
                    self.assertEqual(opts['trainer.resume_mode'], 'disable')
                    self.assertEqual(opts['trainer.save_freq'], opts['trainer.test_freq'])
                    self.assertEqual(opts['trainer.test_freq'], '10')
                    h = 'algorithm.hierarchy.'
                    self.assertEqual(opts[h + 'num_worker_stages'], '4')
                    self.assertEqual(opts[h + 'terminal_worker_as_answer'], 'true')
                    self.assertEqual(opts[h + 'scoped_c3_grpo.balance_worker_functions'], 'true')
                    self.assertEqual(opts[h + 'scoped_c3_grpo.prefix_probe.enable'], 'true')
                    self.assertEqual(opts[h + 'scoped_c3_grpo.prefix_probe.apply_to_training'], str(arm == 'gated').lower())
                    expected_roles = '[decomposer]' if stage == 'decomposers' else '[worker_stage_1,worker_stage_2,worker_stage_3,worker_stage_4]'
                    self.assertEqual(opts[h + 'train_agent_roles'], expected_roles)
                    self.assertEqual(opts[h + 'agent12_curriculum.train_decomposer'], str(stage == 'decomposers').lower())

    def test_preparation_never_submits_and_requires_both_distinct_worker_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'workers'
            pair.prepare(config(), 'workers', path, dict(gated='', ungated=''))
            commands = (path / 'commands.sh').read_text()
            self.assertEqual(commands.count('sbatch '), 2)
            self.assertIn('--nodes=4', commands)
            self.assertNotIn('--exclude', commands)
            for arm in pair.ARMS:
                saved = pair.read_json(path / f'workers-{arm}.json')
                self.assertEqual(saved['worker_remote'], '')
                self.assertEqual(saved['code_sha256'], pair.fingerprint())
            with self.assertRaises(FileExistsError):
                pair.prepare(config(), 'workers', path, dict(gated='', ungated=''))
            for parents in (dict(gated='', ungated=''), dict(gated='s3:bucket/w', ungated='s3:bucket/w')):
                with self.assertRaises(ValueError):
                    pair.prepare(config(), 'decomposers', Path(tmp) / 'bad', parents)
            pair.prepare(config(), 'decomposers', Path(tmp) / 'decomposers',
                         dict(gated='s3:bucket/g/workers/huggingface', ungated='s3:bucket/u/workers/huggingface'))

    def test_code_changes_and_allocation_mismatch_are_rejected_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run.json'
            path.write_text(json.dumps(dict(run(), code_sha256={})))
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/eac3_fresh_pair.py'),
                                     'environment', '--run', str(path)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('changed since preparation', result.stderr)


if __name__ == '__main__':
    unittest.main()

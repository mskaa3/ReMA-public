#!/usr/bin/env python3
"""Prepare two teacher-assisted curriculum arms, without submitting GPU jobs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / 'agent12-curriculum-trainer-multinode.sh'
ARMS = ('gated', 'ungated')


def read_json(path):
    return json.loads(Path(path).read_text())


def validate(cfg):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', cfg['study']):
        raise ValueError('Use a filesystem-safe study name')
    for key in ('worker_updates', 'decomposer_updates', 'decomposer_transfer_updates',
                'rollout_step_limit', 'test_freq', 'save_freq', 'base_question_batch_size',
                'teacher_attempts', 'rollout_n', 'continuations_per_action', 'nodes', 'gpus_per_node'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(cfg['seed']) is not int or cfg['seed'] < 0:
        raise ValueError('seed must be a non-negative integer')
    if type(cfg['balance_worker_functions']) is not bool:
        raise ValueError('balance_worker_functions must be boolean')
    if cfg['rollout_step_limit'] < max(cfg['worker_updates'], cfg['decomposer_updates']):
        raise ValueError('rollout_step_limit must cover both update budgets')
    if cfg['decomposer_transfer_updates'] >= cfg['decomposer_updates']:
        raise ValueError('Leave some decomposer updates after teacher access reaches zero')
    n, m = cfg['rollout_n'], cfg['continuations_per_action']
    if n % m or n // m < 2:
        raise ValueError('C3 requires complete suffix groups and at least two actions')
    pool = cfg['nodes'] * cfg['gpus_per_node'] // 2
    batch = cfg['base_question_batch_size'] * cfg['teacher_attempts'] * n
    if cfg['gpus_per_node'] % 2 or batch % pool:
        raise ValueError('Use two equal GPU pools and a divisible expanded trajectory batch')
    for key in ('remote_root', 'teacher_shards_remote'):
        remote(cfg[key])
    for key in ('decomposer_model', 'worker_model'):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', cfg[key]):
            raise ValueError(f'{key} must be a base Hugging Face model ID, not an RL export')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', cfg['partition']):
        raise ValueError('Invalid partition')
    if not re.fullmatch(r'(?:[0-9]+-)?[0-9]+:[0-9]{2}:[0-9]{2}', cfg['time']):
        raise ValueError('Invalid Slurm time limit')


def remote(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+:[A-Za-z0-9_./+-]+', value):
        raise ValueError('Expected an explicit rclone remote path')
    if any(marker in value for marker in ('SET_ME', 'YOUR_', '..')):
        raise ValueError('Replace placeholder remote paths')
    return value.rstrip('/')


def fingerprint():
    paths = [COMMON, Path(__file__), ROOT / 'eac3-fresh-pair-multinode.sh',
             ROOT / 'agent1-only-rl-trainer-multinode.sh', ROOT / 'agent2-only-rl-trainer-multinode.sh',
             ROOT / 'config/rema-rl.yaml', ROOT / 'scripts/merge_agent12_teacher_shards.py']
    paths += sorted((ROOT / 'src/verl/verl').rglob('*.py'))
    paths += sorted((ROOT / 'prompt').rglob('*.py'))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def environment(run, job_id, nodes):
    cfg, stage, arm = run['config'], run['stage'], run['arm']
    validate(cfg)
    if stage not in ('workers', 'decomposers') or arm not in ARMS:
        raise ValueError('Invalid stage or arm')
    if not re.fullmatch(r'[0-9]+', str(job_id)) or int(nodes) != cfg['nodes']:
        raise ValueError('Submit with the node count recorded in the manifest')
    worker_remote = run['worker_remote']
    if stage == 'workers' and worker_remote:
        raise ValueError('Fresh workers must not load a previous RL export')
    if stage == 'decomposers' and not remote(worker_remote).endswith('/workers/huggingface'):
        raise ValueError('Decomposer training needs this branch\'s workers/huggingface export')
    gated = arm == 'gated'
    updates = cfg['worker_updates'] if stage == 'workers' else cfg['decomposer_updates']
    name = f"{cfg['study']}-{stage}-{arm}-s{cfg['seed']}-{job_id}"
    scratch = f'/mnt/lscratch/slurm/{job_id}/agent12'
    overrides = {
        'data.seed': cfg['seed'],
        'actor_rollout_ref.rollout.seed': cfg['seed'],
        'actor_rollout_ref.rollout.max_num_turns': 1,
        'actor_rollout_ref.rollout.val_kwargs.n': 1,
        'actor_rollout_ref.rollout.val_kwargs.do_sample': False,
        'actor_rollout_ref.rollout.val_kwargs.temperature': 0.0,
        'algorithm.hierarchy.routing_mode': 'sequential_plan',
        'algorithm.hierarchy.accept_revise.enable': False,
        'algorithm.hierarchy.worker_context_mode': 'full_question',
        'algorithm.hierarchy.pass_question_to_workers': True,
        'algorithm.hierarchy.scoped_c3_grpo.enable': True,
        'algorithm.hierarchy.scoped_c3_grpo.positive_only_nonterminal_workers': True,
        'algorithm.hierarchy.scoped_c3_grpo.prefix_probe.solver_role': 'decomposer',
        'trainer.resume_mode': 'disable',
        'trainer.val_before_train': True,
        'trainer.max_actor_updates': updates,
        'trainer.validation_output_dir': '${trainer.default_local_dir}/evaluations',
    }
    env = {
        'GPUS_PER_NODE': cfg['gpus_per_node'], 'POOL_GPUS_PER_NODE': cfg['gpus_per_node'] // 2,
        'TRAINING_NNODES': cfg['nodes'],
        'DECOMPOSER_MODEL_PATH': cfg['decomposer_model'], 'DECOMPOSER_MODEL_REMOTE': '',
        'WORKER_MODEL_PATH': cfg['worker_model'], 'WORKER_MODEL_REMOTE': worker_remote,
        'TRAIN_DECOMPOSER': str(stage == 'decomposers').lower(),
        'TRAIN_AGENT_ROLES': ('[decomposer]' if stage == 'decomposers' else
                              '[worker_stage_1,worker_stage_2,worker_stage_3,worker_stage_4]'),
        'TERMINAL_WORKER_AS_ANSWER': 'true', 'NUM_WORKER_STAGES': 4,
        'MAX_PLANNED_SUBTASKS': 4, 'WORKER_CHECKPOINT_ROLE': 'worker_stage_4',
        'AGENT12_RUN_NAME': name, 'AGENT12_WANDB_RUN_ID': name,
        'AGENT12_S3_REMOTE': cfg['remote_root'],
        'JOB_TMP': scratch, 'CHECKPOINT_ROOT': scratch + '/checkpoints/agent12',
        'TEACHER_TRAIN_SHARDS_REMOTE': cfg['teacher_shards_remote'],
        'TEACHER_ROLLOUT_N': cfg['teacher_attempts'],
        'GENERATE_TEACHER_DATA': 0, 'ONLINE_TEACHER_GENERATION': 'false', 'AGENT2_PILOT': 'false',
        'BASE_QUESTION_BATCH_SIZE': cfg['base_question_batch_size'],
        'OPTIMIZER_PROMPT_BATCH_SIZE': cfg['base_question_batch_size'] * cfg['teacher_attempts'],
        'ROLLOUT_N': cfg['rollout_n'], 'C3_CONTINUATIONS_PER_ACTION': cfg['continuations_per_action'],
        'C3_BALANCE_WORKER_FUNCTIONS': str(cfg['balance_worker_functions']).lower(),
        'C3_SELECT_AVAILABLE_WORKER': 'true',
        'PREFIX_PROBE_ENABLE': 'true', 'PREFIX_PROBE_APPLY_TO_TRAINING': str(gated).lower(),
        'C3_REQUIRE_ELIGIBLE_SUCCESS': str(gated).lower(),
        'PREFIX_PROBE_SOLVER_BACKEND': 'role', 'VALIDATION_PROBE_BACKEND': 'off',
        'ROLLOUT_PROBE_PIPELINE': 'false', 'TEACHER_ASSISTED_VALIDATION': 'false',
        'AGENT1_ELIGIBILITY_MODE': 'eac3' if gated else 'baseline',
        'WORKER_BOOTSTRAP_STEPS': cfg['rollout_step_limit'] if stage == 'workers' else 0,
        'DECOMPOSER_TRANSFER_STEPS': cfg['decomposer_transfer_updates'] if stage == 'decomposers' else 0,
        'WORKER_QUESTION_BOOTSTRAP_PROBABILITY': 1.0, 'WORKER_QUESTION_FINAL_PROBABILITY': 1.0,
        'WORKER_QUESTION_EVAL_PROBABILITY': 1.0, 'WORKER_QUESTION_FADE_STEPS': 0, 'JOINT_STEPS': 0,
        'TOTAL_STEPS': cfg['rollout_step_limit'], 'TEST_FREQ': cfg['test_freq'], 'SAVE_FREQ': cfg['save_freq'],
        'AGENT12_EXPERIMENT_OVERRIDES': ' '.join(shlex.quote(
            key + '=' + (value if isinstance(value, str) else json.dumps(value))
        ) for key, value in overrides.items()),
    }
    return {key: str(value) for key, value in env.items()}


def shell_environment(run, job_id, nodes):
    # Reset launcher knobs, not credentials. env.sh is sourced before this file
    # again in the common launcher, so stale site/user model settings cannot win.
    # Later command strings also contain exports (TMPDIR, PYTHONPATH, etc.).
    # They are runtime variables, not experiment settings to clear on the host.
    settings = re.findall(
        r'^# BEGIN_AGENT12_LAUNCH_SETTINGS\n(.*?)^# END_AGENT12_LAUNCH_SETTINGS$',
        COMMON.read_text(), re.M | re.S,
    )
    if len(settings) != 1:
        raise ValueError('Expected exactly one marked Agent 1/2 launcher settings section')
    reset = set(re.findall(r'^export ([A-Z][A-Z0-9_]*)=', settings[0], re.M))
    reset -= {'SIF_REMOTE', 'SIF_NAME'}
    reset.update(('EAC3_RUN_FILE', 'AGENT12_EXPERIMENT_OVERRIDES', 'WANDB_RUN_ID', 'WANDB_RESUME'))
    return '\n'.join(['unset ' + ' '.join(sorted(reset))] + [
        f'export {key}={shlex.quote(value)}' for key, value in environment(run, job_id, nodes).items()
    ]) + '\n'


def prepare(cfg, stage, output, workers):
    validate(cfg)
    if stage == 'workers' and any(workers.values()):
        raise ValueError('Do not supply worker checkpoints for the fresh worker stage')
    if stage == 'decomposers':
        for value in workers.values():
            remote(value)
        if workers['gated'] == workers['ungated']:
            raise ValueError('Supply distinct trained worker exports for the two branches')
    runs = [dict(schema_version=1, config=cfg, stage=stage, arm=arm,
                 worker_remote=workers[arm] or '') for arm in ARMS]
    for run in runs:
        environment(run, '0', cfg['nodes'])
    hashes = fingerprint()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    commands = ['#!/bin/bash', 'set -euo pipefail',
                '# Run from the repository root. These submit two independent jobs.']
    for run in runs:
        run.update(code_sha256=hashes, git_sha=commit)
        path = output / f'{stage}-{run["arm"]}.json'
        path.write_text(json.dumps(run, indent=2) + '\n')
        commands.append(shlex.join([
            'sbatch', '--export=ALL', '--parsable', f'--nodes={cfg["nodes"]}',
            f'--partition={cfg["partition"]}', f'--time={cfg["time"]}',
            f'--gres=gpu:hopper:{cfg["gpus_per_node"]},storage:local:200G',
            f'--job-name={cfg["study"]}-{stage}-{run["arm"]}',
            'eac3-fresh-pair-multinode.sh', str(path),
        ]))
    (output / 'commands.sh').write_text('\n'.join(commands) + '\n')
    print(f'Prepared {output}/commands.sh. No jobs submitted.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('prepare')
    plan.add_argument('--config', default='config/eac3-fresh-pair.json')
    plan.add_argument('--stage', choices=('workers', 'decomposers'), required=True)
    plan.add_argument('--output-dir', required=True)
    plan.add_argument('--gated-workers', default='')
    plan.add_argument('--ungated-workers', default='')
    env = sub.add_parser('environment')
    env.add_argument('--run', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            prepare(read_json(args.config), args.stage, args.output_dir,
                    dict(gated=args.gated_workers, ungated=args.ungated_workers))
        else:
            run = read_json(args.run)
            if run.get('code_sha256') != fingerprint():
                raise ValueError('Training code/prompts changed since preparation; regenerate both manifests')
            print(shell_environment(run, os.environ.get('SLURM_JOB_ID', ''),
                                    os.environ.get('SLURM_NNODES', '0')), end='')
    except (ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Build opt-in EAC3 research runs; never submit jobs automatically."""

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
sys.path.insert(0, str(ROOT / 'src/verl/verl/rema_separated_trainer/ppo'))
from experiment_io import question_id, plan_id  # noqa: E402

PHASES = ('planners', 'workers', 'transfer', 'cross', 'adapt-planners', 'adapt-workers')
WORKER_ROLES = '[worker_stage_1,worker_stage_2,worker_stage_3,worker_stage_4]'


def read_json(path):
    return json.loads(Path(path).read_text())


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repository_data_files(cfg):
    files = {}
    for split in ('train', 'eval'):
        value = cfg.get(split + '_file', '')
        path = Path(value)
        if (not re.fullmatch(r'[A-Za-z0-9_./\-]+', value)
                or path.is_absolute() or '..' in path.parts
                or len(path.parts) < 3 or path.parts[0] != 'data'
                or path.parts[1] not in ('MATH', 'overall_math')
                or path.suffix != '.parquet'):
            raise ValueError(f'{split}_file must be a parquet under data/MATH or data/overall_math')
        files[split] = path.as_posix()
    return files


def validate_study(cfg):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', cfg['study']):
        raise ValueError('study must be a filesystem-safe, unique study name')
    planners = cfg['planners']
    if not planners or not set(planners).issubset({'baseline', 'eac3'}):
        raise ValueError('Specify baseline and/or eac3 planner exports')
    if len(planners) == 2 and planners['baseline'] == planners['eac3']:
        raise ValueError('Baseline and EAC3 must identify different planner artifacts')
    for key in ('worker_updates', 'rollout_step_limit', 'test_freq', 'save_freq',
                'train_question_batch_size', 'rollout_n', 'continuations_per_action',
                'nodes', 'gpus_per_node'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['rollout_step_limit'] < cfg['worker_updates']:
        raise ValueError('Rollout cap cannot be smaller than the optimizer-update target')
    if 'planner_updates' in cfg:
        if type(cfg['planner_updates']) is not int or cfg['planner_updates'] <= 0:
            raise ValueError('planner_updates must be a positive integer')
        if cfg['rollout_step_limit'] < cfg['planner_updates']:
            raise ValueError('Rollout cap cannot be smaller than planner_updates')
    n, m = cfg['rollout_n'], cfg['continuations_per_action']
    if n % m or n // m < 2:
        raise ValueError('C3 needs at least two actions and complete continuation groups')
    gpu = cfg['gpus_per_node']
    if gpu % 2 or (cfg['train_question_batch_size'] * n) % (cfg['nodes'] * gpu // 2):
        raise ValueError('Two equal GPU pools and a divisible trajectory batch are required')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Specify distinct training seeds')
    if any(type(seed) is not int or seed < 0 for seed in cfg['seeds'] + [cfg['eval_seed']]):
        raise ValueError('Seeds must be non-negative integers')
    if cfg.get('data_source', 'remote') not in ('remote', 'repository'):
        raise ValueError('data_source must be remote or repository')
    if cfg.get('data_source') == 'repository':
        repository_data_files(cfg)
    if cfg.get('worker_gate_mode', 'off') not in ('off', 'diagnostic', 'gated'):
        raise ValueError('worker_gate_mode must be off, diagnostic, or gated')
    test_steps = cfg.get('test_update_steps', [])
    if (not isinstance(test_steps, list)
            or any(type(step) is not int or step <= 0 for step in test_steps)
            or len(set(test_steps)) != len(test_steps)):
        raise ValueError('test_update_steps must be a list of distinct positive update numbers')
    probe = cfg.get('local_validation_probe')
    if probe is not None:
        if not isinstance(probe, dict):
            raise ValueError('local_validation_probe must be a configuration object')
        if not re.fullmatch(r'[0-9a-fA-F]{40}', str(probe.get('revision', ''))):
            raise ValueError('local_validation_probe.revision must be a fixed 40-character HF commit hash')
        for key, default in (('extra_nodes', 1), ('samples_per_dataset', 32),
                             ('max_new_tokens', 256), ('max_model_len', 16384), ('batch_size', 4)):
            value = probe.get(key, default)
            if type(value) is not int or value < 1:
                raise ValueError(f'local_validation_probe.{key} must be a positive integer')
        if probe.get('max_model_len', 16384) <= probe.get('max_new_tokens', 256):
            raise ValueError('Local probe context must exceed its output budget')
        if not re.fullmatch(r'[A-Za-z0-9_./\-]+', probe.get('model', 'Qwen/Qwen3-4B-Instruct-2507')):
            raise ValueError('local_validation_probe.model must identify a Hugging Face model')


def allocated_nodes(cfg):
    probe = cfg.get('local_validation_probe')
    return cfg['nodes'] + (probe.get('extra_nodes', 1) if probe is not None else 0)


def run_name(cfg, phase, arm, seed):
    return f"{cfg['study']}--{phase}-{arm}-s{seed}"


def worker_export(cfg, arm, seed, phase='workers'):
    return f"{cfg['remote_root'].rstrip('/')}/{run_name(cfg, phase, arm, seed)}/workers/huggingface"


def adapted_planner_export(cfg, arm, seed):
    return f"{cfg['remote_root'].rstrip('/')}/{run_name(cfg, 'adapt-planners', arm, seed)}/decomposer/huggingface"


def make_runs(cfg, phase):
    validate_study(cfg)
    if phase in ('adapt-planners', 'adapt-workers'):
        if set(cfg['planners']) != {'eac3'} or 'planner_updates' not in cfg:
            raise ValueError('Alternating phases require one eac3 planner and explicit planner_updates')
        if cfg.get('data_source') != 'repository':
            raise ValueError('Alternating phases require question-only repository data')
    jobs = []
    evaluation = (repository_data_files(cfg)['eval'] if cfg.get('data_source') == 'repository'
                  else cfg['eval_remote'])

    def add(arm, seed, planner, worker, mode):
        jobs.append({
            'schema_version': 1, 'run_name': run_name(cfg, phase, arm, seed),
            'phase': phase, 'arm': arm, 'seed': seed, 'mode': mode,
            'planner': planner, 'worker': worker,
            'evaluation_remote': cfg['common_plan_remote'] if mode == 'replay_eval' else evaluation,
            'study_config': cfg,
        })

    if phase == 'planners':
        for arm, planner in cfg['planners'].items():
            add(arm, cfg['eval_seed'], planner, cfg['initial_worker'], 'planner_eval')
    elif phase == 'workers':
        for seed in cfg['seeds']:
            for arm, planner in cfg['planners'].items():
                add(arm, seed, planner, cfg['initial_worker'], 'worker_train')
    elif phase in ('adapt-planners', 'adapt-workers'):
        for seed in cfg['seeds']:
            planner = (cfg['planners']['eac3'] if phase == 'adapt-planners'
                       else adapted_planner_export(cfg, 'eac3', seed))
            mode = 'planner_train' if phase == 'adapt-planners' else 'worker_train'
            add('eac3', seed, planner, cfg['initial_worker'], mode)
    elif phase == 'transfer':
        # D is loaded for compatibility but never generates in fixed-plan evaluation.
        planner = next(iter(cfg['planners'].values()))
        add('initial', cfg['eval_seed'], planner, cfg['initial_worker'], 'replay_eval')
        for seed in cfg['seeds']:
            for arm in cfg['planners']:
                add(arm, seed, planner, worker_export(cfg, arm, seed), 'replay_eval')
    elif phase == 'cross':
        for seed in cfg['seeds']:
            for worker_arm in cfg['planners']:
                for planner_arm, planner in cfg['planners'].items():
                    add(f'{worker_arm}-under-{planner_arm}', seed, planner,
                        worker_export(cfg, worker_arm, seed), 'planner_eval')
    else:
        raise ValueError(f'Unknown phase: {phase}')
    return jobs


def model_env(prefix, artifact):
    if ':' in artifact:
        return {prefix + '_MODEL_REMOTE': artifact}
    return {prefix + '_MODEL_REMOTE': '', prefix + '_MODEL_PATH': artifact}


def run_environment(run, *, check_allocation=True):
    cfg = run['study_config']
    validate_study(cfg)
    expected = {item['run_name']: item for item in make_runs(cfg, run['phase'])}
    if run['run_name'] not in expected or any(
        run.get(key) != value for key, value in expected[run['run_name']].items()
    ):
        raise ValueError('Run differs from its declared experimental design; regenerate it')
    repository_data = cfg.get('data_source') == 'repository'
    data_files = repository_data_files(cfg) if repository_data else {}
    train_source = data_files['train'] if repository_data else cfg['train_remote']
    for artifact in (run['planner'], run['worker'], train_source, run['evaluation_remote'], cfg['remote_root']):
        if not artifact or 'SET_ME' in artifact or not re.fullmatch(r'[A-Za-z0-9_./:@+\-]+', artifact):
            raise ValueError(f'Replace the unresolved/invalid artifact before launch: {artifact!r}')
    if check_allocation and int(os.environ.get('SLURM_NNODES', allocated_nodes(cfg))) != allocated_nodes(cfg):
        raise ValueError('Slurm node allocation differs from the experiment manifest')
    train_planner = run['mode'] == 'planner_train'
    train = train_planner or run['mode'] == 'worker_train'
    gate_mode = cfg.get('worker_gate_mode', 'off')
    overrides = {
        'algorithm.hierarchy.agent12_curriculum.enable': False,
        'algorithm.hierarchy.start_agent': 'decomposer' if train_planner else 'worker_stage_1',
        'algorithm.hierarchy.scoped_c3_grpo.enable': True,
        'algorithm.hierarchy.scoped_c3_grpo.positive_only_nonterminal_workers': True,
        'algorithm.hierarchy.scoped_c3_grpo.prefix_probe.solver_role': 'decomposer',
        'algorithm.hierarchy.accept_revise.enable': False,
        'actor_rollout_ref.rollout.max_num_turns': 1,
        'algorithm.hierarchy.worker_context_mode': 'full_question',
        'algorithm.hierarchy.pass_question_to_workers': True,
        'actor_rollout_ref.rollout.seed': run['seed'],
        'data.seed': run['seed'],
        'actor_rollout_ref.rollout.val_kwargs.n': 1,
        'actor_rollout_ref.rollout.val_kwargs.do_sample': False,
        'data.replay_plans': run['mode'] == 'replay_eval',
        'trainer.val_only': not train,
        'trainer.val_before_train': True,
        'trainer.resume_mode': 'disable',
        'trainer.max_actor_updates': (cfg['planner_updates'] if train_planner
                                      else cfg['worker_updates'] if train else None),
        'trainer.test_update_steps': cfg.get('test_update_steps', []),
        'trainer.validation_output_dir': '${trainer.default_local_dir}/evaluations',
    }
    env = {
        'GPUS_PER_NODE': cfg['gpus_per_node'], 'POOL_GPUS_PER_NODE': cfg['gpus_per_node'] // 2,
        'TRAINING_NNODES': cfg['nodes'],
        'AGENT12_RUN_NAME': run['run_name'], 'AGENT12_S3_REMOTE': cfg['remote_root'],
        'AGENT12_WANDB_RUN_ID': run['run_name'] + '-' + os.environ.get('SLURM_JOB_ID', 'local'),
        'TRAIN_DECOMPOSER': str(train_planner).lower(),
        'TRAIN_AGENT_ROLES': '[decomposer]' if train_planner else WORKER_ROLES,
        'TERMINAL_WORKER_AS_ANSWER': 'true', 'NUM_WORKER_STAGES': 4,
        'MAX_PLANNED_SUBTASKS': 4, 'WORKER_CHECKPOINT_ROLE': 'worker_stage_4',
        'DECOMPOSER_USE_REMOVE_PADDING': 'true', 'WORKER_USE_REMOVE_PADDING': 'true',
        # Both adaptation chains use EAC3 for D, independent of their worker gate.
        'PREFIX_PROBE_ENABLE': str(train_planner or gate_mode != 'off').lower(),
        'PREFIX_PROBE_APPLY_TO_TRAINING': str(train_planner or (train and gate_mode == 'gated')).lower(),
        # Preserve the existing successful-group requirement in every arm.
        'C3_REQUIRE_ELIGIBLE_SUCCESS': 'true',
        'PREFIX_PROBE_MAX_NEW_TOKENS': 256,
        'PREFIX_PROBE_VALIDATION_MAX_SAMPLES': 128,
        'AGENT2_PILOT': 'false', 'GENERATE_TEACHER_DATA': 0, 'ONLINE_TEACHER_GENERATION': 'false',
        'TEACHER_TRAIN_REMOTE': '' if repository_data else train_source, 'TEACHER_TRAIN_SHARDS_REMOTE': '',
        'TEACHER_ASSISTED_VALIDATION': 'false', 'TEACHER_ROLLOUT_N': 1, 'TEACHER_SAMPLES_PER_CALL': 1,
        'BASE_QUESTION_BATCH_SIZE': cfg['train_question_batch_size'],
        'OPTIMIZER_PROMPT_BATCH_SIZE': cfg['train_question_batch_size'],
        'ROLLOUT_N': cfg['rollout_n'], 'C3_CONTINUATIONS_PER_ACTION': cfg['continuations_per_action'],
        'TOTAL_STEPS': cfg['rollout_step_limit'], 'TEST_FREQ': cfg['test_freq'], 'SAVE_FREQ': cfg['save_freq'],
        'EXPERIMENT_MODE': run['mode'], 'EXPERIMENT_VAL_REMOTE': run['evaluation_remote'],
        'EXPERIMENT_DATA_SOURCE': 'repository' if repository_data else 'remote',
        # The common launcher copies data/MATH and data/overall_math into JOB_TMP.
        'EXPERIMENT_TRAIN_RELATIVE': data_files['train'][len('data/'):] if repository_data else '',
        'EXPERIMENT_EVAL_RELATIVE': data_files['eval'][len('data/'):] if repository_data else '',
    }
    env.update(model_env('DECOMPOSER', run['planner']))
    env.update(model_env('WORKER', run['worker']))
    probe = cfg.get('local_validation_probe')
    if probe is not None:
        env.update({
            'VALIDATION_PROBE_BACKEND': 'local', 'VALIDATION_PROBE_URL': '',
            'VALIDATION_PROBE_MODEL': probe.get('model', 'Qwen/Qwen3-4B-Instruct-2507'),
            'VALIDATION_PROBE_REVISION': probe['revision'],
            'VALIDATION_PROBE_SAMPLES_PER_DATASET': probe.get('samples_per_dataset', 32),
            'VALIDATION_PROBE_MAX_NEW_TOKENS': probe.get('max_new_tokens', 256),
            'VALIDATION_PROBE_MAX_MODEL_LEN': probe.get('max_model_len', 16384),
            'VALIDATION_PROBE_BATCH_SIZE': probe.get('batch_size', 4),
        })
    # Shell quoting protects Hydra interpolations and strings from bash expansion.
    env['AGENT12_EXPERIMENT_OVERRIDES'] = ' '.join(shlex.quote(
        key + '=' + (json.dumps(value) if not isinstance(value, str) else value)
    ) for key, value in overrides.items())
    return env


def write_runs(cfg, phase, directory):
    runs = make_runs(cfg, phase)
    # Fail on unresolved artifacts before writing submission commands.
    for run in runs:
        run_environment(run, check_allocation=False)
    data_fingerprint = {}
    if cfg.get('data_source') == 'repository':
        data_fingerprint = {name: sha_file(ROOT / name) for name in repository_data_files(cfg).values()}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    code_files = [ROOT / 'agent12-curriculum-trainer-multinode.sh', ROOT / 'config/rema-rl.yaml',
                  ROOT / 'eac3-experiment-multinode.sh', Path(__file__)]
    code_files += sorted((ROOT / 'src/verl/verl/rema_separated_trainer/ppo').glob('*.py'))
    code_files += sorted((ROOT / 'prompt').rglob('*.py'))
    code_files += [ROOT / name for name in (
        'src/verl/verl/rema_separated_trainer/main_ppo.py',
        'src/verl/verl/workers/reward_manager/rema.py',
        'src/verl/verl/workers/fsdp_rema_workers.py',
        'src/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py',
        'src/verl/verl/utils/dataset/rema_dataset.py',
    )]
    fingerprint = {str(path.relative_to(ROOT)): sha_file(path) for path in code_files}
    commands = []
    for run in runs:
        run.update(git_sha=git_sha, code_sha256=fingerprint)
        if data_fingerprint:
            run['data_sha256'] = data_fingerprint
        path = directory / (run['run_name'] + '.json')
        path.write_text(json.dumps(run, indent=2) + '\n')
        command = ['sbatch', '--export=ALL', f"--nodes={allocated_nodes(cfg)}",
                   f"--gres=gpu:hopper:{cfg['gpus_per_node']},storage:local:200G",
                   'eac3-experiment-multinode.sh', str(path)]
        commands.append(shlex.join(command))
    (directory / 'commands.txt').write_text('\n'.join(commands) + '\n')
    print('\n'.join(commands))
    print(f'Wrote {len(runs)} run manifests. Nothing has been submitted.', file=sys.stderr)


def data_frames(train_path, eval_path, *, replay=False, deduplicate_train=False):
    import pandas as pd
    train, evaluation = pd.read_parquet(train_path), pd.read_parquet(eval_path)
    if deduplicate_train:
        for _, group in train.groupby(train.question.map(question_id)):
            if len({str(value['ground_truth']) for value in group.reward_model}) != 1:
                raise ValueError('Duplicate training questions have conflicting labels')
        train = train.loc[~train.question.map(question_id).duplicated()].copy()
    for label, frame in [('train', train), ('eval', evaluation)]:
        required = {'question', 'reward_model', 'data_source'}
        if frame.empty or not required.issubset(frame.columns):
            raise ValueError(f'{label}: non-empty data with {sorted(required)} required')
        if not all(isinstance(q, str) and q.strip() for q in frame.question):
            raise ValueError(f'{label}: questions must be non-empty strings')
        if not all(isinstance(r, dict) and 'ground_truth' in r for r in frame.reward_model):
            raise ValueError(f'{label}: missing ground truth for the existing verifier')
        if (label == 'train' or not replay) and frame.question.map(question_id).duplicated().any():
            raise ValueError(f'{label}: duplicate questions; deduplicate explicitly before this experiment')
    if set(train.question.map(question_id)) & set(evaluation.question.map(question_id)):
        raise ValueError('Training and held-out evaluation questions overlap')
    if replay:
        required = {'fixed_decomposer_output', 'eval_item_id', 'answer_free_reviewed'}
        if not required.issubset(evaluation.columns):
            raise ValueError(f'Replay bank requires {sorted(required)}')
        if evaluation.eval_item_id.duplicated().any():
            raise ValueError('Replay bank has duplicate evaluation items')
        if not all(isinstance(p, str) and p.strip() for p in evaluation.fixed_decomposer_output):
            raise ValueError('Replay bank contains missing plans')
        if not evaluation.answer_free_reviewed.eq(True).all():
            raise ValueError('Transfer test requires independently reviewed answer-free plans')
    return train, evaluation


def prepare_data(train_path, eval_path, directory, deduplicate_train=False):
    train, evaluation = data_frames(train_path, eval_path, deduplicate_train=deduplicate_train)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name, frame in [('train', train), ('eval', evaluation)]:
        # No candidate answers/teacher traces in the experimental planner input.
        columns = [key for key in ('question', 'reward_model', 'data_source', 'subset', 'extra_info') if key in frame]
        frame[columns].to_parquet(directory / f'{name}.parquet', index=False)
    manifest = {
        'train_questions': len(train), 'eval_questions': len(evaluation),
        'sha256': {name: sha_file(directory / f'{name}.parquet') for name in ('train', 'eval')},
        'split_rule': 'explicit input splits; normalized question overlap rejected; no outcome filtering',
        'deduplicate_train': deduplicate_train,
    }
    (directory / 'data.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--config', default='config/eac3-experiments.json')
    plan.add_argument('--phase', choices=PHASES, required=True)
    plan.add_argument('--output-dir', required=True)
    plan.add_argument('--local-probe-revision', help='Enable a frozen local Qwen3-4B probe at this HF commit')
    env = sub.add_parser('environment')
    env.add_argument('--run', required=True)
    prepare = sub.add_parser('prepare-data')
    check = sub.add_parser('validate-data')
    for command in (prepare, check):
        command.add_argument('--train', required=True)
        command.add_argument('--eval', required=True)
    prepare.add_argument('--output-dir', required=True)
    prepare.add_argument('--deduplicate-train', action='store_true')
    check.add_argument('--mode', choices=['worker_train', 'planner_train', 'planner_eval', 'replay_eval'], required=True)
    check.add_argument('--output-manifest')
    args = parser.parse_args()
    if args.command == 'plan':
        cfg = read_json(args.config)
        if args.local_probe_revision:
            cfg['local_validation_probe'] = {
                **(cfg.get('local_validation_probe') or {}),
                'revision': args.local_probe_revision,
            }
        write_runs(cfg, args.phase, args.output_dir)
    elif args.command == 'environment':
        run = read_json(args.run)
        for name, digest in run.get('code_sha256', {}).items():
            if sha_file(ROOT / name) != digest:
                raise ValueError(f'Code changed after manifest creation: {name}; regenerate manifests')
        for name, digest in run.get('data_sha256', {}).items():
            if sha_file(ROOT / name) != digest:
                raise ValueError(f'Data changed after manifest creation: {name}; regenerate manifests')
        # Drop inherited node-local paths; each independent experiment starts fresh.
        print('unset JOB_TMP RAY_NODE_TMP CHECKPOINT_ROOT TEACHER_TRAIN_FILE TEACHER_CANDIDATES_FILE')
        for name, value in run_environment(run).items():
            print(f'export {name}={shlex.quote(str(value))}')
    elif args.command == 'prepare-data':
        prepare_data(args.train, args.eval, args.output_dir, args.deduplicate_train)
    else:
        train, evaluation = data_frames(args.train, args.eval, replay=args.mode == 'replay_eval')
        print(f'Validated disjoint train={len(train)}, eval={len(evaluation)}; mode={args.mode}')
        if args.output_manifest:
            path = Path(args.output_manifest)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                'train_rows': len(train), 'eval_rows': len(evaluation),
                'train_sha256': sha_file(args.train), 'eval_sha256': sha_file(args.eval),
            }, indent=2) + '\n')


if __name__ == '__main__':
    main()

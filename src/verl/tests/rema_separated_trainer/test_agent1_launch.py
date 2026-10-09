"""Agent-1-only launch, frozen-worker scheduling, and model staging checks."""

import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

from verl.rema_separated_trainer.ppo.ray_trainer import RayReMASeparatedTrainer


ROOT = Path(__file__).resolve().parents[4]
WRAPPER = ROOT / "agent1-only-rl-trainer-multinode.sh"
LAUNCHER = (ROOT / "agent12-curriculum-trainer-multinode.sh").read_text()
VALIDATOR = ROOT / "scripts/validate_hf_model_dir.py"
validate_model_dir = runpy.run_path(str(VALIDATOR))["validate_model_dir"]


@pytest.fixture
def agent1_trainer(tmp_path):
    wrapper = WRAPPER.read_text()
    exports = LAUNCHER[:LAUNCHER.index("\nif (( GPUS_PER_NODE")].replace(
        "source ./env.sh", ": # No cluster credentials in tests"
    )
    training_function = LAUNCHER[
        LAUNCHER.index("run_agent12_training() {"):
        LAUNCHER.index('\nif [[ "$ONLINE_TEACHER_GENERATION"')
    ]
    capture = tmp_path / "training-command.txt"
    shell = "\n".join([
        wrapper[:wrapper.index('exec bash')], exports,
        # The staging block separately verifies this exact path on every node.
        'export WORKER_MODEL_PATH=${JOB_TMP}/worker_model',
        'srun() { printf "%s\\n" "${!#}" > "$CAPTURE"; }',
        'HEAD_NODE=head; IP_HEAD=head:6379; COMMON_MOUNTS=(--nv)',
        training_function,
        'run_agent12_training /tmp/teacher.train.parquet "$TOTAL_STEPS" True True',
    ])
    subprocess.run(["bash", "-c", shell], check=True, capture_output=True, text=True, env={
        "PATH": os.environ["PATH"], "SLURM_JOB_ID": "123", "SLURM_NNODES": "6",
        "SLURM_SUBMIT_DIR": str(ROOT), "CAPTURE": str(capture),
        "WORKER_MODEL_REMOTE": "s3:bucket/prior-run/workers/huggingface",
    })
    command = shlex.split(capture.read_text().rsplit(";", 1)[-1])
    config = OmegaConf.load(ROOT / "config/rema-rl.yaml")
    OmegaConf.set_struct(config, True)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist([
        arg for arg in command[3:] if not arg.startswith("--")
    ]))
    OmegaConf.resolve(config)
    trainer = RayReMASeparatedTrainer.__new__(RayReMASeparatedTrainer)
    trainer.config, trainer.use_critic, trainer.use_rm = config, False, False
    trainer._validate_config()
    trainer._init_scoped_c3_grpo()
    trainer._init_prefix_probe()
    return trainer


def test_agent1_actual_launcher_configuration(agent1_trainer):
    cfg = agent1_trainer.config
    hierarchy = agent1_trainer._get_hierarchy_config()
    assert agent1_trainer._get_train_agent_roles() == ["decomposer"]
    assert hierarchy["agent12_curriculum"]["train_decomposer"]
    assert cfg.algorithm.switch_agent.model_paths == [
        "Qwen/Qwen2.5-7B-Instruct", "/mnt/lscratch/slurm/123/agent12/worker_model",
    ]
    assert len(hierarchy["stage_roles"]) == hierarchy["max_planned_subtasks"] == 4
    assert hierarchy["terminal_worker_as_answer"]
    assert agent1_trainer.prefix_probe_enabled and agent1_trainer.scoped_c3_grpo_enabled
    assert cfg.actor_rollout_ref.rollout.n == 32
    assert agent1_trainer.scoped_c3_grpo_config["continuations_per_action"] == 4
    assert cfg.trainer.nnodes == 6 and cfg.trainer.n_gpus_per_node == 2
    assert cfg.trainer.total_training_steps == cfg.trainer.session_stop_step == 800
    assert cfg.trainer.wandb_run_id == "agent1-only-123"
    assert cfg.trainer.val_before_train and not cfg.data.teacher_assisted_validation
    curriculum = hierarchy["agent12_curriculum"]
    assert curriculum["worker_bootstrap_steps"] == 0
    assert curriculum["decomposer_transfer_steps"] == 200
    assert curriculum["worker_question_eval_probability"] == 1.0


@pytest.mark.parametrize("rollout_step,updates,probability", [
    (1, 0, 1.0), (300, 0, 1.0), (301, 100, .5), (600, 200, 0.0), (800, 400, 0.0),
])
def test_workers_never_selected_even_after_transfer_or_skipped_updates(
    agent1_trainer, rollout_step, updates, probability,
):
    trainer = agent1_trainer
    trainer.global_steps, trainer.actor_update_steps = rollout_step, updates
    trainer._current_train_agent = None
    trainer._update_current_train_agent()
    assert trainer._current_train_agent == "decomposer"
    state = trainer._get_agent12_curriculum_state()
    assert state.teacher_attempt_probability == probability
    assert state.worker_question_probability == 1.0


def test_decomposer_only_rejects_worker_bootstrap(agent1_trainer):
    agent1_trainer.config.algorithm.hierarchy.agent12_curriculum.worker_bootstrap_steps = 1
    with pytest.raises(ValueError, match="worker_bootstrap_steps=0"):
        agent1_trainer._validate_config()


def test_wrapper_requires_explicit_worker_export():
    result = subprocess.run(["bash", str(WRAPPER)], capture_output=True, text=True, env={
        "PATH": os.environ["PATH"], "SLURM_SUBMIT_DIR": str(ROOT), "SLURM_JOB_ID": "123",
    })
    assert result.returncode != 0
    assert "Set WORKER_MODEL_REMOTE" in result.stderr


def test_wrapper_from_slurm_spool_reuses_data_and_disables_worker_updates(tmp_path):
    capture = tmp_path / "environment.txt"
    (tmp_path / "agent12-curriculum-trainer-multinode.sh").write_text('env > "$CAPTURE"\n')
    result = subprocess.run(["bash", str(WRAPPER)], capture_output=True, text=True, env={
        "PATH": os.environ["PATH"], "SLURM_SUBMIT_DIR": str(tmp_path), "SLURM_JOB_ID": "999",
        "WORKER_MODEL_REMOTE": "s3:bucket/workers/huggingface", "CAPTURE": str(capture),
        # These conflicting inherited values must not turn Agent 2 training back on.
        "AGENT2_PILOT": "true", "ONLINE_TEACHER_GENERATION": "true", "GENERATE_TEACHER_DATA": "1",
        "TRAIN_DECOMPOSER": "false", "TRAIN_AGENT_ROLES": "[worker_stage_1]",
    })
    assert result.returncode == 0, result.stderr
    env = dict(line.split("=", 1) for line in capture.read_text().splitlines())
    assert env["TRAIN_AGENT_ROLES"] == "[decomposer]" and env["TRAIN_DECOMPOSER"] == "true"
    assert env["GENERATE_TEACHER_DATA"] == "0" and env["ONLINE_TEACHER_GENERATION"] == "false"
    assert env["AGENT2_PILOT"] == "false" and env["JOINT_STEPS"] == "0"
    assert env["AGENT12_RUN_NAME"] == "agent1-only-999"


@pytest.mark.parametrize("path_case", ["old_job", "current_job", "shared", "mixed"])
def test_launcher_drops_only_paths_from_other_slurm_jobs(path_case):
    defaults = {
        "JOB_TMP": "",
        "RAY_NODE_TMP": "/ray",
        "CHECKPOINT_ROOT": "/checkpoints/agent12",
        "TEACHER_TRAIN_FILE": "/agent12_data/train.parquet",
        "TEACHER_CANDIDATES_FILE": "/agent12_data/teacher_candidates.parquet",
    }
    root = {
        "old_job": "/mnt/lscratch/slurm/6055460/agent12",
        "current_job": "/mnt/lscratch/slurm/999/custom",
        "shared": "/shared/custom",
        "mixed": "/mnt/lscratch/slurm/6055460/agent12",
    }[path_case]
    overrides = {key: root + suffix for key, suffix in defaults.items()}
    if path_case == "mixed":
        overrides["JOB_TMP"] = "/mnt/lscratch/slurm/999/custom"
    expected_root = (
        "/mnt/lscratch/slurm/999/agent12" if path_case == "old_job"
        else overrides["JOB_TMP"]
    )
    shell = LAUNCHER[:LAUNCHER.index("\nif (( GPUS_PER_NODE")].replace(
        "source ./env.sh", ": # No cluster credentials in tests"
    )
    worker_remote = "s3:bucket/agent2-only-6055460/workers/huggingface"
    result = subprocess.run(["bash", "-c", shell + "\nenv"],
        check=True, capture_output=True, text=True, env={
            "PATH": os.environ["PATH"], "SLURM_JOB_ID": "999",
            "WORKER_MODEL_REMOTE": worker_remote, **overrides,
        })
    env = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    for key, suffix in defaults.items():
        assert env[key] == expected_root + suffix
    assert env["WORKER_MODEL_REMOTE"] == worker_remote
    if path_case in ("old_job", "mixed"):
        assert "Ignoring" in result.stdout
    else:
        assert "Ignoring" not in result.stdout


def _model(directory, sharded=False):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text('{"model_type":"qwen2"}')
    (directory / "tokenizer_config.json").write_text('{}')
    (directory / "tokenizer.json").write_text('{}')
    if sharded:
        (directory / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"},
        }))
        for i in (1, 2):
            (directory / f"model-{i:05d}-of-00002.safetensors").write_bytes(b"test weight file")
    else:
        (directory / "model.safetensors").write_bytes(b"test weight file")


@pytest.mark.parametrize("sharded", [False, True])
def test_model_preflight_accepts_complete_file_sets(tmp_path, sharded):
    _model(tmp_path, sharded)
    validate_model_dir(tmp_path)


@pytest.mark.parametrize("missing", ["config.json", "tokenizer.json", "model-00002-of-00002.safetensors"])
def test_model_preflight_rejects_missing_files(tmp_path, missing):
    _model(tmp_path, True)
    (tmp_path / missing).unlink()
    with pytest.raises(ValueError):
        validate_model_dir(tmp_path)


def test_model_preflight_rejects_empty_or_external_shards(tmp_path):
    _model(tmp_path, True)
    shard = tmp_path / "model-00002-of-00002.safetensors"
    shard.write_bytes(b"")
    with pytest.raises(ValueError, match="Missing, empty"):
        validate_model_dir(tmp_path)
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="Missing, empty"):
        validate_model_dir(tmp_path)


@pytest.mark.parametrize("missing,download_failure", [(False, False), (True, False), (False, True)])
def test_staging_runs_on_every_node_and_fails_closed(tmp_path, missing, download_failure):
    remote = tmp_path / "remote"
    _model(remote, True)
    if missing:
        (remote / "model-00002-of-00002.safetensors").unlink()
    log = tmp_path / "calls.txt"
    staging = LAUNCHER[
        LAUNCHER.index('\nif [[ -n "$WORKER_MODEL_REMOTE"'):
        LAUNCHER.index('\nRAY_JOB_PIDS=')
    ]
    shell = '''
set -euo pipefail
COMMON_MOUNTS=(--nv)
rclone() {
    echo "download:$JOB_TMP" >> "$CALLS"
    if [[ "$DOWNLOAD_FAILURE" == 1 ]]; then return 1; fi
    cp -R "$2/." "$3/"
}
export -f rclone
srun() {
    echo "$*" >> "$CALLS"
    while [[ "$1" == --* ]]; do shift; done
    local node
    for node in 0 1; do
        (
            export JOB_TMP="$ROOT_TMP/node-$node"
            export WORKER_MODEL_PATH="$JOB_TMP/worker_model"
            mkdir -p "$JOB_TMP"
            if [[ "$1" == bash ]]; then
                # Run the actual staging body, without a login shell resetting PATH.
                bash -c "$3"
            else
                "$PYTHON" "$VALIDATOR" "$WORKER_MODEL_PATH"
            fi
        ) || return $?
    done
}
'''
    result = subprocess.run(["bash", "-c", shell + staging], capture_output=True, text=True, env={
        "PATH": os.environ["PATH"], "ROOT_TMP": str(tmp_path), "JOB_TMP": str(tmp_path / "node-0"),
        "WORKER_MODEL_REMOTE": str(remote), "SLURM_NNODES": "2", "SIF_NAME": "test.sif",
        "CALLS": str(log), "PYTHON": sys.executable, "VALIDATOR": str(VALIDATOR),
        "DOWNLOAD_FAILURE": str(int(download_failure)),
    })
    assert (result.returncode == 0) is not (missing or download_failure), result.stderr
    calls = log.read_text()
    assert "--nodes=2 --ntasks=2" in calls
    if not missing and not download_failure:
        for node in (0, 1):
            validate_model_dir(tmp_path / f"node-{node}" / "worker_model")
            assert f"download:{tmp_path}/node-{node}" in calls

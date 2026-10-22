"""Launcher storage safety tests; no Slurm, GPU, container or network required."""
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[4]
HELPER = ROOT / "scripts/shared-training-storage.sh"


@pytest.fixture
def env(tmp_path):
    result = {key: value for key, value in os.environ.items()
              if not key.startswith(("SLURM_", "APPTAINER_", "REMA_", "TMPDIR"))
              and key not in {"SHARED_RUNTIME_MODEL_ROOT", "DISTRIBUTED_OFFLINE_INPUT_ROOT"}}
    for name, directory in (("HOME", "home"), ("TMPDIR_LUSTRE", "lustre"),
                            ("TMPDIR_LOCAL", "local")):
        path = tmp_path / directory
        path.mkdir()
        result[name] = str(path)
    result.update(JOB_ID="123", APPTAINER_BINDPATH="/existing:/existing")
    return result


def run(env, command="rema_configure_shared_training_storage"):
    return subprocess.run(
        ["bash", "-c", 'set -euo pipefail; source "$1"; ' + command, "bash", str(HELPER)],
        env=env, text=True, capture_output=True, timeout=15,
    )


def test_missing_lustre_fails_without_home_fallback(env):
    env.pop("TMPDIR_LUSTRE")
    result = run(env)
    assert result.returncode != 0
    assert "TMPDIR_LUSTRE is required" in result.stderr
    assert not list(Path(env["HOME"]).iterdir())


def test_defaults_and_container_bind_preserve_existing_bind(env):
    result = run(env, '''rema_configure_shared_training_storage
rema_configure_shared_training_storage
printf 'BIND=%s\n' "$APPTAINER_BINDPATH"
printf 'MODEL=%s\nINPUT=%s\n' "$SHARED_RUNTIME_MODEL_ROOT" "$DISTRIBUTED_OFFLINE_INPUT_ROOT"
''')
    assert result.returncode == 0, result.stderr
    base = str(Path(env["TMPDIR_LUSTRE"]).resolve())
    assert f"MODEL={base}/hierarchical_rema_train_123/_runtime_active_models" in result.stdout
    assert f"INPUT={base}/hierarchical_rema_train_123/_distributed_offline_grpo" in result.stdout
    assert f"BIND=/existing:/existing,{base}:{base}\n" in result.stdout


@pytest.mark.parametrize("variable", ["SHARED_RUNTIME_MODEL_ROOT", "DISTRIBUTED_OFFLINE_INPUT_ROOT"])
@pytest.mark.parametrize("location", ["home", "local", "relative", "escape", "base"])
def test_unsafe_overrides_rejected(env, variable, location):
    paths = {
        "home": env["HOME"] + "/old_checkpoint",
        "local": env["TMPDIR_LOCAL"] + "/checkpoint",
        "relative": "outputs/checkpoint",
        "escape": env["TMPDIR_LUSTRE"] + "/../home/checkpoint",
        "base": env["TMPDIR_LUSTRE"],
    }
    env[variable] = paths[location]
    result = run(env)
    assert result.returncode != 0
    assert variable in result.stderr


@pytest.mark.parametrize("target", ["HOME", "TMPDIR_LOCAL"])
def test_symlink_escape_rejected(env, target):
    link = Path(env["TMPDIR_LUSTRE"]) / "outside"
    link.symlink_to(env[target], target_is_directory=True)
    env["SHARED_RUNTIME_MODEL_ROOT"] = str(link / "models")
    result = run(env)
    assert result.returncode != 0
    assert "strictly inside TMPDIR_LUSTRE" in result.stderr


@pytest.mark.parametrize("target", ["HOME", "TMPDIR_LOCAL"])
def test_lustre_root_cannot_itself_resolve_to_unsafe_storage(env, target):
    env["TMPDIR_LUSTRE"] = env[target]
    result = run(env)
    assert result.returncode != 0
    assert "[storage]" in result.stderr


def test_missing_mount_rejected(env):
    env["TMPDIR_LUSTRE"] += "/not-mounted"
    result = run(env)
    assert result.returncode != 0
    assert "missing or not writable" in result.stderr


@pytest.mark.parametrize("nested", [False, True])
def test_roots_cannot_overlap(env, nested):
    env["SHARED_RUNTIME_MODEL_ROOT"] = env["TMPDIR_LUSTRE"] + "/models"
    env["DISTRIBUTED_OFFLINE_INPUT_ROOT"] = env["SHARED_RUNTIME_MODEL_ROOT"] + ("/replay" if nested else "")
    result = run(env)
    assert result.returncode != 0
    assert "non-nested directories" in result.stderr


def test_explicit_shared_subdirectories_supported(env):
    env["SHARED_RUNTIME_MODEL_ROOT"] = env["TMPDIR_LUSTRE"] + "/custom/models"
    env["DISTRIBUTED_OFFLINE_INPUT_ROOT"] = env["TMPDIR_LUSTRE"] + "/custom/replay"
    result = run(env)
    assert result.returncode == 0, result.stderr
    assert env["SHARED_RUNTIME_MODEL_ROOT"] in result.stdout
    assert env["DISTRIBUTED_OFFLINE_INPUT_ROOT"] in result.stdout


@pytest.fixture
def cluster(env, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    srun = bin_dir / "srun"
    srun.write_text('''#!/usr/bin/env bash
set -euo pipefail
while [[ "$1" == --* ]]; do shift; done
for rank in 0 1; do
    if [[ "${PROBE_TEST_FAILURE:-}" == skip_peer && "$rank" == 1 ]]; then continue; fi
    if [[ "${PROBE_TEST_FAILURE:-}" == peer_error && "$rank" == 1 ]]; then exit 7; fi
    SLURM_PROCID="$rank" "$@"
done
''')
    apptainer = bin_dir / "apptainer"
    apptainer.write_text('''#!/usr/bin/env bash
set -euo pipefail
[[ "$APPTAINER_BINDPATH" == *"$TMPDIR_LUSTRE:$TMPDIR_LUSTRE"* ]]
[[ "$1" == exec && "$2" == --writable-tmpfs ]]
shift 3
if [[ "${PROBE_TEST_FAILURE:-}" == container_error ]]; then exit 9; fi
exec "$@"
''')
    srun.chmod(0o755)
    apptainer.chmod(0o755)
    env.update(PATH=str(bin_dir) + os.pathsep + env["PATH"], SLURM_JOB_ID="123",
               SLURM_JOB_NUM_NODES="2", LOCAL_SIF_IMAGE_PATH="/mock/image.sif")
    return env


PREFLIGHT = '''rema_configure_shared_training_storage
mkdir -p "$SHARED_RUNTIME_MODEL_ROOT" "$DISTRIBUTED_OFFLINE_INPUT_ROOT"
rema_verify_shared_training_storage host
rema_verify_shared_training_storage container
'''


def test_two_node_host_and_container_read_write_probe(cluster):
    result = run(cluster, PREFLIGHT)
    assert result.returncode == 0, result.stderr
    assert "host preflight passed on 2 node(s)" in result.stdout
    assert "container preflight passed on 2 node(s)" in result.stdout
    assert not list(Path(cluster["TMPDIR_LUSTRE"]).rglob(".rema-shared-probe.*"))


@pytest.mark.parametrize("failure", ["skip_peer", "peer_error", "container_error"])
def test_preflight_requires_every_peer_and_container_access(cluster, failure):
    cluster["PROBE_TEST_FAILURE"] = failure
    result = run(cluster, PREFLIGHT)
    assert result.returncode != 0
    assert "preflight failed" in result.stderr
    assert "Refusing home/local fallback" in result.stderr
    assert not list(Path(cluster["TMPDIR_LUSTRE"]).rglob(".rema-shared-probe.*"))


def test_launchers_request_lustre_and_preflight_before_ray():
    for name in ("hierarchical-rema-train.sh", "hierarchical-rema-trainer.sh"):
        text = (ROOT / name).read_text()
        assert "#SBATCH --gres=storage:local:500G,storage:lustre:1" in text
        subprocess.run(["bash", "-n", str(ROOT / name)], check=True)
    text = (ROOT / "hierarchical-rema-trainer.sh").read_text()
    assert text.index("rema_configure_shared_training_storage") < text.index('mkdir -p "$RUN_ROOT"')
    assert text.index("rema_verify_shared_training_storage host") < text.index("stage_runtime_on_allocated_nodes()")
    assert text.index("rema_verify_shared_training_storage container") < text.index("\nstart_ray_cluster\n")

#!/usr/bin/env bash
# Sourced by the HPC launcher after env.sh; no home-directory fallback for training.

rema_configure_shared_training_storage() {
    local config value binding
    local -a paths=()
    config=$(python3 - "${TMPDIR_LUSTRE:-}" "${JOB_ID:-manual}" \
        "${SHARED_RUNTIME_MODEL_ROOT:-}" "${DISTRIBUTED_OFFLINE_INPUT_ROOT:-}" <<'PY'
import os
from pathlib import Path
import re
import sys


def fail(message):
    sys.exit('[hierarchical-rema][storage] ' + message)


def resolve(value, name):
    if not Path(value).is_absolute() or any(c in value for c in '\n\r,:'):
        fail(f'{name} must be an absolute path without newlines, commas or colons.')
    return Path(value).resolve()


raw_base, job_id, raw_models, raw_inputs = sys.argv[1:]
if not raw_base:
    fail('TMPDIR_LUSTRE is required for training. Request '
         '--gres=storage:local:500G,storage:lustre:1 and submit through sbatch. '
         'Refusing to put shared checkpoints or learner inputs in home/local scratch.')
if not re.fullmatch(r'[A-Za-z0-9_-]+', job_id):
    fail('Invalid job ID for shared storage.')
base = resolve(raw_base, 'TMPDIR_LUSTRE')
home = Path.home().resolve()
local = os.environ.get('TMPDIR_LOCAL')
local = Path(local).resolve() if local else None
if base == Path('/') or base == home or home in base.parents:
    fail('TMPDIR_LUSTRE must not resolve to the home directory or one of its children.')
if local and (base == local or local in base.parents):
    fail('TMPDIR_LUSTRE resolves to node-local TMPDIR_LOCAL, not shared storage.')
if not base.is_dir() or not os.access(base, os.W_OK | os.X_OK):
    fail(f'Shared Lustre storage is missing or not writable: {base}')
job_root = base / f'hierarchical_rema_train_{job_id}'
paths = []
for name, raw, suffix in (
    ('SHARED_RUNTIME_MODEL_ROOT', raw_models, '_runtime_active_models'),
    ('DISTRIBUTED_OFFLINE_INPUT_ROOT', raw_inputs, '_distributed_offline_grpo'),
):
    path = resolve(raw or str(job_root / suffix), name)
    if base not in path.parents:
        fail(f'{name}={path} must be strictly inside TMPDIR_LUSTRE={base}. '
             'Unset the old override or choose a shared-scratch subdirectory.')
    if local and (path == local or local in path.parents):
        fail(f'{name} resolves to node-local storage.')
    if path.exists() and (not path.is_dir() or not os.access(path, os.W_OK | os.X_OK)):
        fail(f'{name} is not a writable directory: {path}')
    paths.append(path)
if paths[0] == paths[1] or paths[0] in paths[1].parents or paths[1] in paths[0].parents:
    fail('Model and learner-input roots must be separate, non-nested directories.')
for path in (base, job_root, *paths):
    print(path)
PY
    ) || return 1
    while IFS= read -r value; do paths+=("$value"); done <<< "$config"
    REMA_SHARED_STORAGE_BASE=${paths[0]}
    DEFAULT_SHARED_RUNTIME_MODEL_ROOT="${paths[1]}/_runtime_active_models"
    DEFAULT_DISTRIBUTED_OFFLINE_INPUT_ROOT="${paths[1]}/_distributed_offline_grpo"
    SHARED_RUNTIME_MODEL_ROOT=${paths[2]}
    DISTRIBUTED_OFFLINE_INPUT_ROOT=${paths[3]}
    binding="$REMA_SHARED_STORAGE_BASE:$REMA_SHARED_STORAGE_BASE"
    case ",${APPTAINER_BINDPATH:-}," in
        *",$binding,"*) ;;
        *) APPTAINER_BINDPATH="${APPTAINER_BINDPATH:+${APPTAINER_BINDPATH},}$binding" ;;
    esac
    export REMA_SHARED_STORAGE_BASE SHARED_RUNTIME_MODEL_ROOT DISTRIBUTED_OFFLINE_INPUT_ROOT
    export APPTAINER_BINDPATH
    echo "[hierarchical-rema][storage] model_root=$SHARED_RUNTIME_MODEL_ROOT"
    echo "[hierarchical-rema][storage] learner_inputs=$DISTRIBUTED_OFFLINE_INPUT_ROOT"
}

rema_shared_storage_peer_probe() {
    local token=$1 probe
    shift
    for probe in "$@"; do
        if [[ ! -r "$probe/driver" || "$(< "$probe/driver")" != "$token" ]]; then
            echo "[hierarchical-rema][storage] driver marker not visible on $(hostname): $probe" >&2
            return 1
        fi
        printf '%s\n' "$token" > "$probe/node-${SLURM_PROCID:-0}" || return 1
    done
}

rema_verify_shared_training_storage() {
    local mode=$1 root probe rank status=0
    local nodes=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}
    local token="${JOB_ID:-manual}-$$-${RANDOM}"
    local -a probes=() command=()
    if [[ ! "$nodes" =~ ^[1-9][0-9]*$ || ! "$mode" =~ ^(host|container)$ ]]; then
        echo '[hierarchical-rema][storage] Invalid preflight mode or allocated node count' >&2
        return 1
    fi
    if [[ -z "${SLURM_JOB_ID:-}" && "$nodes" != 1 ]]; then
        echo '[hierarchical-rema][storage] Multi-node preflight requires a Slurm allocation' >&2
        return 1
    fi
    for root in "$SHARED_RUNTIME_MODEL_ROOT" "$DISTRIBUTED_OFFLINE_INPUT_ROOT"; do
        if ! probe=$(mktemp -d "$root/.rema-shared-probe.XXXXXX"); then
            if [[ ${#probes[@]} -gt 0 ]]; then rm -rf -- "${probes[@]}"; fi
            return 1
        fi
        probes+=("$probe")
        if ! printf '%s\n' "$token" > "$probe/driver"; then
            rm -rf -- "${probes[@]}"
            return 1
        fi
    done
    command=(bash -c "$(declare -f rema_shared_storage_peer_probe); rema_shared_storage_peer_probe \"\$@\""
        bash "$token" "${probes[@]}")
    if [[ "$mode" == container ]]; then
        command=(apptainer exec --writable-tmpfs "$LOCAL_SIF_IMAGE_PATH" "${command[@]}")
    fi
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        command=(srun --overlap --immediate=60 --time=00:02:00 --kill-on-bad-exit=1
            --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 --cpus-per-task=1
            --gres=none "${command[@]}")
    fi
    if ! "${command[@]}"; then status=1; fi
    # A successful step is insufficient: verify every node's write from the driver.
    for probe in "${probes[@]}"; do
        for ((rank=0; rank<nodes; rank++)); do
            if [[ ! -r "$probe/node-$rank" || "$(< "$probe/node-$rank")" != "$token" ]]; then
                status=1
            fi
        done
    done
    rm -rf -- "${probes[@]}"
    if [[ "$status" != 0 ]]; then
        echo "[hierarchical-rema][storage] $mode preflight failed: both storage roots must be shared and writable on all $nodes nodes. Refusing home/local fallback." >&2
        return 1
    fi
    echo "[hierarchical-rema][storage] $mode preflight passed on $nodes node(s)"
}

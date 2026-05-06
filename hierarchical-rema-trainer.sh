#!/bin/bash
#SBATCH --job-name=hierarchical-rema
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --mem=200gb
#SBATCH -p lem-gpu-short
#SBATCH --gpus-per-node=hopper:4
#SBATCH --verbose

set -euo pipefail

if [[ -f ./env.sh ]]; then
    source ./env.sh
fi

JOB_ID=${SLURM_JOB_ID:-manual}
RUN_ROOT=${RUN_ROOT:-$TMPDIR/hierarchical_rema_${JOB_ID}}
LOCAL_OUTPUT_DIR=${LOCAL_OUTPUT_DIR:-$RUN_ROOT/output}
PERSIST_LOCAL_DIR=${PERSIST_LOCAL_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/outputs/hierarchical_rema/${JOB_ID}}

# Override this at submission time if you want a different destination.
S3_OUTPUT_PATH=${S3_OUTPUT_PATH:-s3v2:s3min-tomasznaskret-1712063354/user/jmoska/hierarchical_rema/output/${JOB_ID}}
SIF_IMAGE_PATH=${SIF_IMAGE_PATH:-s3v2:s3min-tomasznaskret-1712063354/user/dmotyka/sif_images/verl-rema-v3.sif}

BACKEND=${BACKEND:-mock}
TASK=${TASK:-all}
MODE=${MODE:-joint}
PHASE=${PHASE:-selector}
PARAMETER_SHARING=${PARAMETER_SHARING:-false}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
DECOMPOSER_MODEL_PATH=${DECOMPOSER_MODEL_PATH:-$MODEL_PATH}
SELECTOR_MODEL_PATH=${SELECTOR_MODEL_PATH:-$MODEL_PATH}
WORKER_BASE_MODEL_PATH=${WORKER_BASE_MODEL_PATH:-$MODEL_PATH}

NUM_DECOMPOSITIONS=${NUM_DECOMPOSITIONS:-3}
NUM_SELECTIONS=${NUM_SELECTIONS:-2}
SOFT_MAX_HOPS=${SOFT_MAX_HOPS:-3}
HARD_MAX_HOPS=${HARD_MAX_HOPS:-5}
SOFT_HOP_PENALTY=${SOFT_HOP_PENALTY:-0.1}

TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
CONTROLLER_MAX_NEW_TOKENS=${CONTROLLER_MAX_NEW_TOKENS:-768}
WORKER_MAX_NEW_TOKENS=${WORKER_MAX_NEW_TOKENS:-256}
BEST_K=${BEST_K:-10}

mkdir -p "$RUN_ROOT" "$LOCAL_OUTPUT_DIR" "$PERSIST_LOCAL_DIR"

cp -r ./src/verl "$TMPDIR/verl"
rclone copy "$SIF_IMAGE_PATH" "$TMPDIR/"

export HF_HOME=${HF_HOME:-$TMPDIR/hf_home}

PARAMETER_SHARING_FLAG=""
if [[ "$PARAMETER_SHARING" == "1" || "$PARAMETER_SHARING" == "true" || "$PARAMETER_SHARING" == "True" ]]; then
    PARAMETER_SHARING_FLAG="--parameter-sharing"
fi

persist_outputs() {
    set +e

    if [[ -d "$LOCAL_OUTPUT_DIR" ]]; then
        mkdir -p "$PERSIST_LOCAL_DIR"
        cp -r "$LOCAL_OUTPUT_DIR"/. "$PERSIST_LOCAL_DIR"/ 2>/dev/null || true

        if [[ -n "${S3_OUTPUT_PATH:-}" ]]; then
            rclone copy "$LOCAL_OUTPUT_DIR" "$S3_OUTPUT_PATH" || \
                echo "Warning: failed to upload outputs to ${S3_OUTPUT_PATH}" >&2
        fi
    fi

    if [[ -n "${TMPDIR:-}" && -d "${TMPDIR:-}" ]]; then
        rm -rf "$TMPDIR"/*
    fi
}

COMMAND="unset ROCR_VISIBLE_DEVICES; \
export HF_HOME=$TMPDIR/hf_home; \
export PYTHONPATH=/verl/verl:\$PYTHONPATH; \
mkdir -p ${LOCAL_OUTPUT_DIR}; \
python3 -m hierarchical_rema.demo \
  --backend ${BACKEND} \
  --task ${TASK} \
  --mode ${MODE} \
  --phase ${PHASE} \
  ${PARAMETER_SHARING_FLAG} \
  --decomposer-model-path ${DECOMPOSER_MODEL_PATH} \
  --selector-model-path ${SELECTOR_MODEL_PATH} \
  --worker-base-model-path ${WORKER_BASE_MODEL_PATH} \
  --num-decompositions ${NUM_DECOMPOSITIONS} \
  --num-selections ${NUM_SELECTIONS} \
  --soft-max-hops ${SOFT_MAX_HOPS} \
  --hard-max-hops ${HARD_MAX_HOPS} \
  --soft-hop-penalty ${SOFT_HOP_PENALTY} \
  --temperature ${TEMPERATURE} \
  --top-p ${TOP_P} \
  --controller-max-new-tokens ${CONTROLLER_MAX_NEW_TOKENS} \
  --worker-max-new-tokens ${WORKER_MAX_NEW_TOKENS} \
  --output-dir ${LOCAL_OUTPUT_DIR} \
  --best-k ${BEST_K} \
  > ${LOCAL_OUTPUT_DIR}/demo_stdout.json"

set +e
srun apptainer exec --nv --writable-tmpfs \
    --mount type=bind,src=$TMPDIR,dst=$TMPDIR \
    --mount type=bind,src=$TMPDIR,dst=/root/tmpdir \
    --mount type=bind,src=$TMPDIR/verl,dst=/verl \
    "$TMPDIR/verl-rema-v3.sif" \
    bash -c "$COMMAND"
RUN_EXIT_CODE=$?
set -e

persist_outputs
exit $RUN_EXIT_CODE

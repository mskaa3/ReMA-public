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

SOURCE_DIR=${SLURM_SUBMIT_DIR:-$PWD}

DEBUG_LAUNCHER=${DEBUG_LAUNCHER:-false}
if [[ "$DEBUG_LAUNCHER" == "1" || "$DEBUG_LAUNCHER" == "true" || "$DEBUG_LAUNCHER" == "True" ]]; then
    set -x
fi

if [[ -f "$SOURCE_DIR/env.sh" ]]; then
    source "$SOURCE_DIR/env.sh"
fi

JOB_ID=${SLURM_JOB_ID:-manual}
RUN_KIND=${RUN_KIND:-rollout}  # rollout | train
RUN_ROOT=${RUN_ROOT:-$TMPDIR/hierarchical_rema_${RUN_KIND}_${JOB_ID}}
TASK_SOURCE_ARG=${1:-}

case "$RUN_KIND" in
    rollout)
        OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-hierarchical_rema}
        DEFAULT_S3_OUTPUT_PATH="s3v2:s3min-tomasznaskret-1712063354/user/jmoska/hierarchical_rema/output/${JOB_ID}"
        ;;
    train)
        OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-hierarchical_rema_train}
        DEFAULT_S3_OUTPUT_PATH="s3v2:s3min-tomasznaskret-1712063354/user/jmoska/hierarchical_rema/train/${JOB_ID}"
        ;;
    *)
        echo "Unsupported RUN_KIND=${RUN_KIND}. Use rollout or train." >&2
        exit 1
        ;;
esac

LOCAL_OUTPUT_DIR=${LOCAL_OUTPUT_DIR:-$RUN_ROOT/output}
PERSIST_LOCAL_DIR=${PERSIST_LOCAL_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/outputs/${OUTPUT_SUBDIR}/${JOB_ID}}
RUNTIME_LOG=${RUNTIME_LOG:-$LOCAL_OUTPUT_DIR/runtime.log}
RUN_METADATA_FILE=${RUN_METADATA_FILE:-$LOCAL_OUTPUT_DIR/run_metadata.txt}

S3_OUTPUT_PATH=${S3_OUTPUT_PATH:-$DEFAULT_S3_OUTPUT_PATH}
SIF_IMAGE_PATH=${SIF_IMAGE_PATH:-s3v2:s3min-tomasznaskret-1712063354/user/dmotyka/sif_images/verl-rema-v3.sif}

# Frequently adjusted: high-level experiment shape.
BACKEND=${BACKEND:-mock}
TASK=${TASK:-all}
MODE=${MODE:-joint}
PHASE=${PHASE:-selector}
PARAMETER_SHARING=${PARAMETER_SHARING:-false}

# Frequently adjusted: base model assignment.
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
DECOMPOSER_MODEL_PATH=${DECOMPOSER_MODEL_PATH:-$MODEL_PATH}
SELECTOR_MODEL_PATH=${SELECTOR_MODEL_PATH:-$MODEL_PATH}
WORKER_BASE_MODEL_PATH=${WORKER_BASE_MODEL_PATH:-$MODEL_PATH}

normalize_name_component() {
    local value="$1"
    value="${value##*/}"
    value="${value//\//-}"
    value="${value// /-}"
    value="${value//:/-}"
    value="${value//,/-}"
    value="${value//=/-}"
    value="${value//+/-}"
    value="${value//(/-}"
    value="${value//)/-}"
    while [[ "$value" == *"--"* ]]; do
        value="${value//--/-}"
    done
    value="${value#-}"
    value="${value%-}"
    if [[ -z "$value" ]]; then
        value="unknown"
    fi
    printf '%s' "$value"
}

if [[ "$PARAMETER_SHARING" == "1" || "$PARAMETER_SHARING" == "true" || "$PARAMETER_SHARING" == "True" ]]; then
    PARAMETER_SHARING_TAG="ps-true"
else
    PARAMETER_SHARING_TAG="ps-false"
fi
MODE_TAG="mode-$(normalize_name_component "$MODE")"
if [[ "$DECOMPOSER_MODEL_PATH" == "$SELECTOR_MODEL_PATH" ]]; then
    MODEL_NAME_TAG="$(normalize_name_component "$DECOMPOSER_MODEL_PATH")"
else
    MODEL_NAME_TAG="dec-$(normalize_name_component "$DECOMPOSER_MODEL_PATH")-sel-$(normalize_name_component "$SELECTOR_MODEL_PATH")"
fi

# Frequently adjusted: search width and DAG complexity.
NUM_DECOMPOSITIONS=${NUM_DECOMPOSITIONS:-3}
NUM_SELECTIONS=${NUM_SELECTIONS:-2}
SOFT_MAX_HOPS=${SOFT_MAX_HOPS:-5}
HARD_MAX_HOPS=${HARD_MAX_HOPS:-8}
MAX_NODES_PER_DECOMPOSITION=${MAX_NODES_PER_DECOMPOSITION:-$HARD_MAX_HOPS}
SOFT_HOP_PENALTY=${SOFT_HOP_PENALTY:-0.1}

# Frequently adjusted: generation budget / decoding.
TEMPERATURE=${TEMPERATURE:-0.5}
CONTROLLER_TEMPERATURE=${CONTROLLER_TEMPERATURE:-$TEMPERATURE}
WORKER_TEMPERATURE=${WORKER_TEMPERATURE:-$TEMPERATURE}
TOP_P=${TOP_P:-0.95}
CONTROLLER_CONSTRAINED_DECODING=${CONTROLLER_CONSTRAINED_DECODING:-true}
CONTROLLER_MAX_NEW_TOKENS=${CONTROLLER_MAX_NEW_TOKENS:-768}
DECOMPOSER_MAX_NEW_TOKENS=${DECOMPOSER_MAX_NEW_TOKENS:-0}
SELECTOR_MAX_NEW_TOKENS=${SELECTOR_MAX_NEW_TOKENS:-256}
WORKER_MAX_NEW_TOKENS=${WORKER_MAX_NEW_TOKENS:-256}
CONTROLLER_BATCH_SIZE=${CONTROLLER_BATCH_SIZE:-8}
WORKER_BATCH_SIZE=${WORKER_BATCH_SIZE:-16}
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-2048}

# Usually leave alone unless changing cluster/runtime behavior.
SLURM_GPUS_RAW=${SLURM_GPUS_ON_NODE:-1}
if [[ "$SLURM_GPUS_RAW" == *:* ]]; then
    DEFAULT_RAY_N_GPUS_PER_NODE=${SLURM_GPUS_RAW##*:}
else
    DEFAULT_RAY_N_GPUS_PER_NODE=${SLURM_GPUS_RAW}
fi
RAY_NNODES=${RAY_NNODES:-${SLURM_JOB_NUM_NODES:-1}}
RAY_N_GPUS_PER_NODE=${RAY_N_GPUS_PER_NODE:-$DEFAULT_RAY_N_GPUS_PER_NODE}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.8}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-2048}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-}
RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RAY_NAMESPACE=${RAY_NAMESPACE:-verl}
OFFLINE_GRPO_DISTRIBUTED=${OFFLINE_GRPO_DISTRIBUTED:-false}
OFFLINE_GRPO_NNODES=${OFFLINE_GRPO_NNODES:-$RAY_NNODES}
OFFLINE_GRPO_GPUS_PER_NODE=${OFFLINE_GRPO_GPUS_PER_NODE:-$RAY_N_GPUS_PER_NODE}
OFFLINE_GRPO_MASTER_PORT=${OFFLINE_GRPO_MASTER_PORT:-29501}
BEST_K=${BEST_K:-10}
PRINT_MODE=${PRINT_MODE:-summary}
# Number of training tasks sent through one hierarchical rollout inference chunk.
ROLLOUT_TASK_BATCH_SIZE=${ROLLOUT_TASK_BATCH_SIZE:-32}
# Semi-online cadence: run GRPO after this many rollout batches.
UPDATE_EVERY_N_ROLLOUT_BATCHES=${UPDATE_EVERY_N_ROLLOUT_BATCHES:-4}
ROLLOUT_PROGRESS_EVERY=${ROLLOUT_PROGRESS_EVERY:-10}
ROLLOUT_LOG_MODE=${ROLLOUT_LOG_MODE:-best}
ROLLOUT_LOG_DETAIL=${ROLLOUT_LOG_DETAIL:-compact}

TASK_SOURCE=${TASK_SOURCE:-$TASK_SOURCE_ARG}
if [[ "$RUN_KIND" == "train" ]]; then
    TASK_SOURCE=${TASK_SOURCE:-$SOURCE_DIR/data/MATH/train_lv3to5_8k.parquet}
else
    TASK_SOURCE=${TASK_SOURCE:-$SOURCE_DIR/data/overall_math/all_test_data.jsonl}
fi

# Frequently adjusted: outer-loop data cadence.
TASK_FORMAT=${TASK_FORMAT:-auto}
PROMPT_KEY=${PROMPT_KEY:-question}
ANSWER_KEY=${ANSWER_KEY:-answer}
TASK_ID_KEY=${TASK_ID_KEY:-idx}
MAX_TASKS=${MAX_TASKS:-0}
TRAIN_SUBSET_SIZE=${TRAIN_SUBSET_SIZE:-${TASKS_PER_EPOCH:-160}}
TASKS_PER_EPOCH=${TASKS_PER_EPOCH:-$TRAIN_SUBSET_SIZE}
SHUFFLE_TASKS=${SHUFFLE_TASKS:-false}
TASK_SOURCE_STAGE=${TASK_SOURCE_STAGE:-$RUN_ROOT/task_source}
TASK_SOURCE_RUNTIME=${TASK_SOURCE_RUNTIME:-$TASK_SOURCE}
VAL_TASK_SOURCE=${VAL_TASK_SOURCE:-$SOURCE_DIR/data/overall_math/test.parquet}
VAL_TASK_FORMAT=${VAL_TASK_FORMAT:-auto}
VAL_PROMPT_KEY=${VAL_PROMPT_KEY:-question}
VAL_ANSWER_KEY=${VAL_ANSWER_KEY:-answer}
VAL_TASK_ID_KEY=${VAL_TASK_ID_KEY:-idx}
MAX_VAL_TASKS=${MAX_VAL_TASKS:-0}
# How many validation tasks to run when validation is triggered.
VAL_TASKS_PER_EPOCH=${VAL_TASKS_PER_EPOCH:-128}
# How many validation tasks to take per subset/dataset before any global cap.
VAL_TASKS_PER_SUBSET=${VAL_TASKS_PER_SUBSET:-128}
# Run external validation periodically instead of every outer round.
EXTERNAL_VALIDATION_EVERY_SUBSET_ROUNDS=${EXTERNAL_VALIDATION_EVERY_SUBSET_ROUNDS:-10}
VAL_TASK_SOURCE_STAGE=${VAL_TASK_SOURCE_STAGE:-$RUN_ROOT/val_task_source}
VAL_TASK_SOURCE_RUNTIME=${VAL_TASK_SOURCE_RUNTIME:-$VAL_TASK_SOURCE}
DISABLE_EXTERNAL_VALIDATION=${DISABLE_EXTERNAL_VALIDATION:-false}
VAL_NUM_DECOMPOSITIONS=${VAL_NUM_DECOMPOSITIONS:-1}
VAL_NUM_SELECTIONS=${VAL_NUM_SELECTIONS:-1}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.0}
VAL_CONTROLLER_TEMPERATURE=${VAL_CONTROLLER_TEMPERATURE:-$VAL_TEMPERATURE}
VAL_WORKER_TEMPERATURE=${VAL_WORKER_TEMPERATURE:-$VAL_TEMPERATURE}
VAL_TOP_P=${VAL_TOP_P:-1.0}
VAL_CONTROLLER_MAX_NEW_TOKENS=${VAL_CONTROLLER_MAX_NEW_TOKENS:-0}
VAL_DECOMPOSER_MAX_NEW_TOKENS=${VAL_DECOMPOSER_MAX_NEW_TOKENS:-0}
VAL_SELECTOR_MAX_NEW_TOKENS=${VAL_SELECTOR_MAX_NEW_TOKENS:-0}
VAL_WORKER_MAX_NEW_TOKENS=${VAL_WORKER_MAX_NEW_TOKENS:-0}
# Validation inference chunk size; separate from total validation slice size.
VAL_ROLLOUT_TASK_BATCH_SIZE=${VAL_ROLLOUT_TASK_BATCH_SIZE:-64}

# Occasionally adjusted: replay filtering / controller penalties.
TRAIN_ROLE=${TRAIN_ROLE:-both}
TRAIN_POLICY_ID=${TRAIN_POLICY_ID:-}
TRAIN_VAL_RATIO=${TRAIN_VAL_RATIO:-0.05}
TRAIN_MIN_REWARD=${TRAIN_MIN_REWARD:-0.21}
TRAIN_MIN_ADVANTAGE=${TRAIN_MIN_ADVANTAGE:-}
DECOMPOSER_REWARD_AGGREGATION=${DECOMPOSER_REWARD_AGGREGATION:-best}
DECOMPOSER_NO_CORRECT_SELECTION_SCALE=${DECOMPOSER_NO_CORRECT_SELECTION_SCALE:-0.25}
CONTROLLER_FORMAT_RETRY_PENALTY=${CONTROLLER_FORMAT_RETRY_PENALTY:-0.05}
CONTROLLER_FORMAT_FALLBACK_PENALTY=${CONTROLLER_FORMAT_FALLBACK_PENALTY:-0.25}
FINAL_ANSWER_CORRECTNESS_REWARD_ONLY=${FINAL_ANSWER_CORRECTNESS_REWARD_ONLY:-false}
CONFIDENCE_REWARD_WEIGHT=${CONFIDENCE_REWARD_WEIGHT:-0.05}
COMPATIBILITY_REWARD_WEIGHT=${COMPATIBILITY_REWARD_WEIGHT:-0.1}
TRACK_WORKERS_HISTORY=${TRACK_WORKERS_HISTORY:-true}
TRAIN_WORKER_MODEL=${TRAIN_WORKER_MODEL:-false}
SAVE_REPLAY_COPY=${SAVE_REPLAY_COPY:-false}
SEED=${SEED:-42}
TRAIN_SUBSET_ROUNDS=${TRAIN_SUBSET_ROUNDS:-${NUM_EPOCHS:-80}}
NUM_EPOCHS=${NUM_EPOCHS:-$TRAIN_SUBSET_ROUNDS}

# Frequently adjusted: offline GRPO optimizer/update cadence.
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
GRPO_PASSES_PER_SUBSET=${GRPO_PASSES_PER_SUBSET:-${EPOCHS:-1}}
EPOCHS=${EPOCHS:-$GRPO_PASSES_PER_SUBSET}
MAX_LENGTH=${MAX_LENGTH:-3072}
TRUNCATION=${TRUNCATION:-left}
CLIP_RANGE=${CLIP_RANGE:-0.2}
CLIP_RATIO_C=${CLIP_RATIO_C:-3.0}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.01}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-200}
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-10}
CHECKPOINT_MODE=${CHECKPOINT_MODE:-final}
PRUNE_STALE_POLICY_MODELS=${PRUNE_STALE_POLICY_MODELS:-false}
PRUNE_UPLOADED_LOCAL_CHECKPOINTS=${PRUNE_UPLOADED_LOCAL_CHECKPOINTS:-false}
STRIP_LOCAL_MODELS_AFTER_SYNC=${STRIP_LOCAL_MODELS_AFTER_SYNC:-false}
DEVICE=${DEVICE:-cuda}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-false}
ENABLE_WANDB=${ENABLE_WANDB:-false}
WANDB_PROJECT=${WANDB_PROJECT:-multi-grpo-rema}
WANDB_EXPERIMENT_NAME=${WANDB_EXPERIMENT_NAME:-${MODE_TAG}-${PARAMETER_SHARING_TAG}-${MODEL_NAME_TAG}-${JOB_ID}}
DISABLE_ROLLOUT_LOGGING=${DISABLE_ROLLOUT_LOGGING:-false}
EPOCH_S3_SYNC=${EPOCH_S3_SYNC:-false}
EPOCH_S3_SYNC_INTERVAL=${EPOCH_S3_SYNC_INTERVAL:-300}
OFFLINE_GRPO_PYTORCH_CUDA_ALLOC_CONF=${OFFLINE_GRPO_PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Usually leave alone: runtime paths / artifact plumbing.
LOCAL_VERL_DIR=${LOCAL_VERL_DIR:-$TMPDIR/verl}
LOCAL_SIF_IMAGE_PATH=${LOCAL_SIF_IMAGE_PATH:-$TMPDIR/verl-rema-v3.sif}
RAY_LOCAL_TMPDIR=${RAY_LOCAL_TMPDIR:-/tmp/${USER:-user}/hierarchical_rema_${JOB_ID}/ray}

S3_EPOCHS_PATH=${S3_EPOCHS_PATH:-${S3_OUTPUT_PATH}/epochs}
S3_BEST_SO_FAR_MODELS_PATH=${S3_BEST_SO_FAR_MODELS_PATH:-${S3_OUTPUT_PATH}/best_so_far_models}
S3_BEST_VAL_MODELS_PATH=${S3_BEST_VAL_MODELS_PATH:-${S3_OUTPUT_PATH}/best_val_models}
S3_FINAL_MODELS_PATH=${S3_FINAL_MODELS_PATH:-${S3_OUTPUT_PATH}/final_models}

mkdir -p "$RUN_ROOT" "$LOCAL_OUTPUT_DIR" "$PERSIST_LOCAL_DIR"

if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    RAY_CPUS_PER_NODE=${RAY_CPUS_PER_NODE:-$SLURM_CPUS_PER_TASK}
elif [[ -n "${SLURM_CPUS_PER_GPU:-}" ]]; then
    RAY_CPUS_PER_NODE=${RAY_CPUS_PER_NODE:-$(( SLURM_CPUS_PER_GPU * RAY_N_GPUS_PER_NODE ))}
else
    RAY_CPUS_PER_NODE=${RAY_CPUS_PER_NODE:-1}
fi

export HF_HOME=${HF_HOME:-$TMPDIR/hf_home}
export PYTHONUNBUFFERED=1

PARAMETER_SHARING_FLAG=""
if [[ "$PARAMETER_SHARING" == "1" || "$PARAMETER_SHARING" == "true" || "$PARAMETER_SHARING" == "True" ]]; then
    PARAMETER_SHARING_FLAG="--parameter-sharing"
fi

SAVE_REPLAY_COPY_FLAG=""
if [[ "$SAVE_REPLAY_COPY" == "1" || "$SAVE_REPLAY_COPY" == "true" || "$SAVE_REPLAY_COPY" == "True" ]]; then
    SAVE_REPLAY_COPY_FLAG="--save-replay-copy"
fi

FINAL_ANSWER_CORRECTNESS_REWARD_ONLY_FLAG=""
if [[ "$FINAL_ANSWER_CORRECTNESS_REWARD_ONLY" == "1" || "$FINAL_ANSWER_CORRECTNESS_REWARD_ONLY" == "true" || "$FINAL_ANSWER_CORRECTNESS_REWARD_ONLY" == "True" ]]; then
    FINAL_ANSWER_CORRECTNESS_REWARD_ONLY_FLAG="--final-answer-correctness-reward-only"
fi

TRACK_WORKERS_HISTORY_FLAG="--track-workers-history"
if [[ "$TRACK_WORKERS_HISTORY" == "0" || "$TRACK_WORKERS_HISTORY" == "false" || "$TRACK_WORKERS_HISTORY" == "False" ]]; then
    TRACK_WORKERS_HISTORY_FLAG="--no-track-workers-history"
fi

TRAIN_WORKER_MODEL_FLAG=""
if [[ "$TRAIN_WORKER_MODEL" == "1" || "$TRAIN_WORKER_MODEL" == "true" || "$TRAIN_WORKER_MODEL" == "True" ]]; then
    TRAIN_WORKER_MODEL_FLAG="--train-worker-model"
fi

OFFLINE_GRPO_DISTRIBUTED_FLAG=""
if [[ "$OFFLINE_GRPO_DISTRIBUTED" == "1" || "$OFFLINE_GRPO_DISTRIBUTED" == "true" || "$OFFLINE_GRPO_DISTRIBUTED" == "True" ]]; then
    OFFLINE_GRPO_DISTRIBUTED_FLAG="--offline-grpo-distributed"
fi

GRADIENT_CHECKPOINTING_FLAG=""
if [[ "$GRADIENT_CHECKPOINTING" == "1" || "$GRADIENT_CHECKPOINTING" == "true" || "$GRADIENT_CHECKPOINTING" == "True" ]]; then
    GRADIENT_CHECKPOINTING_FLAG="--gradient-checkpointing"
fi

TRUST_REMOTE_CODE_FLAG=""
if [[ "$TRUST_REMOTE_CODE" == "1" || "$TRUST_REMOTE_CODE" == "true" || "$TRUST_REMOTE_CODE" == "True" ]]; then
    TRUST_REMOTE_CODE_FLAG="--trust-remote-code"
fi

CONTROLLER_CONSTRAINED_DECODING_FLAG="--controller-constrained-decoding"
if [[ "$CONTROLLER_CONSTRAINED_DECODING" == "0" || "$CONTROLLER_CONSTRAINED_DECODING" == "false" || "$CONTROLLER_CONSTRAINED_DECODING" == "False" ]]; then
    CONTROLLER_CONSTRAINED_DECODING_FLAG="--disable-controller-constrained-decoding"
fi

ENABLE_WANDB_FLAG=""
if [[ "$ENABLE_WANDB" == "1" || "$ENABLE_WANDB" == "true" || "$ENABLE_WANDB" == "True" ]]; then
    ENABLE_WANDB_FLAG="--enable-wandb"
fi

SHUFFLE_TASKS_FLAG=""
if [[ "$SHUFFLE_TASKS" == "1" || "$SHUFFLE_TASKS" == "true" || "$SHUFFLE_TASKS" == "True" ]]; then
    SHUFFLE_TASKS_FLAG="--shuffle-tasks"
fi

DISABLE_ROLLOUT_LOGGING_FLAG=""
if [[ "$DISABLE_ROLLOUT_LOGGING" == "1" || "$DISABLE_ROLLOUT_LOGGING" == "true" || "$DISABLE_ROLLOUT_LOGGING" == "True" ]]; then
    DISABLE_ROLLOUT_LOGGING_FLAG="--disable-rollout-logging"
fi

DISABLE_EXTERNAL_VALIDATION_FLAG=""
if [[ "$DISABLE_EXTERNAL_VALIDATION" == "1" || "$DISABLE_EXTERNAL_VALIDATION" == "true" || "$DISABLE_EXTERNAL_VALIDATION" == "True" ]]; then
    DISABLE_EXTERNAL_VALIDATION_FLAG="--disable-external-validation"
fi

PRUNE_STALE_POLICY_MODELS_FLAG=""
if [[ "$PRUNE_STALE_POLICY_MODELS" == "1" || "$PRUNE_STALE_POLICY_MODELS" == "true" || "$PRUNE_STALE_POLICY_MODELS" == "True" ]]; then
    PRUNE_STALE_POLICY_MODELS_FLAG="--prune-stale-policy-models"
fi

WANDB_EXPERIMENT_NAME_FLAG=""
if [[ -n "$WANDB_EXPERIMENT_NAME" ]]; then
    WANDB_EXPERIMENT_NAME_FLAG="--experiment-name ${WANDB_EXPERIMENT_NAME}"
fi

TRAIN_MODEL_FLAGS="--model-path ${MODEL_PATH}"
if [[ "${DECOMPOSER_MODEL_PATH}" != "${MODEL_PATH}" || "${SELECTOR_MODEL_PATH}" != "${MODEL_PATH}" ]]; then
    TRAIN_MODEL_FLAGS="--decomposer-model-path ${DECOMPOSER_MODEL_PATH} --selector-model-path ${SELECTOR_MODEL_PATH}"
fi

TRAIN_POLICY_ID_FLAG=""
if [[ -n "$TRAIN_POLICY_ID" ]]; then
    TRAIN_POLICY_ID_FLAG="--policy-id ${TRAIN_POLICY_ID}"
fi

TRAIN_MIN_REWARD_FLAG=""
if [[ -n "$TRAIN_MIN_REWARD" ]]; then
    TRAIN_MIN_REWARD_FLAG="--min-reward ${TRAIN_MIN_REWARD}"
fi

TRAIN_MIN_ADVANTAGE_FLAG=""
if [[ -n "$TRAIN_MIN_ADVANTAGE" ]]; then
    TRAIN_MIN_ADVANTAGE_FLAG="--min-advantage ${TRAIN_MIN_ADVANTAGE}"
fi

VLLM_MAX_MODEL_LEN_FLAG=""
if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then
    VLLM_MAX_MODEL_LEN_FLAG="--vllm-max-model-len ${VLLM_MAX_MODEL_LEN}"
fi

MULTINODE_RAY_ENABLED=0
if [[ "$BACKEND" == "vllm" && "${RAY_NNODES:-1}" -gt 1 ]]; then
    MULTINODE_RAY_ENABLED=1
fi
if [[ -n "${SLURM_JOB_NUM_NODES:-}" && "${RAY_NNODES:-1}" -gt "${SLURM_JOB_NUM_NODES:-1}" ]]; then
    echo "RAY_NNODES=${RAY_NNODES} exceeds allocated SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}" >&2
    exit 1
fi
if [[ -n "${SLURM_JOB_NUM_NODES:-}" && "${OFFLINE_GRPO_NNODES:-1}" -gt "${SLURM_JOB_NUM_NODES:-1}" ]]; then
    echo "OFFLINE_GRPO_NNODES=${OFFLINE_GRPO_NNODES} exceeds allocated SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}" >&2
    exit 1
fi
RUNTIME_STAGE_NNODES=1
if [[ "$MULTINODE_RAY_ENABLED" == "1" ]]; then
    RUNTIME_STAGE_NNODES=$RAY_NNODES
fi
if [[ "$OFFLINE_GRPO_DISTRIBUTED" == "1" || "$OFFLINE_GRPO_DISTRIBUTED" == "true" || "$OFFLINE_GRPO_DISTRIBUTED" == "True" ]]; then
    if [[ "${OFFLINE_GRPO_NNODES:-1}" -gt "$RUNTIME_STAGE_NNODES" ]]; then
        RUNTIME_STAGE_NNODES=$OFFLINE_GRPO_NNODES
    fi
fi

RAY_CLUSTER_PIDS=()
RAY_HEAD_NODE=""
RAY_HEAD_NODE_IP=""
RAY_ADDRESS_VALUE=""
RAY_DRIVER_NODE_FLAG=""

stage_runtime_payload() {
    local runtime_lock_dir="${LOCAL_VERL_DIR}.stage_lock"
    local runtime_done_file="${LOCAL_VERL_DIR}.stage_complete"
    local image_lock_dir="${LOCAL_SIF_IMAGE_PATH}.stage_lock"
    local image_done_file="${LOCAL_SIF_IMAGE_PATH}.stage_complete"

    mkdir -p "$TMPDIR"
    mkdir -p "$RAY_LOCAL_TMPDIR"

    if [[ ! -f "$runtime_done_file" ]]; then
        while ! mkdir "$runtime_lock_dir" 2>/dev/null; do
            sleep 1
        done
        if [[ ! -f "$runtime_done_file" ]]; then
            rm -rf "$LOCAL_VERL_DIR"
            mkdir -p "$LOCAL_VERL_DIR"
            cp -r "$SOURCE_DIR/src/verl/." "$LOCAL_VERL_DIR"/
            touch "$runtime_done_file"
        fi
        rmdir "$runtime_lock_dir" >/dev/null 2>&1 || true
    fi

    if [[ ! -f "$image_done_file" ]]; then
        while ! mkdir "$image_lock_dir" 2>/dev/null; do
            sleep 1
        done
        if [[ ! -f "$image_done_file" ]]; then
            rm -f "$LOCAL_SIF_IMAGE_PATH"
            rclone copyto "$SIF_IMAGE_PATH" "$LOCAL_SIF_IMAGE_PATH"
            touch "$image_done_file"
        fi
        rmdir "$image_lock_dir" >/dev/null 2>&1 || true
    fi
}

stage_runtime_on_current_node() {
    stage_runtime_payload
}

stage_runtime_on_allocated_nodes() {
    if [[ "$RUNTIME_STAGE_NNODES" -le 1 ]]; then
        stage_runtime_on_current_node
        return
    fi

    srun --nodes="${RUNTIME_STAGE_NNODES}" --ntasks="${RUNTIME_STAGE_NNODES}" bash -lc "
set -euo pipefail
runtime_lock_dir=\"${LOCAL_VERL_DIR}.stage_lock\"
runtime_done_file=\"${LOCAL_VERL_DIR}.stage_complete\"
image_lock_dir=\"${LOCAL_SIF_IMAGE_PATH}.stage_lock\"
image_done_file=\"${LOCAL_SIF_IMAGE_PATH}.stage_complete\"

mkdir -p \"$TMPDIR\"
mkdir -p \"$RAY_LOCAL_TMPDIR\"

if [[ ! -f \"\$runtime_done_file\" ]]; then
    while ! mkdir \"\$runtime_lock_dir\" 2>/dev/null; do
        sleep 1
    done
    if [[ ! -f \"\$runtime_done_file\" ]]; then
        rm -rf \"$LOCAL_VERL_DIR\"
        mkdir -p \"$LOCAL_VERL_DIR\"
        cp -r \"$SOURCE_DIR/src/verl/.\" \"$LOCAL_VERL_DIR\"/
        touch \"\$runtime_done_file\"
    fi
    rmdir \"\$runtime_lock_dir\" >/dev/null 2>&1 || true
fi

if [[ ! -f \"\$image_done_file\" ]]; then
    while ! mkdir \"\$image_lock_dir\" 2>/dev/null; do
        sleep 1
    done
    if [[ ! -f \"\$image_done_file\" ]]; then
        rm -f \"$LOCAL_SIF_IMAGE_PATH\"
        rclone copyto \"$SIF_IMAGE_PATH\" \"$LOCAL_SIF_IMAGE_PATH\"
        touch \"\$image_done_file\"
    fi
    rmdir \"\$image_lock_dir\" >/dev/null 2>&1 || true
fi
"
}

resolve_ray_head_node() {
    if [[ "$MULTINODE_RAY_ENABLED" == "0" ]]; then
        return
    fi

    mapfile -t SLURM_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    RAY_HEAD_NODE="${SLURM_HOSTS[0]}"
    RAY_HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$RAY_HEAD_NODE" bash -lc "hostname -I | awk '{for(i=1;i<=NF;i++) if(\$i !~ /:/){print \$i; exit}}'")
    if [[ -z "$RAY_HEAD_NODE_IP" ]]; then
        RAY_HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$RAY_HEAD_NODE" bash -lc "hostname --ip-address | awk '{print \$1}'")
    fi
    if [[ -z "$RAY_HEAD_NODE_IP" ]]; then
        echo "Failed to resolve an IPv4 address for Ray head node ${RAY_HEAD_NODE}" >&2
        exit 1
    fi
    RAY_ADDRESS_VALUE="${RAY_HEAD_NODE_IP}:${RAY_PORT}"
    RAY_DRIVER_NODE_FLAG="-w ${RAY_HEAD_NODE}"
}

wait_for_ray_head() {
    if [[ "$MULTINODE_RAY_ENABLED" == "0" ]]; then
        return
    fi

    local attempt=0
    local max_attempts=45
    while [[ "$attempt" -lt "$max_attempts" ]]; do
        if srun --overlap --nodes=1 --ntasks=1 -w "$RAY_HEAD_NODE" \
            apptainer exec --nv --writable-tmpfs \
            --mount type=bind,src=$TMPDIR,dst=$TMPDIR \
            --mount type=bind,src=$RAY_LOCAL_TMPDIR,dst=$RAY_LOCAL_TMPDIR \
            --mount type=bind,src=$TMPDIR,dst=/root/tmpdir \
            --mount type=bind,src=$LOCAL_VERL_DIR,dst=/verl \
            "$LOCAL_SIF_IMAGE_PATH" \
            bash -lc "export TMPDIR='${RAY_LOCAL_TMPDIR}'; export RAY_TMPDIR='${RAY_LOCAL_TMPDIR}'; export PYTHONPATH='/verl/verl':\$PYTHONPATH; python3 -m ray.scripts.scripts status --address '${RAY_ADDRESS_VALUE}' >/dev/null 2>&1"; then
            echo "[hierarchical-rema][ray] head is ready address=${RAY_ADDRESS_VALUE}"
            return
        fi
        sleep 2
        attempt=$((attempt + 1))
    done

    echo "[hierarchical-rema][ray] head failed to become ready address=${RAY_ADDRESS_VALUE}" >&2
    exit 1
}

start_ray_cluster() {
    if [[ "$MULTINODE_RAY_ENABLED" == "0" ]]; then
        return
    fi

    resolve_ray_head_node
    export RAY_ADDRESS="$RAY_ADDRESS_VALUE"
    export RAY_NAMESPACE

    echo "[hierarchical-rema][ray] starting head node=${RAY_HEAD_NODE} address=${RAY_ADDRESS_VALUE} nnodes=${RAY_NNODES} gpus_per_node=${RAY_N_GPUS_PER_NODE}"

    srun --nodes=1 --ntasks=1 -w "$RAY_HEAD_NODE" \
        apptainer exec --nv --writable-tmpfs \
        --mount type=bind,src=$TMPDIR,dst=$TMPDIR \
        --mount type=bind,src=$RAY_LOCAL_TMPDIR,dst=$RAY_LOCAL_TMPDIR \
        --mount type=bind,src=$TMPDIR,dst=/root/tmpdir \
        --mount type=bind,src=$LOCAL_VERL_DIR,dst=/verl \
        "$LOCAL_SIF_IMAGE_PATH" \
        bash -lc "export TMPDIR='${RAY_LOCAL_TMPDIR}'; export RAY_TMPDIR='${RAY_LOCAL_TMPDIR}'; export PYTHONPATH='/verl/verl':\$PYTHONPATH; python3 -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true; python3 -m ray.scripts.scripts start --head --node-ip-address='$RAY_HEAD_NODE_IP' --port='${RAY_PORT}' --dashboard-host=0.0.0.0 --dashboard-port='${RAY_DASHBOARD_PORT}' --temp-dir='${RAY_LOCAL_TMPDIR}' --num-cpus='${RAY_CPUS_PER_NODE}' --num-gpus='${RAY_N_GPUS_PER_NODE}' --block" &
    RAY_CLUSTER_PIDS+=("$!")
    wait_for_ray_head

    local worker_num=$((RAY_NNODES - 1))
    local idx=0
    while [[ "$idx" -lt "$worker_num" ]]; do
        local node_i="${SLURM_HOSTS[$((idx + 1))]}"
        echo "[hierarchical-rema][ray] starting worker node=${node_i} address=${RAY_ADDRESS_VALUE}"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            apptainer exec --nv --writable-tmpfs \
            --mount type=bind,src=$TMPDIR,dst=$TMPDIR \
            --mount type=bind,src=$RAY_LOCAL_TMPDIR,dst=$RAY_LOCAL_TMPDIR \
            --mount type=bind,src=$TMPDIR,dst=/root/tmpdir \
            --mount type=bind,src=$LOCAL_VERL_DIR,dst=/verl \
            "$LOCAL_SIF_IMAGE_PATH" \
            bash -lc "export TMPDIR='${RAY_LOCAL_TMPDIR}'; export RAY_TMPDIR='${RAY_LOCAL_TMPDIR}'; export PYTHONPATH='/verl/verl':\$PYTHONPATH; python3 -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true; python3 -m ray.scripts.scripts start --address '${RAY_ADDRESS_VALUE}' --temp-dir='${RAY_LOCAL_TMPDIR}' --num-cpus='${RAY_CPUS_PER_NODE}' --num-gpus='${RAY_N_GPUS_PER_NODE}' --block" &
        RAY_CLUSTER_PIDS+=("$!")
        sleep 5
        idx=$((idx + 1))
    done
}

stop_ray_cluster() {
    if [[ "$MULTINODE_RAY_ENABLED" == "0" ]]; then
        return
    fi

    srun --nodes="${RAY_NNODES}" --ntasks="${RAY_NNODES}" \
        apptainer exec --nv --writable-tmpfs \
        --mount type=bind,src=$TMPDIR,dst=$TMPDIR \
        --mount type=bind,src=$RAY_LOCAL_TMPDIR,dst=$RAY_LOCAL_TMPDIR \
        --mount type=bind,src=$TMPDIR,dst=/root/tmpdir \
        --mount type=bind,src=$LOCAL_VERL_DIR,dst=/verl \
        "$LOCAL_SIF_IMAGE_PATH" \
        bash -lc "export TMPDIR='${RAY_LOCAL_TMPDIR}'; export RAY_TMPDIR='${RAY_LOCAL_TMPDIR}'; export PYTHONPATH='/verl/verl':\$PYTHONPATH; python3 -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true" >/dev/null 2>&1 || true

    local pid=""
    for pid in "${RAY_CLUSTER_PIDS[@]}"; do
        kill "$pid" >/dev/null 2>&1 || true
        wait "$pid" 2>/dev/null || true
    done
    RAY_CLUSTER_PIDS=()
}

stage_task_source() {
    if [[ "$TASK_SOURCE" == "demo" ]]; then
        TASK_SOURCE_RUNTIME="demo"
        return
    fi

    if [[ "$TASK_SOURCE" != /* && "$TASK_SOURCE" != *:* && -e "$SOURCE_DIR/$TASK_SOURCE" ]]; then
        TASK_SOURCE="$SOURCE_DIR/$TASK_SOURCE"
    fi

    mkdir -p "$TASK_SOURCE_STAGE"

    if [[ "$TASK_SOURCE" == *:* ]]; then
        rclone copy "$TASK_SOURCE" "$TASK_SOURCE_STAGE"
    elif [[ -d "$TASK_SOURCE" ]]; then
        cp -r "$TASK_SOURCE"/. "$TASK_SOURCE_STAGE"/
    elif [[ -f "$TASK_SOURCE" ]]; then
        cp "$TASK_SOURCE" "$TASK_SOURCE_STAGE"/
    else
        echo "TASK_SOURCE does not exist: $TASK_SOURCE" >&2
        exit 1
    fi

    if [[ -f "$TASK_SOURCE" ]]; then
        TASK_SOURCE_RUNTIME="$TASK_SOURCE_STAGE/$(basename "$TASK_SOURCE")"
    elif [[ "$TASK_SOURCE" == *:* ]]; then
        remote_name=$(basename "$TASK_SOURCE")
        if [[ "$remote_name" == *.* && -f "$TASK_SOURCE_STAGE/$remote_name" ]]; then
            TASK_SOURCE_RUNTIME="$TASK_SOURCE_STAGE/$remote_name"
        else
            TASK_SOURCE_RUNTIME="$TASK_SOURCE_STAGE"
        fi
    else
        TASK_SOURCE_RUNTIME="$TASK_SOURCE_STAGE"
    fi
}

stage_val_task_source() {
    if [[ "$DISABLE_EXTERNAL_VALIDATION" == "1" || "$DISABLE_EXTERNAL_VALIDATION" == "true" || "$DISABLE_EXTERNAL_VALIDATION" == "True" ]]; then
        VAL_TASK_SOURCE_RUNTIME=""
        return
    fi

    if [[ -z "$VAL_TASK_SOURCE" ]]; then
        VAL_TASK_SOURCE_RUNTIME=""
        return
    fi

    if [[ "$VAL_TASK_SOURCE" == "demo" ]]; then
        VAL_TASK_SOURCE_RUNTIME="demo"
        return
    fi

    if [[ "$VAL_TASK_SOURCE" != /* && "$VAL_TASK_SOURCE" != *:* && -e "$SOURCE_DIR/$VAL_TASK_SOURCE" ]]; then
        VAL_TASK_SOURCE="$SOURCE_DIR/$VAL_TASK_SOURCE"
    fi

    mkdir -p "$VAL_TASK_SOURCE_STAGE"

    if [[ "$VAL_TASK_SOURCE" == *:* ]]; then
        rclone copy "$VAL_TASK_SOURCE" "$VAL_TASK_SOURCE_STAGE"
    elif [[ -d "$VAL_TASK_SOURCE" ]]; then
        cp -r "$VAL_TASK_SOURCE"/. "$VAL_TASK_SOURCE_STAGE"/
    elif [[ -f "$VAL_TASK_SOURCE" ]]; then
        cp "$VAL_TASK_SOURCE" "$VAL_TASK_SOURCE_STAGE"/
    else
        echo "VAL_TASK_SOURCE does not exist: $VAL_TASK_SOURCE" >&2
        exit 1
    fi

    if [[ -f "$VAL_TASK_SOURCE" ]]; then
        VAL_TASK_SOURCE_RUNTIME="$VAL_TASK_SOURCE_STAGE/$(basename "$VAL_TASK_SOURCE")"
    elif [[ "$VAL_TASK_SOURCE" == *:* ]]; then
        remote_name=$(basename "$VAL_TASK_SOURCE")
        if [[ "$remote_name" == *.* && -f "$VAL_TASK_SOURCE_STAGE/$remote_name" ]]; then
            VAL_TASK_SOURCE_RUNTIME="$VAL_TASK_SOURCE_STAGE/$remote_name"
        else
            VAL_TASK_SOURCE_RUNTIME="$VAL_TASK_SOURCE_STAGE"
        fi
    else
        VAL_TASK_SOURCE_RUNTIME="$VAL_TASK_SOURCE_STAGE"
    fi
}

write_run_metadata() {
    cat > "$RUN_METADATA_FILE" <<EOF
JOB_ID=$JOB_ID
RUN_KIND=$RUN_KIND
RUN_ROOT=$RUN_ROOT
LOCAL_OUTPUT_DIR=$LOCAL_OUTPUT_DIR
PERSIST_LOCAL_DIR=$PERSIST_LOCAL_DIR
S3_OUTPUT_PATH=$S3_OUTPUT_PATH
TASK_SOURCE=$TASK_SOURCE
TASK_SOURCE_STAGE=$TASK_SOURCE_STAGE
TASK_SOURCE_RUNTIME=$TASK_SOURCE_RUNTIME
TASK_FORMAT=$TASK_FORMAT
PROMPT_KEY=$PROMPT_KEY
ANSWER_KEY=$ANSWER_KEY
TASK_ID_KEY=$TASK_ID_KEY
MAX_TASKS=$MAX_TASKS
TRAIN_SUBSET_SIZE=$TRAIN_SUBSET_SIZE
TASKS_PER_EPOCH=$TASKS_PER_EPOCH
SHUFFLE_TASKS=$SHUFFLE_TASKS
VAL_TASK_SOURCE=$VAL_TASK_SOURCE
VAL_TASK_SOURCE_STAGE=$VAL_TASK_SOURCE_STAGE
VAL_TASK_SOURCE_RUNTIME=$VAL_TASK_SOURCE_RUNTIME
VAL_TASK_FORMAT=$VAL_TASK_FORMAT
VAL_PROMPT_KEY=$VAL_PROMPT_KEY
VAL_ANSWER_KEY=$VAL_ANSWER_KEY
VAL_TASK_ID_KEY=$VAL_TASK_ID_KEY
MAX_VAL_TASKS=$MAX_VAL_TASKS
VAL_TASKS_PER_EPOCH=$VAL_TASKS_PER_EPOCH
EXTERNAL_VALIDATION_EVERY_SUBSET_ROUNDS=$EXTERNAL_VALIDATION_EVERY_SUBSET_ROUNDS
DISABLE_EXTERNAL_VALIDATION=$DISABLE_EXTERNAL_VALIDATION
VAL_NUM_DECOMPOSITIONS=$VAL_NUM_DECOMPOSITIONS
VAL_NUM_SELECTIONS=$VAL_NUM_SELECTIONS
VAL_TEMPERATURE=$VAL_TEMPERATURE
VAL_CONTROLLER_TEMPERATURE=$VAL_CONTROLLER_TEMPERATURE
VAL_WORKER_TEMPERATURE=$VAL_WORKER_TEMPERATURE
VAL_TOP_P=$VAL_TOP_P
VAL_CONTROLLER_MAX_NEW_TOKENS=$VAL_CONTROLLER_MAX_NEW_TOKENS
VAL_DECOMPOSER_MAX_NEW_TOKENS=$VAL_DECOMPOSER_MAX_NEW_TOKENS
VAL_SELECTOR_MAX_NEW_TOKENS=$VAL_SELECTOR_MAX_NEW_TOKENS
VAL_WORKER_MAX_NEW_TOKENS=$VAL_WORKER_MAX_NEW_TOKENS
VAL_ROLLOUT_TASK_BATCH_SIZE=$VAL_ROLLOUT_TASK_BATCH_SIZE
TRAIN_SUBSET_ROUNDS=$TRAIN_SUBSET_ROUNDS
UPDATE_EVERY_N_ROLLOUT_BATCHES=$UPDATE_EVERY_N_ROLLOUT_BATCHES
NUM_EPOCHS=$NUM_EPOCHS
MODEL_PATH=$MODEL_PATH
DECOMPOSER_MODEL_PATH=$DECOMPOSER_MODEL_PATH
SELECTOR_MODEL_PATH=$SELECTOR_MODEL_PATH
BACKEND=$BACKEND
TASK=$TASK
MODE=$MODE
PHASE=$PHASE
PRINT_MODE=$PRINT_MODE
ROLLOUT_TASK_BATCH_SIZE=$ROLLOUT_TASK_BATCH_SIZE
ROLLOUT_PROGRESS_EVERY=$ROLLOUT_PROGRESS_EVERY
CONTROLLER_BATCH_SIZE=$CONTROLLER_BATCH_SIZE
WORKER_BATCH_SIZE=$WORKER_BATCH_SIZE
TEMPERATURE=$TEMPERATURE
TOP_P=$TOP_P
CONTROLLER_TEMPERATURE=$CONTROLLER_TEMPERATURE
WORKER_TEMPERATURE=$WORKER_TEMPERATURE
CONTROLLER_CONSTRAINED_DECODING=$CONTROLLER_CONSTRAINED_DECODING
CONTROLLER_MAX_NEW_TOKENS=$CONTROLLER_MAX_NEW_TOKENS
DECOMPOSER_MAX_NEW_TOKENS=$DECOMPOSER_MAX_NEW_TOKENS
SELECTOR_MAX_NEW_TOKENS=$SELECTOR_MAX_NEW_TOKENS
CONTROLLER_FORMAT_RETRY_PENALTY=$CONTROLLER_FORMAT_RETRY_PENALTY
CONTROLLER_FORMAT_FALLBACK_PENALTY=$CONTROLLER_FORMAT_FALLBACK_PENALTY
FINAL_ANSWER_CORRECTNESS_REWARD_ONLY=$FINAL_ANSWER_CORRECTNESS_REWARD_ONLY
CONFIDENCE_REWARD_WEIGHT=$CONFIDENCE_REWARD_WEIGHT
COMPATIBILITY_REWARD_WEIGHT=$COMPATIBILITY_REWARD_WEIGHT
TRACK_WORKERS_HISTORY=$TRACK_WORKERS_HISTORY
TRAIN_WORKER_MODEL=$TRAIN_WORKER_MODEL
ROLLOUT_PROMPT_LENGTH=$ROLLOUT_PROMPT_LENGTH
RAY_CPUS_PER_NODE=$RAY_CPUS_PER_NODE
MULTINODE_RAY_ENABLED=$MULTINODE_RAY_ENABLED
RAY_NNODES=$RAY_NNODES
RAY_N_GPUS_PER_NODE=$RAY_N_GPUS_PER_NODE
RAY_PORT=$RAY_PORT
RAY_DASHBOARD_PORT=$RAY_DASHBOARD_PORT
RAY_NAMESPACE=$RAY_NAMESPACE
OFFLINE_GRPO_DISTRIBUTED=$OFFLINE_GRPO_DISTRIBUTED
OFFLINE_GRPO_NNODES=$OFFLINE_GRPO_NNODES
OFFLINE_GRPO_GPUS_PER_NODE=$OFFLINE_GRPO_GPUS_PER_NODE
OFFLINE_GRPO_MASTER_PORT=$OFFLINE_GRPO_MASTER_PORT
VLLM_TENSOR_PARALLEL_SIZE=$VLLM_TENSOR_PARALLEL_SIZE
VLLM_GPU_MEMORY_UTILIZATION=$VLLM_GPU_MEMORY_UTILIZATION
VLLM_MAX_NUM_BATCHED_TOKENS=$VLLM_MAX_NUM_BATCHED_TOKENS
VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS
VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN
ENABLE_WANDB=$ENABLE_WANDB
WANDB_PROJECT=$WANDB_PROJECT
WANDB_EXPERIMENT_NAME=$WANDB_EXPERIMENT_NAME
OFFLINE_GRPO_PYTORCH_CUDA_ALLOC_CONF=$OFFLINE_GRPO_PYTORCH_CUDA_ALLOC_CONF
DISABLE_ROLLOUT_LOGGING=$DISABLE_ROLLOUT_LOGGING
ROLLOUT_LOG_MODE=$ROLLOUT_LOG_MODE
ROLLOUT_LOG_DETAIL=$ROLLOUT_LOG_DETAIL
CHECKPOINT_MODE=$CHECKPOINT_MODE
PRUNE_STALE_POLICY_MODELS=$PRUNE_STALE_POLICY_MODELS
PRUNE_UPLOADED_LOCAL_CHECKPOINTS=$PRUNE_UPLOADED_LOCAL_CHECKPOINTS
STRIP_LOCAL_MODELS_AFTER_SYNC=$STRIP_LOCAL_MODELS_AFTER_SYNC
EPOCH_S3_SYNC=$EPOCH_S3_SYNC
EPOCH_S3_SYNC_INTERVAL=$EPOCH_S3_SYNC_INTERVAL
GRPO_PASSES_PER_SUBSET=$GRPO_PASSES_PER_SUBSET
S3_EPOCHS_PATH=$S3_EPOCHS_PATH
S3_BEST_SO_FAR_MODELS_PATH=$S3_BEST_SO_FAR_MODELS_PATH
S3_BEST_VAL_MODELS_PATH=$S3_BEST_VAL_MODELS_PATH
S3_FINAL_MODELS_PATH=$S3_FINAL_MODELS_PATH
EOF
}

upload_epoch_folder_to_s3() {
    local epoch_dir="$1"
    local epoch_name
    epoch_name=$(basename "$epoch_dir")
    local upload_status=0

    if [[ -z "${S3_OUTPUT_PATH:-}" ]]; then
        return 0
    fi

    echo "[hierarchical-rema][s3] uploading epoch folder ${epoch_name} -> ${S3_EPOCHS_PATH}/${epoch_name}"
    if ! rclone copy "$epoch_dir" "${S3_EPOCHS_PATH}/${epoch_name}"; then
        echo "Warning: failed to upload epoch folder ${epoch_name} to ${S3_EPOCHS_PATH}/${epoch_name}" >&2
        upload_status=1
    fi

    for policy_dir in "$epoch_dir"/train/*; do
        [[ -d "$policy_dir" ]] || continue
        local policy_id
        policy_id=$(basename "$policy_dir")
        if [[ -d "$policy_dir/final" ]]; then
            echo "[hierarchical-rema][s3] syncing best-so-far model ${policy_id} -> ${S3_BEST_SO_FAR_MODELS_PATH}/${policy_id}"
            if ! rclone sync "$policy_dir/final" "${S3_BEST_SO_FAR_MODELS_PATH}/${policy_id}"; then
                echo "Warning: failed to sync best-so-far model ${policy_id}" >&2
                upload_status=1
            fi
        fi
        if [[ -d "$policy_dir/best" ]]; then
            echo "[hierarchical-rema][s3] syncing best-val model ${policy_id} -> ${S3_BEST_VAL_MODELS_PATH}/${policy_id}"
            if ! rclone sync "$policy_dir/best" "${S3_BEST_VAL_MODELS_PATH}/${policy_id}"; then
                echo "Warning: failed to sync best-val model ${policy_id}" >&2
                upload_status=1
            fi
        fi
    done
    return "$upload_status"
}

prune_epoch_local_checkpoints() {
    local epoch_dir="$1"
    for policy_dir in "$epoch_dir"/train/*; do
        [[ -d "$policy_dir" ]] || continue
        rm -rf "$policy_dir"/best 2>/dev/null || true
        rm -rf "$policy_dir"/checkpoint-* 2>/dev/null || true
    done
}

strip_local_model_artifacts() {
    local root_dir="$1"
    local epoch_dir=""
    for epoch_dir in "$root_dir"/epoch_*; do
        [[ -d "$epoch_dir" ]] || continue
        local policy_dir=""
        for policy_dir in "$epoch_dir"/train/*; do
            [[ -d "$policy_dir" ]] || continue
            rm -rf "$policy_dir"/best 2>/dev/null || true
            rm -rf "$policy_dir"/final 2>/dev/null || true
            rm -rf "$policy_dir"/checkpoint-* 2>/dev/null || true
        done
    done
}

upload_final_models_to_s3() {
    local latest_epoch=""
    local epoch_dir=""

    for epoch_dir in "$LOCAL_OUTPUT_DIR"/epoch_*; do
        [[ -d "$epoch_dir" ]] || continue
        latest_epoch="$epoch_dir"
    done

    [[ -n "$latest_epoch" ]] || return 0
    [[ -n "${S3_OUTPUT_PATH:-}" ]] || return 0

    for policy_dir in "$latest_epoch"/train/*; do
        [[ -d "$policy_dir" ]] || continue
        local policy_id
        policy_id=$(basename "$policy_dir")
        if [[ -d "$policy_dir/final" ]]; then
            echo "[hierarchical-rema][s3] syncing final model ${policy_id} -> ${S3_FINAL_MODELS_PATH}/${policy_id}"
            rclone sync "$policy_dir/final" "${S3_FINAL_MODELS_PATH}/${policy_id}" || \
                echo "Warning: failed to sync final model ${policy_id}" >&2
        fi
    done
}

start_epoch_s3_sync_watcher() {
    [[ "$RUN_KIND" == "train" ]] || return 0
    [[ -n "${S3_OUTPUT_PATH:-}" ]] || return 0
    if [[ "$EPOCH_S3_SYNC" != "1" && "$EPOCH_S3_SYNC" != "true" && "$EPOCH_S3_SYNC" != "True" ]]; then
        return 0
    fi

    local marker_dir="$RUN_ROOT/.uploaded_epochs"
    mkdir -p "$marker_dir"

    (
        while true; do
            local epoch_dir=""
            for epoch_dir in "$LOCAL_OUTPUT_DIR"/epoch_*; do
                [[ -d "$epoch_dir" ]] || continue
                [[ -f "$epoch_dir/epoch_summary.json" ]] || continue
                local epoch_name
                epoch_name=$(basename "$epoch_dir")
                [[ -f "$marker_dir/$epoch_name" ]] && continue
                if upload_epoch_folder_to_s3 "$epoch_dir"; then
                    if [[ "$PRUNE_UPLOADED_LOCAL_CHECKPOINTS" == "1" || "$PRUNE_UPLOADED_LOCAL_CHECKPOINTS" == "true" || "$PRUNE_UPLOADED_LOCAL_CHECKPOINTS" == "True" ]]; then
                        prune_epoch_local_checkpoints "$epoch_dir"
                    fi
                    touch "$marker_dir/$epoch_name"
                fi
            done
            sleep "$EPOCH_S3_SYNC_INTERVAL"
        done
    ) &
    EPOCH_S3_SYNC_PID=$!
}

stop_epoch_s3_sync_watcher() {
    if [[ -n "${EPOCH_S3_SYNC_PID:-}" ]]; then
        kill "$EPOCH_S3_SYNC_PID" >/dev/null 2>&1 || true
        wait "$EPOCH_S3_SYNC_PID" 2>/dev/null || true
        unset EPOCH_S3_SYNC_PID
    fi
}

persist_outputs() {
    set +e

    if [[ -d "$LOCAL_OUTPUT_DIR" ]]; then
        local s3_upload_succeeded=0
        if [[ -n "${S3_OUTPUT_PATH:-}" ]]; then
            if rclone copy "$LOCAL_OUTPUT_DIR" "$S3_OUTPUT_PATH"; then
                s3_upload_succeeded=1
            else
                echo "Warning: failed to upload outputs to ${S3_OUTPUT_PATH}" >&2
            fi
            upload_final_models_to_s3
        fi

        if [[ "$s3_upload_succeeded" == "1" && ( "$STRIP_LOCAL_MODELS_AFTER_SYNC" == "1" || "$STRIP_LOCAL_MODELS_AFTER_SYNC" == "true" || "$STRIP_LOCAL_MODELS_AFTER_SYNC" == "True" ) ]]; then
            strip_local_model_artifacts "$LOCAL_OUTPUT_DIR"
        fi

        mkdir -p "$PERSIST_LOCAL_DIR"
        cp -r "$LOCAL_OUTPUT_DIR"/. "$PERSIST_LOCAL_DIR"/ 2>/dev/null || true
    fi

    if [[ -n "${TMPDIR:-}" && -d "${TMPDIR:-}" ]]; then
        rm -rf "$TMPDIR"/*
    fi
}

if [[ "$RUN_KIND" == "train" ]]; then
    stage_task_source
    stage_val_task_source
fi
stage_runtime_on_allocated_nodes
start_ray_cluster

write_run_metadata

echo "[hierarchical-rema] starting run"
echo "[hierarchical-rema] RUN_KIND=$RUN_KIND"
echo "[hierarchical-rema] LOCAL_OUTPUT_DIR=$LOCAL_OUTPUT_DIR"
echo "[hierarchical-rema] S3_OUTPUT_PATH=$S3_OUTPUT_PATH"
if [[ "$MULTINODE_RAY_ENABLED" == "1" ]]; then
    echo "[hierarchical-rema] RAY_ADDRESS=${RAY_ADDRESS_VALUE}"
    echo "[hierarchical-rema] RAY_NAMESPACE=${RAY_NAMESPACE}"
    echo "[hierarchical-rema] RAY_NNODES=${RAY_NNODES}"
    echo "[hierarchical-rema] RAY_N_GPUS_PER_NODE=${RAY_N_GPUS_PER_NODE}"
fi
if [[ "$RUN_KIND" == "train" ]]; then
    echo "[hierarchical-rema] TASK_SOURCE=$TASK_SOURCE"
    echo "[hierarchical-rema] TASK_SOURCE_RUNTIME=$TASK_SOURCE_RUNTIME"
    if [[ "$TASK_SOURCE_RUNTIME" != "demo" ]]; then
        find "$TASK_SOURCE_STAGE" -maxdepth 3 -type f | sort || true
    fi
    echo "[hierarchical-rema] VAL_TASK_SOURCE=$VAL_TASK_SOURCE"
    echo "[hierarchical-rema] VAL_TASK_SOURCE_RUNTIME=$VAL_TASK_SOURCE_RUNTIME"
    if [[ -n "$VAL_TASK_SOURCE_RUNTIME" && "$VAL_TASK_SOURCE_RUNTIME" != "demo" ]]; then
        find "$VAL_TASK_SOURCE_STAGE" -maxdepth 3 -type f | sort || true
    fi
fi

if [[ "$RUN_KIND" == "rollout" ]]; then
    COMMAND="unset ROCR_VISIBLE_DEVICES; \
export HF_HOME=$TMPDIR/hf_home; \
export PYTHONUNBUFFERED=1; \
export PYTHONPATH=/verl/verl:\$PYTHONPATH; \
unset PYTORCH_CUDA_ALLOC_CONF; \
export OFFLINE_GRPO_PYTORCH_CUDA_ALLOC_CONF='${OFFLINE_GRPO_PYTORCH_CUDA_ALLOC_CONF}'; \
export TMPDIR=${RAY_LOCAL_TMPDIR}; \
export RAY_TMPDIR=${RAY_LOCAL_TMPDIR}; \
export RAY_ADDRESS=\${RAY_ADDRESS:-}; \
export RAY_NAMESPACE=\${RAY_NAMESPACE:-}; \
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
  --max-nodes-per-decomposition ${MAX_NODES_PER_DECOMPOSITION} \
  --soft-max-hops ${SOFT_MAX_HOPS} \
  --hard-max-hops ${HARD_MAX_HOPS} \
  --soft-hop-penalty ${SOFT_HOP_PENALTY} \
  --temperature ${TEMPERATURE} \
  --controller-temperature ${CONTROLLER_TEMPERATURE} \
  --worker-temperature ${WORKER_TEMPERATURE} \
  --top-p ${TOP_P} \
  --controller-max-new-tokens ${CONTROLLER_MAX_NEW_TOKENS} \
  --decomposer-max-new-tokens ${DECOMPOSER_MAX_NEW_TOKENS} \
  --selector-max-new-tokens ${SELECTOR_MAX_NEW_TOKENS} \
  --worker-max-new-tokens ${WORKER_MAX_NEW_TOKENS} \
  --rollout-prompt-length ${ROLLOUT_PROMPT_LENGTH} \
  --ray-nnodes ${RAY_NNODES} \
  --ray-n-gpus-per-node ${RAY_N_GPUS_PER_NODE} \
  --ray-cpus-per-node ${RAY_CPUS_PER_NODE} \
  ${OFFLINE_GRPO_DISTRIBUTED_FLAG} \
  --offline-grpo-nnodes ${OFFLINE_GRPO_NNODES} \
  --offline-grpo-gpus-per-node ${OFFLINE_GRPO_GPUS_PER_NODE} \
  --offline-grpo-master-port ${OFFLINE_GRPO_MASTER_PORT} \
  --vllm-tensor-parallel-size ${VLLM_TENSOR_PARALLEL_SIZE} \
  --vllm-gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION} \
  --vllm-max-num-batched-tokens ${VLLM_MAX_NUM_BATCHED_TOKENS} \
  --vllm-max-num-seqs ${VLLM_MAX_NUM_SEQS} \
  ${VLLM_MAX_MODEL_LEN_FLAG} \
  --output-dir ${LOCAL_OUTPUT_DIR} \
  --rollout-log-mode ${ROLLOUT_LOG_MODE} \
  --rollout-log-detail ${ROLLOUT_LOG_DETAIL} \
  --best-k ${BEST_K} \
  --print-mode ${PRINT_MODE} \
  ${DISABLE_ROLLOUT_LOGGING_FLAG}"
else
    COMMAND="unset ROCR_VISIBLE_DEVICES; \
export HF_HOME=$TMPDIR/hf_home; \
export PYTHONUNBUFFERED=1; \
export PYTHONPATH=/verl/verl:\$PYTHONPATH; \
unset PYTORCH_CUDA_ALLOC_CONF; \
export OFFLINE_GRPO_PYTORCH_CUDA_ALLOC_CONF='${OFFLINE_GRPO_PYTORCH_CUDA_ALLOC_CONF}'; \
export TMPDIR=${RAY_LOCAL_TMPDIR}; \
export RAY_TMPDIR=${RAY_LOCAL_TMPDIR}; \
export RAY_ADDRESS=\${RAY_ADDRESS:-}; \
export RAY_NAMESPACE=\${RAY_NAMESPACE:-}; \
mkdir -p ${LOCAL_OUTPUT_DIR}; \
python3 -m hierarchical_rema.train \
  --task-source ${TASK_SOURCE_RUNTIME} \
  --task-format ${TASK_FORMAT} \
  --prompt-key ${PROMPT_KEY} \
  --answer-key ${ANSWER_KEY} \
  --task-id-key ${TASK_ID_KEY} \
  --max-tasks ${MAX_TASKS} \
  --tasks-per-epoch ${TASKS_PER_EPOCH} \
  ${SHUFFLE_TASKS_FLAG} \
  --val-task-source ${VAL_TASK_SOURCE_RUNTIME} \
  --val-task-format ${VAL_TASK_FORMAT} \
  --val-prompt-key ${VAL_PROMPT_KEY} \
  --val-answer-key ${VAL_ANSWER_KEY} \
  --val-task-id-key ${VAL_TASK_ID_KEY} \
  --max-val-tasks ${MAX_VAL_TASKS} \
  --val-tasks-per-epoch ${VAL_TASKS_PER_EPOCH} \
  --val-tasks-per-subset ${VAL_TASKS_PER_SUBSET} \
  --external-validation-every-n-epochs ${EXTERNAL_VALIDATION_EVERY_SUBSET_ROUNDS} \
  ${DISABLE_EXTERNAL_VALIDATION_FLAG} \
  --backend ${BACKEND} \
  --mode ${MODE} \
  --phase ${PHASE} \
  --num-epochs ${NUM_EPOCHS} \
  ${PARAMETER_SHARING_FLAG} \
  --worker-base-model-path ${WORKER_BASE_MODEL_PATH} \
  --num-decompositions ${NUM_DECOMPOSITIONS} \
  --num-selections ${NUM_SELECTIONS} \
  --max-nodes-per-decomposition ${MAX_NODES_PER_DECOMPOSITION} \
  --soft-max-hops ${SOFT_MAX_HOPS} \
  --hard-max-hops ${HARD_MAX_HOPS} \
  --soft-hop-penalty ${SOFT_HOP_PENALTY} \
  --temperature ${TEMPERATURE} \
  --controller-temperature ${CONTROLLER_TEMPERATURE} \
  --worker-temperature ${WORKER_TEMPERATURE} \
  --top-p ${TOP_P} \
  ${CONTROLLER_CONSTRAINED_DECODING_FLAG} \
  --controller-max-new-tokens ${CONTROLLER_MAX_NEW_TOKENS} \
  --decomposer-max-new-tokens ${DECOMPOSER_MAX_NEW_TOKENS} \
  --selector-max-new-tokens ${SELECTOR_MAX_NEW_TOKENS} \
  --worker-max-new-tokens ${WORKER_MAX_NEW_TOKENS} \
  --val-num-decompositions ${VAL_NUM_DECOMPOSITIONS} \
  --val-num-selections ${VAL_NUM_SELECTIONS} \
  --val-temperature ${VAL_TEMPERATURE} \
  --val-controller-temperature ${VAL_CONTROLLER_TEMPERATURE} \
  --val-worker-temperature ${VAL_WORKER_TEMPERATURE} \
  --val-top-p ${VAL_TOP_P} \
  --val-controller-max-new-tokens ${VAL_CONTROLLER_MAX_NEW_TOKENS} \
  --val-decomposer-max-new-tokens ${VAL_DECOMPOSER_MAX_NEW_TOKENS} \
  --val-selector-max-new-tokens ${VAL_SELECTOR_MAX_NEW_TOKENS} \
  --val-worker-max-new-tokens ${VAL_WORKER_MAX_NEW_TOKENS} \
  --val-controller-batch-size ${VAL_CONTROLLER_BATCH_SIZE} \
  --val-worker-batch-size ${VAL_WORKER_BATCH_SIZE} \
  --controller-batch-size ${CONTROLLER_BATCH_SIZE} \
  --worker-batch-size ${WORKER_BATCH_SIZE} \
  --rollout-prompt-length ${ROLLOUT_PROMPT_LENGTH} \
  --ray-nnodes ${RAY_NNODES} \
  --ray-n-gpus-per-node ${RAY_N_GPUS_PER_NODE} \
  --ray-cpus-per-node ${RAY_CPUS_PER_NODE} \
  ${OFFLINE_GRPO_DISTRIBUTED_FLAG} \
  --offline-grpo-nnodes ${OFFLINE_GRPO_NNODES} \
  --offline-grpo-gpus-per-node ${OFFLINE_GRPO_GPUS_PER_NODE} \
  --offline-grpo-master-port ${OFFLINE_GRPO_MASTER_PORT} \
  --vllm-tensor-parallel-size ${VLLM_TENSOR_PARALLEL_SIZE} \
  --vllm-gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION} \
  --vllm-max-num-batched-tokens ${VLLM_MAX_NUM_BATCHED_TOKENS} \
  --vllm-max-num-seqs ${VLLM_MAX_NUM_SEQS} \
  ${VLLM_MAX_MODEL_LEN_FLAG} \
  ${DISABLE_ROLLOUT_LOGGING_FLAG} \
  --rollout-log-mode ${ROLLOUT_LOG_MODE} \
  --rollout-log-detail ${ROLLOUT_LOG_DETAIL} \
  --best-k ${BEST_K} \
  --rollout-task-batch-size ${ROLLOUT_TASK_BATCH_SIZE} \
  --update-every-n-rollout-batches ${UPDATE_EVERY_N_ROLLOUT_BATCHES} \
  --val-rollout-task-batch-size ${VAL_ROLLOUT_TASK_BATCH_SIZE} \
  --rollout-progress-every ${ROLLOUT_PROGRESS_EVERY} \
  --role ${TRAIN_ROLE} \
  ${TRAIN_POLICY_ID_FLAG} \
  ${TRAIN_MIN_REWARD_FLAG} \
  ${TRAIN_MIN_ADVANTAGE_FLAG} \
  --decomposer-reward-aggregation ${DECOMPOSER_REWARD_AGGREGATION} \
  --decomposer-no-correct-selection-scale ${DECOMPOSER_NO_CORRECT_SELECTION_SCALE} \
  --controller-format-retry-penalty ${CONTROLLER_FORMAT_RETRY_PENALTY} \
  --controller-format-fallback-penalty ${CONTROLLER_FORMAT_FALLBACK_PENALTY} \
  --confidence-reward-weight ${CONFIDENCE_REWARD_WEIGHT} \
  --compatibility-reward-weight ${COMPATIBILITY_REWARD_WEIGHT} \
  ${FINAL_ANSWER_CORRECTNESS_REWARD_ONLY_FLAG} \
  ${TRACK_WORKERS_HISTORY_FLAG} \
  ${TRAIN_WORKER_MODEL_FLAG} \
  ${SAVE_REPLAY_COPY_FLAG} \
  ${GRADIENT_CHECKPOINTING_FLAG} \
  ${TRUST_REMOTE_CODE_FLAG} \
  ${ENABLE_WANDB_FLAG} \
  ${WANDB_EXPERIMENT_NAME_FLAG} \
  ${TRAIN_MODEL_FLAGS} \
  --output-dir ${LOCAL_OUTPUT_DIR} \
  --project-name ${WANDB_PROJECT} \
  --val-ratio ${TRAIN_VAL_RATIO} \
  --seed ${SEED} \
  --learning-rate ${LEARNING_RATE} \
  --weight-decay ${WEIGHT_DECAY} \
  --train-batch-size ${TRAIN_BATCH_SIZE} \
  --grad-accum-steps ${GRAD_ACCUM_STEPS} \
  --epochs ${EPOCHS} \
  --max-length ${MAX_LENGTH} \
  --truncation ${TRUNCATION} \
  --clip-range ${CLIP_RANGE} \
  --clip-ratio-c ${CLIP_RATIO_C} \
  --entropy-coeff ${ENTROPY_COEFF} \
  --max-grad-norm ${MAX_GRAD_NORM} \
  --warmup-ratio ${WARMUP_RATIO} \
  --logging-steps ${LOGGING_STEPS} \
  --save-steps ${SAVE_STEPS} \
  --eval-every-steps ${EVAL_EVERY_STEPS} \
  --checkpoint-mode ${CHECKPOINT_MODE} \
  ${PRUNE_STALE_POLICY_MODELS_FLAG} \
  --device ${DEVICE} \
  --torch-dtype ${TORCH_DTYPE}"
fi

start_epoch_s3_sync_watcher

set +e
srun --overlap --nodes=1 --ntasks=1 ${RAY_DRIVER_NODE_FLAG} apptainer exec --nv --writable-tmpfs \
    --mount type=bind,src=$TMPDIR,dst=$TMPDIR \
    --mount type=bind,src=$RAY_LOCAL_TMPDIR,dst=$RAY_LOCAL_TMPDIR \
    --mount type=bind,src=$TMPDIR,dst=/root/tmpdir \
    --mount type=bind,src=$LOCAL_VERL_DIR,dst=/verl \
    "$LOCAL_SIF_IMAGE_PATH" \
    bash -c "$COMMAND" 2>&1 | tee "$RUNTIME_LOG"
RUN_EXIT_CODE=${PIPESTATUS[0]}
set -e

stop_epoch_s3_sync_watcher
stop_ray_cluster

persist_outputs
exit $RUN_EXIT_CODE

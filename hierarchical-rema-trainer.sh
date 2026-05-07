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

TEMPERATURE=${TEMPERATURE:-0.5}
TOP_P=${TOP_P:-0.95}
CONTROLLER_MAX_NEW_TOKENS=${CONTROLLER_MAX_NEW_TOKENS:-768}
WORKER_MAX_NEW_TOKENS=${WORKER_MAX_NEW_TOKENS:-256}
CONTROLLER_BATCH_SIZE=${CONTROLLER_BATCH_SIZE:-8}
WORKER_BATCH_SIZE=${WORKER_BATCH_SIZE:-16}
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-2048}
SLURM_GPUS_RAW=${SLURM_GPUS_ON_NODE:-1}
if [[ "$SLURM_GPUS_RAW" == *:* ]]; then
    DEFAULT_RAY_N_GPUS_PER_NODE=${SLURM_GPUS_RAW##*:}
else
    DEFAULT_RAY_N_GPUS_PER_NODE=${SLURM_GPUS_RAW}
fi
RAY_NNODES=${RAY_NNODES:-1}
RAY_N_GPUS_PER_NODE=${RAY_N_GPUS_PER_NODE:-$DEFAULT_RAY_N_GPUS_PER_NODE}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.5}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-1024}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-}
BEST_K=${BEST_K:-10}
PRINT_MODE=${PRINT_MODE:-summary}
ROLLOUT_TASK_BATCH_SIZE=${ROLLOUT_TASK_BATCH_SIZE:-32}
ROLLOUT_PROGRESS_EVERY=${ROLLOUT_PROGRESS_EVERY:-10}
ROLLOUT_LOG_MODE=${ROLLOUT_LOG_MODE:-best}
ROLLOUT_LOG_DETAIL=${ROLLOUT_LOG_DETAIL:-compact}

TASK_SOURCE=${TASK_SOURCE:-$TASK_SOURCE_ARG}
if [[ "$RUN_KIND" == "train" ]]; then
    TASK_SOURCE=${TASK_SOURCE:-$SOURCE_DIR/data/MATH/train_lv3to5_8k.parquet}
else
    TASK_SOURCE=${TASK_SOURCE:-$SOURCE_DIR/data/overall_math/all_test_data.jsonl}
fi
TASK_FORMAT=${TASK_FORMAT:-auto}
PROMPT_KEY=${PROMPT_KEY:-question}
ANSWER_KEY=${ANSWER_KEY:-answer}
TASK_ID_KEY=${TASK_ID_KEY:-idx}
MAX_TASKS=${MAX_TASKS:-0}
TASKS_PER_EPOCH=${TASKS_PER_EPOCH:-0}
SHUFFLE_TASKS=${SHUFFLE_TASKS:-false}
TASK_SOURCE_STAGE=${TASK_SOURCE_STAGE:-$RUN_ROOT/task_source}
TASK_SOURCE_RUNTIME=${TASK_SOURCE_RUNTIME:-$TASK_SOURCE}
VAL_TASK_SOURCE=${VAL_TASK_SOURCE:-$SOURCE_DIR/data/overall_math/test.parquet}
VAL_TASK_FORMAT=${VAL_TASK_FORMAT:-auto}
VAL_PROMPT_KEY=${VAL_PROMPT_KEY:-question}
VAL_ANSWER_KEY=${VAL_ANSWER_KEY:-answer}
VAL_TASK_ID_KEY=${VAL_TASK_ID_KEY:-idx}
MAX_VAL_TASKS=${MAX_VAL_TASKS:-0}
VAL_TASKS_PER_EPOCH=${VAL_TASKS_PER_EPOCH:-0}
VAL_TASK_SOURCE_STAGE=${VAL_TASK_SOURCE_STAGE:-$RUN_ROOT/val_task_source}
VAL_TASK_SOURCE_RUNTIME=${VAL_TASK_SOURCE_RUNTIME:-$VAL_TASK_SOURCE}
DISABLE_EXTERNAL_VALIDATION=${DISABLE_EXTERNAL_VALIDATION:-false}
VAL_NUM_DECOMPOSITIONS=${VAL_NUM_DECOMPOSITIONS:-1}
VAL_NUM_SELECTIONS=${VAL_NUM_SELECTIONS:-1}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.0}
VAL_TOP_P=${VAL_TOP_P:-1.0}
VAL_CONTROLLER_MAX_NEW_TOKENS=${VAL_CONTROLLER_MAX_NEW_TOKENS:-0}
VAL_WORKER_MAX_NEW_TOKENS=${VAL_WORKER_MAX_NEW_TOKENS:-0}
VAL_ROLLOUT_TASK_BATCH_SIZE=${VAL_ROLLOUT_TASK_BATCH_SIZE:-0}
TRAIN_ROLE=${TRAIN_ROLE:-both}
TRAIN_POLICY_ID=${TRAIN_POLICY_ID:-}
TRAIN_VAL_RATIO=${TRAIN_VAL_RATIO:-0.05}
TRAIN_MIN_REWARD=${TRAIN_MIN_REWARD:-}
TRAIN_MIN_ADVANTAGE=${TRAIN_MIN_ADVANTAGE:-}
SAVE_REPLAY_COPY=${SAVE_REPLAY_COPY:-false}
SEED=${SEED:-42}
NUM_EPOCHS=${NUM_EPOCHS:-1}

LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
EPOCHS=${EPOCHS:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
TRUNCATION=${TRUNCATION:-left}
CLIP_RANGE=${CLIP_RANGE:-0.2}
CLIP_RATIO_C=${CLIP_RATIO_C:-3.0}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-200}
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-10}
CHECKPOINT_MODE=${CHECKPOINT_MODE:-final}
PRUNE_STALE_POLICY_MODELS=${PRUNE_STALE_POLICY_MODELS:-false}
DEVICE=${DEVICE:-cuda}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-false}
ENABLE_WANDB=${ENABLE_WANDB:-false}
WANDB_PROJECT=${WANDB_PROJECT:-multi-grpo-rema}
WANDB_EXPERIMENT_NAME=${WANDB_EXPERIMENT_NAME:-}
DISABLE_ROLLOUT_LOGGING=${DISABLE_ROLLOUT_LOGGING:-false}
EPOCH_S3_SYNC=${EPOCH_S3_SYNC:-false}
EPOCH_S3_SYNC_INTERVAL=${EPOCH_S3_SYNC_INTERVAL:-300}

S3_EPOCHS_PATH=${S3_EPOCHS_PATH:-${S3_OUTPUT_PATH}/epochs}
S3_BEST_SO_FAR_MODELS_PATH=${S3_BEST_SO_FAR_MODELS_PATH:-${S3_OUTPUT_PATH}/best_so_far_models}
S3_BEST_VAL_MODELS_PATH=${S3_BEST_VAL_MODELS_PATH:-${S3_OUTPUT_PATH}/best_val_models}
S3_FINAL_MODELS_PATH=${S3_FINAL_MODELS_PATH:-${S3_OUTPUT_PATH}/final_models}

mkdir -p "$RUN_ROOT" "$LOCAL_OUTPUT_DIR" "$PERSIST_LOCAL_DIR"

cp -r "$SOURCE_DIR/src/verl" "$TMPDIR/verl"
rclone copy "$SIF_IMAGE_PATH" "$TMPDIR/"

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

GRADIENT_CHECKPOINTING_FLAG=""
if [[ "$GRADIENT_CHECKPOINTING" == "1" || "$GRADIENT_CHECKPOINTING" == "true" || "$GRADIENT_CHECKPOINTING" == "True" ]]; then
    GRADIENT_CHECKPOINTING_FLAG="--gradient-checkpointing"
fi

TRUST_REMOTE_CODE_FLAG=""
if [[ "$TRUST_REMOTE_CODE" == "1" || "$TRUST_REMOTE_CODE" == "true" || "$TRUST_REMOTE_CODE" == "True" ]]; then
    TRUST_REMOTE_CODE_FLAG="--trust-remote-code"
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
DISABLE_EXTERNAL_VALIDATION=$DISABLE_EXTERNAL_VALIDATION
VAL_NUM_DECOMPOSITIONS=$VAL_NUM_DECOMPOSITIONS
VAL_NUM_SELECTIONS=$VAL_NUM_SELECTIONS
VAL_TEMPERATURE=$VAL_TEMPERATURE
VAL_TOP_P=$VAL_TOP_P
VAL_CONTROLLER_MAX_NEW_TOKENS=$VAL_CONTROLLER_MAX_NEW_TOKENS
VAL_WORKER_MAX_NEW_TOKENS=$VAL_WORKER_MAX_NEW_TOKENS
VAL_ROLLOUT_TASK_BATCH_SIZE=$VAL_ROLLOUT_TASK_BATCH_SIZE
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
ROLLOUT_PROMPT_LENGTH=$ROLLOUT_PROMPT_LENGTH
RAY_NNODES=$RAY_NNODES
RAY_N_GPUS_PER_NODE=$RAY_N_GPUS_PER_NODE
VLLM_TENSOR_PARALLEL_SIZE=$VLLM_TENSOR_PARALLEL_SIZE
VLLM_GPU_MEMORY_UTILIZATION=$VLLM_GPU_MEMORY_UTILIZATION
VLLM_MAX_NUM_BATCHED_TOKENS=$VLLM_MAX_NUM_BATCHED_TOKENS
VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS
VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN
ENABLE_WANDB=$ENABLE_WANDB
WANDB_PROJECT=$WANDB_PROJECT
WANDB_EXPERIMENT_NAME=$WANDB_EXPERIMENT_NAME
DISABLE_ROLLOUT_LOGGING=$DISABLE_ROLLOUT_LOGGING
ROLLOUT_LOG_MODE=$ROLLOUT_LOG_MODE
ROLLOUT_LOG_DETAIL=$ROLLOUT_LOG_DETAIL
CHECKPOINT_MODE=$CHECKPOINT_MODE
PRUNE_STALE_POLICY_MODELS=$PRUNE_STALE_POLICY_MODELS
EPOCH_S3_SYNC=$EPOCH_S3_SYNC
EPOCH_S3_SYNC_INTERVAL=$EPOCH_S3_SYNC_INTERVAL
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

    if [[ -z "${S3_OUTPUT_PATH:-}" ]]; then
        return 0
    fi

    echo "[hierarchical-rema][s3] uploading epoch folder ${epoch_name} -> ${S3_EPOCHS_PATH}/${epoch_name}"
    rclone copy "$epoch_dir" "${S3_EPOCHS_PATH}/${epoch_name}" || \
        echo "Warning: failed to upload epoch folder ${epoch_name} to ${S3_EPOCHS_PATH}/${epoch_name}" >&2

    for policy_dir in "$epoch_dir"/train/*; do
        [[ -d "$policy_dir" ]] || continue
        local policy_id
        policy_id=$(basename "$policy_dir")
        if [[ -d "$policy_dir/final" ]]; then
            echo "[hierarchical-rema][s3] syncing best-so-far model ${policy_id} -> ${S3_BEST_SO_FAR_MODELS_PATH}/${policy_id}"
            rclone sync "$policy_dir/final" "${S3_BEST_SO_FAR_MODELS_PATH}/${policy_id}" || \
                echo "Warning: failed to sync best-so-far model ${policy_id}" >&2
        fi
        if [[ -d "$policy_dir/best" ]]; then
            echo "[hierarchical-rema][s3] syncing best-val model ${policy_id} -> ${S3_BEST_VAL_MODELS_PATH}/${policy_id}"
            rclone sync "$policy_dir/best" "${S3_BEST_VAL_MODELS_PATH}/${policy_id}" || \
                echo "Warning: failed to sync best-val model ${policy_id}" >&2
        fi
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
                upload_epoch_folder_to_s3 "$epoch_dir"
                touch "$marker_dir/$epoch_name"
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
        mkdir -p "$PERSIST_LOCAL_DIR"
        cp -r "$LOCAL_OUTPUT_DIR"/. "$PERSIST_LOCAL_DIR"/ 2>/dev/null || true

        if [[ -n "${S3_OUTPUT_PATH:-}" ]]; then
            rclone copy "$LOCAL_OUTPUT_DIR" "$S3_OUTPUT_PATH" || \
                echo "Warning: failed to upload outputs to ${S3_OUTPUT_PATH}" >&2
            upload_final_models_to_s3
        fi
    fi

    if [[ -n "${TMPDIR:-}" && -d "${TMPDIR:-}" ]]; then
        rm -rf "$TMPDIR"/*
    fi
}

if [[ "$RUN_KIND" == "train" ]]; then
    stage_task_source
    stage_val_task_source
fi

write_run_metadata

echo "[hierarchical-rema] starting run"
echo "[hierarchical-rema] RUN_KIND=$RUN_KIND"
echo "[hierarchical-rema] LOCAL_OUTPUT_DIR=$LOCAL_OUTPUT_DIR"
echo "[hierarchical-rema] S3_OUTPUT_PATH=$S3_OUTPUT_PATH"
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
  --rollout-prompt-length ${ROLLOUT_PROMPT_LENGTH} \
  --ray-nnodes ${RAY_NNODES} \
  --ray-n-gpus-per-node ${RAY_N_GPUS_PER_NODE} \
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
  ${DISABLE_EXTERNAL_VALIDATION_FLAG} \
  --backend ${BACKEND} \
  --mode ${MODE} \
  --phase ${PHASE} \
  --num-epochs ${NUM_EPOCHS} \
  ${PARAMETER_SHARING_FLAG} \
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
  --val-num-decompositions ${VAL_NUM_DECOMPOSITIONS} \
  --val-num-selections ${VAL_NUM_SELECTIONS} \
  --val-temperature ${VAL_TEMPERATURE} \
  --val-top-p ${VAL_TOP_P} \
  --val-controller-max-new-tokens ${VAL_CONTROLLER_MAX_NEW_TOKENS} \
  --val-worker-max-new-tokens ${VAL_WORKER_MAX_NEW_TOKENS} \
  --controller-batch-size ${CONTROLLER_BATCH_SIZE} \
  --worker-batch-size ${WORKER_BATCH_SIZE} \
  --rollout-prompt-length ${ROLLOUT_PROMPT_LENGTH} \
  --ray-nnodes ${RAY_NNODES} \
  --ray-n-gpus-per-node ${RAY_N_GPUS_PER_NODE} \
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
  --val-rollout-task-batch-size ${VAL_ROLLOUT_TASK_BATCH_SIZE} \
  --rollout-progress-every ${ROLLOUT_PROGRESS_EVERY} \
  --role ${TRAIN_ROLE} \
  ${TRAIN_POLICY_ID_FLAG} \
  ${TRAIN_MIN_REWARD_FLAG} \
  ${TRAIN_MIN_ADVANTAGE_FLAG} \
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
srun apptainer exec --nv --writable-tmpfs \
    --mount type=bind,src=$TMPDIR,dst=$TMPDIR \
    --mount type=bind,src=$TMPDIR,dst=/root/tmpdir \
    --mount type=bind,src=$TMPDIR/verl,dst=/verl \
    "$TMPDIR/verl-rema-v3.sif" \
    bash -c "$COMMAND" 2>&1 | tee "$RUNTIME_LOG"
RUN_EXIT_CODE=${PIPESTATUS[0]}
set -e

stop_epoch_s3_sync_watcher

persist_outputs
exit $RUN_EXIT_CODE

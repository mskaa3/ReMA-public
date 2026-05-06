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

TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
CONTROLLER_MAX_NEW_TOKENS=${CONTROLLER_MAX_NEW_TOKENS:-768}
WORKER_MAX_NEW_TOKENS=${WORKER_MAX_NEW_TOKENS:-256}
BEST_K=${BEST_K:-10}
PRINT_MODE=${PRINT_MODE:-summary}

TASK_SOURCE=${TASK_SOURCE:-$TASK_SOURCE_ARG}
TASK_SOURCE=${TASK_SOURCE:-$SOURCE_DIR/data/overall_math/all_test_data.jsonl}
TASK_FORMAT=${TASK_FORMAT:-auto}
PROMPT_KEY=${PROMPT_KEY:-question}
ANSWER_KEY=${ANSWER_KEY:-answer}
TASK_ID_KEY=${TASK_ID_KEY:-idx}
MAX_TASKS=${MAX_TASKS:-0}
TASKS_PER_EPOCH=${TASKS_PER_EPOCH:-0}
SHUFFLE_TASKS=${SHUFFLE_TASKS:-false}
TASK_SOURCE_STAGE=${TASK_SOURCE_STAGE:-$RUN_ROOT/task_source}
TASK_SOURCE_RUNTIME=${TASK_SOURCE_RUNTIME:-$TASK_SOURCE}
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
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-0}
DEVICE=${DEVICE:-cuda}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-false}
ENABLE_WANDB=${ENABLE_WANDB:-false}
WANDB_PROJECT=${WANDB_PROJECT:-hierarchical-rema}
WANDB_EXPERIMENT_NAME=${WANDB_EXPERIMENT_NAME:-}
DISABLE_ROLLOUT_LOGGING=${DISABLE_ROLLOUT_LOGGING:-false}

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
NUM_EPOCHS=$NUM_EPOCHS
MODEL_PATH=$MODEL_PATH
DECOMPOSER_MODEL_PATH=$DECOMPOSER_MODEL_PATH
SELECTOR_MODEL_PATH=$SELECTOR_MODEL_PATH
BACKEND=$BACKEND
TASK=$TASK
MODE=$MODE
PHASE=$PHASE
PRINT_MODE=$PRINT_MODE
ENABLE_WANDB=$ENABLE_WANDB
WANDB_PROJECT=$WANDB_PROJECT
WANDB_EXPERIMENT_NAME=$WANDB_EXPERIMENT_NAME
DISABLE_ROLLOUT_LOGGING=$DISABLE_ROLLOUT_LOGGING
EOF
}

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

if [[ "$RUN_KIND" == "train" ]]; then
    stage_task_source
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
  --output-dir ${LOCAL_OUTPUT_DIR} \
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
  ${DISABLE_ROLLOUT_LOGGING_FLAG} \
  --best-k ${BEST_K} \
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
  --device ${DEVICE} \
  --torch-dtype ${TORCH_DTYPE}"
fi

set +e
srun apptainer exec --nv --writable-tmpfs \
    --mount type=bind,src=$TMPDIR,dst=$TMPDIR \
    --mount type=bind,src=$TMPDIR,dst=/root/tmpdir \
    --mount type=bind,src=$TMPDIR/verl,dst=/verl \
    "$TMPDIR/verl-rema-v3.sif" \
    bash -c "$COMMAND" 2>&1 | tee "$RUNTIME_LOG"
RUN_EXIT_CODE=${PIPESTATUS[0]}
set -e

persist_outputs
exit $RUN_EXIT_CODE

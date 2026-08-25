#!/bin/bash
#SBATCH --job-name=agent12-curriculum
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=200gb
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:4,storage:local:200G
#SBATCH --verbose

set -euo pipefail

source ./env.sh

export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export POOL_GPUS_PER_NODE=${POOL_GPUS_PER_NODE:-$((GPUS_PER_NODE / 2))}
export RAY_PORT=${RAY_PORT:-6379}

export SIF_NAME=${SIF_NAME:-verl-rema-v3.sif}
export SIF_REMOTE=${SIF_REMOTE:-s3v2:s3min-tomasznaskret-1712063354/user/dmotyka/sif_images/${SIF_NAME}}

export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
export DECOMPOSER_MODEL_PATH=${DECOMPOSER_MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
export WORKER_MODEL_PATH=${WORKER_MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
export TEACHER_ROLLOUT_N=${TEACHER_ROLLOUT_N:-16}
export TEACHER_SAMPLES_PER_CALL=${TEACHER_SAMPLES_PER_CALL:-${TEACHER_ROLLOUT_N}}
export TEACHER_CHECKPOINT_EVERY_BATCHES=${TEACHER_CHECKPOINT_EVERY_BATCHES:-5}
export TEACHER_GENERATION_MAX_ATTEMPTS=${TEACHER_GENERATION_MAX_ATTEMPTS:-3}
export ROLLOUT_N=${ROLLOUT_N:-16}
export TEACHER_ATTEMPT_MAX_CHARS=${TEACHER_ATTEMPT_MAX_CHARS:-8000}
export BASE_QUESTION_BATCH_SIZE=${BASE_QUESTION_BATCH_SIZE:-3}
export OPTIMIZER_PROMPT_BATCH_SIZE=${OPTIMIZER_PROMPT_BATCH_SIZE:-$((BASE_QUESTION_BATCH_SIZE * TEACHER_ROLLOUT_N))}
export TRAIN_DECOMPOSER=${TRAIN_DECOMPOSER:-true}
export TRAIN_AGENT_ROLES=${TRAIN_AGENT_ROLES:-'[decomposer,worker_stage_1,worker_stage_2,worker_stage_3,worker_stage_4,worker_stage_5]'}

export WORKER_BOOTSTRAP_STEPS=${WORKER_BOOTSTRAP_STEPS:-200}
export DECOMPOSER_TRANSFER_STEPS=${DECOMPOSER_TRANSFER_STEPS:-200}
export WORKER_QUESTION_FADE_STEPS=${WORKER_QUESTION_FADE_STEPS:-200}
export JOINT_STEPS=${JOINT_STEPS:-400}
export TOTAL_STEPS=${TOTAL_STEPS:-$((WORKER_BOOTSTRAP_STEPS + DECOMPOSER_TRANSFER_STEPS + JOINT_STEPS))}
export TEST_FREQ=${TEST_FREQ:-10}
export SAVE_FREQ=${SAVE_FREQ:-50}

export AGENT12_S3_REMOTE=${AGENT12_S3_REMOTE:-s3v2:s3min-tomasznaskret-1712063354/user/ajanz/agent12}
export AGENT12_RUN_NAME=${AGENT12_RUN_NAME:-agent12-curriculum-${SLURM_JOB_ID}}
export AGENT12_WANDB_RUN_ID=${AGENT12_WANDB_RUN_ID:-${AGENT12_RUN_NAME}}
export REMOTE_RUN=${AGENT12_S3_REMOTE%/}/${AGENT12_RUN_NAME}
export GENERATE_TEACHER_DATA=${GENERATE_TEACHER_DATA:-1}
export ONLINE_TEACHER_GENERATION=${ONLINE_TEACHER_GENERATION:-false}
export TEACHER_SHARD_QUESTIONS=${TEACHER_SHARD_QUESTIONS:-512}
export TEACHER_TRAIN_REMOTE=${TEACHER_TRAIN_REMOTE:-${AGENT12_S3_REMOTE%/}/teacher_data/math-qwen25-1.5b-n${TEACHER_ROLLOUT_N}-all-attempt-groups.parquet}
export TEACHER_CANDIDATES_REMOTE=${TEACHER_CANDIDATES_REMOTE:-${TEACHER_TRAIN_REMOTE%.parquet}.generation-progress.parquet}
export TEACHER_SHARD_REMOTE_ROOT=${TEACHER_SHARD_REMOTE_ROOT:-${TEACHER_TRAIN_REMOTE%.parquet}.shards/q${TEACHER_SHARD_QUESTIONS}}

export JOB_TMP=${JOB_TMP:-/mnt/lscratch/slurm/${SLURM_JOB_ID}/agent12}
export RAY_NODE_TMP=${RAY_NODE_TMP:-${JOB_TMP}/ray}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${JOB_TMP}/checkpoints/agent12}
export TEACHER_TRAIN_FILE=${TEACHER_TRAIN_FILE:-${JOB_TMP}/agent12_data/train.parquet}
export TEACHER_CANDIDATES_FILE=${TEACHER_CANDIDATES_FILE:-${JOB_TMP}/agent12_data/teacher_candidates.parquet}

if (( GPUS_PER_NODE % 2 != 0 )); then
    echo "GPUS_PER_NODE must be even because Agent 1 and Agent 2 use separate pools" >&2
    exit 1
fi
if (( POOL_GPUS_PER_NODE * 2 > GPUS_PER_NODE )); then
    echo "Two model pools request more GPUs than the SLURM allocation" >&2
    exit 1
fi
if (( OPTIMIZER_PROMPT_BATCH_SIZE != BASE_QUESTION_BATCH_SIZE * TEACHER_ROLLOUT_N )); then
    echo "OPTIMIZER_PROMPT_BATCH_SIZE must equal BASE_QUESTION_BATCH_SIZE * TEACHER_ROLLOUT_N for full attempt expansion" >&2
    exit 1
fi
if (( TEACHER_ROLLOUT_N % TEACHER_SAMPLES_PER_CALL != 0 )); then
    echo "TEACHER_ROLLOUT_N must be divisible by TEACHER_SAMPLES_PER_CALL" >&2
    exit 1
fi
if (( TEACHER_SHARD_QUESTIONS <= 0 )); then
    echo "TEACHER_SHARD_QUESTIONS must be positive" >&2
    exit 1
fi
echo "Agent 1/2 batch: ${BASE_QUESTION_BATCH_SIZE} questions x ${TEACHER_ROLLOUT_N} attempts x ${ROLLOUT_N} rollouts = $((BASE_QUESTION_BATCH_SIZE * TEACHER_ROLLOUT_N * ROLLOUT_N)) trajectories"
echo "Train decomposer: ${TRAIN_DECOMPOSER}; train roles: ${TRAIN_AGENT_ROLES}"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE=${NODES[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
if [[ "$HEAD_NODE_IP" == *" "* ]]; then
    read -ra ADDR <<< "$HEAD_NODE_IP"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        HEAD_NODE_IP=${ADDR[1]}
    else
        HEAD_NODE_IP=${ADDR[0]}
    fi
fi
IP_HEAD=${HEAD_NODE_IP}:${RAY_PORT}

echo "Preparing Agent 1/2 workspace on all nodes"
srun --label --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" bash -lc '
    set -euo pipefail
    mkdir -p "$JOB_TMP" "$RAY_NODE_TMP"
    rm -rf "$JOB_TMP/MATH" "$JOB_TMP/overall_math" "$JOB_TMP/agent0_data" \
        "$JOB_TMP/agent12_data" "$JOB_TMP/verl" "$JOB_TMP/prompt" \
        "$JOB_TMP/scripts" "$JOB_TMP/config"
    mkdir -p "$JOB_TMP/agent12_data"
    cp -r "$SLURM_SUBMIT_DIR/data/MATH" "$JOB_TMP/MATH"
    cp -r "$SLURM_SUBMIT_DIR/data/overall_math" "$JOB_TMP/overall_math"
    cp -r "$SLURM_SUBMIT_DIR/src/verl" "$JOB_TMP/verl"
    cp -r "$SLURM_SUBMIT_DIR/prompt" "$JOB_TMP/prompt"
    cp -r "$SLURM_SUBMIT_DIR/scripts" "$JOB_TMP/scripts"
    cp -r "$SLURM_SUBMIT_DIR/config" "$JOB_TMP/config"
    rclone copy "$SIF_REMOTE" "$JOB_TMP/" --stats=30s --stats-one-line
'

COMMON_MOUNTS=(
    --mount "type=bind,src=${JOB_TMP},dst=${JOB_TMP}"
    --mount "type=bind,src=${JOB_TMP},dst=/root/tmpdir"
    --mount "type=bind,src=${JOB_TMP}/verl,dst=/verl"
    --mount "type=bind,src=${JOB_TMP}/verl,dst=/root/ReMA-public/src/verl"
    --mount "type=bind,src=${JOB_TMP}/prompt,dst=/root/ReMA-public/prompt"
    --mount "type=bind,src=${JOB_TMP}/scripts,dst=/root/ReMA-public/scripts"
    --mount "type=bind,src=${JOB_TMP}/config,dst=/root/ReMA-public/config"
    --mount "type=bind,src=${RAY_NODE_TMP},dst=${RAY_NODE_TMP}"
)

RAY_JOB_PIDS=()

start_ray() {
    echo "Starting Ray cluster at ${IP_HEAD}"
    srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" bash -lc '
        rm -rf "$RAY_NODE_TMP"
        mkdir -p "$RAY_NODE_TMP"
    '

    srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
        ray start --head --node-ip-address="$HEAD_NODE_IP" --port="$RAY_PORT" \
            --temp-dir="$RAY_NODE_TMP" --num-cpus="${SLURM_CPUS_PER_TASK}" \
            --num-gpus="$GPUS_PER_NODE" --block &
    RAY_JOB_PIDS+=("$!")
    sleep 15

    for ((i = 1; i < SLURM_NNODES; i++)); do
        NODE=${NODES[$i]}
        srun --nodes=1 --ntasks=1 -w "$NODE" \
            apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
            ray start --address "$IP_HEAD" --temp-dir="$RAY_NODE_TMP" \
                --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus="$GPUS_PER_NODE" --block &
        RAY_JOB_PIDS+=("$!")
        sleep 5
    done

    local expected_gpus=$((SLURM_NNODES * GPUS_PER_NODE))
    local elapsed=0
    local timeout=${RAY_WAIT_TIMEOUT:-300}
    local interval=${RAY_WAIT_INTERVAL:-10}
    while true; do
        available_gpus=$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
            apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
            python3 -c "import ray; ray.init(address='${IP_HEAD}', ignore_reinit_error=True, logging_level=40); print(int(ray.cluster_resources().get('GPU', 0)))" \
            2>/dev/null || echo 0)
        available_gpus=$(echo "$available_gpus" | tail -n 1 | tr -d '[:space:]')
        if [[ "$available_gpus" =~ ^[0-9]+$ ]] && (( available_gpus >= expected_gpus )); then
            echo "Ray cluster ready: ${available_gpus}/${expected_gpus} GPUs"
            break
        fi
        if (( elapsed >= timeout )); then
            echo "Timed out waiting for Ray: ${available_gpus:-0}/${expected_gpus} GPUs" >&2
            exit 1
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
}

stop_ray() {
    echo "Stopping Ray cluster"
    srun --overlap --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" \
        apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
        ray stop --force || true
    for pid in "${RAY_JOB_PIDS[@]}"; do
        wait "$pid" || true
    done
    RAY_JOB_PIDS=()
}

download_remote_file_on_head() {
    local remote_path=$1
    local local_path=$2
    local description=$3
    local result

    if [[ -z "$remote_path" ]]; then
        echo "A remote path is required to download ${description}" >&2
        return 2
    fi

    export DOWNLOAD_REMOTE_PATH="$remote_path"
    export DOWNLOAD_LOCAL_PATH="$local_path"
    if ! result=$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc '
            set -euo pipefail
            mkdir -p "$(dirname "$DOWNLOAD_LOCAL_PATH")"
            temporary_path="${DOWNLOAD_LOCAL_PATH}.partial.$$"
            rm -f "$temporary_path"
            if rclone cat "$DOWNLOAD_REMOTE_PATH" > "$temporary_path" 2>/dev/null \
                    && [[ -s "$temporary_path" ]]; then
                mv -f "$temporary_path" "$DOWNLOAD_LOCAL_PATH"
                size_bytes=$(stat -c %s "$DOWNLOAD_LOCAL_PATH")
                echo "__REMOTE_FILE_PRESENT__:${size_bytes}"
            else
                rm -f "$temporary_path"
                echo "__REMOTE_FILE_MISSING__"
            fi
        '); then
        echo "Failed to check ${description} on ${HEAD_NODE}" >&2
        return 2
    fi

    if [[ "$result" == *"__REMOTE_FILE_PRESENT__:"* ]]; then
        size_bytes=${result##*__REMOTE_FILE_PRESENT__:}
        echo "Downloaded ${description} to ${HEAD_NODE} (${size_bytes} bytes)"
        return 0
    fi
    return 1
}

run_agent12_training() {
    local train_file=$1
    local session_stop_step=$2
    local val_before_train=$3
    local restore_dataloader=$4
    local train_command

    echo "Training Agent 1/2 on ${train_file} through step ${session_stop_step}"
    train_command="unset ROCR_VISIBLE_DEVICES; \
export TMPDIR=${RAY_NODE_TMP}; \
export HF_HOME=/root/tmpdir/hf_home; \
export PYTHONPATH=/root/ReMA-public:/root/ReMA-public/src/verl:/verl:\$PYTHONPATH; \
export RAY_ADDRESS=${IP_HEAD}; \
export WANDB_RUN_ID=${AGENT12_WANDB_RUN_ID}; \
export WANDB_RESUME=allow; \
python3 -m verl.rema_separated_trainer.main_ppo \
  --config-path=/root/ReMA-public/config \
  --config-name=rema-rl.yaml \
  data.train_files=${train_file} \
  data.train_batch_size=${BASE_QUESTION_BATCH_SIZE} \
  actor_rollout_ref.model.path=${DECOMPOSER_MODEL_PATH} \
  algorithm.switch_agent.model_paths=[${DECOMPOSER_MODEL_PATH},${WORKER_MODEL_PATH}] \
  algorithm.hierarchy.agent12_curriculum.enable=True \
  algorithm.hierarchy.agent12_curriculum.train_decomposer=${TRAIN_DECOMPOSER} \
  algorithm.hierarchy.agent12_curriculum.expand_all_teacher_attempts=True \
  algorithm.hierarchy.agent12_curriculum.teacher_attempts_per_question=${TEACHER_ROLLOUT_N} \
  algorithm.hierarchy.agent12_curriculum.optimizer_prompt_batch_size=${OPTIMIZER_PROMPT_BATCH_SIZE} \
  algorithm.hierarchy.agent12_curriculum.teacher_attempt_max_chars=${TEACHER_ATTEMPT_MAX_CHARS} \
  algorithm.hierarchy.agent12_curriculum.worker_bootstrap_steps=${WORKER_BOOTSTRAP_STEPS} \
  algorithm.hierarchy.agent12_curriculum.decomposer_transfer_steps=${DECOMPOSER_TRANSFER_STEPS} \
  algorithm.hierarchy.agent12_curriculum.worker_question_fade_steps=${WORKER_QUESTION_FADE_STEPS} \
  algorithm.hierarchy.agent12_curriculum.worker_question_final_probability=0.0 \
  algorithm.hierarchy.agent12_curriculum.worker_question_eval_probability=0.0 \
  algorithm.hierarchy.train_agent_roles=${TRAIN_AGENT_ROLES} \
  algorithm.hierarchy.reward_composer.enable=False \
  algorithm.hierarchy.reward_composer.online.enable=False \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  trainer.nnodes=${SLURM_NNODES} \
  trainer.n_gpus_per_node=${POOL_GPUS_PER_NODE} \
  trainer.total_epochs=100 \
  trainer.total_training_steps=${TOTAL_STEPS} \
  trainer.session_stop_step=${session_stop_step} \
  trainer.restore_dataloader_on_resume=${restore_dataloader} \
  trainer.val_before_train=${val_before_train} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.experiment_name=${AGENT12_RUN_NAME} \
  trainer.wandb_run_id=${AGENT12_WANDB_RUN_ID} \
  trainer.wandb_resume=allow \
  trainer.default_local_dir=${CHECKPOINT_ROOT}"

    PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
        bash -c "$train_command"
}

if [[ "$ONLINE_TEACHER_GENERATION" == "1" || "$ONLINE_TEACHER_GENERATION" == "true" ]]; then
    echo "Chunked online teacher generation enabled: ${TEACHER_SHARD_QUESTIONS} questions per shard"
    echo "Preparing frozen Agent 0 prompts"
    srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        apptainer exec --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
        bash -lc 'export PYTHONPATH=/root/ReMA-public:/root/ReMA-public/src/verl:/verl:$PYTHONPATH; \
            python3 /root/ReMA-public/scripts/prepare_agent0_data.py \
            --train-input /root/tmpdir/MATH/train_lv3to5_8k.parquet \
            --val-input /root/tmpdir/overall_math/test.parquet \
            --output-dir /root/tmpdir/agent0_data'

    TOTAL_TEACHER_QUESTIONS=$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        apptainer exec --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
        python3 -c "import pandas as pd; print(len(pd.read_parquet('/root/tmpdir/agent0_data/train.parquet')))" \
        | tail -n 1 | tr -d '[:space:]')
    if [[ ! "$TOTAL_TEACHER_QUESTIONS" =~ ^[0-9]+$ ]] || (( TOTAL_TEACHER_QUESTIONS == 0 )); then
        echo "Could not determine the number of Agent 0 training questions" >&2
        exit 1
    fi
    export TOTAL_TEACHER_QUESTIONS
    TEACHER_SHARD_COUNT=$(((TOTAL_TEACHER_QUESTIONS + TEACHER_SHARD_QUESTIONS - 1) / TEACHER_SHARD_QUESTIONS))
    export TEACHER_SHARD_COUNT
    echo "Online pipeline: ${TOTAL_TEACHER_QUESTIONS} questions in ${TEACHER_SHARD_COUNT} shards"

    srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        mkdir -p "$JOB_TMP/agent12_data/shards"

    for ((shard_index = 0; shard_index < TEACHER_SHARD_COUNT; shard_index++)); do
        shard_id=$(printf 'shard-%05d' "$shard_index")
        shard_start=$((shard_index * TEACHER_SHARD_QUESTIONS))
        shard_end=$((shard_start + TEACHER_SHARD_QUESTIONS))
        if (( shard_end > TOTAL_TEACHER_QUESTIONS )); then
            shard_end=$TOTAL_TEACHER_QUESTIONS
        fi
        shard_size=$((shard_end - shard_start))

        shard_candidates_file=${JOB_TMP}/agent12_data/shards/${shard_id}.candidates.parquet
        shard_train_file=${JOB_TMP}/agent12_data/shards/${shard_id}.train.parquet
        shard_candidates_container=/root/tmpdir/agent12_data/shards/${shard_id}.candidates.parquet
        shard_train_container=/root/tmpdir/agent12_data/shards/${shard_id}.train.parquet
        shard_candidates_remote=${TEACHER_SHARD_REMOTE_ROOT}/candidates/${shard_id}.parquet
        shard_train_remote=${TEACHER_SHARD_REMOTE_ROOT}/train/${shard_id}.parquet
        export TEACHER_CANDIDATES_FILE=$shard_candidates_file
        export TEACHER_CANDIDATES_REMOTE=$shard_candidates_remote

        echo "Teacher shard $((shard_index + 1))/${TEACHER_SHARD_COUNT}: questions [${shard_start}, ${shard_end})"
        if download_remote_file_on_head \
                "$shard_train_remote" "$shard_train_file" "$shard_id"; then
            echo "Reusing complete teacher shard ${shard_train_remote}"
        else
            download_status=$?
            if (( download_status == 2 )); then
                exit 1
            fi
            echo "No complete teacher shard found at ${shard_train_remote}"
            if [[ "$GENERATE_TEACHER_DATA" != "1" && "$GENERATE_TEACHER_DATA" != "true" ]]; then
                echo "Missing ${shard_train_remote} and teacher generation is disabled" >&2
                exit 1
            fi

            if download_remote_file_on_head \
                    "$shard_candidates_remote" "$shard_candidates_file" \
                    "${shard_id} candidate progress"; then
                echo "Resuming candidate shard from ${shard_candidates_remote}"
            else
                download_status=$?
                if (( download_status == 2 )); then
                    exit 1
                fi
                echo "No candidate progress for ${shard_id}; generating it from scratch"
            fi

            start_ray
            teacher_command="unset ROCR_VISIBLE_DEVICES; \
export TMPDIR=${RAY_NODE_TMP}; \
export HF_HOME=/root/tmpdir/hf_home; \
export PYTHONPATH=/root/ReMA-public:/root/ReMA-public/src/verl:/verl:\$PYTHONPATH; \
export RAY_ADDRESS=${IP_HEAD}; \
python3 -m verl.trainer.main_generation \
  --config-path=/root/ReMA-public/config \
  --config-name=agent0-teacher-generation.yaml \
  trainer.nnodes=${SLURM_NNODES} \
  trainer.n_gpus_per_node=${GPUS_PER_NODE} \
  model.path=${TEACHER_MODEL_PATH} \
  data.output_path=${shard_candidates_container} \
  data.start_index=${shard_start} \
  data.max_examples=${shard_size} \
  data.n_samples=${TEACHER_ROLLOUT_N} \
  data.samples_per_call=${TEACHER_SAMPLES_PER_CALL} \
  data.checkpoint_every_batches=${TEACHER_CHECKPOINT_EVERY_BATCHES} \
  rollout.n=${TEACHER_SAMPLES_PER_CALL}"

            generation_status=1
            for ((generation_attempt = 1; generation_attempt <= TEACHER_GENERATION_MAX_ATTEMPTS; generation_attempt++)); do
                echo "${shard_id} generation attempt ${generation_attempt}/${TEACHER_GENERATION_MAX_ATTEMPTS}"
                if PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
                    apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
                    bash -c "$teacher_command"; then
                    generation_status=0
                else
                    generation_status=$?
                fi

                if srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
                    bash -lc 'test -s "$TEACHER_CANDIDATES_FILE" && rclone copyto "$TEACHER_CANDIDATES_FILE" "$TEACHER_CANDIDATES_REMOTE" --s3-no-check-bucket'; then
                    echo "Candidate progress uploaded to ${shard_candidates_remote}"
                else
                    echo "No completed batch is available for ${shard_id} progress upload"
                fi

                if (( generation_status == 0 )); then
                    break
                fi
                if (( generation_attempt < TEACHER_GENERATION_MAX_ATTEMPTS )); then
                    echo "Restarting Ray before resuming ${shard_id}"
                    stop_ray
                    start_ray
                fi
            done
            if (( generation_status != 0 )); then
                stop_ray
                echo "Teacher generation failed for ${shard_id}; partial data is at ${shard_candidates_remote}" >&2
                exit "$generation_status"
            fi

            srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
                apptainer exec --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
                python3 /root/ReMA-public/scripts/prepare_agent12_teacher_data.py \
                    --input "$shard_candidates_container" \
                    --output "$shard_train_container"
            srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
                rclone copyto "$shard_train_file" "$shard_train_remote" --s3-no-check-bucket
            echo "Complete teacher shard uploaded to ${shard_train_remote}"
            stop_ray
        fi

        session_stop_step=$(((shard_end * TOTAL_STEPS + TOTAL_TEACHER_QUESTIONS - 1) / TOTAL_TEACHER_QUESTIONS))
        if [[ -f "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt" ]]; then
            completed_training_step=$(cat "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt")
        else
            completed_training_step=0
        fi
        if (( session_stop_step > completed_training_step )); then
            if (( completed_training_step == 0 )); then
                val_before_train=True
            else
                val_before_train=False
            fi
            start_ray
            run_agent12_training "$shard_train_file" "$session_stop_step" "$val_before_train" False
            stop_ray
        else
            echo "Training already reached step ${completed_training_step}; skipping shard target ${session_stop_step}"
        fi
    done

    echo "Merging all online teacher shards"
    srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        apptainer exec --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
        python3 /root/ReMA-public/scripts/merge_agent12_teacher_shards.py \
            --input-dir /root/tmpdir/agent12_data/shards \
            --output /root/tmpdir/agent12_data/train.parquet \
            --expected-rows "$TOTAL_TEACHER_QUESTIONS"
    srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        rclone copyto "$TEACHER_TRAIN_FILE" "$TEACHER_TRAIN_REMOTE" --s3-no-check-bucket
    echo "Merged teacher data uploaded to ${TEACHER_TRAIN_REMOTE}"
else
if [[ "$GENERATE_TEACHER_DATA" == "1" || "$GENERATE_TEACHER_DATA" == "true" ]]; then
    start_ray
    echo "Preparing frozen Agent 0 prompts"
    srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        apptainer exec --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
        bash -lc 'export PYTHONPATH=/root/ReMA-public:/root/ReMA-public/src/verl:/verl:$PYTHONPATH; \
            python3 /root/ReMA-public/scripts/prepare_agent0_data.py \
            --train-input /root/tmpdir/MATH/train_lv3to5_8k.parquet \
            --val-input /root/tmpdir/overall_math/test.parquet \
            --output-dir /root/tmpdir/agent0_data'

    echo "Generating ${TEACHER_ROLLOUT_N} frozen Agent 0 candidates per question"
    if download_remote_file_on_head \
            "$TEACHER_CANDIDATES_REMOTE" "$TEACHER_CANDIDATES_FILE" \
            "teacher generation progress"; then
        echo "Downloaded resumable teacher progress from ${TEACHER_CANDIDATES_REMOTE}"
    else
        download_status=$?
        if (( download_status == 2 )); then
            exit 1
        fi
        echo "No resumable teacher progress found; starting from the first example"
    fi
    TEACHER_COMMAND="unset ROCR_VISIBLE_DEVICES; \
export TMPDIR=${RAY_NODE_TMP}; \
export HF_HOME=/root/tmpdir/hf_home; \
export PYTHONPATH=/root/ReMA-public:/root/ReMA-public/src/verl:/verl:\$PYTHONPATH; \
export RAY_ADDRESS=${IP_HEAD}; \
python3 -m verl.trainer.main_generation \
  --config-path=/root/ReMA-public/config \
  --config-name=agent0-teacher-generation.yaml \
  trainer.nnodes=${SLURM_NNODES} \
  trainer.n_gpus_per_node=${GPUS_PER_NODE} \
  model.path=${TEACHER_MODEL_PATH} \
  data.n_samples=${TEACHER_ROLLOUT_N} \
  data.samples_per_call=${TEACHER_SAMPLES_PER_CALL} \
  data.checkpoint_every_batches=${TEACHER_CHECKPOINT_EVERY_BATCHES} \
  rollout.n=${TEACHER_SAMPLES_PER_CALL}"

    generation_status=1
    for ((generation_attempt = 1; generation_attempt <= TEACHER_GENERATION_MAX_ATTEMPTS; generation_attempt++)); do
        echo "Teacher generation attempt ${generation_attempt}/${TEACHER_GENERATION_MAX_ATTEMPTS}"
        if PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
            apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
            bash -c "$TEACHER_COMMAND"; then
            generation_status=0
        else
            generation_status=$?
        fi

        if srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
            bash -lc 'test -s "$TEACHER_CANDIDATES_FILE" && rclone copyto "$TEACHER_CANDIDATES_FILE" "$TEACHER_CANDIDATES_REMOTE" --s3-no-check-bucket'; then
            echo "Teacher generation progress uploaded to ${TEACHER_CANDIDATES_REMOTE}"
        else
            echo "No completed teacher batch is available for progress upload"
        fi

        if (( generation_status == 0 )); then
            break
        fi
        if (( generation_attempt < TEACHER_GENERATION_MAX_ATTEMPTS )); then
            echo "Teacher generation failed with status ${generation_status}; restarting Ray and resuming"
            stop_ray
            start_ray
        fi
    done
    if (( generation_status != 0 )); then
        stop_ray
        echo "Teacher generation failed after ${TEACHER_GENERATION_MAX_ATTEMPTS} attempts." >&2
        echo "Partial data, when available, is stored at ${TEACHER_CANDIDATES_REMOTE}" >&2
        exit "$generation_status"
    fi

    echo "Collecting all teacher attempts while retaining every task"
    srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        apptainer exec --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
        bash -lc 'export PYTHONPATH=/root/ReMA-public:/root/ReMA-public/src/verl:/verl:$PYTHONPATH; \
            python3 /root/ReMA-public/scripts/prepare_agent12_teacher_data.py \
            --input /root/tmpdir/agent12_data/teacher_candidates.parquet \
            --output /root/tmpdir/agent12_data/train.parquet'

    if [[ -n "$TEACHER_TRAIN_REMOTE" ]]; then
        srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
            rclone copyto "$TEACHER_TRAIN_FILE" "$TEACHER_TRAIN_REMOTE" --s3-no-check-bucket
        echo "Teacher data uploaded to ${TEACHER_TRAIN_REMOTE}"
    fi
    stop_ray
else
    if [[ -z "$TEACHER_TRAIN_REMOTE" ]]; then
        echo "TEACHER_TRAIN_REMOTE is required when GENERATE_TEACHER_DATA=0" >&2
        exit 1
    fi
    echo "Reusing teacher data from ${TEACHER_TRAIN_REMOTE}"
    if ! download_remote_file_on_head \
            "$TEACHER_TRAIN_REMOTE" "$TEACHER_TRAIN_FILE" \
            "the full teacher dataset"; then
        echo "Teacher data is missing or empty at ${TEACHER_TRAIN_REMOTE}" >&2
        exit 1
    fi
fi

srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    test -s "$TEACHER_TRAIN_FILE"

start_ray
run_agent12_training "$TEACHER_TRAIN_FILE" "$TOTAL_STEPS" True True
stop_ray
fi

LATEST_STEP=$(srun --overlap --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" \
    bash -lc 'if [[ -f "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt" ]]; then cat "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"; fi' \
    | grep -E '^[0-9]+$' | sort -n | tail -n 1)
if [[ -z "$LATEST_STEP" ]]; then
    echo "Could not locate the final Agent 1/2 checkpoint" >&2
    exit 1
fi
export LATEST_STEP

echo "Collecting distributed Agent 1/2 checkpoints at step ${LATEST_STEP}"
srun --overlap --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" bash -lc '
    set -euo pipefail
    node_name=${SLURMD_NODENAME:-$(hostname)}
    for spec in "decomposer:decomposer" "workers:worker_stage_5"; do
        export_name=${spec%%:*}
        role=${spec##*:}
        actor_dir="$CHECKPOINT_ROOT/global_step_${LATEST_STEP}/${role}/actor"
        if [[ -d "$actor_dir" ]]; then
            rclone copy "$actor_dir" "$REMOTE_RUN/raw/${export_name}/${node_name}/actor" \
                --s3-no-check-bucket \
                --include "/model_world_size_*_rank_*.pt" \
                --include "/huggingface/**" --exclude "*" \
                --stats=30s --stats-one-line
        fi
    done
'

export MERGE_ROOT=${JOB_TMP}/agent12_merge
for export_name in decomposer workers; do
    export EXPORT_NAME=$export_name
    echo "Merging ${export_name} checkpoint"
    srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc '
        set -euo pipefail
        target="$MERGE_ROOT/$EXPORT_NAME"
        rm -rf "$target"
        mkdir -p "$target/raw" "$target/actor/huggingface"
        rclone copy "$REMOTE_RUN/raw/$EXPORT_NAME" "$target/raw" --stats=30s --stats-one-line
        find "$target/raw" -type f -name "model_world_size_*_rank_*.pt" -exec cp {} "$target/actor/" \;
        hf_source=$(find "$target/raw" -type d -name huggingface | head -n 1)
        test -n "$hf_source"
        cp -r "$hf_source/." "$target/actor/huggingface/"
        first_shard=$(find "$target/actor" -maxdepth 1 -type f -name "model_world_size_*_rank_0.pt" | head -n 1)
        expected=$(basename "$first_shard" | sed -E "s/model_world_size_([0-9]+)_rank_0.pt/\\1/")
        actual=$(find "$target/actor" -maxdepth 1 -type f -name "model_world_size_*_rank_*.pt" | wc -l | tr -d " ")
        if [[ -z "$expected" || "$actual" != "$expected" ]]; then
            echo "Incomplete $EXPORT_NAME checkpoint: $actual/${expected:-unknown} shards" >&2
            exit 1
        fi
    '
    srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        apptainer exec --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
        python3 /verl/scripts/model_merger.py --local_dir "$MERGE_ROOT/$export_name/actor"
    srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        rclone copy "$MERGE_ROOT/$export_name/actor/huggingface" \
        "$REMOTE_RUN/$export_name/huggingface" --s3-no-check-bucket \
        --stats=30s --stats-one-line
done

srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc '
    cat > "$MERGE_ROOT/metadata.txt" <<EOF
job_id=${SLURM_JOB_ID}
step=${LATEST_STEP}
teacher_model=${TEACHER_MODEL_PATH}
teacher_attempt_max_chars=${TEACHER_ATTEMPT_MAX_CHARS}
base_question_batch_size=${BASE_QUESTION_BATCH_SIZE}
teacher_attempts_per_question=${TEACHER_ROLLOUT_N}
rollouts_per_attempt=${ROLLOUT_N}
optimizer_prompt_batch_size=${OPTIMIZER_PROMPT_BATCH_SIZE}
train_decomposer=${TRAIN_DECOMPOSER}
train_agent_roles=${TRAIN_AGENT_ROLES}
decomposer_base=${DECOMPOSER_MODEL_PATH}
worker_base=${WORKER_MODEL_PATH}
worker_bootstrap_steps=${WORKER_BOOTSTRAP_STEPS}
decomposer_transfer_steps=${DECOMPOSER_TRANSFER_STEPS}
worker_question_fade_steps=${WORKER_QUESTION_FADE_STEPS}
joint_steps=${JOINT_STEPS}
teacher_data=${TEACHER_TRAIN_REMOTE}
online_teacher_generation=${ONLINE_TEACHER_GENERATION}
teacher_shard_questions=${TEACHER_SHARD_QUESTIONS}
teacher_shard_remote_root=${TEACHER_SHARD_REMOTE_ROOT}
EOF
    rclone copyto "$MERGE_ROOT/metadata.txt" "$REMOTE_RUN/metadata.txt" --s3-no-check-bucket
'

echo "Agent 1 model: ${REMOTE_RUN}/decomposer/huggingface"
echo "Agent 2 model: ${REMOTE_RUN}/workers/huggingface"
echo "Agent 1/2 curriculum completed"

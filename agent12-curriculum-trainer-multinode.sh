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
export ROLLOUT_N=${ROLLOUT_N:-16}
export TEACHER_ATTEMPT_MAX_CHARS=${TEACHER_ATTEMPT_MAX_CHARS:-8000}

export WORKER_BOOTSTRAP_STEPS=${WORKER_BOOTSTRAP_STEPS:-200}
export DECOMPOSER_TRANSFER_STEPS=${DECOMPOSER_TRANSFER_STEPS:-200}
export WORKER_QUESTION_FADE_STEPS=${WORKER_QUESTION_FADE_STEPS:-200}
export JOINT_STEPS=${JOINT_STEPS:-400}
export TOTAL_STEPS=${TOTAL_STEPS:-$((WORKER_BOOTSTRAP_STEPS + DECOMPOSER_TRANSFER_STEPS + JOINT_STEPS))}
export TEST_FREQ=${TEST_FREQ:-10}
export SAVE_FREQ=${SAVE_FREQ:-50}

export AGENT12_S3_REMOTE=${AGENT12_S3_REMOTE:-s3v2:s3min-tomasznaskret-1712063354/user/ajanz/agent12}
export AGENT12_RUN_NAME=${AGENT12_RUN_NAME:-agent12-curriculum-${SLURM_JOB_ID}}
export REMOTE_RUN=${AGENT12_S3_REMOTE%/}/${AGENT12_RUN_NAME}
export GENERATE_TEACHER_DATA=${GENERATE_TEACHER_DATA:-1}
export TEACHER_TRAIN_REMOTE=${TEACHER_TRAIN_REMOTE:-${AGENT12_S3_REMOTE%/}/teacher_data/math-qwen25-1.5b-n${TEACHER_ROLLOUT_N}-all-attempts.parquet}

export JOB_TMP=${JOB_TMP:-/mnt/lscratch/slurm/${SLURM_JOB_ID}/agent12}
export RAY_NODE_TMP=${RAY_NODE_TMP:-${JOB_TMP}/ray}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${JOB_TMP}/checkpoints/agent12}
export TEACHER_TRAIN_FILE=${TEACHER_TRAIN_FILE:-${JOB_TMP}/agent12_data/train.parquet}

if (( GPUS_PER_NODE % 2 != 0 )); then
    echo "GPUS_PER_NODE must be even because Agent 1 and Agent 2 use separate pools" >&2
    exit 1
fi
if (( POOL_GPUS_PER_NODE * 2 > GPUS_PER_NODE )); then
    echo "Two model pools request more GPUs than the SLURM allocation" >&2
    exit 1
fi

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
  data.n_samples=${TEACHER_ROLLOUT_N}"
    PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
        bash -c "$TEACHER_COMMAND"

    echo "Collecting all teacher attempts while retaining every task"
    srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        apptainer exec --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
        bash -lc 'export PYTHONPATH=/root/ReMA-public:/root/ReMA-public/src/verl:/verl:$PYTHONPATH; \
            python3 /root/ReMA-public/scripts/prepare_agent12_teacher_data.py \
            --input /root/tmpdir/agent12_data/teacher_candidates.parquet \
            --output /root/tmpdir/agent12_data/train.parquet'

    if [[ -n "$TEACHER_TRAIN_REMOTE" ]]; then
        srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
            rclone copyto "$TEACHER_TRAIN_FILE" "$TEACHER_TRAIN_REMOTE"
        echo "Teacher data uploaded to ${TEACHER_TRAIN_REMOTE}"
    fi
    stop_ray
else
    if [[ -z "$TEACHER_TRAIN_REMOTE" ]]; then
        echo "TEACHER_TRAIN_REMOTE is required when GENERATE_TEACHER_DATA=0" >&2
        exit 1
    fi
    echo "Reusing teacher data from ${TEACHER_TRAIN_REMOTE}"
    srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        rclone copyto "$TEACHER_TRAIN_REMOTE" "$TEACHER_TRAIN_FILE"
fi

start_ray

echo "Training Agent 1 (decomposer) and Agent 2 (workers/final)"
TRAIN_COMMAND="unset ROCR_VISIBLE_DEVICES; \
export TMPDIR=${RAY_NODE_TMP}; \
export HF_HOME=/root/tmpdir/hf_home; \
export PYTHONPATH=/root/ReMA-public:/root/ReMA-public/src/verl:/verl:\$PYTHONPATH; \
export RAY_ADDRESS=${IP_HEAD}; \
python3 -m verl.rema_separated_trainer.main_ppo \
  --config-path=/root/ReMA-public/config \
  --config-name=rema-rl.yaml \
  data.train_files=${TEACHER_TRAIN_FILE} \
  actor_rollout_ref.model.path=${DECOMPOSER_MODEL_PATH} \
  algorithm.switch_agent.model_paths=[${DECOMPOSER_MODEL_PATH},${WORKER_MODEL_PATH}] \
  algorithm.hierarchy.agent12_curriculum.enable=True \
  algorithm.hierarchy.agent12_curriculum.teacher_attempt_max_chars=${TEACHER_ATTEMPT_MAX_CHARS} \
  algorithm.hierarchy.agent12_curriculum.worker_bootstrap_steps=${WORKER_BOOTSTRAP_STEPS} \
  algorithm.hierarchy.agent12_curriculum.decomposer_transfer_steps=${DECOMPOSER_TRANSFER_STEPS} \
  algorithm.hierarchy.agent12_curriculum.worker_question_fade_steps=${WORKER_QUESTION_FADE_STEPS} \
  algorithm.hierarchy.agent12_curriculum.worker_question_final_probability=0.0 \
  algorithm.hierarchy.agent12_curriculum.worker_question_eval_probability=0.0 \
  algorithm.hierarchy.train_agent_roles=[decomposer,worker_stage_1,worker_stage_2,worker_stage_3,worker_stage_4,worker_stage_5] \
  algorithm.hierarchy.reward_composer.enable=False \
  algorithm.hierarchy.reward_composer.online.enable=False \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  trainer.nnodes=${SLURM_NNODES} \
  trainer.n_gpus_per_node=${POOL_GPUS_PER_NODE} \
  trainer.total_epochs=100 \
  trainer.total_training_steps=${TOTAL_STEPS} \
  trainer.val_before_train=True \
  trainer.test_freq=${TEST_FREQ} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.experiment_name=${AGENT12_RUN_NAME} \
  trainer.default_local_dir=${CHECKPOINT_ROOT}"

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
    bash -c "$TRAIN_COMMAND"

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
        "$REMOTE_RUN/$export_name/huggingface" --stats=30s --stats-one-line
done

srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc '
    cat > "$MERGE_ROOT/metadata.txt" <<EOF
job_id=${SLURM_JOB_ID}
step=${LATEST_STEP}
teacher_model=${TEACHER_MODEL_PATH}
teacher_attempt_max_chars=${TEACHER_ATTEMPT_MAX_CHARS}
decomposer_base=${DECOMPOSER_MODEL_PATH}
worker_base=${WORKER_MODEL_PATH}
worker_bootstrap_steps=${WORKER_BOOTSTRAP_STEPS}
decomposer_transfer_steps=${DECOMPOSER_TRANSFER_STEPS}
worker_question_fade_steps=${WORKER_QUESTION_FADE_STEPS}
joint_steps=${JOINT_STEPS}
teacher_data=${TEACHER_TRAIN_REMOTE}
EOF
    rclone copyto "$MERGE_ROOT/metadata.txt" "$REMOTE_RUN/metadata.txt"
'

echo "Agent 1 model: ${REMOTE_RUN}/decomposer/huggingface"
echo "Agent 2 model: ${REMOTE_RUN}/workers/huggingface"
echo "Agent 1/2 curriculum completed"

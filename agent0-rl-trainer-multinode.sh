#!/bin/bash
#SBATCH --job-name=agent0-serial
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

export AGENT0_S3_REMOTE=${AGENT0_S3_REMOTE:-s3v2:s3min-tomasznaskret-1712063354/user/ajanz/checkpoints}

export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export RAY_PORT=${RAY_PORT:-6379}
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
export ROLLOUT_N=${ROLLOUT_N:-16}
export TEST_FREQ=${TEST_FREQ:-10}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TOTAL_STEPS=${TOTAL_STEPS:-4000}

export SIF_NAME=${SIF_NAME:-verl-rema-v3.sif}
export SIF_REMOTE=${SIF_REMOTE:-s3v2:s3min-tomasznaskret-1712063354/user/dmotyka/sif_images/${SIF_NAME}}
export JOB_TMP=${JOB_TMP:-/mnt/lscratch/slurm/${SLURM_JOB_ID}/agent0}
export RAY_NODE_TMP=${RAY_NODE_TMP:-${JOB_TMP}/ray}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${JOB_TMP}/checkpoints/agent0-serial}
export AGENT0_RUN_NAME=${AGENT0_RUN_NAME:-agent0-serial-${SLURM_JOB_ID}}
export REMOTE_RUN=${AGENT0_S3_REMOTE%/}/${AGENT0_RUN_NAME}

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

echo "Preparing Agent 0 workspace on all nodes"
srun --label --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" bash -lc '
    set -euo pipefail
    mkdir -p "$JOB_TMP"
    rm -rf "$JOB_TMP/MATH" "$JOB_TMP/overall_math" "$JOB_TMP/agent0_data" \
        "$JOB_TMP/verl" "$JOB_TMP/prompt" "$JOB_TMP/scripts" "$JOB_TMP/config"
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

echo "Preparing Ray temporary directories"
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" bash -lc '
    rm -rf "$RAY_NODE_TMP"
    mkdir -p "$RAY_NODE_TMP"
'

echo "Building single-agent parquet files on all nodes"
srun --label --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" \
    apptainer exec --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
    bash -lc 'export PYTHONPATH=/root/ReMA-public:/verl:$PYTHONPATH; \
        python3 /root/ReMA-public/scripts/prepare_agent0_data.py \
        --train-input /root/tmpdir/MATH/train_lv3to5_8k.parquet \
        --val-input /root/tmpdir/overall_math/test.parquet \
        --output-dir /root/tmpdir/agent0_data'

echo "Starting Ray head on ${HEAD_NODE} (${IP_HEAD})"
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
    ray start --head --node-ip-address="$HEAD_NODE_IP" --port="$RAY_PORT" \
        --temp-dir="$RAY_NODE_TMP" --num-cpus="${SLURM_CPUS_PER_TASK}" \
        --num-gpus="${GPUS_PER_NODE}" --block &
sleep 15

for ((i = 1; i < SLURM_NNODES; i++)); do
    NODE=${NODES[$i]}
    echo "Starting Ray worker on ${NODE}"
    srun --nodes=1 --ntasks=1 -w "$NODE" \
        apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
        ray start --address "$IP_HEAD" --temp-dir="$RAY_NODE_TMP" \
            --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus="${GPUS_PER_NODE}" --block &
    sleep 5
done

EXPECTED_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))
RAY_WAIT_TIMEOUT=${RAY_WAIT_TIMEOUT:-300}
RAY_WAIT_INTERVAL=${RAY_WAIT_INTERVAL:-10}
RAY_WAIT_ELAPSED=0

while true; do
    AVAILABLE_GPUS=$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
        python3 -c "import ray; ray.init(address='${IP_HEAD}', ignore_reinit_error=True, logging_level=40); print(int(ray.cluster_resources().get('GPU', 0)))" \
        2>/dev/null || echo 0)
    AVAILABLE_GPUS=$(echo "$AVAILABLE_GPUS" | tail -n 1 | tr -d '[:space:]')
    if [[ "$AVAILABLE_GPUS" =~ ^[0-9]+$ ]] && (( AVAILABLE_GPUS >= EXPECTED_GPUS )); then
        echo "Ray cluster ready: ${AVAILABLE_GPUS}/${EXPECTED_GPUS} GPUs"
        break
    fi
    if (( RAY_WAIT_ELAPSED >= RAY_WAIT_TIMEOUT )); then
        echo "Timed out waiting for Ray: ${AVAILABLE_GPUS:-0}/${EXPECTED_GPUS} GPUs" >&2
        exit 1
    fi
    sleep "$RAY_WAIT_INTERVAL"
    RAY_WAIT_ELAPSED=$((RAY_WAIT_ELAPSED + RAY_WAIT_INTERVAL))
done

COMMAND="unset ROCR_VISIBLE_DEVICES; \
export TMPDIR=${RAY_NODE_TMP}; \
export HF_HOME=/root/tmpdir/hf_home; \
export PYTHONPATH=/root/ReMA-public:/root/ReMA-public/src:/verl:\$PYTHONPATH; \
export RAY_ADDRESS=${IP_HEAD}; \
python3 -m verl.trainer.main_ppo \
  --config-path=/root/ReMA-public/config \
  --config-name=agent0-rl.yaml \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  trainer.nnodes=${SLURM_NNODES} \
  trainer.n_gpus_per_node=${GPUS_PER_NODE} \
  trainer.total_training_steps=${TOTAL_STEPS} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.default_local_dir=${CHECKPOINT_ROOT}"

echo "Training Agent 0"
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
    bash -c "$COMMAND"

# The driver may be scheduled on any Ray node. Find the step marker wherever it
# was written, then upload each node's uniquely named FSDP shards to S3.
LATEST_STEP=$(srun --overlap --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" \
    bash -lc 'if [[ -f "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt" ]]; then cat "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"; fi' \
    | grep -E '^[0-9]+$' | sort -n | tail -n 1)
if [[ -z "$LATEST_STEP" ]]; then
    echo "Could not locate the final Agent 0 checkpoint step" >&2
    exit 1
fi
export LATEST_STEP
echo "Collecting global_step_${LATEST_STEP} through ${REMOTE_RUN}/raw"

srun --overlap --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" bash -lc '
    set -euo pipefail
    actor_dir="$CHECKPOINT_ROOT/global_step_${LATEST_STEP}/actor"
    if [[ -d "$actor_dir" ]]; then
        node_name=${SLURMD_NODENAME:-$(hostname)}
        rclone copy "$actor_dir" "$REMOTE_RUN/raw/$node_name/actor" \
            --include "/model_world_size_*_rank_*.pt" \
            --include "/huggingface/**" --exclude "*" \
            --stats=30s --stats-one-line
    fi
'

export MERGE_ROOT=${JOB_TMP}/agent0_merge
echo "Downloading all FSDP shards to ${HEAD_NODE} for Hugging Face merge"
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc '
    set -euo pipefail
    rm -rf "$MERGE_ROOT"
    mkdir -p "$MERGE_ROOT/raw" "$MERGE_ROOT/actor/huggingface"
    rclone copy "$REMOTE_RUN/raw" "$MERGE_ROOT/raw" --stats=30s --stats-one-line

    find "$MERGE_ROOT/raw" -type f -name "model_world_size_*_rank_*.pt" \
        -exec cp {} "$MERGE_ROOT/actor/" \;
    hf_source=$(find "$MERGE_ROOT/raw" -type d -name huggingface | head -n 1)
    if [[ -z "$hf_source" ]]; then
        echo "No Hugging Face metadata found in the raw checkpoint" >&2
        exit 1
    fi
    cp -r "$hf_source/." "$MERGE_ROOT/actor/huggingface/"

    first_shard=$(find "$MERGE_ROOT/actor" -maxdepth 1 -type f \
        -name "model_world_size_*_rank_0.pt" | head -n 1)
    expected=$(basename "$first_shard" | sed -E "s/model_world_size_([0-9]+)_rank_0.pt/\\1/")
    actual=$(find "$MERGE_ROOT/actor" -maxdepth 1 -type f \
        -name "model_world_size_*_rank_*.pt" | wc -l | tr -d " ")
    if [[ -z "$expected" || "$actual" != "$expected" ]]; then
        echo "Incomplete FSDP checkpoint: found $actual of ${expected:-unknown} shards" >&2
        exit 1
    fi
'

echo "Merging Agent 0 checkpoint"
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    apptainer exec --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
    python3 /verl/scripts/model_merger.py --local_dir "$MERGE_ROOT/actor"

echo "Uploading ready-to-load Agent 0 model to ${REMOTE_RUN}/huggingface"
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc '
    set -euo pipefail
    cat > "$MERGE_ROOT/metadata.txt" <<EOF
job_id=${SLURM_JOB_ID}
step=${LATEST_STEP}
base_model=${MODEL_PATH}
rollout_n=${ROLLOUT_N}
EOF
    rclone copy "$MERGE_ROOT/actor/huggingface" "$REMOTE_RUN/huggingface" --stats=30s --stats-one-line
    rclone copyto "$MERGE_ROOT/metadata.txt" "$REMOTE_RUN/metadata.txt"
    rclone copyto "$SLURM_SUBMIT_DIR/prompt/math/serial_solver.py" "$REMOTE_RUN/serial_solver.py"
    echo "Agent 0 model: $REMOTE_RUN/huggingface"
'

echo "Agent 0 training and S3 export completed"

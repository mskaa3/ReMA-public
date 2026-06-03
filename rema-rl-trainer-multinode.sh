#!/bin/bash
#SBATCH --job-name=verl-trainer-mn
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
# This trainer creates two GPU pools, so this is GPUs per node per pool.
export POOL_GPUS_PER_NODE=${POOL_GPUS_PER_NODE:-$((GPUS_PER_NODE / 2))}
export RAY_PORT=${RAY_PORT:-6379}
export RAY_NODE_TMP=${RAY_NODE_TMP:-/tmp/ray-${USER}-${SLURM_JOB_ID}}

export SIF_NAME=${SIF_NAME:-verl-rema-v3.sif}
export SIF_REMOTE=${SIF_REMOTE:-s3v2:s3min-tomasznaskret-1712063354/user/dmotyka/sif_images/${SIF_NAME}}

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}

export JOB_TMP=${JOB_TMP:-/mnt/lscratch/slurm/${SLURM_JOB_ID}/rema}

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

echo "Preparing node-local JOB_TMP at ${JOB_TMP} on all nodes"
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" bash -lc '
    set -euo pipefail
    mkdir -p "$JOB_TMP"
    rm -rf "$JOB_TMP/overall_math" "$JOB_TMP/MATH" "$JOB_TMP/verl"
    cp -r "$SLURM_SUBMIT_DIR/data/overall_math" "$JOB_TMP/overall_math"
    cp -r "$SLURM_SUBMIT_DIR/data/MATH" "$JOB_TMP/MATH"
    cp -r "$SLURM_SUBMIT_DIR/src/verl" "$JOB_TMP/verl"
    rclone copy "$SIF_REMOTE" "$JOB_TMP/"
'

echo "Preparing node-local Ray temp dir at ${RAY_NODE_TMP}"
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" bash -lc '
    rm -rf "'"${RAY_NODE_TMP}"'"
    mkdir -p "'"${RAY_NODE_TMP}"'"
'

COMMON_MOUNTS=(
    --mount "type=bind,src=${JOB_TMP},dst=${JOB_TMP}"
    --mount "type=bind,src=${JOB_TMP},dst=/root/tmpdir"
    --mount "type=bind,src=${JOB_TMP}/verl,dst=/verl"
    --mount "type=bind,src=${JOB_TMP}/verl,dst=/root/ReMA-public/src/verl"
    --mount "type=bind,src=${RAY_NODE_TMP},dst=${RAY_NODE_TMP}"
)

echo "Starting Ray head on ${HEAD_NODE} (${IP_HEAD})"
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
    ray start --head --node-ip-address="$HEAD_NODE_IP" --port="$RAY_PORT" \
        --temp-dir="$RAY_NODE_TMP" \
        --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus="${GPUS_PER_NODE}" --block &
sleep 15

WORKER_COUNT=$((SLURM_NNODES - 1))
for ((i = 1; i <= WORKER_COUNT; i++)); do
    NODE=${NODES[$i]}
    echo "Starting Ray worker ${i} on ${NODE}"
    srun --nodes=1 --ntasks=1 -w "$NODE" \
        apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
        ray start --address "$IP_HEAD" \
            --temp-dir="$RAY_NODE_TMP" \
            --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus="${GPUS_PER_NODE}" --block &
    sleep 5
done

EXPECTED_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))
RAY_WAIT_TIMEOUT=${RAY_WAIT_TIMEOUT:-300}
RAY_WAIT_INTERVAL=${RAY_WAIT_INTERVAL:-10}
RAY_WAIT_ELAPSED=0

echo "Waiting for Ray cluster to register ${EXPECTED_GPUS} GPUs"
while true; do
    AVAILABLE_GPUS=$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
        python3 -c "import ray; ray.init(address='${IP_HEAD}', ignore_reinit_error=True, logging_level=40); print(int(ray.cluster_resources().get('GPU', 0)))" \
        2>/dev/null || echo 0)
    AVAILABLE_GPUS=$(echo "$AVAILABLE_GPUS" | tail -n 1 | tr -d '[:space:]')

    if [[ "$AVAILABLE_GPUS" =~ ^[0-9]+$ ]] && (( AVAILABLE_GPUS >= EXPECTED_GPUS )); then
        echo "Ray cluster ready: ${AVAILABLE_GPUS}/${EXPECTED_GPUS} GPUs available"
        break
    fi

    if (( RAY_WAIT_ELAPSED >= RAY_WAIT_TIMEOUT )); then
        echo "Timed out waiting for Ray cluster: ${AVAILABLE_GPUS:-0}/${EXPECTED_GPUS} GPUs available"
        srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
            apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
            ray status --address "$IP_HEAD" || true
        exit 1
    fi

    echo "Ray cluster not ready yet: ${AVAILABLE_GPUS:-0}/${EXPECTED_GPUS} GPUs available; waiting ${RAY_WAIT_INTERVAL}s"
    sleep "$RAY_WAIT_INTERVAL"
    RAY_WAIT_ELAPSED=$((RAY_WAIT_ELAPSED + RAY_WAIT_INTERVAL))
done

COMMAND="unset ROCR_VISIBLE_DEVICES; \
export TMPDIR=${RAY_NODE_TMP}; \
export HF_HOME=/root/tmpdir/hf_home; \
export PYTHONPATH=/root/ReMA-public/src:/verl:\$PYTHONPATH; \
export RAY_ADDRESS=${IP_HEAD}; \
python3 -m verl.rema_separated_trainer.main_ppo \
  --config-path=/home/ajanz/projects/ReMA-public/config \
  --config-name=rema-rl.yaml \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  trainer.nnodes=${SLURM_NNODES} \
  trainer.n_gpus_per_node=${POOL_GPUS_PER_NODE}"

echo "Submitting trainer on Ray head"
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
    bash -c "$COMMAND"

if [[ -n ${JOB_TMP:-} ]]; then
    srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" bash -lc 'rm -rf "$JOB_TMP"/*'
fi
if [[ -n ${RAY_NODE_TMP:-} ]]; then
    srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" bash -lc 'rm -rf "'"${RAY_NODE_TMP}"'"'
fi

#!/bin/bash
#SBATCH --job-name=verl-trainer-mn
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=72:00:00
#SBATCH --mem=200gb
#SBATCH -p lem-gpu-short
#SBATCH --gpus-per-node=hopper:4
#SBATCH --gres=storage:local:200G
#SBATCH --verbose

set -euo pipefail

source ./env.sh

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
# This trainer creates two GPU pools, so this is GPUs per node per pool.
POOL_GPUS_PER_NODE=${POOL_GPUS_PER_NODE:-$((GPUS_PER_NODE / 2))}
RAY_PORT=${RAY_PORT:-6379}
RAY_NODE_TMP=${RAY_NODE_TMP:-/tmp/ray-${USER}-${SLURM_JOB_ID}}

SIF_NAME=${SIF_NAME:-verl-rema-v3.sif}
SIF_REMOTE=${SIF_REMOTE:-s3v2:s3min-tomasznaskret-1712063354/user/dmotyka/sif_images/${SIF_NAME}}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}

export JOB_TMP=${JOB_TMP:-${TMPDIR:-/tmp/${USER}/rema-${SLURM_JOB_ID}}}

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

echo "Preparing shared JOB_TMP at ${JOB_TMP}"
mkdir -p "$JOB_TMP"
rm -rf "$JOB_TMP/overall_math" "$JOB_TMP/MATH" "$JOB_TMP/verl"
cp -r ./data/overall_math "$JOB_TMP/overall_math"
cp -r ./data/MATH "$JOB_TMP/MATH"
cp -r ./src/verl "$JOB_TMP/verl"
rclone copy "$SIF_REMOTE" "$JOB_TMP/"

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
    rm -rf "$JOB_TMP"/*
fi
if [[ -n ${RAY_NODE_TMP:-} ]]; then
    srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" bash -lc 'rm -rf "'"${RAY_NODE_TMP}"'"'
fi

#!/bin/bash
#SBATCH --job-name=verl-trainer-mn
#SBATCH --nodes=8
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
export PRD_EXPORT_ENABLE=${PRD_EXPORT_ENABLE:-0}
export PRD_EXPORT_MAX_RECORDS=${PRD_EXPORT_MAX_RECORDS:-50000}
export PRD_EXPORT_LOCAL=${PRD_EXPORT_LOCAL:-${JOB_TMP}/prd_reward_composer/prd_records_${SLURM_JOB_ID}.jsonl}
export PRD_EXPORT_REMOTE=${PRD_EXPORT_REMOTE:-}
export PRD_ONLINE_ENABLE=${PRD_ONLINE_ENABLE:-1}
export PRD_ONLINE_MODEL_TYPE=${PRD_ONLINE_MODEL_TYPE:-role}
export PRD_ONLINE_LR=${PRD_ONLINE_LR:-3.0e-4}
export PRD_ROUTING_FLOOR=${PRD_ROUTING_FLOOR:-0.05}
if [[ -z "${PRD_ROUTING_ACTIVATION:-}" ]]; then
    if [[ "${PRD_ONLINE_MODEL_TYPE}" == "role" ]]; then
        export PRD_ROUTING_ACTIVATION=softmax
    else
        export PRD_ROUTING_ACTIVATION=sigmoid
    fi
else
    export PRD_ROUTING_ACTIVATION
fi
export PRD_ROLE_RANK_LOSS_WEIGHT=${PRD_ROLE_RANK_LOSS_WEIGHT:-0.0}
export PRD_ROLE_ACTIVITY_MARGIN=${PRD_ROLE_ACTIVITY_MARGIN:-0.05}
export PRD_ROLE_RANK_TEMPERATURE=${PRD_ROLE_RANK_TEMPERATURE:-1.0}
export PRD_IMPLICIT_CF_LOSS_WEIGHT=${PRD_IMPLICIT_CF_LOSS_WEIGHT:-0.0}
export PRD_IMPLICIT_CF_ROLE_DISTANCE_MARGIN=${PRD_IMPLICIT_CF_ROLE_DISTANCE_MARGIN:-0.05}
export PRD_IMPLICIT_CF_HUBER_DELTA=${PRD_IMPLICIT_CF_HUBER_DELTA:-1.0}
export PRD_ONLINE_WARMUP_STEPS=${PRD_ONLINE_WARMUP_STEPS:-150}
export PRD_ONLINE_BLEND_ALPHA=${PRD_ONLINE_BLEND_ALPHA:-0.20}
export PRD_ONLINE_BLEND_ALPHA_MAX=${PRD_ONLINE_BLEND_ALPHA_MAX:-0.20}
export PRD_ONLINE_BLEND_RAMP_STEPS=${PRD_ONLINE_BLEND_RAMP_STEPS:-200}
export PRD_ONLINE_SYNC_INTERVAL=${PRD_ONLINE_SYNC_INTERVAL:-1}
export PRD_ONLINE_EMA_BETA=${PRD_ONLINE_EMA_BETA:-0.9}
export PRD_GRAPH_PRIOR_MODE=${PRD_GRAPH_PRIOR_MODE:-soft}
export PRD_GRAPH_PRIOR_SOFT_DISTANCE_PENALTY=${PRD_GRAPH_PRIOR_SOFT_DISTANCE_PENALTY:-1.0}
export PRD_GRAPH_PRIOR_REVERSE_DISTANCE_PENALTY=${PRD_GRAPH_PRIOR_REVERSE_DISTANCE_PENALTY:-3.0}
export ROLLOUT_N=${ROLLOUT_N:-32}
export TEST_FREQ=${TEST_FREQ:-10}

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

PRD_EXPORT_OVERRIDES=""
if [[ "${PRD_EXPORT_ENABLE}" == "1" || "${PRD_EXPORT_ENABLE}" == "true" ]]; then
    PRD_EXPORT_OVERRIDES="algorithm.hierarchy.reward_composer.export_jsonl_path=${PRD_EXPORT_LOCAL} algorithm.hierarchy.reward_composer.export_jsonl_max_records=${PRD_EXPORT_MAX_RECORDS}"
    echo "PRD reward-composer export enabled: ${PRD_EXPORT_LOCAL}"
    if [[ -n "${PRD_EXPORT_REMOTE}" ]]; then
        echo "PRD export will be uploaded to: ${PRD_EXPORT_REMOTE}"
    else
        echo "PRD_EXPORT_REMOTE is not set; export will remain only on node-local scratch until cleanup."
    fi
fi

PRD_ONLINE_OVERRIDES="algorithm.hierarchy.reward_composer.enable=False actor_rollout_ref.rollout.n=${ROLLOUT_N}"
if [[ "${PRD_ONLINE_ENABLE}" == "1" || "${PRD_ONLINE_ENABLE}" == "true" ]]; then
    PRD_ONLINE_OVERRIDES="${PRD_ONLINE_OVERRIDES} algorithm.hierarchy.reward_composer.graph_prior_mode=${PRD_GRAPH_PRIOR_MODE} algorithm.hierarchy.reward_composer.graph_prior_soft_distance_penalty=${PRD_GRAPH_PRIOR_SOFT_DISTANCE_PENALTY} algorithm.hierarchy.reward_composer.graph_prior_reverse_distance_penalty=${PRD_GRAPH_PRIOR_REVERSE_DISTANCE_PENALTY} algorithm.hierarchy.reward_composer.online.enable=True algorithm.hierarchy.reward_composer.online.model_type=${PRD_ONLINE_MODEL_TYPE} algorithm.hierarchy.reward_composer.online.lr=${PRD_ONLINE_LR} algorithm.hierarchy.reward_composer.online.routing_activation=${PRD_ROUTING_ACTIVATION} algorithm.hierarchy.reward_composer.online.routing_floor=${PRD_ROUTING_FLOOR} algorithm.hierarchy.reward_composer.online.role_rank_loss_weight=${PRD_ROLE_RANK_LOSS_WEIGHT} algorithm.hierarchy.reward_composer.online.role_activity_margin=${PRD_ROLE_ACTIVITY_MARGIN} algorithm.hierarchy.reward_composer.online.role_rank_temperature=${PRD_ROLE_RANK_TEMPERATURE} algorithm.hierarchy.reward_composer.online.implicit_cf_loss_weight=${PRD_IMPLICIT_CF_LOSS_WEIGHT} algorithm.hierarchy.reward_composer.online.implicit_cf_role_distance_margin=${PRD_IMPLICIT_CF_ROLE_DISTANCE_MARGIN} algorithm.hierarchy.reward_composer.online.implicit_cf_huber_delta=${PRD_IMPLICIT_CF_HUBER_DELTA} algorithm.hierarchy.reward_composer.online.warmup_steps=${PRD_ONLINE_WARMUP_STEPS} algorithm.hierarchy.reward_composer.online.blend_alpha=${PRD_ONLINE_BLEND_ALPHA} algorithm.hierarchy.reward_composer.online.blend_alpha_max=${PRD_ONLINE_BLEND_ALPHA_MAX} algorithm.hierarchy.reward_composer.online.blend_ramp_steps=${PRD_ONLINE_BLEND_RAMP_STEPS} algorithm.hierarchy.reward_composer.online.mixed_groups_only=True algorithm.hierarchy.reward_composer.online.sync_interval=${PRD_ONLINE_SYNC_INTERVAL} algorithm.hierarchy.reward_composer.online.ema_beta=${PRD_ONLINE_EMA_BETA} algorithm.hierarchy.reward_composer.online.graph_prior_mode=${PRD_GRAPH_PRIOR_MODE} algorithm.hierarchy.reward_composer.online.graph_prior_soft_distance_penalty=${PRD_GRAPH_PRIOR_SOFT_DISTANCE_PENALTY} algorithm.hierarchy.reward_composer.online.graph_prior_reverse_distance_penalty=${PRD_GRAPH_PRIOR_REVERSE_DISTANCE_PENALTY}"
    echo "Online PRD reward composer enabled with model_type=${PRD_ONLINE_MODEL_TYPE}, objective=cpcr_replay, rollout.n=${ROLLOUT_N}, alpha=${PRD_ONLINE_BLEND_ALPHA}, warmup=${PRD_ONLINE_WARMUP_STEPS}, lr=${PRD_ONLINE_LR}, routing=${PRD_ROUTING_ACTIVATION}, graph_prior=${PRD_GRAPH_PRIOR_MODE}, implicit_cf=${PRD_IMPLICIT_CF_LOSS_WEIGHT}, legacy_role_rank=${PRD_ROLE_RANK_LOSS_WEIGHT}"
else
    PRD_ONLINE_OVERRIDES="${PRD_ONLINE_OVERRIDES} algorithm.hierarchy.reward_composer.online.enable=False"
    echo "Online PRD reward composer disabled; rollout.n=${ROLLOUT_N}"
fi

echo "Preparing node-local JOB_TMP at ${JOB_TMP} on all nodes"
srun --label --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" bash -lc '
    set -euo pipefail
    echo "[$(hostname)] prepare: mkdir ${JOB_TMP}"
    mkdir -p "$JOB_TMP"
    echo "[$(hostname)] prepare: cleanup old staged data"
    rm -rf "$JOB_TMP/overall_math" "$JOB_TMP/MATH" "$JOB_TMP/verl"
    echo "[$(hostname)] prepare: copy data/overall_math"
    cp -r "$SLURM_SUBMIT_DIR/data/overall_math" "$JOB_TMP/overall_math"
    echo "[$(hostname)] prepare: copy data/MATH"
    cp -r "$SLURM_SUBMIT_DIR/data/MATH" "$JOB_TMP/MATH"
    echo "[$(hostname)] prepare: copy src/verl"
    cp -r "$SLURM_SUBMIT_DIR/src/verl" "$JOB_TMP/verl"
    echo "[$(hostname)] prepare: copy sif ${SIF_REMOTE}"
    rclone copy "$SIF_REMOTE" "$JOB_TMP/" --stats=30s --stats-one-line
    echo "[$(hostname)] prepare: done"
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
	  trainer.n_gpus_per_node=${POOL_GPUS_PER_NODE} \
	  trainer.val_before_train=False \
	  trainer.test_freq=${TEST_FREQ} \
	  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
	  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
	  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
	  algorithm.hierarchy.num_worker_stages=3 \
      ${PRD_ONLINE_OVERRIDES} \
      ${PRD_EXPORT_OVERRIDES}"

echo "Submitting trainer on Ray head"
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    apptainer exec --nv --writable-tmpfs "${COMMON_MOUNTS[@]}" "$JOB_TMP/${SIF_NAME}" \
    bash -c "$COMMAND"

if [[ ("${PRD_EXPORT_ENABLE}" == "1" || "${PRD_EXPORT_ENABLE}" == "true") && -n "${PRD_EXPORT_REMOTE}" ]]; then
    echo "Uploading PRD reward-composer export from head node"
    srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc '
        set -euo pipefail
        if [[ ! -s "'"${PRD_EXPORT_LOCAL}"'" ]]; then
            echo "No PRD export found at '"${PRD_EXPORT_LOCAL}"'; skipping upload"
            exit 0
        fi
        gzip -c "'"${PRD_EXPORT_LOCAL}"'" > "'"${PRD_EXPORT_LOCAL}"'.gz"
        rclone copyto "'"${PRD_EXPORT_LOCAL}"'.gz" "'"${PRD_EXPORT_REMOTE}"'/prd_records_'"${SLURM_JOB_ID}"'.jsonl.gz"
        echo "Uploaded PRD export to '"${PRD_EXPORT_REMOTE}"'/prd_records_'"${SLURM_JOB_ID}"'.jsonl.gz"
    '
fi

if [[ -n ${JOB_TMP:-} ]]; then
    srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" bash -lc 'rm -rf "$JOB_TMP"/*'
fi
if [[ -n ${RAY_NODE_TMP:-} ]]; then
    srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" bash -lc 'rm -rf "'"${RAY_NODE_TMP}"'"'
fi

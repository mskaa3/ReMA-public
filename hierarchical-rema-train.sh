#!/bin/bash
#SBATCH --job-name=hierarchical-rema-train
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --mem=200gb
#SBATCH -p lem-gpu-short
#SBATCH --gpus-per-node=hopper:4
#SBATCH --verbose

set -euo pipefail

export RUN_KIND=${RUN_KIND:-train}
export BACKEND=${BACKEND:-vllm}
export MODE=${MODE:-alternating}
export TASKS_PER_EPOCH=${TASKS_PER_EPOCH:-32}
export NUM_DECOMPOSITIONS=${NUM_DECOMPOSITIONS:-2}
export NUM_SELECTIONS=${NUM_SELECTIONS:-2}
export TEMPERATURE=${TEMPERATURE:-0.5}
export CONTROLLER_MAX_NEW_TOKENS=${CONTROLLER_MAX_NEW_TOKENS:-256}
export WORKER_MAX_NEW_TOKENS=${WORKER_MAX_NEW_TOKENS:-128}
export ROLLOUT_TASK_BATCH_SIZE=${ROLLOUT_TASK_BATCH_SIZE:-8}
export CONTROLLER_BATCH_SIZE=${CONTROLLER_BATCH_SIZE:-8}
export WORKER_BATCH_SIZE=${WORKER_BATCH_SIZE:-16}
export ROLLOUT_PROGRESS_EVERY=${ROLLOUT_PROGRESS_EVERY:-1}

SOURCE_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
exec bash "$SOURCE_DIR/hierarchical-rema-trainer.sh" "$@"

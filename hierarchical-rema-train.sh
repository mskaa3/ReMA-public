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
export BACKEND=${BACKEND:-vllm} # rollout backend used during training: mock | hf | vllm
export MODE=${MODE:-alternating} # alternating = epoch 1 selector, epoch 2 decomposer, then switch back and forth

export NUM_EPOCHS=${NUM_EPOCHS:-20} # outer RL epochs: rollout -> GRPO update -> benchmark validation
export EPOCHS=${EPOCHS:-1} # inner GRPO passes over replay from one outer epoch; keep small for on-policy freshness

export TASKS_PER_EPOCH=${TASKS_PER_EPOCH:-853} # train tasks used to collect fresh rollouts in one outer epoch
export NUM_DECOMPOSITIONS=${NUM_DECOMPOSITIONS:-16} # controller rollout width during training
export NUM_SELECTIONS=${NUM_SELECTIONS:-16} # selector samples per decomposition during training
export TEMPERATURE=${TEMPERATURE:-0.5} # controller/worker sampling temperature during training rollouts
export CONTROLLER_MAX_NEW_TOKENS=${CONTROLLER_MAX_NEW_TOKENS:-1024} # max tokens for decomposer/selector outputs
export WORKER_MAX_NEW_TOKENS=${WORKER_MAX_NEW_TOKENS:-512} # max tokens for worker outputs
export ROLLOUT_TASK_BATCH_SIZE=${ROLLOUT_TASK_BATCH_SIZE:-16} # number of train tasks rolled out together
export CONTROLLER_BATCH_SIZE=${CONTROLLER_BATCH_SIZE:-16} # generation batch size for decomposer/selector calls
export WORKER_BATCH_SIZE=${WORKER_BATCH_SIZE:-16} # generation batch size for worker calls
export ROLLOUT_PROGRESS_EVERY=${ROLLOUT_PROGRESS_EVERY:-1} # print progress after every completed task

export VAL_NUM_DECOMPOSITIONS=${VAL_NUM_DECOMPOSITIONS:-1} # validation uses a single decomposition candidate per task
export VAL_NUM_SELECTIONS=${VAL_NUM_SELECTIONS:-1} # validation uses a single selector sample per task
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.0} # deterministic validation rollout
export VAL_TOP_P=${VAL_TOP_P:-1.0} # deterministic validation rollout
export MAX_VAL_TASKS=${MAX_VAL_TASKS:-0} # 0 = load the full overall_math validation benchmark
export VAL_TASKS_PER_EPOCH=${VAL_TASKS_PER_EPOCH:-0} # 0 = validate on all loaded validation tasks each outer epoch
export VAL_ROLLOUT_TASK_BATCH_SIZE=${VAL_ROLLOUT_TASK_BATCH_SIZE:-8} # number of validation tasks rolled out together

export ROLLOUT_LOG_MODE=${ROLLOUT_LOG_MODE:-best} # best = keep only top-K rollout records instead of every rollout
export ROLLOUT_LOG_DETAIL=${ROLLOUT_LOG_DETAIL:-compact} # compact = smaller JSONL artifacts
export BEST_K=${BEST_K:-1} # keep only the single best decomposition/selection record per epoch
export CHECKPOINT_MODE=${CHECKPOINT_MODE:-all} # all = enable best/ checkpoints when eval is on; final = only final/
export PRUNE_STALE_POLICY_MODELS=${PRUNE_STALE_POLICY_MODELS:-true} # delete older local policy checkpoints after newer ones are produced
export SAVE_STEPS=${SAVE_STEPS:-0} # disable checkpoint-N step snapshots to save disk
export EPOCH_S3_SYNC=${EPOCH_S3_SYNC:-true} # upload completed epoch folders to S3 while the job is still running
export EPOCH_S3_SYNC_INTERVAL=${EPOCH_S3_SYNC_INTERVAL:-120} # seconds between checks for finished epoch folders
export ENABLE_WANDB=${ENABLE_WANDB:-true} 

SOURCE_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
exec bash "$SOURCE_DIR/hierarchical-rema-trainer.sh" "$@"

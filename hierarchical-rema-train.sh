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
export PARAMETER_SHARING=${PARAMETER_SHARING:-false} # if true, selector and decomposer share the same weights; if false, they have separate weights and can specialize 
# Preferred subset-based names for the hierarchical training loop.
# TRAIN_SUBSET_ROUNDS: how many times we pick a train subset and run the full loop on it.
# TRAIN_SUBSET_SIZE: how many train tasks are in that subset.
# GRPO_PASSES_PER_SUBSET: how many update passes we make on rollout samples generated from that one subset.
# Legacy aliases still work underneath: NUM_EPOCHS, TASKS_PER_EPOCH, EPOCHS.
#
# Default alternating setup below assumes the original train split is about 8523 tasks.
# With TRAIN_SUBSET_SIZE=426 and alternating mode, TRAIN_SUBSET_ROUNDS=40 gives each controller
# about 20 subset rounds, i.e. roughly one full pass over the train set per role:
#   20 * 426 ~= 8520
export TRAIN_SUBSET_ROUNDS=${TRAIN_SUBSET_ROUNDS:-${NUM_EPOCHS:-40}}
export GRPO_PASSES_PER_SUBSET=${GRPO_PASSES_PER_SUBSET:-${EPOCHS:-1}}

export TRAIN_SUBSET_SIZE=${TRAIN_SUBSET_SIZE:-${TASKS_PER_EPOCH:-426}}
export NUM_DECOMPOSITIONS=${NUM_DECOMPOSITIONS:-16} # controller rollout width during training; 16 is expensive but keeps broad exploration
export NUM_SELECTIONS=${NUM_SELECTIONS:-16} # selector samples per decomposition; 16 is expensive but matches the broad exploration setting
export TEMPERATURE=${TEMPERATURE:-0.3} # shared fallback temperature; workers usually benefit from staying around 0.5
export CONTROLLER_TEMPERATURE=${CONTROLLER_TEMPERATURE:-$TEMPERATURE} # decomposer/selector temperature; can be lowered independently if structured output is unstable
export WORKER_TEMPERATURE=${WORKER_TEMPERATURE:-$TEMPERATURE} # worker temperature; separate from controller so reasoning diversity can stay higher
export CONTROLLER_CONSTRAINED_DECODING=${CONTROLLER_CONSTRAINED_DECODING:-true} # enable controller constrained decoding hints (regex/structured outputs/stop tags when backend supports them)
export CONTROLLER_MAX_NEW_TOKENS=${CONTROLLER_MAX_NEW_TOKENS:-1024} # max tokens for decomposer/selector outputs; structured plans should stay short
export DECOMPOSER_MAX_NEW_TOKENS=${DECOMPOSER_MAX_NEW_TOKENS:-1024} # decomposer plans can legitimately be longer because they carry node fields
export SELECTOR_MAX_NEW_TOKENS=${SELECTOR_MAX_NEW_TOKENS:-512} # selector should stay compact: one worker assignment per node
export WORKER_MAX_NEW_TOKENS=${WORKER_MAX_NEW_TOKENS:-512} # max tokens for worker outputs
export ROLLOUT_TASK_BATCH_SIZE=${ROLLOUT_TASK_BATCH_SIZE:-16} # number of train tasks rolled out together
export CONTROLLER_BATCH_SIZE=${CONTROLLER_BATCH_SIZE:-16} # generation batch size for decomposer/selector calls
export WORKER_BATCH_SIZE=${WORKER_BATCH_SIZE:-16} # generation batch size for worker calls
export ROLLOUT_PROGRESS_EVERY=${ROLLOUT_PROGRESS_EVERY:-1} # print progress after every completed task
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4} # replay microbatch size during GRPO updates
export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4} # effective replay samples per optimizer step = TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS
export CONTROLLER_FORMAT_RETRY_PENALTY=${CONTROLLER_FORMAT_RETRY_PENALTY:-0.05} # subtract from controller reward/advantage when output needed repair
export CONTROLLER_FORMAT_FALLBACK_PENALTY=${CONTROLLER_FORMAT_FALLBACK_PENALTY:-0.25} # stronger subtract when parser fallback output was used
export FINAL_ANSWER_CORRECTNESS_REWARD_ONLY=${FINAL_ANSWER_CORRECTNESS_REWARD_ONLY:-false} # if true, worker/selection reward is exactly final-answer correctness; if false, keep the current blended reward

export VAL_NUM_DECOMPOSITIONS=${VAL_NUM_DECOMPOSITIONS:-1} # validation uses a single decomposition candidate per task
export VAL_NUM_SELECTIONS=${VAL_NUM_SELECTIONS:-1} # validation uses a single selector sample per task
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.0} # deterministic validation rollout
export VAL_CONTROLLER_TEMPERATURE=${VAL_CONTROLLER_TEMPERATURE:-$VAL_TEMPERATURE} # deterministic by default, but can be overridden separately for controllers
export VAL_WORKER_TEMPERATURE=${VAL_WORKER_TEMPERATURE:-$VAL_TEMPERATURE} # deterministic by default, but can be overridden separately for workers
export VAL_TOP_P=${VAL_TOP_P:-1.0} # deterministic validation rollout
export VAL_DECOMPOSER_MAX_NEW_TOKENS=${VAL_DECOMPOSER_MAX_NEW_TOKENS:-0} # 0 = reuse training decomposer setting
export VAL_SELECTOR_MAX_NEW_TOKENS=${VAL_SELECTOR_MAX_NEW_TOKENS:-0} # 0 = reuse training selector setting
export MAX_VAL_TASKS=${MAX_VAL_TASKS:-0} # 0 = load the full overall_math validation benchmark
export VAL_TASKS_PER_EPOCH=${VAL_TASKS_PER_EPOCH:-0} # 0 = validate on all loaded validation tasks each outer epoch
export VAL_ROLLOUT_TASK_BATCH_SIZE=${VAL_ROLLOUT_TASK_BATCH_SIZE:-32} # number of validation tasks rolled out together

export ROLLOUT_LOG_MODE=${ROLLOUT_LOG_MODE:-best} # best = keep only top-K rollout records instead of every rollout
export ROLLOUT_LOG_DETAIL=${ROLLOUT_LOG_DETAIL:-compact} # compact = smaller JSONL artifacts
export BEST_K=${BEST_K:-1} # keep only the single best decomposition/selection record per epoch
export CHECKPOINT_MODE=${CHECKPOINT_MODE:-all} # all = enable best/ checkpoints when eval is on; final = only final/
export PRUNE_STALE_POLICY_MODELS=${PRUNE_STALE_POLICY_MODELS:-true} # delete older local policy checkpoints after newer ones are produced
export SAVE_STEPS=${SAVE_STEPS:-100} # save checkpoint-N during GRPO so long runs visibly progress
export EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-100} # replay-val cadence during GRPO; full benchmark still runs only after the subset update finishes
export EPOCH_S3_SYNC=${EPOCH_S3_SYNC:-true} # upload completed epoch folders to S3 while the job is still running
export EPOCH_S3_SYNC_INTERVAL=${EPOCH_S3_SYNC_INTERVAL:-120} # seconds between checks for finished epoch folders
export PRUNE_UPLOADED_LOCAL_CHECKPOINTS=${PRUNE_UPLOADED_LOCAL_CHECKPOINTS:-true} # after epoch upload, delete local best/ and checkpoint-* dirs; keep final/ for continued training
export STRIP_LOCAL_MODELS_AFTER_SYNC=${STRIP_LOCAL_MODELS_AFTER_SYNC:-true} # after final S3 sync, remove local best/final/checkpoint-* before copying outputs home
export ENABLE_WANDB=${ENABLE_WANDB:-true} # log rollout, training, and validation metrics to W&B

SOURCE_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
exec bash "$SOURCE_DIR/hierarchical-rema-trainer.sh" "$@"

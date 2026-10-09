#!/bin/bash
#SBATCH --job-name=agent1-only
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=200gb
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:4,storage:local:200G
#SBATCH --verbose

set -euo pipefail

# Slurm runs a spool copy; resolve the common launcher from the submitted repo.
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
CURRICULUM_SCRIPT=${SCRIPT_DIR}/agent12-curriculum-trainer-multinode.sh
if [[ ! -f "$CURRICULUM_SCRIPT" ]]; then
    echo "ERROR: Submit this job from the ReMA repository root." >&2
    exit 1
fi
if [[ -z "${WORKER_MODEL_REMOTE:-}" ]]; then
    echo "ERROR: Set WORKER_MODEL_REMOTE to the trained Agent 2 workers/huggingface directory on S3." >&2
    exit 1
fi
export WORKER_MODEL_REMOTE

# Only the decomposer is an optimization target. Workers still generate and
# are checkpointed, but never receive an optimizer update in this run.
export TRAIN_DECOMPOSER=true
export TRAIN_AGENT_ROLES='[decomposer]'
export WORKER_BOOTSTRAP_STEPS=0
export DECOMPOSER_TRANSFER_STEPS=${DECOMPOSER_TRANSFER_STEPS:-200}
export JOINT_STEPS=0
# The fade counts completed updates; allow additional rollout attempts when
# gates skip batches, then continue question-only with workers still frozen.
export TOTAL_STEPS=${TOTAL_STEPS:-800}
if [[ ! "$DECOMPOSER_TRANSFER_STEPS" =~ ^[0-9]+$ || ! "$TOTAL_STEPS" =~ ^[0-9]+$ ]] || \
        (( DECOMPOSER_TRANSFER_STEPS <= 0 || TOTAL_STEPS <= 0 )); then
    echo "ERROR: DECOMPOSER_TRANSFER_STEPS and TOTAL_STEPS must be positive integers." >&2
    exit 1
fi

export TERMINAL_WORKER_AS_ANSWER=true
export NUM_WORKER_STAGES=4
export MAX_PLANNED_SUBTASKS=4
export WORKER_CHECKPOINT_ROLE=worker_stage_4
export PREFIX_PROBE_ENABLE=true
export C3_CONTINUATIONS_PER_ACTION=${C3_CONTINUATIONS_PER_ACTION:-4}
export ROLLOUT_N=${ROLLOUT_N:-32}
export PREFIX_PROBE_MAX_NEW_TOKENS=${PREFIX_PROBE_MAX_NEW_TOKENS:-256}
export WORKER_QUESTION_BOOTSTRAP_PROBABILITY=1.0
export WORKER_QUESTION_FINAL_PROBABILITY=1.0
export WORKER_QUESTION_EVAL_PROBABILITY=1.0
export WORKER_QUESTION_FADE_STEPS=0

# Reuse all cached teacher attempts, including incorrect ones. No Agent 0
# generation or Agent-2 pilot behavior is enabled by this wrapper.
export GENERATE_TEACHER_DATA=0
export ONLINE_TEACHER_GENERATION=false
export AGENT2_PILOT=false
export TEACHER_ASSISTED_VALIDATION=false
export AGENT12_RUN_NAME=${AGENT12_RUN_NAME:-agent1-only-${SLURM_JOB_ID}}

exec bash "$CURRICULUM_SCRIPT"

#!/bin/bash
#SBATCH --job-name=agent2-only
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=200gb
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:4,storage:local:200G
#SBATCH --verbose

set -euo pipefail

# Slurm executes a private spool copy of the submitted script, so BASH_SOURCE
# does not point at the repository once the job starts.
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
CURRICULUM_SCRIPT=${SCRIPT_DIR}/agent12-curriculum-trainer-multinode.sh

if [[ ! -f "${CURRICULUM_SCRIPT}" ]]; then
    echo "ERROR: ${CURRICULUM_SCRIPT} was not found." >&2
    echo "Submit this job from the ReMA repository root." >&2
    exit 1
fi

# Keep the complete run in worker bootstrap. Agent 1 generates plans from the
# Agent 0 attempts, while only the shared worker/final model is optimized.
export TRAIN_DECOMPOSER=false
export TRAIN_AGENT_ROLES='[worker_stage_1,worker_stage_2,worker_stage_3,worker_stage_4,worker_stage_5]'
export WORKER_BOOTSTRAP_STEPS=${WORKER_BOOTSTRAP_STEPS:-800}
export DECOMPOSER_TRANSFER_STEPS=0
export WORKER_QUESTION_FADE_STEPS=0
export JOINT_STEPS=0
export TOTAL_STEPS=${TOTAL_STEPS:-${WORKER_BOOTSTRAP_STEPS}}
export AGENT12_RUN_NAME=${AGENT12_RUN_NAME:-agent2-only-${SLURM_JOB_ID}}
export ONLINE_TEACHER_GENERATION=${ONLINE_TEACHER_GENERATION:-true}
export TEACHER_SHARD_QUESTIONS=${TEACHER_SHARD_QUESTIONS:-512}

exec bash "${CURRICULUM_SCRIPT}"

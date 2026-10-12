#!/bin/bash
#SBATCH --job-name=agent1-baseline
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=200gb
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:4,storage:local:200G
#SBATCH --verbose

set -euo pipefail

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
AGENT1_SCRIPT=${SCRIPT_DIR}/agent1-only-rl-trainer-multinode.sh
if [[ ! -f "$AGENT1_SCRIPT" ]]; then
    echo "ERROR: Submit this job from the ReMA repository root." >&2
    exit 1
fi
if [[ -n "${AGENT12_EXPERIMENT_OVERRIDES:-}" || -n "${EAC3_RUN_FILE:-}" ]]; then
    echo "ERROR: Unset AGENT12_EXPERIMENT_OVERRIDES and EAC3_RUN_FILE for the Agent-1 baseline." >&2
    exit 1
fi

# Reuse the EAC3 launcher so only the training objective and run identity differ.
# Keep probes for measurement, never for baseline reward or update selection.
export AGENT1_ELIGIBILITY_MODE=baseline
export AGENT12_RUN_NAME=${BASELINE_RUN_NAME:-agent1-baseline-${SLURM_JOB_ID}}
export AGENT12_WANDB_RUN_ID=${AGENT12_RUN_NAME}

exec bash "$AGENT1_SCRIPT"

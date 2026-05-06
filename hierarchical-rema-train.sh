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

SOURCE_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
exec bash "$SOURCE_DIR/hierarchical-rema-trainer.sh"

#!/bin/bash
#SBATCH --job-name=eac3-experiment
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=200gb
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:4,storage:local:200G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ $# != 1 || ! -f "$1" ]]; then
    echo "Usage: sbatch eac3-experiment-multinode.sh <generated-run.json>" >&2
    exit 1
fi
export EAC3_RUN_FILE
EAC3_RUN_FILE=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1")
environment_file=$(mktemp)
trap 'rm -f "$environment_file"' EXIT
# Generate quoted exports rather than source user-authored configuration as code.
python3 scripts/eac3_experiments.py environment --run "$EAC3_RUN_FILE" > "$environment_file"
source "$environment_file"
rm -f "$environment_file"
trap - EXIT
exec bash ./agent12-curriculum-trainer-multinode.sh

#!/bin/bash
#SBATCH --job-name=eac3-fresh-pair
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200gb
#SBATCH --partition=lem-gpu-short
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:hopper:4,storage:local:200G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ $# != 1 || ! -f "$1" ]]; then
    echo "Usage: sbatch eac3-fresh-pair-multinode.sh <generated-run.json>" >&2
    exit 1
fi
export EAC3_FRESH_RUN_FILE
EAC3_FRESH_RUN_FILE=$(realpath "$1")
export AGENT12_PINNED_ENV_FILE
AGENT12_PINNED_ENV_FILE=$(mktemp)
trap 'rm -f "$AGENT12_PINNED_ENV_FILE"' EXIT
python3 -B scripts/eac3_fresh_pair.py environment --run "$EAC3_FRESH_RUN_FILE" > "$AGENT12_PINNED_ENV_FILE"
source "$AGENT12_PINNED_ENV_FILE"
stage=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage"])' "$EAC3_FRESH_RUN_FILE")
export REMOTE_RUN="${AGENT12_S3_REMOTE%/}/${AGENT12_RUN_NAME}"
echo "Fresh pair: ${AGENT12_RUN_NAME}; gate=${PREFIX_PROBE_APPLY_TO_TRAINING}; balancing=${C3_BALANCE_WORKER_FUNCTIONS}"
echo "Final model export: ${REMOTE_RUN}/workers/huggingface or decomposer/huggingface"

if [[ "$stage" == workers ]]; then
    bash ./agent2-only-rl-trainer-multinode.sh
else
    bash ./agent1-only-rl-trainer-multinode.sh
fi

# The common launcher has exported the model. Keep the trajectories, manifest,
# teacher-shard hashes and completed-update counter alongside it for comparison.
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
srun --overlap --nodes=1 --ntasks=1 -w "${nodes[0]}" bash -lc '
    set -euo pipefail
    mkdir -p "$CHECKPOINT_ROOT/evaluations"
    cp "$EAC3_FRESH_RUN_FILE" "$CHECKPOINT_ROOT/evaluations/run.json"
    step=$(cat "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt")
    cp "$CHECKPOINT_ROOT/global_step_${step}/training_progress.json" "$CHECKPOINT_ROOT/evaluations/training_progress.json"
    cd "$JOB_TMP/agent12_data/cached_teacher_shards"
    sha256sum shard-*.parquet > "$CHECKPOINT_ROOT/evaluations/teacher-shards.sha256"
    cd "$JOB_TMP/overall_math"
    sha256sum test.parquet > "$CHECKPOINT_ROOT/evaluations/eval-data.sha256"
    rclone copy "$CHECKPOINT_ROOT/evaluations" "$REMOTE_RUN/evaluations" --s3-no-check-bucket
    python3 - "$EAC3_FRESH_RUN_FILE" "$CHECKPOINT_ROOT/evaluations/training_progress.json" <<"PY"
import json, sys
run, progress = (json.load(open(path)) for path in sys.argv[1:])
target = run["config"]["worker_updates" if run["stage"] == "workers" else "decomposer_updates"]
actual = progress["actor_update_steps"]
print(f"Completed actor updates: {actual}/{target}")
if actual != target:
    raise SystemExit("Update budget not met: do not treat this run as a matched-budget result.")
PY
'

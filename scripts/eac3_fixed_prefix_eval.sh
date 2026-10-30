#!/bin/bash
#SBATCH --job-name=eac3-prefix-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1,storage:local:80G

set -euo pipefail

# Submit from the repository root. Never reuse an inherited training JOB_TMP.
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
: "${SLURM_JOB_ID:?Submit this script with sbatch from the repository root}"
CONFIG=${1:-config/eac3-fixed-prefix-pilot.json}
cd "$REPO"
test -f scripts/eac3_fixed_prefix_eval.py
test -f "$CONFIG"
BASE="/mnt/lscratch/slurm/${SLURM_JOB_ID}/fixed-prefix-eval"
mkdir -p "$BASE"
WORK=$(mktemp -d "$BASE/run.XXXXXX")
OUT=${PREFIX_EVAL_OUTPUT:-$REPO/results/fixed-prefix-${SLURM_JOB_ID}}
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"  # Existing results are never silently overwritten.
OUT=$(cd "$OUT" && pwd)
cp "$CONFIG" "$OUT/config.json"

python3 - "$REPO" "$CONFIG" "$WORK" <<'PY'
import json, pathlib, re, sys
root, config, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
cfg = json.loads(config.read_text())
dataset = (root / cfg['dataset']).resolve()
if not dataset.is_relative_to(root.resolve()) or not dataset.is_file():
    raise ValueError('Dataset must be an existing file inside this repository')
models = cfg['models']
ids = [m['id'] for m in models]
if len(ids) < 2 or len(set(ids)) != len(ids) or any(not re.fullmatch(r'[a-zA-Z0-9_-]+', i) for i in ids):
    raise ValueError('Need at least two uniquely named models')
for model in models:
    if not re.fullmatch(r'[a-zA-Z0-9_-]+:[^\s]+', model['source']):
        raise ValueError('Model sources must be rclone remote paths, not placeholder or local paths')
(work / 'dataset.txt').write_text(str(dataset.relative_to(root.resolve())))
(work / 'models.tsv').write_text(''.join(f"{m['id']}\t{m['source']}\n" for m in models))
PY
DATASET=$(cat "$WORK/dataset.txt")
python3 -B scripts/eac3_fixed_prefix_eval.py check --dataset "$DATASET"

SIF=${PREFIX_EVAL_SIF:-$WORK/verl-rema-v3.sif}
if [[ ! -s "$SIF" ]]; then
    if [[ -n "${PREFIX_EVAL_SIF:-}" ]]; then
        echo "PREFIX_EVAL_SIF does not exist: $SIF" >&2
        exit 1
    fi
    rclone copyto "${SIF_REMOTE:-s3v2:s3min-tomasznaskret-1712063354/user/dmotyka/sif_images/verl-rema-v3.sif}" "$SIF"
fi

# Independent processes release each model's GPU memory before the next load.
container=(apptainer exec --nv --writable-tmpfs
    --bind "$REPO:/repo:ro" --bind "$WORK:/work" --bind "$OUT:/output"
    --env PYTHONPATH=/repo/src/verl --env PYTHONDONTWRITEBYTECODE=1
    "$SIF")
eval_py=(python3 -B -u /repo/scripts/eac3_fixed_prefix_eval.py)
"${container[@]}" "${eval_py[@]}" prepare --dataset "/repo/$DATASET" --output /output/prepared.json
run_dirs=()
while IFS=$'\t' read -r id source; do
    echo "Staging $id from $source"
    rclone copy "$source" "$WORK/models/$id" --stats=30s --stats-one-line
    "${container[@]}" python3 -B /repo/scripts/validate_hf_model_dir.py "/work/models/$id"
    "${container[@]}" "${eval_py[@]}" run --prepared /output/prepared.json \
        --model-id "$id" --model-source "$source" --model-dir "/work/models/$id" \
        --output "/output/$id" --max-new-tokens 1024 --max-model-len 8192 --batch-size 8
    "${container[@]}" "${eval_py[@]}" score --prepared /output/prepared.json --directory "/output/$id"
    run_dirs+=("/output/$id")
    if [[ -n "${PREFIX_EVAL_REMOTE:-}" ]]; then
        rclone copy "$OUT" "${PREFIX_EVAL_REMOTE%/}/job-${SLURM_JOB_ID}" --s3-no-check-bucket
    fi
done < "$WORK/models.tsv"
"${container[@]}" "${eval_py[@]}" report --prepared /output/prepared.json \
    --runs "${run_dirs[@]}" --output /output/comparison
if [[ -n "${PREFIX_EVAL_REMOTE:-}" ]]; then
    rclone copy "$OUT" "${PREFIX_EVAL_REMOTE%/}/job-${SLURM_JOB_ID}" --s3-no-check-bucket
fi
echo "Complete. Report: $OUT/comparison/report.md"
echo "Full inputs and paired outputs: $OUT/comparison/review.jsonl"

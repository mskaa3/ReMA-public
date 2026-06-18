#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${INTERVAL:-5}"
SIF="${1:-${SIF_LOCAL:-${SIF_PATH:-${SIF_REMOTE:-}}}}"

if [[ -z "${SIF}" ]]; then
  echo "Usage: $0 /path/to/container.sif" >&2
  echo "Or set one of: SIF_LOCAL, SIF_PATH, SIF_REMOTE" >&2
  exit 1
fi

if command -v apptainer >/dev/null 2>&1; then
  RUNTIME="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  RUNTIME="singularity"
else
  echo "Could not find apptainer or singularity on PATH." >&2
  exit 1
fi

while true; do
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') host=$(hostname) ====="
  "${RUNTIME}" exec --nv "${SIF}" nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits \
    | awk -F', ' '{printf "gpu=%s name=%s mem=%s/%s MiB util=%s%%\n", $1, $2, $3, $4, $5}'
  sleep "${INTERVAL}"
done

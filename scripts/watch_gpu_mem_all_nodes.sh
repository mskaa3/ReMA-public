#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${INTERVAL:-5}"
JOBID="${JOBID:-${SLURM_JOB_ID:-}}"

if [[ $# -gt 0 && "${1}" =~ ^[0-9]+$ ]]; then
  JOBID="$1"
  shift
fi

SIF="${1:-${SIF_LOCAL:-${SIF_PATH:-${SIF_REMOTE:-}}}}"

if [[ -z "${JOBID}" ]]; then
  echo "Usage: $0 <jobid> /path/to/container.sif" >&2
  echo "Or set JOBID / SLURM_JOB_ID." >&2
  exit 1
fi

if [[ -z "${SIF}" ]]; then
  echo "Usage: $0 <jobid> /path/to/container.sif" >&2
  echo "Or set one of: SIF_LOCAL, SIF_PATH, SIF_REMOTE." >&2
  exit 1
fi

if ! command -v srun >/dev/null 2>&1; then
  echo "Could not find srun on PATH. Run this on the Slurm cluster/login node." >&2
  exit 1
fi

NODELIST=""
if command -v squeue >/dev/null 2>&1; then
  NODELIST="$(squeue -j "${JOBID}" -h -o '%N' | head -n 1 || true)"
fi

NNODES=""
if [[ -n "${NODELIST}" && "${NODELIST}" != "(None)" ]] && command -v scontrol >/dev/null 2>&1; then
  NNODES="$(scontrol show hostnames "${NODELIST}" | wc -l | tr -d ' ')"
fi

if [[ -z "${NNODES}" || "${NNODES}" == "0" ]]; then
  NNODES="${SLURM_JOB_NUM_NODES:-1}"
fi

echo "Monitoring job ${JOBID} on ${NNODES} node(s), interval=${INTERVAL}s"
echo "Container: ${SIF}"
echo "Press Ctrl-C to stop."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCH_SCRIPT="${SCRIPT_DIR}/watch_gpu_mem.sh"

srun --jobid="${JOBID}" --overlap --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 \
  bash -lc "INTERVAL='${INTERVAL}' '${WATCH_SCRIPT}' '${SIF}'"

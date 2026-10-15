#!/usr/bin/env bash
# Run inside an interactive Slurm allocation:
#   bash scripts/recover_archived_model.sh 6099558 6099557
# Inspect the archive without merging or uploading:
#   bash scripts/recover_archived_model.sh --list 6099558 6099557
# Or source this file and call recover_model JOB_ID individually.
# Requires Bash, Apptainer, rclone, and the repository in ~/projects/ReMA-public.
# Optional path overrides: ARCHIVE_ROOT, RECOVERY_REPO, RECOVERY_SCRATCH.

recover_model() (
    set -euo pipefail
    local job="${1:-}" mode="${2:-recover}" step role source file name world rank remote hf_source=""
    local actor_dirs candidate
    [[ "$job" =~ ^[0-9]+$ ]] || { echo 'Expected a numeric failed job ID' >&2; exit 1; }
    local archive="${ARCHIVE_ROOT:-/lustre/tmp/slurm/finished_jobs}/$job"
    local repo="${RECOVERY_REPO:-$HOME/projects/ReMA-public}"
    local -a sources rank0 shards
    test -d "$archive" || { echo "Missing archive: $archive" >&2; exit 1; }
    # Node archives and checkpoint directories may be symlinks. Do not hide
    # traversal errors in process substitution and continue with a partial list.
    if ! actor_dirs=$(find -L "$archive" -type d -path '*/global_step_*/*/actor' \
        -print -prune | sort -u); then
        echo "Checkpoint search failed under $archive; check the find errors above (permissions or broken archive links)." >&2
        exit 1
    fi
    if [[ -z "$actor_dirs" ]]; then
        echo "No checkpoint actor directories found under $archive (including symlink targets)." >&2
        echo 'Expected: LOCAL/<node>/agent12/checkpoints/agent12/global_step_<step>/<role>/actor' >&2
        echo 'No step or role can be selected yet. Check for saved model shards with:' >&2
        printf '  find -L %q -name %q -print\n' "$archive" 'model_world_size_*_rank_*.pt' >&2
        echo 'If none are found, inspect the archive for broken links, a different layout, or missing checkpoint files.' >&2
        exit 1
    fi
    echo "Available checkpoints for job $job:"
    printf '%s\n' "$actor_dirs" \
        | sed -E 's@.*/(global_step_[0-9]+/[^/]+/actor)$@\1@' | sort -Vu
    if [[ "$mode" == list ]]; then
        echo 'Source directories (shard completeness is checked during recovery):'
        printf '%s\n' "$actor_dirs"
        exit 0
    fi
    read -r -p 'Checkpoint step (number): ' step
    read -r -p 'Role directory (e.g. worker_stage_4 or decomposer): ' role
    [[ "$step" =~ ^[0-9]+$ && "$role" =~ ^(decomposer|worker_stage_[0-9]+)$ ]] \
        || { echo 'Invalid step or role' >&2; exit 1; }
    sources=()
    while IFS= read -r candidate; do
        if [[ "$candidate" == */global_step_"${step}"/"${role}"/actor ]]; then
            sources+=("$candidate")
        fi
    done <<< "$actor_dirs"
    ((${#sources[@]})) || { echo "No checkpoint for step $step, role $role; choose a pair from the list above." >&2; exit 1; }
    : "${SLURM_JOB_ID:?Run inside your current interactive allocation}"
    local scratch="${RECOVERY_SCRATCH:-/mnt/lscratch/slurm/$SLURM_JOB_ID}/model-recovery"
    mkdir -p "$scratch"
    local work
    work=$(mktemp -d "$scratch/job-${job}-step-${step}-${role}.XXXXXX")
    local actor="$work/actor"
    mkdir -p "$actor/huggingface"
    echo "Recovery workspace: $work"
    shopt -s nullglob
    for source in "${sources[@]}"; do
        for file in "$source"/model_world_size_*_rank_*.pt; do
            test -s "$file" || { echo "Empty shard: $file" >&2; exit 1; }
            name=${file##*/}
            if [[ -e "$actor/$name" ]]; then
                cmp -s "$file" "$actor/$name" \
                    || { echo "Conflicting duplicate shard: $file" >&2; exit 1; }
            else
                cp "$file" "$actor/$name"
            fi
        done
        if [[ -z "$hf_source" && -s "$source/huggingface/config.json" ]]; then
            hf_source="$source/huggingface"
        fi
    done
    [[ -n "$hf_source" ]] || { echo 'Missing Hugging Face metadata' >&2; exit 1; }
    cp -a "$hf_source/." "$actor/huggingface/"
    rank0=("$actor"/model_world_size_*_rank_0.pt)
    ((${#rank0[@]} == 1)) || { echo 'Missing or ambiguous rank 0' >&2; exit 1; }
    name=${rank0[0]##*/}; world=${name#model_world_size_}; world=${world%_rank_0.pt}
    shards=("$actor"/model_world_size_*_rank_*.pt)
    ((${#shards[@]} == world)) || { echo "Incomplete/mixed checkpoint: ${#shards[@]}/$world shards" >&2; exit 1; }
    for ((rank=0; rank<world; rank++)); do
        test -s "$actor/model_world_size_${world}_rank_${rank}.pt" \
            || { echo "Missing rank $rank" >&2; exit 1; }
    done
    local sif
    sif=$(find -L "$archive" -type f -name verl-rema-v3.sif -print -quit)
    [[ -n "$sif" ]] || { echo 'Container image missing from archive' >&2; exit 1; }
    test -f "$repo/src/verl/scripts/model_merger.py"
    test -f "$repo/scripts/validate_hf_model_dir.py"
    echo "Merging $world ranks on CPU"
    apptainer exec --writable-tmpfs \
        --bind "$work:$work" --bind "$repo:/recovery-repo:ro" \
        "$sif" python3 /recovery-repo/src/verl/scripts/model_merger.py --local_dir "$actor"
    apptainer exec --writable-tmpfs \
        --bind "$work:$work" --bind "$repo:/recovery-repo:ro" \
        "$sif" python3 /recovery-repo/scripts/validate_hf_model_dir.py "$actor/huggingface"
    local export_name="$role"
    [[ "$role" == worker_stage_* ]] && export_name=workers
    remote="s3v2:s3min-tomasznaskret-1712063354/user/ajanz/agent12/recovered-${job}/snapshots/global_step_${step}/${export_name}/huggingface"
    rclone copy "$actor/huggingface" "$remote" --s3-no-check-bucket \
        --immutable --stats=30s --stats-one-line
    rclone check "$actor/huggingface" "$remote" --one-way --size-only
    echo "Hugging Face export uploaded (file sizes verified): $remote"
)

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -e
    recovery_mode=recover
    if [[ "${1:-}" == --list ]]; then
        recovery_mode=list
        shift
    fi
    if (($# == 0)); then
        echo "Usage: bash $0 [--list] FAILED_JOB_ID [FAILED_JOB_ID ...]" >&2
        exit 2
    fi
    recovery_status=0
    for failed_job in "$@"; do
        if [[ "$recovery_mode" == list ]]; then
            # Report all requested archives even if one has no checkpoints.
            recover_model "$failed_job" list || recovery_status=1
        else
            recover_model "$failed_job"
        fi
    done
    exit "$recovery_status"
fi

"""Reuse completed teacher shards without regenerating or filtering attempts."""

import os
from pathlib import Path
import runpy
import subprocess
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[4]
MERGER = ROOT / "scripts/merge_agent12_teacher_shards.py"
merge_shards = runpy.run_path(str(MERGER))["merge_shards"]
LAUNCHER = (ROOT / "agent12-curriculum-trainer-multinode.sh").read_text()


def write_shard(path, question="question", attempts=None):
    if attempts is None:
        attempts = ["correct attempt", "incorrect attempt", ""] + ["attempt"] * 13
    pd.DataFrame([{"question": question, "teacher_attempts": attempts}]).to_parquet(path)


@pytest.mark.parametrize("suffix", [".parquet", ".train.parquet"])
def test_merge_retains_all_attempts_and_available_questions(tmp_path, suffix, capsys):
    write_shard(tmp_path / f"shard-00000{suffix}", "first")
    write_shard(tmp_path / f"shard-00002{suffix}", "third")
    # In-progress candidate files are not completed teacher shards.
    (tmp_path / "shard-00001.candidates.parquet").write_text("not a completed shard")
    output = tmp_path / "merged.parquet"
    merge_shards(tmp_path, output, expected_attempts=16)
    frame = pd.read_parquet(output)
    assert frame["question"].tolist() == ["first", "third"]
    assert frame.iloc[0]["teacher_attempts"].tolist() == [
        "correct attempt", "incorrect attempt", "",
    ] + ["attempt"] * 13
    assert "full-dataset coverage has not been verified" in capsys.readouterr().out


def test_original_full_dataset_merge_still_checks_row_count(tmp_path):
    write_shard(tmp_path / "shard-00000.train.parquet")
    output = tmp_path / "merged.parquet"
    with pytest.raises(ValueError, match="expected 2"):
        merge_shards(tmp_path, output, expected_rows=2)
    assert not output.exists()
    merge_shards(tmp_path, output, expected_rows=1)
    assert len(pd.read_parquet(output)) == 1


@pytest.mark.parametrize("failure", [
    "no_shards", "empty_shard", "corrupt", "candidate_schema", "attempt_count",
    "attempt_string", "attempt_null", "duplicate_id", "different_columns",
])
def test_bad_shards_do_not_replace_existing_output(tmp_path, failure):
    path = tmp_path / "shard-00000.parquet"
    if failure != "no_shards":
        write_shard(path)
    if failure == "empty_shard":
        pd.DataFrame(columns=["question", "teacher_attempts"]).to_parquet(path)
    elif failure == "corrupt":
        path.write_text("broken parquet")
    elif failure == "candidate_schema":
        pd.DataFrame([{"question": "q", "responses": ["candidate"]}]).to_parquet(path)
    elif failure == "attempt_count":
        write_shard(path, attempts=["only one"])
    elif failure == "attempt_string":
        write_shard(path, attempts="x" * 16)
    elif failure == "attempt_null":
        write_shard(path, attempts=[None] * 16)
    elif failure == "duplicate_id":
        write_shard(tmp_path / "shard-00000.train.parquet")
    elif failure == "different_columns":
        pd.DataFrame([{"teacher_attempts": ["a"] * 16}]).to_parquet(
            tmp_path / "shard-00001.parquet"
        )
    output = tmp_path / "merged.parquet"
    output.write_bytes(b"previous valid cache")
    with pytest.raises((ValueError, OSError)):
        merge_shards(tmp_path, output, expected_attempts=16)
    assert output.read_bytes() == b"previous valid cache"


@pytest.mark.parametrize("mode", ["shards", "file", "download_failure", "empty_directory", "bad_count"])
def test_launcher_reuses_selected_cache_without_teacher_generation(tmp_path, mode):
    remote = tmp_path / "remote"
    remote.mkdir()
    if mode != "empty_directory":
        write_shard(remote / "shard-00000.parquet", attempts=(
            ["one"] if mode == "bad_count" else None
        ))
    if mode == "shards":
        write_shard(remote / "shard-00001.parquet", "second question")
    job = tmp_path / "job"
    output = job / "agent12_data/train.parquet"
    calls = tmp_path / "calls.txt"
    loader = LAUNCHER[
        LAUNCHER.index("reuse_teacher_data() {"):
        LAUNCHER.index("run_agent12_training() {")
    ]
    shell = r'''
set -euo pipefail
HEAD_NODE=head
COMMON_MOUNTS=(--bind /tmp:/tmp)
rclone() {
    echo "rclone:$*" >> "$CALLS"
    [[ "$FAIL_DOWNLOAD" != 1 ]] || return 7
    [[ "$1" == copy ]] || return 8
    shopt -s nullglob
    local files=("$2"/*.parquet)
    if ((${#files[@]})); then cp "${files[@]}" "$3/"; fi
}
export -f rclone
apptainer() {
    while [[ "$1" != python3 ]]; do shift; done
    shift
    [[ "$1" == /root/ReMA-public/scripts/merge_agent12_teacher_shards.py ]]
    shift
    "$PYTHON" "$MERGER" "$@"
}
srun() {
    echo "srun:$*" >> "$CALLS"
    while [[ "$1" != bash && "$1" != apptainer ]]; do shift; done
    if [[ "$1" == bash ]]; then bash -c "$3"; else "$@"; fi
}
download_remote_file_on_head() {
    echo "single-file:$1" >> "$CALLS"
    mkdir -p "$(dirname "$2")"
    cp "$1" "$2"
}
'''
    result = subprocess.run(["bash", "-c", shell + loader + "\nreuse_teacher_data"],
        capture_output=True, text=True, env={
            "PATH": os.environ["PATH"], "JOB_TMP": str(job), "SIF_NAME": "test.sif",
            "TEACHER_TRAIN_SHARDS_REMOTE": "" if mode == "file" else str(remote),
            "TEACHER_TRAIN_REMOTE": str(remote / "shard-00000.parquet"),
            "TEACHER_TRAIN_FILE": str(output), "TEACHER_ROLLOUT_N": "16",
            "CALLS": str(calls), "PYTHON": sys.executable, "MERGER": str(MERGER),
            "FAIL_DOWNLOAD": str(int(mode == "download_failure")),
        })
    successful = mode in ("shards", "file")
    assert (result.returncode == 0) == successful, result.stdout + result.stderr
    assert output.exists() == successful
    call_text = calls.read_text()
    if mode == "file":
        assert "single-file:" in call_text and "rclone:" not in call_text
    else:
        assert "single-file:" not in call_text
        assert "--nodes=1 --ntasks=1 -w head" in call_text
    if successful:
        frame = pd.read_parquet(output)
        assert len(frame) == (2 if mode == "shards" else 1)
        assert len(frame.iloc[0]["teacher_attempts"]) == 16

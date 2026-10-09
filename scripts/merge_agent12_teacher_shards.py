#!/usr/bin/env python3
"""Validate and merge chunked Agent 0 teacher data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd


def merge_shards(
    input_dir: Path,
    output_path: Path,
    expected_rows: int | None = None,
    expected_attempts: int | None = None,
) -> None:
    shard_paths = []
    shard_ids = set()
    for path in sorted(input_dir.glob("shard-*.parquet")):
        match = re.fullmatch(r"shard-(\d+)(?:\.train)?\.parquet", path.name)
        if match is None:
            continue
        shard_id = int(match.group(1))
        if shard_id in shard_ids:
            raise ValueError(f"Duplicate teacher shard ID {shard_id}: {path}")
        shard_ids.add(shard_id)
        shard_paths.append(path)
    if not shard_paths:
        raise ValueError(f"No completed teacher shards found in {input_dir}")
    if expected_attempts is not None and expected_attempts <= 0:
        raise ValueError("expected_attempts must be positive")

    frames = []
    for shard_path in shard_paths:
        frame = pd.read_parquet(shard_path)
        if frame.empty:
            raise ValueError(f"Teacher shard is empty: {shard_path}")
        if "teacher_attempts" not in frame.columns:
            raise ValueError(
                f"Teacher shard has no teacher_attempts column: {shard_path}"
            )
        if frames and set(frame.columns) != set(frames[0].columns):
            raise ValueError(f"Teacher shard columns do not match: {shard_path}")
        if expected_attempts is not None:
            for row, attempts in enumerate(frame["teacher_attempts"]):
                if not isinstance(attempts, (list, tuple, np.ndarray)) or len(attempts) != expected_attempts:
                    raise ValueError(
                        f"{shard_path.name} row {row}: expected {expected_attempts} teacher attempts"
                    )
                if not all(isinstance(attempt, str) for attempt in attempts):
                    raise ValueError(f"{shard_path.name} row {row}: teacher attempts must be strings")
        frames.append(frame)
        print(f"Validated {shard_path.name}: {len(frame)} rows")

    merged = pd.concat(frames, ignore_index=True)
    if expected_rows is not None and len(merged) != expected_rows:
        raise ValueError(
            f"Merged teacher data has {len(merged)} rows; "
            f"expected {expected_rows}"
        )
    if expected_rows is None:
        print("Using available shards only; full-dataset coverage has not been verified")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    merged.to_parquet(temporary_path, index=False)
    os.replace(temporary_path, output_path)
    print(
        f"Merged {len(shard_paths)} teacher shards and {len(merged)} rows "
        f"into {output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, help="Require full-dataset row count when known")
    parser.add_argument("--expected-attempts", type=int, help="Validate every row's teacher-attempt count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_shards(args.input_dir, args.output, args.expected_rows, args.expected_attempts)


if __name__ == "__main__":
    main()

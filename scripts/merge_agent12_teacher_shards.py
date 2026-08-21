#!/usr/bin/env python3
"""Validate and merge chunked Agent 0 teacher data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def merge_shards(input_dir: Path, output_path: Path, expected_rows: int) -> None:
    shard_paths = sorted(input_dir.glob("shard-*.train.parquet"))
    if not shard_paths:
        raise ValueError(f"No completed teacher shards found in {input_dir}")

    frames = []
    for shard_path in shard_paths:
        frame = pd.read_parquet(shard_path)
        if frame.empty:
            raise ValueError(f"Teacher shard is empty: {shard_path}")
        if "teacher_attempts" not in frame.columns:
            raise ValueError(
                f"Teacher shard has no teacher_attempts column: {shard_path}"
            )
        frames.append(frame)
        print(f"Validated {shard_path.name}: {len(frame)} rows")

    merged = pd.concat(frames, ignore_index=True)
    if len(merged) != expected_rows:
        raise ValueError(
            f"Merged teacher data has {len(merged)} rows; "
            f"expected {expected_rows}"
        )

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
    parser.add_argument("--expected-rows", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_shards(args.input_dir, args.output, args.expected_rows)


if __name__ == "__main__":
    main()

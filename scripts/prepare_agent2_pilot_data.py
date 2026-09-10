#!/usr/bin/env python3
"""Create a fixed pilot split from one cached teacher-attempt shard."""

import argparse
import json
from pathlib import Path

import pandas as pd


def prepare_pilot(input_path, output_dir, train_questions=128, val_questions=64, seed=42):
    frame = pd.read_parquet(input_path).reset_index(drop=True)
    required = {"question", "teacher_attempts", "reward_model", "data_source"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing pilot columns: {sorted(required - set(frame.columns))}")
    if train_questions <= 0 or val_questions <= 0:
        raise ValueError("Pilot train and validation counts must be positive")
    keys = frame["question"].map(lambda value: " ".join(str(value).split()))
    if keys.duplicated().any():
        raise ValueError("Teacher shard contains duplicate questions; split unique questions to avoid leakage")
    if len(frame) < train_questions + val_questions:
        raise ValueError("Teacher shard is too small for the requested disjoint pilot split")
    if any(isinstance(values, str) or not hasattr(values, '__iter__') for values in frame['teacher_attempts']):
        raise ValueError("teacher_attempts must contain lists of attempts")

    # Pick by fixed random order, never by a teacher's correctness or score.
    shuffled = frame.sample(frac=1, random_state=seed)
    attempts = shuffled["teacher_attempts"].map(
        lambda values: next((v for v in values if isinstance(v, str) and v.strip()), "")
    )
    val_indices = attempts[attempts != ""].index[:val_questions]
    if len(val_indices) < val_questions:
        raise ValueError("Not enough non-empty teacher attempts for paired held-out evaluation")
    validation = shuffled.loc[val_indices].copy()
    validation["teacher_attempt"] = attempts.loc[val_indices]
    if "subset" not in validation:
        validation["subset"] = validation["data_source"]
    # Both validation passes use these same rows; only prompt visibility changes.
    validation = validation.drop(columns=["teacher_attempts"])
    training = shuffled.drop(index=val_indices).iloc[:train_questions].copy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training.to_parquet(output_dir / "train.parquet", index=False)
    validation.to_parquet(output_dir / "val.parquet", index=False)
    manifest = {
        "input": str(input_path), "seed": seed,
        "train_questions": len(training), "validation_questions": len(validation),
        "teacher_selection": "first non-empty attempt, without correctness filtering",
        "validation_contexts": ["question_only", "teacher_assisted"],
    }
    (output_dir / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Pilot: {len(training)} training and {len(validation)} held-out questions in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-questions", type=int, default=128)
    parser.add_argument("--val-questions", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_pilot(args.input, args.output_dir, args.train_questions, args.val_questions, args.seed)

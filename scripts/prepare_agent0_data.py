#!/usr/bin/env python3
"""Build single-agent Agent 0 parquet files from the existing ReMA datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prompt.math.serial_solver import SERIAL_SOLVER_SYSTEM_PROMPT


def _question_from_row(row: pd.Series) -> str:
    question = row.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()

    prompt = row.get("prompt")
    if isinstance(prompt, (list, tuple)) and prompt:
        last_message = prompt[-1]
        if isinstance(last_message, dict) and isinstance(last_message.get("content"), str):
            return last_message["content"].strip()

    raise ValueError("Every Agent 0 row must contain a non-empty question or chat prompt")


def convert_parquet(input_path: Path, output_path: Path) -> None:
    frame = pd.read_parquet(input_path).copy()
    prompts = []
    questions = []

    for _, row in frame.iterrows():
        question = _question_from_row(row)
        questions.append(question)
        prompts.append(
            [
                {"role": "system", "content": SERIAL_SOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
        )

    frame["question"] = questions
    frame["prompt"] = prompts
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    print(f"Wrote {len(frame)} Agent 0 examples to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--val-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_parquet(args.train_input, args.output_dir / "train.parquet")
    convert_parquet(args.val_input, args.output_dir / "val.parquet")


if __name__ == "__main__":
    main()

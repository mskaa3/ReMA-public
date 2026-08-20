#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attach every Agent 0 generation slot to Agent 1/2 examples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def clean_teacher_response(response: object) -> str:
    if not isinstance(response, str):
        return ""
    cleaned = response
    for token in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def collect_teacher_attempts(responses: Iterable[object]) -> List[str]:
    """Return all generated attempts without correctness labels."""
    if isinstance(responses, str):
        responses = [responses]
    return [clean_teacher_response(response) for response in responses]


def build_teacher_dataset(
    input_path: Path,
    output_path: Path,
) -> None:
    frame = pd.read_parquet(input_path)
    selected_rows = []
    total_attempt_count = 0
    usable_attempt_count = 0
    rows_without_usable_attempts = 0

    for _, row in frame.iterrows():
        responses = row.get("responses")
        if responses is None:
            raise ValueError("Teacher candidate parquet has no responses column")
        attempts = collect_teacher_attempts(responses)
        selected = row.to_dict()
        selected.pop("responses", None)
        selected["teacher_attempts"] = attempts
        selected_rows.append(selected)
        total_attempt_count += len(attempts)
        usable_attempt_count += sum(bool(attempt) for attempt in attempts)
        rows_without_usable_attempts += int(not any(attempts))

    if not selected_rows:
        raise ValueError("Teacher candidate parquet contains no training rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_rows).to_parquet(output_path, index=False)
    print(
        f"Retained all {len(selected_rows)} examples and all "
        f"{total_attempt_count} Agent 0 attempts; {usable_attempt_count} are "
        f"non-empty and {rows_without_usable_attempts} examples have no "
        f"usable attempt. "
        f"Wrote {output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_teacher_dataset(args.input, args.output)


if __name__ == "__main__":
    main()

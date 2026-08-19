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

"""Attach scored Agent 0 traces to every Agent 1/2 training example."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import pandas as pd


ScoreFn = Callable[[str, str, str, object], float]


def clean_teacher_response(response: object) -> str:
    if not isinstance(response, str):
        return ""
    cleaned = response
    for token in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def select_first_correct_trace(
    responses: Iterable[object],
    *,
    data_source: str,
    ground_truth: str,
    extra_info: object,
    score_fn: ScoreFn,
) -> Tuple[Optional[str], Optional[int], float]:
    """Return the first positively scored trace and its candidate index."""
    best_score = 0.0
    if isinstance(responses, str):
        responses = [responses]
    for candidate_index, response in enumerate(responses):
        cleaned = clean_teacher_response(response)
        if not cleaned:
            continue
        try:
            score = float(
                score_fn(data_source, cleaned, ground_truth, extra_info)
            )
        except Exception as exc:
            print(
                f"Skipping ungradable teacher candidate {candidate_index}: "
                f"{type(exc).__name__}: {exc}"
            )
            score = 0.0
        best_score = max(best_score, score)
        if score > 0.0:
            return cleaned, candidate_index, score
    return None, None, best_score


def select_teacher_trace(
    responses: Iterable[object],
    *,
    data_source: str,
    ground_truth: str,
    extra_info: object,
    score_fn: ScoreFn,
) -> Tuple[Optional[str], Optional[int], float, bool]:
    """Prefer a correct trace, otherwise retain the first usable attempt."""
    fallback_trace = None
    fallback_index = None
    fallback_score = 0.0

    if isinstance(responses, str):
        responses = [responses]
    for candidate_index, response in enumerate(responses):
        cleaned = clean_teacher_response(response)
        if not cleaned:
            continue
        try:
            score = float(
                score_fn(data_source, cleaned, ground_truth, extra_info)
            )
        except Exception as exc:
            print(
                f"Keeping ungradable teacher candidate {candidate_index} as "
                f"an unsuccessful attempt: {type(exc).__name__}: {exc}"
            )
            score = 0.0

        if fallback_trace is None:
            fallback_trace = cleaned
            fallback_index = candidate_index
            fallback_score = score
        if score > 0.0:
            return cleaned, candidate_index, score, True

    return fallback_trace, fallback_index, fallback_score, False


def _ground_truth(row: pd.Series) -> str:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        return str(reward_model["ground_truth"])
    for key in ("ground_truth", "answer", "target"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise ValueError("Teacher candidate row has no ground truth")


def build_teacher_dataset(
    input_path: Path,
    output_path: Path,
    score_fn: ScoreFn,
) -> None:
    frame = pd.read_parquet(input_path)
    selected_rows = []
    correct_trace_count = 0

    for _, row in frame.iterrows():
        responses = row.get("responses")
        if responses is None:
            raise ValueError("Teacher candidate parquet has no responses column")
        data_source = str(row.get("data_source", "ReMA-math"))
        extra_info = row.get("extra_info", {})
        trace, candidate_index, score, is_correct = select_teacher_trace(
            responses,
            data_source=data_source,
            ground_truth=_ground_truth(row),
            extra_info=extra_info,
            score_fn=score_fn,
        )
        selected = row.to_dict()
        selected.pop("responses", None)
        selected["teacher_solution"] = trace or ""
        selected["teacher_candidate_index"] = (
            int(candidate_index) if candidate_index is not None else -1
        )
        selected["teacher_score"] = float(score)
        selected["teacher_solution_correct"] = bool(is_correct)
        selected_rows.append(selected)
        correct_trace_count += int(is_correct)

    if not selected_rows:
        raise ValueError("Teacher candidate parquet contains no training rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_rows).to_parquet(output_path, index=False)
    correct_coverage = correct_trace_count / max(len(frame), 1)
    print(
        f"Retained all {len(selected_rows)} teacher examples; "
        f"{correct_trace_count} have a correct Agent 0 trace "
        f"(correct_coverage={correct_coverage:.3f}). Wrote {output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    from verl.utils.reward_score import _default_compute_score

    args = parse_args()
    build_teacher_dataset(args.input, args.output, _default_compute_score)


if __name__ == "__main__":
    main()

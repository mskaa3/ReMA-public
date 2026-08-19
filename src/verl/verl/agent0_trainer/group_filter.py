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

"""Utilities for retaining informative GRPO rollout groups."""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class GroupFilterResult:
    """Classification of prompt groups by within-group reward variation."""

    mixed_uids: tuple[str, ...]
    all_zero_count: int
    all_one_count: int
    homogeneous_other_count: int
    total_count: int


def classify_rollout_groups(
    uids: Sequence[object],
    scores: Sequence[float],
    *,
    atol: float = 1e-8,
) -> GroupFilterResult:
    """Return groups with non-zero GRPO signal and homogeneous diagnostics."""

    if len(uids) != len(scores):
        raise ValueError(f"uids and scores differ in length: {len(uids)} != {len(scores)}")

    grouped_scores: dict[str, list[float]] = {}
    for uid, score in zip(uids, scores):
        grouped_scores.setdefault(str(uid), []).append(float(score))

    mixed_uids: list[str] = []
    all_zero_count = 0
    all_one_count = 0
    homogeneous_other_count = 0

    for uid, group_scores in grouped_scores.items():
        values = np.asarray(group_scores, dtype=np.float64)
        if np.max(values) - np.min(values) > atol:
            mixed_uids.append(uid)
        elif np.all(np.isclose(values, 0.0, atol=atol, rtol=0.0)):
            all_zero_count += 1
        elif np.all(np.isclose(values, 1.0, atol=atol, rtol=0.0)):
            all_one_count += 1
        else:
            homogeneous_other_count += 1

    return GroupFilterResult(
        mixed_uids=tuple(mixed_uids),
        all_zero_count=all_zero_count,
        all_one_count=all_one_count,
        homogeneous_other_count=homogeneous_other_count,
        total_count=len(grouped_scores),
    )


def indices_for_uids(uids: Sequence[object], selected_uids: Iterable[object]) -> list[int]:
    """Return trajectory indices belonging to selected prompt groups."""

    selected = {str(uid) for uid in selected_uids}
    return [index for index, uid in enumerate(uids) if str(uid) in selected]


def ordered_unique_uids(uids: Sequence[object]) -> list[str]:
    """Return prompt IDs in first-occurrence order."""

    return list(dict.fromkeys(str(uid) for uid in uids))


def usable_partial_prompt_count(available: int, target: int, minibatch: int) -> int:
    """Choose the largest optimizer-compatible partial prompt batch."""

    if target <= 0 or minibatch <= 0:
        raise ValueError("target and minibatch must be positive")
    return min(target, available - (available % minibatch))

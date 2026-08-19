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

import numpy as np
import pytest

from verl.agent0_trainer.group_filter import (
    classify_rollout_groups,
    indices_for_uids,
    ordered_unique_uids,
    usable_partial_prompt_count,
)


def test_classify_rollout_groups_keeps_only_varying_rewards():
    uids = np.asarray(["zero"] * 4 + ["mixed"] * 4 + ["one"] * 4, dtype=object)
    scores = [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1]

    result = classify_rollout_groups(uids, scores)

    assert result.mixed_uids == ("mixed",)
    assert result.all_zero_count == 1
    assert result.all_one_count == 1
    assert result.homogeneous_other_count == 0
    assert result.total_count == 3
    assert indices_for_uids(uids, result.mixed_uids) == [4, 5, 6, 7]


def test_homogeneous_shaped_score_is_not_mislabeled_as_binary():
    result = classify_rollout_groups(["a", "a"], [0.25, 0.25])

    assert result.mixed_uids == ()
    assert result.homogeneous_other_count == 1


def test_order_and_partial_batch_alignment():
    assert ordered_unique_uids(["b", "a", "b", "c"]) == ["b", "a", "c"]
    assert usable_partial_prompt_count(47, target=48, minibatch=12) == 36
    assert usable_partial_prompt_count(48, target=48, minibatch=12) == 48
    assert usable_partial_prompt_count(11, target=48, minibatch=12) == 0


def test_partial_batch_rejects_invalid_sizes():
    with pytest.raises(ValueError):
        usable_partial_prompt_count(10, target=0, minibatch=12)

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

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss


def _policy_loss(mode: str) -> torch.Tensor:
    old_log_prob = torch.zeros((2, 3))
    log_prob = torch.zeros((2, 3))
    advantages = torch.tensor([[1.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
    response_mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)

    loss, *_ = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=response_mask,
        cliprange=0.2,
        loss_agg_mode=mode,
    )
    return loss


def test_trajectory_aggregation_gives_responses_equal_weight():
    # Token aggregation weights the three-token response three times as much.
    assert _policy_loss('token').item() == pytest.approx(0.5)
    assert _policy_loss('trajectory').item() == pytest.approx(0.0)


def test_policy_loss_rejects_unknown_aggregation_mode():
    with pytest.raises(ValueError, match='Unsupported loss_agg_mode'):
        _policy_loss('turn')

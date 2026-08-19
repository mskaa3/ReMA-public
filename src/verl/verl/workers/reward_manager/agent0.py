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

"""Correctness reward manager for the standalone serial solver."""

import torch

from verl import DataProto
from verl.workers.reward_manager.naive import NaiveRewardManager


class Agent0RewardManager(NaiveRewardManager):
    """Expose raw sequence correctness as both token reward and ``acc``."""

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = super().__call__(data)
        data.batch["acc"] = reward_tensor.sum(dim=-1)
        return reward_tensor

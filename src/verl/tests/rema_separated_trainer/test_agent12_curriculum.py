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

from verl.rema_separated_trainer.ppo.multi_agent_rollout import (
    curriculum_context_is_visible,
)
from verl.rema_separated_trainer.ppo.ray_trainer import (
    compute_agent12_curriculum_state,
)


def test_curriculum_context_visibility_is_group_deterministic():
    values = {
        curriculum_context_is_visible(
            "shared-uid", 17, 0.5, salt="teacher_solution"
        )
        for _ in range(16)
    }
    assert len(values) == 1


def test_curriculum_context_visibility_respects_probability_bounds():
    assert not curriculum_context_is_visible("uid", 1, 0.0, salt="context")
    assert curriculum_context_is_visible("uid", 1, 1.0, salt="context")


def _state(step):
    return compute_agent12_curriculum_state(
        step,
        worker_bootstrap_steps=2,
        decomposer_transfer_steps=2,
        worker_question_fade_steps=2,
        worker_question_final_probability=0.0,
    )


def test_agent12_curriculum_phase_boundaries():
    assert _state(1).phase == "worker_bootstrap"
    assert _state(2).phase == "worker_bootstrap"
    assert _state(3).phase == "decomposer_transfer"
    assert _state(3).teacher_solution_probability == 1.0
    assert _state(4).teacher_solution_probability == 0.5
    assert _state(5).phase == "joint"
    assert _state(5).worker_question_probability == 1.0
    assert _state(7).worker_question_probability == 0.0

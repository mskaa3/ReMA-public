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

from verl.rema_separated_trainer.ppo.multi_agent_rollout import (
    curriculum_context_is_visible,
)
from verl.rema_separated_trainer.ppo.ray_trainer import (
    compute_agent12_curriculum_state,
    expand_agent12_teacher_attempt_batch,
    select_agent12_training_role,
)


def test_curriculum_context_visibility_is_group_deterministic():
    values = {
        curriculum_context_is_visible(
            "shared-uid", 17, 0.5, salt="teacher_attempt"
        )
        for _ in range(16)
    }
    assert len(values) == 1


def test_curriculum_context_visibility_respects_probability_bounds():
    assert not curriculum_context_is_visible("uid", 1, 0.0, salt="context")
    assert curriculum_context_is_visible("uid", 1, 1.0, salt="context")


def test_teacher_attempt_expansion_builds_one_group_per_attempt():
    expanded, base_batch_size = expand_agent12_teacher_attempt_batch(
        {
            "question": np.asarray(["q1", "q2"], dtype=object),
            "data_source": np.asarray(["d1", "d2"], dtype=object),
            "teacher_attempts": np.asarray(
                [["a1", "a2", "a3"], ["b1", "b2", "b3"]],
                dtype=object,
            ),
        },
        attempts_key="teacher_attempts",
        attempt_key="teacher_attempt",
        expected_attempts=3,
    )

    assert base_batch_size == 2
    assert expanded["question"].tolist() == ["q1"] * 3 + ["q2"] * 3
    assert expanded["teacher_attempt"].tolist() == [
        "a1", "a2", "a3", "b1", "b2", "b3"
    ]
    assert expanded["teacher_attempt_index"].tolist() == [0, 1, 2, 0, 1, 2]
    assert expanded["data_source"].tolist() == ["d1"] * 3 + ["d2"] * 3


def test_teacher_attempt_expansion_rejects_incomplete_attempt_sets():
    with pytest.raises(ValueError, match="expected 3"):
        expand_agent12_teacher_attempt_batch(
            {
                "question": np.asarray(["q1"], dtype=object),
                "teacher_attempts": np.asarray([["a1", "a2"]], dtype=object),
            },
            attempts_key="teacher_attempts",
            attempt_key="teacher_attempt",
            expected_attempts=3,
        )


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
    assert _state(3).teacher_attempt_probability == 1.0
    assert _state(4).teacher_attempt_probability == 0.5
    assert _state(5).phase == "joint"
    assert _state(5).worker_question_probability == 1.0
    assert _state(7).worker_question_probability == 0.0


def test_frozen_decomposer_is_never_selected_for_training():
    worker_roles = ["worker_stage_1", "worker_stage_2"]
    selected_roles = {
        select_agent12_training_role(
            _state(step),
            decomposer_role="decomposer",
            worker_roles=worker_roles,
            switch_freq=1,
            train_decomposer=False,
        )
        for step in range(1, 9)
    }

    assert selected_roles == set(worker_roles)


def test_trainable_decomposer_is_selected_during_transfer():
    assert select_agent12_training_role(
        _state(3),
        decomposer_role="decomposer",
        worker_roles=["worker_stage_1"],
        switch_freq=1,
        train_decomposer=True,
    ) == "decomposer"

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

from scripts.prepare_agent12_teacher_data import (
    clean_teacher_response,
    collect_teacher_attempts,
)


def test_collect_teacher_attempts_keeps_every_generated_slot():
    attempts = collect_teacher_attempts(
        ["wrong", "correct<|im_end|>", "", None, "another wrong attempt"]
    )

    assert attempts == ["wrong", "correct", "", "", "another wrong attempt"]


def test_collect_teacher_attempts_accepts_single_response():
    assert collect_teacher_attempts("one attempt") == ["one attempt"]


def test_clean_teacher_response_removes_generation_tokens():
    assert clean_teacher_response(" answer <|im_end|> ") == "answer"

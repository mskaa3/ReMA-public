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
    select_first_correct_trace,
)


def test_select_first_correct_trace_uses_first_positive_candidate():
    def score_fn(_source, response, _ground_truth, _extra_info):
        return float("correct" in response)

    trace, candidate_index, score = select_first_correct_trace(
        ["wrong", "correct<|im_end|>", "also correct"],
        data_source="ReMA-math",
        ground_truth="1",
        extra_info={},
        score_fn=score_fn,
    )

    assert trace == "correct"
    assert candidate_index == 1
    assert score == 1.0


def test_select_first_correct_trace_reports_failure():
    trace, candidate_index, score = select_first_correct_trace(
        ["", None, "wrong"],
        data_source="ReMA-math",
        ground_truth="1",
        extra_info={},
        score_fn=lambda *_args: 0.0,
    )

    assert trace is None
    assert candidate_index is None
    assert score == 0.0


def test_clean_teacher_response_removes_generation_tokens():
    assert clean_teacher_response(" answer <|im_end|> ") == "answer"

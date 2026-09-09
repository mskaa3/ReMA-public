from types import SimpleNamespace

import pytest
import torch

from verl.rema_separated_trainer.ppo.multi_agent_rollout import (
    MultiAgentRollout,
    _encode_latest_conversation,
    _parse_decomposer_decision,
)
from verl.rema_separated_trainer.ppo.ray_trainer import (
    build_trainable_rank_partitions,
    carry_forward_round_scores,
    compute_usable_filtered_prompt_count,
    compute_round_transition_metrics,
    compute_token_level_scores,
    extract_round_score_role_outputs,
    select_score_role_rewards,
)
from verl.workers.reward_manager.rema import _compute_turn_worker_metrics


class _CharacterTokenizer:
    eos_token_id = 3
    pad_token_id = 0
    truncation_side = "right"

    @staticmethod
    def apply_chat_template(messages, add_generation_prompt, tokenize):
        assert add_generation_prompt
        assert not tokenize
        content = "|".join(
            f"{message['role']}:{message['content']}"
            for message in messages
        )
        return content + "|assistant:"

    @staticmethod
    def encode(text, add_special_tokens):
        assert add_special_tokens
        return [ord(character) for character in text]


def test_validation_selects_each_dynamic_terminal_role_reward():
    rewards = {
        'worker_stage_1_turn_level_reward': torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
        ]),
        'worker_stage_2_turn_level_reward': torch.tensor([
            [5.0, 6.0],
            [7.0, 8.0],
        ]),
    }

    selected = select_score_role_rewards(
        rewards,
        'worker_stage_2',
        ['worker_stage_1', 'worker_stage_2'],
        ['worker_stage_1', 'worker_stage_2'],
    )

    torch.testing.assert_close(
        selected,
        torch.tensor([[1.0, 2.0], [7.0, 8.0]]),
    )


def test_filtered_prompt_fallback_keeps_complete_minibatches():
    assert compute_usable_filtered_prompt_count(14, 48, 12) == 12
    assert compute_usable_filtered_prompt_count(48, 48, 12) == 48
    assert compute_usable_filtered_prompt_count(7, 48, 12) == 0
    assert compute_usable_filtered_prompt_count(
        8,
        48,
        12,
        allow_sub_minibatch=True,
    ) == 8


def test_sparse_rank_partitions_put_trainable_sample_on_every_rank():
    partitions = build_trainable_rank_partitions(
        [8, 7, 6, 5, 4, 3, 2, 1],
        world_size=4,
        trainable_mask=[True, True, True, True, False, False, False, False],
    )

    assert partitions is not None
    assert sorted(idx for partition in partitions for idx in partition) == list(range(8))
    assert all(len(partition) == 2 for partition in partitions)
    assert all(any(idx < 4 for idx in partition) for partition in partitions)


def test_sparse_rank_partitions_reject_insufficient_rank_coverage():
    assert build_trainable_rank_partitions(
        [4, 3, 2, 1],
        world_size=4,
        trainable_mask=[True, True, True, False],
    ) is None


def test_latest_round_encoding_preserves_action_and_original_turn_index():
    conversation = [
        {"role": "system", "content": "role"},
        {"role": "user", "content": "a deliberately long latest prompt"},
        {"role": "assistant", "content": "answer"},
    ]

    (
        input_ids,
        labels,
        step_ids,
        num_gen_tokens,
        encoded_stop_reason,
        prompt_was_truncated,
        response_was_truncated,
    ) = _encode_latest_conversation(
        conversation,
        _CharacterTokenizer(),
        stop_reason="stop",
        turn_idx=2,
        max_prompt_length=24,
        max_total_length=64,
    )

    assert len(input_ids) == len(labels) == len(step_ids)
    assert prompt_was_truncated
    assert not response_was_truncated
    assert set(step_id for step_id in step_ids if step_id >= 0) == {2}
    assert labels[-1] == _CharacterTokenizer.eos_token_id
    assert num_gen_tokens == len("answer") + 1
    assert encoded_stop_reason == "stop"


def test_retokenized_response_truncation_does_not_add_eos_target():
    conversation = [
        {"role": "system", "content": "role"},
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": "long answer"},
    ]

    (
        _,
        labels,
        _,
        num_gen_tokens,
        encoded_stop_reason,
        _,
        response_was_truncated,
    ) = _encode_latest_conversation(
        conversation,
        _CharacterTokenizer(),
        stop_reason="stop",
        turn_idx=1,
        max_prompt_length=24,
        max_total_length=28,
    )

    assert response_was_truncated
    assert encoded_stop_reason == "length"
    assert labels[-1] == -100
    assert num_gen_tokens > 0


def test_decomposer_accept_requires_an_existing_candidate():
    assert _parse_decomposer_decision(
        "DECISION: ACCEPT\nREASONING: looks good\nPLAN:",
        has_candidate=True,
    ) == ("ACCEPT", "ACCEPT", True, False)
    assert _parse_decomposer_decision(
        "DECISION: ACCEPT\nREASONING: too early\nPLAN:",
        has_candidate=False,
    ) == ("ACCEPT", "REVISE", True, True)
    assert _parse_decomposer_decision(
        "REASONING: missing protocol\nPLAN:\n- S1: retry",
        has_candidate=True,
    ) == ("REVISE", "REVISE", False, False)


def test_derive_verify_routing_normalizes_and_fills_missing_verification():
    subtasks, stages = MultiAgentRollout._build_derive_verify_stages(
        [("S7", "Compute the key expression.")],
        ["worker_stage_1", "worker_stage_2", "worker_stage_3"],
        "general_math_worker",
    )

    assert subtasks[0] == ("S1", "Compute the key expression.")
    assert subtasks[1][0] == "S2"
    assert "Use the S1 result" in subtasks[1][1]
    assert stages == [
        ("worker_stage_1", "general_math_worker", [subtasks[0]]),
        ("worker_stage_2", "general_math_worker", [subtasks[1]]),
        ("worker_stage_3", "general_math_worker", []),
    ]


def test_strategy_and_checks_are_normalized_to_derive_verify_subtasks():
    subtasks = MultiAgentRollout._extract_subtasks(
        """REASONING:
Use modular structure.

STRATEGY:
Rewrite 99 as -1 modulo 100.

CHECKS:
Check the parity of the exponent and the requested residue range.""",
        max_subtasks=2,
    )

    assert subtasks == [
        ("S1", "Rewrite 99 as -1 modulo 100."),
        ("S2", "Check the parity of the exponent and the requested residue range."),
    ]


def test_derive_verify_routing_requires_three_stages():
    with pytest.raises(ValueError, match="exactly three worker stages"):
        MultiAgentRollout._build_derive_verify_stages(
            [("S1", "derive"), ("S2", "verify")],
            ["worker_stage_1", "worker_stage_2"],
            "general_math_worker",
        )


def test_sequential_plan_routing_executes_only_planned_worker_stages():
    subtasks, stages = MultiAgentRollout._build_sequential_plan_stages(
        [
            ("S7", "Derive the first equation."),
            ("S9", "Use S1 to simplify the equation."),
        ],
        [
            "worker_stage_1",
            "worker_stage_2",
            "worker_stage_3",
            "worker_stage_4",
            "worker_stage_5",
        ],
        "general_math_worker",
    )

    assert [subtask_id for subtask_id, _ in subtasks] == ["S1", "S2"]
    assert subtasks[0][1] == "Derive the first equation."
    assert subtasks[1][1] == "Use S1 to simplify the equation."
    assert [stage_role for stage_role, _, _ in stages] == [
        "worker_stage_1",
        "worker_stage_5",
    ]
    assert all(len(stage[2]) == 1 for stage in stages)
    assert stages[-1][2][0][0] == "S2"


def test_sequential_plan_routing_falls_back_to_one_worker_for_empty_plan():
    subtasks, stages = MultiAgentRollout._build_sequential_plan_stages(
        [],
        ["worker_stage_1", "worker_stage_2", "worker_stage_5"],
        "general_math_worker",
    )

    assert len(subtasks) == 1
    assert subtasks[0][0] == "S1"
    assert [stage_role for stage_role, _, _ in stages] == ["worker_stage_5"]


def test_terminal_worker_mode_maps_plan_directly_without_finalizer():
    subtasks, stages = MultiAgentRollout._build_sequential_plan_stages(
        [
            ("S7", "Derive an intermediate quantity."),
            ("S9", "Use S1 to compute the requested answer."),
            ("S12", "Check the result and return it."),
        ],
        [
            "worker_stage_1",
            "worker_stage_2",
            "worker_stage_3",
            "worker_stage_4",
        ],
        "general_math_worker",
        terminal_worker_as_answer=True,
    )

    assert [subtask_id for subtask_id, _ in subtasks] == ["S1", "S2", "S3"]
    assert [stage_role for stage_role, _, _ in stages] == [
        "worker_stage_1",
        "worker_stage_2",
        "worker_stage_3",
    ]
    assert [stage[2][0][0] for stage in stages] == ["S1", "S2", "S3"]


def test_terminal_worker_mode_uses_first_worker_for_single_subtask():
    _, stages = MultiAgentRollout._build_sequential_plan_stages(
        [("S1", "Solve the complete problem.")],
        ["worker_stage_1", "worker_stage_2"],
        "general_math_worker",
        terminal_worker_as_answer=True,
    )

    assert len(stages) == 1
    assert stages[0][0] == "worker_stage_1"
    assert stages[0][2][0][0] == "S1"


def test_terminal_worker_mode_scores_the_actual_last_worker_output():
    latest_outputs = ["W1 terminal answer", "W3 terminal answer"]
    histories = [
        [
            {"role": "worker_stage_1", "content": "W1 terminal answer"},
            {"role": "worker_stage_4", "content": ""},
        ],
        [
            {"role": "worker_stage_3", "content": "W3 terminal answer"},
            {"role": "worker_stage_4", "content": ""},
        ],
    ]

    selected = MultiAgentRollout._select_scoring_outputs(
        latest_outputs,
        histories,
        {
            "score_role": "worker_stage_4",
            "terminal_worker_as_answer": True,
        },
    )

    assert selected == latest_outputs


def test_previous_worker_context_contains_local_results_not_reasoning():
    completed_results = [(
        "worker_stage_1",
        "general_math_worker",
        "S1",
        "REASONING:\nlong derivation\n\nLOCAL_RESULT: \\boxed{89}",
    )]

    context = MultiAgentRollout._format_previous_local_results(completed_results)

    assert context == "PREVIOUS LOCAL RESULTS:\nS1 LOCAL_RESULT: \\boxed{89}"
    assert "long derivation" not in context


def test_local_result_falls_back_to_last_boxed_expression():
    output = "REASONING:\ncalculation\nTherefore \\boxed{(3, \\frac{pi}{2})}."

    assert MultiAgentRollout._extract_local_result(output) == (
        r"\boxed{(3, \frac{pi}{2})}"
    )


def test_deterministic_selector_assignment_is_complete_but_not_generated():
    output = MultiAgentRollout._format_deterministic_assignments(
        "general_math_worker"
    )

    assert output == (
        "ASSIGNMENTS:\n"
        "- S1 -> general_math_worker\n"
        "- S2 -> general_math_worker"
    )


def test_deterministic_selector_assignment_supports_four_subtasks():
    output = MultiAgentRollout._format_deterministic_assignments(
        "general_math_worker",
        [(f"S{idx}", f"task {idx}") for idx in range(1, 5)],
    )

    assert output.splitlines() == [
        "ASSIGNMENTS:",
        "- S1 -> general_math_worker",
        "- S2 -> general_math_worker",
        "- S3 -> general_math_worker",
        "- S4 -> general_math_worker",
    ]


def test_deterministic_selector_slot_has_no_trainable_tokens():
    rollout = object.__new__(MultiAgentRollout)
    rollout.config = SimpleNamespace(prompt_length=128, response_length=128)
    role = "selector"
    assignment = MultiAgentRollout._format_deterministic_assignments(
        "general_math_worker"
    )
    tensor_dict = rollout._build_tensor_dict(
        last_round_responses=[{role: assignment}],
        conversation_history={
            role: [[
                {"role": "system", "content": "selector"},
                {"role": "user", "content": "fixed routing"},
            ]],
        },
        tokenizers={role: _CharacterTokenizer()},
        num_gen_token_lst={role: [[0]]},
        stop_reason_lst={role: [["stop"]]},
        max_num_turns=1,
        finish_reason=[None],
        latest_round_only=True,
        last_round_executed=[{role: False}],
    )[role]

    assert torch.all(tensor_dict["labels"] == -100)
    assert torch.all(tensor_dict["step_ids"] == -100)
    assert tensor_dict["num_gen_tokens"].sum().item() == 0


def test_carried_final_answer_has_no_trainable_tokens():
    rollout = object.__new__(MultiAgentRollout)
    rollout.config = SimpleNamespace(prompt_length=64, response_length=64)
    role = "worker_stage_2"
    tensor_dict = rollout._build_tensor_dict(
        last_round_responses=[{role: "final \\boxed{7}"}],
        conversation_history={
            role: [[
                {"role": "system", "content": "finalizer"},
                {"role": "user", "content": "candidate accepted"},
            ]],
        },
        tokenizers={role: _CharacterTokenizer()},
        num_gen_token_lst={role: [[0, 0]]},
        stop_reason_lst={role: [["stop", "stop"]]},
        max_num_turns=3,
        finish_reason=["decomposer_accept"],
        latest_round_only=True,
        last_round_executed=[{role: False}],
    )

    assert torch.all(tensor_dict[role]["labels"] == -100)
    assert torch.all(tensor_dict[role]["step_ids"] == -100)
    assert tensor_dict[role]["num_gen_tokens"].sum().item() == 0
    assert tensor_dict[role]["turn_finished"].item() == 5


def test_c3_branch_action_remains_trainable_after_later_accept():
    rollout = object.__new__(MultiAgentRollout)
    rollout.config = SimpleNamespace(prompt_length=64, response_length=64)
    role = "decomposer"
    branch_chat = [
        {"role": "system", "content": "decomposer"},
        {"role": "user", "content": "review"},
    ]
    tensor_dict = rollout._build_tensor_dict(
        last_round_responses=[{role: "DECISION: ACCEPT"}],
        conversation_history={role: [[*branch_chat]]},
        tokenizers={role: _CharacterTokenizer()},
        num_gen_token_lst={role: [[10, 10]]},
        stop_reason_lst={role: [["stop", "stop"]]},
        max_num_turns=3,
        finish_reason=["decomposer_accept"],
        latest_round_only=True,
        c3_focal_role=role,
        c3_action_records=[{
            "role": role,
            "turn_idx": 1,
            "chat": branch_chat,
            "output": "DECISION: ACCEPT",
            "num_gen_tokens": 10,
            "stop_reason": "stop",
            "token_ids": [1, 2],
        }],
        last_round_executed=[{role: True}],
    )

    assert torch.any(tensor_dict[role]["labels"] != -100)
    assert set(
        tensor_dict[role]["step_ids"][0][
            tensor_dict[role]["step_ids"][0] >= 0
        ].tolist()
    ) == {1}


def test_round_feedback_preserves_worker_reasoning_and_local_result():
    rollout = object.__new__(MultiAgentRollout)
    long_reasoning = "derive " + ("middle " * 80) + "detect the sign error"
    feedback = rollout._format_hierarchical_feedback(
        plan="PLAN:\n- S1: derive",
        assignments="ASSIGNMENTS:\n- S1 -> algebra_worker",
        worker_results={
            "worker_stage_1": (
                f"REASONING:\n{long_reasoning}\n\n"
                "LOCAL_RESULT: \\boxed{-7}"
            ),
            "worker_stage_2": "",
        },
        last_worker_output="Unable to synthesize.",
        worker_roles=["worker_stage_1", "worker_stage_2"],
        worker_reasoning_max_chars=120,
    )

    assert "REASONING:" in feedback
    assert "derive" in feedback
    assert "detect the sign error" in feedback
    assert "...[middle omitted]..." in feedback
    assert "LOCAL_RESULT:\n\\boxed{-7}" in feedback


def test_final_context_uses_parsed_plan_and_worker_results_without_question():
    rollout = object.__new__(MultiAgentRollout)
    original_question = "This full original question must stay hidden."

    assert rollout._format_final_question_block(
        original_question,
        "plan_and_worker_results",
    ) == ""
    assert rollout._format_final_question_block(
        original_question,
        "worker_results_only",
    ) == ""
    assert original_question in rollout._format_final_question_block(
        original_question,
        "full_question",
    )

    context = rollout._format_plan_and_worker_results_for_final(
        [("S1", "Compute the intermediate value."), ("S2", "Check S1.")],
        [
            (
                "worker_stage_1",
                "algebra_worker",
                "S1",
                "REASONING: compute\nLOCAL_RESULT: \\boxed{7}",
            ),
        ],
    )

    assert "PLAN:" in context
    assert "- S1: Compute the intermediate value." in context
    assert "WORK SO FAR:\nS1:\nREASONING: compute" in context
    assert "\\boxed{7}" in context
    assert original_question not in context


def test_round_feedback_preserves_final_diagnosis_and_answer_separately():
    rollout = object.__new__(MultiAgentRollout)
    feedback = rollout._format_hierarchical_feedback(
        plan="PLAN:\n- S1: derive",
        assignments="ASSIGNMENTS:\n- S1 -> algebra_worker",
        worker_results={"worker_stage_1": "", "worker_stage_2": ""},
        last_worker_output=(
            "The S1 and S2 results conflict. S2 must be recomputed before "
            "the answer can be trusted. Final answer: \\boxed{7}"
        ),
        worker_roles=["worker_stage_1", "worker_stage_2"],
        final_reasoning_max_chars=500,
    )

    assert "PREVIOUS FINAL DIAGNOSIS:" in feedback
    assert "S1 and S2 results conflict" in feedback
    assert "S2 must be recomputed" in feedback
    assert "PREVIOUS FINAL ANSWER:\n\\boxed{7}" in feedback


def test_turn_score_assignment_does_not_stop_at_missing_earlier_turns():
    data = SimpleNamespace(
        meta_info={"max_num_turns": 3},
        batch={
            "input_ids": torch.zeros((1, 8), dtype=torch.long),
            "step_ids": torch.tensor(
                [[-100, -100, -100, -100, 2, 2, 2, -100]]
            ),
            "turn_level_return": torch.tensor([[0.1, 0.2, 0.7]]),
        },
    )

    scores = compute_token_level_scores(data)

    assert scores[0, 6].item() == pytest.approx(0.7)
    assert torch.count_nonzero(scores).item() == 1


def test_duplicate_worker_results_are_not_counted_as_diversity_or_dependency():
    turn_history = [
        {
            "role": "worker_stage_1",
            "content": "REASONING: derive it\nLOCAL_RESULT: \\boxed{7}",
            "assigned_subtasks": ["S1"],
        },
        {
            "role": "worker_stage_2",
            "content": "REASONING: derive it again\nLOCAL_RESULT: \\boxed{7}",
            "assigned_subtasks": ["S2"],
        },
    ]

    metrics = _compute_turn_worker_metrics(
        turn_history,
        {"worker_stage_1", "worker_stage_2"},
    )

    assert metrics["unique_local_result_rate"] == 0.0
    assert metrics["dependency_usage_rate"] == 0.0
    assert metrics["duplicate_result_count"] == 1
    assert metrics["distinct_worker_result_gate"] == 0.0


def test_round_score_outputs_preserve_stopped_samples():
    agent_roles = ["decomposer", "selector", "worker_stage_1", "worker_stage_2"]
    histories = [
        [
            {"role": "decomposer", "content": "round 1 plan"},
            {"role": "selector", "content": "round 1 assignments"},
            {"role": "worker_stage_1", "content": "round 1 result"},
            {"role": "worker_stage_2", "content": "round 1 final"},
            {"role": "decomposer", "content": "round 2 plan"},
            {"role": "selector", "content": "round 2 assignments"},
            {"role": "worker_stage_1", "content": "round 2 result"},
            {"role": "worker_stage_2", "content": "round 2 final"},
        ],
        [
            {"role": "decomposer", "content": "short plan"},
            {"role": "selector", "content": "short assignments"},
            {"role": "worker_stage_1", "content": "short result"},
            {"role": "worker_stage_2", "content": "stable final"},
        ],
    ]

    outputs, executed = extract_round_score_role_outputs(
        histories,
        num_turns=[2, 1],
        agent_roles=agent_roles,
        score_role="worker_stage_2",
        max_num_turns=3,
    )

    assert outputs == [
        ["round 1 final", "stable final"],
        ["round 2 final", ""],
        ["", ""],
    ]
    assert executed.tolist() == [[True, True, False], [True, False, False]]

    candidate_scores = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    state_scores = carry_forward_round_scores(candidate_scores, executed)
    assert state_scores.tolist() == [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]


def test_carried_score_role_slot_is_not_an_executed_answer_attempt():
    agent_roles = ["decomposer", "selector", "worker_stage_1", "worker_stage_2"]
    histories = [[
        {"role": "decomposer", "content": "DECISION: REVISE"},
        {"role": "selector", "content": "assign"},
        {"role": "worker_stage_1", "content": "result"},
        {"role": "worker_stage_2", "content": "candidate", "executed": True},
        {"role": "decomposer", "content": "DECISION: ACCEPT"},
        {"role": "selector", "content": "", "executed": False},
        {"role": "worker_stage_1", "content": "", "executed": False},
        {
            "role": "worker_stage_2",
            "content": "candidate",
            "executed": False,
            "carried_forward": True,
        },
    ]]

    outputs, executed = extract_round_score_role_outputs(
        histories,
        num_turns=[2],
        agent_roles=agent_roles,
        score_role="worker_stage_2",
        max_num_turns=3,
    )

    assert outputs[0][0] == "candidate"
    assert outputs[1][0] == "candidate"
    assert executed.tolist() == [[True, False, False]]


def test_round_output_extraction_accepts_per_sample_terminal_roles():
    agent_roles = [
        "decomposer",
        "selector",
        "worker_stage_1",
        "worker_stage_2",
        "worker_stage_3",
    ]
    histories = [
        [
            {"role": "decomposer", "content": "plan"},
            {"role": "selector", "content": "assign"},
            {"role": "worker_stage_1", "content": "answer from W1"},
            {"role": "worker_stage_2", "content": "", "executed": False},
            {"role": "worker_stage_3", "content": "", "executed": False},
        ],
        [
            {"role": "decomposer", "content": "plan"},
            {"role": "selector", "content": "assign"},
            {"role": "worker_stage_1", "content": "intermediate"},
            {"role": "worker_stage_2", "content": "answer from W2"},
            {"role": "worker_stage_3", "content": "", "executed": False},
        ],
    ]

    outputs, executed = extract_round_score_role_outputs(
        histories,
        [1, 1],
        agent_roles,
        score_role="worker_stage_3",
        max_num_turns=1,
        score_roles=["worker_stage_1", "worker_stage_2"],
    )

    assert outputs[0] == ["answer from W1", "answer from W2"]
    assert executed[:, 0].tolist() == [True, True]


def test_round_transition_metrics_only_use_executed_next_rounds():
    state_scores = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # regression in round 2
            [0.0, 1.0, 1.0],  # repair in round 2, then stops
            [1.0, 1.0, 0.0],  # regression in round 3
            [1.0, 1.0, 1.0],  # stops after round 1; never eligible
        ]
    )
    executed = torch.tensor(
        [
            [True, True, True],
            [True, True, False],
            [True, True, True],
            [True, False, False],
        ]
    )

    metrics = compute_round_transition_metrics(state_scores, executed)

    assert metrics[
        "val/transitions/round_1_to_2/correct_to_wrong_rate"
    ] == pytest.approx(0.5)
    assert metrics[
        "val/transitions/round_1_to_2/wrong_to_correct_rate"
    ] == pytest.approx(1.0)
    assert metrics[
        "val/transitions/round_2_to_3/correct_to_wrong_rate"
    ] == pytest.approx(1.0)
    assert metrics["val/transitions/correct_to_wrong_count"] == 2.0
    assert metrics["val/transitions/wrong_to_correct_count"] == 1.0
    assert metrics["val/transitions/previous_correct_count"] == 3.0
    assert metrics["val/transitions/previous_wrong_count"] == 2.0

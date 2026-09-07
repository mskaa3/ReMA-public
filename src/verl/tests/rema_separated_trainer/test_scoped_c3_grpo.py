from types import SimpleNamespace

import torch

from verl.rema_separated_trainer.ppo.multi_agent_rollout import (
    MultiAgentRollout,
    _select_group_generation_sources,
)
from verl.rema_separated_trainer.ppo.scoped_c3_grpo import (
    estimate_scoped_c3_grpo,
)
from verl.rema_separated_trainer.ppo.prefix_probe import (
    apply_prefix_probe_gate,
    collect_prefix_probe_requests,
)


def test_group_generation_uses_one_source_for_coupled_uid():
    generation_indices, source_by_sample = _select_group_generation_sources(
        [0, 1, 2, 3, 4],
        ["a", "a", "b", "b", "c"],
        {"a", "b"},
    )

    assert generation_indices == [0, 2, 4]
    assert source_by_sample == {0: 0, 1: 0, 2: 2, 3: 2, 4: 4}


def test_group_generation_keeps_branched_uid_independent():
    generation_indices, source_by_sample = _select_group_generation_sources(
        [0, 1, 2, 3],
        ["a", "a", "b", "b"],
        {"b"},
    )

    assert generation_indices == [0, 1, 2]
    assert source_by_sample == {0: 0, 1: 1, 2: 2, 3: 2}


def test_prefix_probe_collects_plan_and_only_focal_nonterminal_worker():
    histories = [
        [
            {"role": "decomposer", "content": "plan-a", "executed": True},
            {"role": "worker_stage_1", "content": "local-a", "executed": True},
            {"role": "worker_stage_2", "content": "final-a", "executed": True},
        ],
        [
            {"role": "decomposer", "content": "plan-b", "executed": True},
            {"role": "worker_stage_1", "content": "final-b", "executed": True},
        ],
    ]

    requests = collect_prefix_probe_requests(
        histories,
        ["worker_stage_2", "worker_stage_1"],
        focal_role="worker_stage_1",
        decomposer_role="decomposer",
        stage_roles=["worker_stage_1", "worker_stage_2"],
    )

    assert [
        (request.sample_index, request.source_kind, request.message)
        for request in requests
    ] == [
        (0, "decomposer", "plan-a"),
        (0, "nonterminal_worker", "local-a"),
        (1, "decomposer", "plan-b"),
    ]


def test_prefix_probe_uses_latest_executed_message():
    requests = collect_prefix_probe_requests(
        [[
            {"role": "decomposer", "content": "old"},
            {"role": "decomposer", "content": "skipped", "executed": False},
            {"role": "decomposer", "content": "new", "executed": True},
        ]],
        ["worker_stage_1"],
        focal_role="decomposer",
        decomposer_role="decomposer",
        stage_roles=["worker_stage_1"],
    )

    assert len(requests) == 1
    assert requests[0].message == "new"


def test_prefix_probe_collects_only_workers_upstream_of_terminal():
    requests = collect_prefix_probe_requests(
        [[
            {"role": "decomposer", "content": "plan"},
            {"role": "worker_stage_1", "content": "candidate"},
            {"role": "worker_stage_2", "content": "check"},
            {"role": "worker_stage_3", "content": "final"},
        ]],
        ["worker_stage_3"],
        focal_role="worker_stage_3",
        decomposer_role="decomposer",
        stage_roles=[
            "worker_stage_1",
            "worker_stage_2",
            "worker_stage_3",
        ],
    )

    assert [
        (request.source_kind, request.source_role, request.message)
        for request in requests
    ] == [
        ("decomposer", "decomposer", "plan"),
        ("upstream_worker", "worker_stage_1", "candidate"),
        ("upstream_worker", "worker_stage_2", "check"),
    ]


def test_prefix_probe_gate_zeros_decomposer_leakage():
    gate = apply_prefix_probe_gate(
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [float("nan")] * 3,
        [0.0] * 3,
        ["worker_stage_2"] * 3,
        focal_role="decomposer",
        decomposer_role="decomposer",
        selector_role="selector",
        stage_roles=["worker_stage_1", "worker_stage_2"],
    )

    assert gate.outcome_scores == [1.0, 0.0, 0.0]
    assert gate.valid_mask == [True, True, True]


def test_prefix_probe_gate_masks_leaky_upstream_for_worker():
    gate = apply_prefix_probe_gate(
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        ["worker_stage_2"] * 3,
        focal_role="worker_stage_1",
        decomposer_role="decomposer",
        selector_role="selector",
        stage_roles=["worker_stage_1", "worker_stage_2"],
    )

    assert gate.outcome_scores == [1.0, 0.0, 0.0]
    assert gate.valid_mask == [True, False, True]


def test_prefix_probe_gate_exempts_terminal_worker_output():
    gate = apply_prefix_probe_gate(
        [1.0],
        [0.0],
        [float("nan")],
        [0.0],
        ["worker_stage_1"],
        focal_role="worker_stage_1",
        decomposer_role="decomposer",
        selector_role="selector",
        stage_roles=["worker_stage_1"],
    )

    assert gate.outcome_scores == [1.0]
    assert gate.valid_mask == [True]


def test_prefix_probe_gate_masks_leaky_worker_before_terminal():
    gate = apply_prefix_probe_gate(
        [1.0],
        [0.0],
        [float("nan")],
        [1.0],
        ["worker_stage_2"],
        focal_role="worker_stage_2",
        decomposer_role="decomposer",
        selector_role="selector",
        stage_roles=["worker_stage_1", "worker_stage_2"],
    )

    assert gate.outcome_scores == [0.0]
    assert gate.valid_mask == [False]


class _ByteTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    truncation_side = "left"

    @staticmethod
    def apply_chat_template(messages, add_generation_prompt, tokenize=False):
        del tokenize
        rendered = "".join(
            f"<{message['role']}>{message['content']}"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    @staticmethod
    def encode(text, add_special_tokens=True):
        del add_special_tokens
        return list(text.encode("utf-8"))


def test_tensor_builder_optimizes_branch_action_instead_of_latest_action():
    rollout = MultiAgentRollout.__new__(MultiAgentRollout)
    rollout.config = SimpleNamespace(prompt_length=256, response_length=64)
    branch_chat = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "round one"},
    ]
    latest_chat = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "round three"},
    ]
    branch_output = "branch action"
    tensor_dict = rollout._build_tensor_dict(
        last_round_responses=[{"decomposer": "latest action"}],
        conversation_history={"decomposer": [latest_chat]},
        tokenizers={"decomposer": _ByteTokenizer()},
        num_gen_token_lst={"decomposer": [[3, 3, 3]]},
        stop_reason_lst={"decomposer": [["stop", "stop", "stop"]]},
        max_num_turns=3,
        finish_reason=["reach_max_turn"],
        latest_round_only=True,
        c3_focal_role="decomposer",
        c3_action_records=[{
            "role": "decomposer",
            "turn_idx": 0,
            "chat": branch_chat,
            "output": branch_output,
            "num_gen_tokens": len(branch_output),
            "stop_reason": "stop",
            "token_ids": list(branch_output.encode("utf-8")),
            "assigned_subtasks": [],
        }],
    )["decomposer"]

    labels = tensor_dict["labels"][0]
    step_ids = tensor_dict["step_ids"][0]
    action_labels = labels[step_ids == 0]
    assert action_labels[:-1].tolist() == list(branch_output.encode("utf-8"))
    assert int(action_labels[-1].item()) == _ByteTokenizer.eos_token_id
    assert not bool((step_ids == 2).any().item())
    assert int(tensor_dict["num_gen_tokens"][0, 0].item()) > 0
    assert int(tensor_dict["num_gen_tokens"][0, 2].item()) == 0


def test_c3_uses_leave_one_out_outcome_credit():
    estimate = estimate_scoped_c3_grpo(
        torch.tensor([1.0, 0.0, 0.0]),
        ["problem", "problem", "problem"],
        torch.tensor([True, True, True]),
        normalize=False,
    )

    torch.testing.assert_close(
        estimate.advantage,
        torch.tensor([1.0, -0.5, -0.5]),
    )
    assert estimate.effective_mask.all()


def test_constant_and_invalid_groups_do_not_update_policy():
    estimate = estimate_scoped_c3_grpo(
        torch.tensor([1.0, 1.0, 1.0, 0.0]),
        ["constant", "constant", "constant", "single"],
        torch.tensor([True, True, True, True]),
    )

    assert not estimate.effective_mask.any()
    assert not estimate.advantage.any()


def test_invalid_candidate_is_excluded_from_loo_baseline():
    estimate = estimate_scoped_c3_grpo(
        torch.tensor([1.0, 0.0, 100.0]),
        ["problem", "problem", "problem"],
        torch.tensor([True, True, False]),
        normalize=False,
    )

    torch.testing.assert_close(
        estimate.advantage,
        torch.tensor([1.0, -1.0, 0.0]),
    )
    assert estimate.effective_mask.tolist() == [True, True, False]


def test_non_updatable_candidate_remains_a_causal_baseline_donor():
    estimate = estimate_scoped_c3_grpo(
        torch.tensor([1.0, 0.0, 0.0]),
        ["problem", "problem", "problem"],
        torch.tensor([True, True, True]),
        update_mask=torch.tensor([True, False, True]),
        normalize=False,
    )

    torch.testing.assert_close(
        estimate.advantage,
        torch.tensor([1.0, -0.5, -0.5]),
    )
    assert estimate.effective_mask.tolist() == [True, False, True]

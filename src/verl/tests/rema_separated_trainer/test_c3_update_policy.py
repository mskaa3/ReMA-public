"""Conservative worker updates without modifying factual C3 outcomes."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.rema_separated_trainer.ppo.ray_trainer import RayReMASeparatedTrainer
from verl.rema_separated_trainer.ppo.scoped_c3_grpo import estimate_scoped_c3_grpo


@pytest.mark.parametrize("normalize", [False, True])
def test_positive_only_keeps_signed_baseline_and_original_normalization(normalize):
    scores = torch.tensor([1., 0., 0.])
    valid = torch.ones(3, dtype=torch.bool)
    original = estimate_scoped_c3_grpo(scores, ["q"] * 3, valid, normalize=normalize)
    positive = estimate_scoped_c3_grpo(
        scores, ["q"] * 3, valid, positive_only_mask=valid,
        require_eligible_success=True, normalize=normalize,
    )
    torch.testing.assert_close(positive.advantage, original.advantage)
    assert positive.effective_mask.tolist() == [True, False, False]
    assert positive.eligible_success_group_mask.all()
    # The sole retained positive must not be re-centered to zero.
    assert positive.advantage[0] > 0
    assert scores.tolist() == [1., 0., 0.]


def test_success_rejected_by_gate_disables_negative_only_group():
    result = estimate_scoped_c3_grpo(
        torch.tensor([1., 0., 0.]), ["q"] * 3, torch.ones(3, dtype=torch.bool),
        update_mask=torch.tensor([False, True, True]),
        require_eligible_success=True, normalize=False,
    )
    torch.testing.assert_close(result.advantage, torch.tensor([1., -.5, -.5]))
    assert not result.effective_mask.any()
    assert not result.eligible_success_group_mask.any()


def test_rejected_success_remains_in_baseline_when_another_success_is_eligible():
    result = estimate_scoped_c3_grpo(
        torch.tensor([1., 1., 0.]), ["q"] * 3, torch.ones(3, dtype=torch.bool),
        update_mask=torch.tensor([True, False, True]),
        positive_only_mask=torch.ones(3, dtype=torch.bool),
        require_eligible_success=True, normalize=False,
    )
    torch.testing.assert_close(result.advantage, torch.tensor([.5, .5, -1.]))
    assert result.effective_mask.tolist() == [True, False, False]


@pytest.mark.parametrize("scores,valid", [
    ([1., 1., 1.], [True, True, True]),
    ([0., 0., 0.], [True, True, True]),
    ([1., 0., 0.], [False, True, True]),
])
def test_no_contrast_or_invalid_success_cannot_qualify_group(scores, valid):
    result = estimate_scoped_c3_grpo(
        torch.tensor(scores), ["q"] * 3, torch.tensor(valid),
        require_eligible_success=True,
    )
    assert not result.effective_mask.any()
    assert not result.eligible_success_group_mask.any()


def test_eligible_success_in_another_group_does_not_rescue_rejected_group():
    result = estimate_scoped_c3_grpo(
        torch.tensor([1., 0., 0., 1., 0., 0.]), ["a"] * 3 + ["b"] * 3,
        torch.ones(6, dtype=torch.bool),
        update_mask=torch.tensor([True, True, True, False, True, True]),
        require_eligible_success=True, normalize=False,
    )
    assert result.effective_mask.tolist() == [True] * 3 + [False] * 3
    assert result.eligible_success_group_mask.tolist() == [True] * 3 + [False] * 3
    torch.testing.assert_close(result.advantage, torch.tensor([1., -.5, -.5] * 2))


def _objects(values):
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def _trainer_and_batch(*, role="worker_stage_2", terminals=None, eligible=None):
    trainer = RayReMASeparatedTrainer.__new__(RayReMASeparatedTrainer)
    trainer.scoped_c3_grpo_enabled = True
    trainer.prefix_probe_enabled = True
    trainer.scoped_c3_grpo_config = {
        "branch_turn": 0, "normalize_advantages": False,
        "require_eligible_success": True, "positive_only_nonterminal_workers": True,
    }
    trainer._current_train_agent = role
    trainer.global_steps = 1
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(max_num_turns=1)),
    )
    trainer._get_hierarchy_config = lambda: {
        "stage_roles": [f"worker_stage_{i}" for i in range(1, 5)],
        "score_role": "worker_stage_4", "terminal_worker_as_answer": True,
    }
    if terminals is None:
        terminals = ["worker_stage_3"] * 3 + ["worker_stage_2"] * 3
    size = len(terminals)
    scores = torch.tensor([1., 0., 0.] * (size // 3))
    batch = DataProto.from_dict(
        tensors={
            "labels": torch.ones(size, 2, dtype=torch.long),
            "step_ids": torch.zeros(size, 2, dtype=torch.long),
            "token_level_rewards": torch.zeros(size, 2),
            "prefix_probe_gate_valid": torch.ones(size, dtype=torch.bool),
            "prefix_probe_collaboration_eligible": torch.tensor(
                eligible if eligible is not None else [True] * size,
            ),
        },
        non_tensors={
            "uid": _objects([f"q{i // 3}" for i in range(size)]),
            "terminal_stage_role": _objects(terminals),
            "c3_action_turn": _objects([0] * size),
            f"{role}_action_token_ids": _objects([[2]] * size),
            f"{role}_conversation_history": _objects([
                [{"role": "user", "content": f"shared prefix {i // 3}"},
                 {"role": "assistant", "content": f"action {i}"}]
                for i in range(size)
            ]),
        },
    )
    return trainer, batch, scores


def test_trainer_uses_dynamic_terminal_role_and_masks_actor_tokens(capsys):
    trainer, batch, raw = _trainer_and_batch()
    trainer._attach_scoped_c3_grpo_signals(batch, {"acc": raw}, {})
    # Stage 2 is intermediate in q0 but terminal in q1, despite 4 stage slots.
    assert batch.batch["scoped_c3_positive_only_mask"].tolist() == [True] * 3 + [False] * 3
    assert batch.batch["scoped_c3_causal_valid"].all()
    assert batch.batch["scoped_c3_gate_update_mask"].all()
    expected = [True, False, False, True, True, True]
    assert batch.batch["scoped_c3_update_mask"].tolist() == expected
    metrics = {}
    trainer._compute_scoped_c3_grpo_advantage(batch, metrics)
    assert batch.batch["labels"].ne(-100).any(-1).tolist() == expected
    assert batch.batch["step_ids"].ne(-100).any(-1).tolist() == expected
    torch.testing.assert_close(
        batch.batch["advantages"][:, 0], torch.tensor([1., 0., 0., 1., -.5, -.5]),
    )
    torch.testing.assert_close(batch.batch["returns"], batch.batch["advantages"])
    torch.testing.assert_close(batch.batch["scoped_c3_outcome_score"], raw)
    torch.testing.assert_close(batch.batch["token_level_rewards"].sum(-1), raw)
    prefix = "reward/c3/roles/worker_stage_2"
    assert metrics[f"{prefix}/negative_after_gate_count"] == 4
    assert metrics[f"{prefix}/negative_after_policy_count"] == 2
    assert metrics[f"{prefix}/negative_policy_removed_count"] == 2
    assert metrics[f"{prefix}/positive_after_policy_count"] == 2
    assert "positive=2 negative=2" in capsys.readouterr().out


@pytest.mark.parametrize("agg_mode,clip_mode", [
    ("token", "token"), ("turn", "token"), ("trajectory", "token"),
    ("turn", "turn"), ("trajectory", "turn"),
])
@pytest.mark.parametrize("world_size", [2, 4])
def test_actor_loss_has_no_gradient_for_negative_nonterminal_actions(agg_mode, clip_mode, world_size):
    from verl.rema_separated_trainer.ppo.actor_batch import pad_scoped_actor_batch
    from verl.rema_trainer.ppo.core_algos import compute_policy_loss

    trainer, batch, raw = _trainer_and_batch()
    trainer._attach_scoped_c3_grpo_signals(batch, {"acc": raw}, {})
    trainer._compute_scoped_c3_grpo_advantage(batch, {})
    batch, padding = pad_scoped_actor_batch(batch, world_size)
    assert padding == (0 if world_size == 2 else 2)
    # Match dp_rema_actor's path, including the zero-transport-padding case.
    real_rows = ~batch.batch["actor_padding_mask"] & batch.batch["labels"].ne(-100).any(-1)
    log_probs = torch.zeros(len(batch), 2, requires_grad=True)
    loss = compute_policy_loss(
        old_log_prob=torch.zeros_like(log_probs)[real_rows], log_prob=log_probs[real_rows],
        advantages=batch.batch["advantages"][real_rows],
        eos_mask=batch.batch["labels"][real_rows].ne(-100),
        step_id=batch.batch["step_ids"][real_rows], cliprange=.2,
        agg_mode=agg_mode, clip_mode=clip_mode,
    )[0]
    grad = torch.autograd.grad(loss, log_probs)[0]
    assert torch.isfinite(grad).all()
    assert not grad[~real_rows].any()
    assert (grad[[0, 3]] < 0).all()  # Descent reinforces eligible successes.
    assert (grad[4:6] > 0).all()  # Terminal failures retain negative credit.


@pytest.mark.parametrize("terminal", ["worker_stage_2", "worker_stage_3"])
def test_early_filter_and_actor_both_reject_group_with_only_masked_success(terminal):
    trainer, batch, raw = _trainer_and_batch(
        terminals=[terminal] * 3, eligible=[False, True, True],
    )
    metrics = {}
    trainer._attach_scoped_c3_grpo_signals(batch, {"acc": raw}, metrics)
    assert batch.batch["scoped_c3_causal_valid"].all()
    # fit() retains a UID only if this early mask has an eligible action.
    assert not batch.batch["scoped_c3_update_mask"].any()
    prefix = "reward/c3/roles/worker_stage_2"
    assert metrics[f"{prefix}/no_eligible_success_group_count"] == 1
    trainer._compute_scoped_c3_grpo_advantage(batch, metrics)
    assert (batch.batch["labels"] == -100).all()
    assert not batch.batch["advantages"].any()
    assert not batch.batch["token_level_rewards"].any()
    assert metrics[f"{prefix}/negative_after_gate_count"] == 2
    assert metrics[f"{prefix}/negative_after_policy_count"] == 0
    assert metrics[f"{prefix}/positive_removed_count"] == 1
    torch.testing.assert_close(batch.batch["scoped_c3_outcome_score"], raw)


def test_decomposer_is_not_treated_as_a_positive_only_worker():
    trainer, batch, raw = _trainer_and_batch(role="decomposer")
    trainer._attach_scoped_c3_grpo_signals(batch, {"acc": raw}, {})
    assert not batch.batch["scoped_c3_positive_only_mask"].any()
    assert batch.batch["scoped_c3_update_mask"].all()


@pytest.mark.parametrize("role,expected", [
    ("worker_stage_2", [True, False, False] * 2),
    ("worker_stage_4", [True] * 6),
])
def test_fixed_finalizer_uses_configured_score_role(role, expected):
    trainer, batch, raw = _trainer_and_batch(role=role)
    hierarchy = trainer._get_hierarchy_config()
    hierarchy["terminal_worker_as_answer"] = False
    trainer._get_hierarchy_config = lambda: hierarchy
    del batch.non_tensor_batch["terminal_stage_role"]
    trainer._attach_scoped_c3_grpo_signals(batch, {"acc": raw}, {})
    assert batch.batch["scoped_c3_update_mask"].tolist() == expected


@pytest.mark.parametrize("bad_metadata", [None, ["unknown"] * 6])
def test_dynamic_worker_policy_rejects_missing_or_invalid_terminal_metadata(bad_metadata):
    trainer, batch, raw = _trainer_and_batch()
    if bad_metadata is None:
        del batch.non_tensor_batch["terminal_stage_role"]
    else:
        batch.non_tensor_batch["terminal_stage_role"] = _objects(bad_metadata)
    with pytest.raises(ValueError, match="terminal_stage_role"):
        trainer._attach_scoped_c3_grpo_signals(batch, {"acc": raw}, {})


def test_disabling_new_policy_restores_signed_updates_and_old_group_selection():
    trainer, batch, raw = _trainer_and_batch(eligible=[False, True, True] * 2)
    trainer.scoped_c3_grpo_config.update(
        require_eligible_success=False, positive_only_nonterminal_workers=False,
    )
    trainer._attach_scoped_c3_grpo_signals(batch, {"acc": raw}, {})
    assert batch.batch["scoped_c3_update_mask"].tolist() == [False, True, True] * 2
    trainer._compute_scoped_c3_grpo_advantage(batch, {})
    torch.testing.assert_close(
        batch.batch["advantages"][:, 0], torch.tensor([0., -.5, -.5] * 2),
    )

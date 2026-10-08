"""Repeated-suffix Monte Carlo credit, without duplicate actor updates."""

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from verl import DataProto
from verl.rema_separated_trainer.ppo.ray_trainer import RayReMASeparatedTrainer

from verl.rema_separated_trainer.ppo.scoped_c3_grpo import (
    aggregate_scoped_c3_actions,
    estimate_scoped_c3_grpo,
)


def _aggregate(scores, *, gates=None, valid=None, groups=None, tokens=None, suffixes=None):
    n = len(scores)
    return aggregate_scoped_c3_actions(
        torch.tensor(scores, dtype=torch.float32), groups or ["q"] * n,
        [i // 4 for i in range(n)], suffixes or list(range(4)) * (n // 4),
        tokens or [[i // 4 + 100] for i in range(n)],
        torch.tensor(valid if valid is not None else [True] * n),
        torch.tensor(gates if gates is not None else [True] * n),
        continuations_per_action=4, positive_only_mask=torch.ones(n, dtype=torch.bool),
    )


def _estimate(result, *, normalize=False):
    return estimate_scoped_c3_grpo(
        result.outcome_scores, ["q"] * len(result.outcome_scores), result.valid_mask,
        update_mask=result.update_mask, positive_only_mask=torch.ones_like(result.valid_mask),
        require_eligible_success=True, normalize=normalize,
    )


def test_eight_actions_four_suffixes_have_eight_baseline_entries_not_32():
    successes = [4, 3, 2, 1, 0, 4, 3, 1]
    scores = [score for count in successes for score in [1.] * count + [0.] * (4 - count)]
    result = _aggregate(scores)
    means = torch.tensor(successes) / 4
    representatives = torch.arange(0, 32, 4)
    assert result.valid_mask.sum() == 8
    torch.testing.assert_close(result.outcome_scores[representatives], means)
    for normalize in (False, True):
        actual = _estimate(result, normalize=normalize)
        expected = estimate_scoped_c3_grpo(
            means, ["q"] * 8, torch.ones(8, dtype=torch.bool),
            positive_only_mask=torch.ones(8, dtype=torch.bool),
            require_eligible_success=True, normalize=normalize,
        )
        torch.testing.assert_close(actual.advantage[representatives], expected.advantage)
        assert torch.equal(actual.effective_mask[representatives], expected.effective_mask)
        assert not actual.effective_mask[~result.representative_mask].any()
        assert not actual.advantage[~result.representative_mask].any()


def test_lucky_single_success_is_not_credited_above_reliable_alternative():
    result = _aggregate([1., 0., 0., 0., 1., 1., 1., 1.])
    estimate = _estimate(result)
    assert estimate.advantage[0] == -.75
    assert estimate.advantage[4] == .75
    assert estimate.effective_mask.tolist() == [False] * 4 + [True, False, False, False]


def test_equal_action_means_have_no_signal_despite_mixed_raw_outcomes():
    result = _aggregate([1., 1., 0., 0., 0., 0., 1., 1.])
    assert not _estimate(result).effective_mask.any()


def test_gate_veto_does_not_cherry_pick_outcomes_or_remove_baseline_donor():
    result = _aggregate([1., 1., 0., 0., 1., 1., 1., 1.],
                        gates=[True, True, False, True] + [True] * 4)
    assert result.outcome_scores[0] == .5
    assert result.valid_mask[0]
    assert not result.update_mask[0]
    assert result.gate_disagreement_mask[0]
    assert _estimate(result).advantage[4] == .5


def test_incomplete_causal_action_is_not_estimated_from_surviving_suffixes():
    result = _aggregate([1., 0., 0., 0., 1., 1., 1., 1.],
                        valid=[True, False, True, True] + [True] * 4)
    assert not result.valid_mask[:4].any()
    assert not _estimate(result).effective_mask.any()


@pytest.mark.parametrize("kwargs,match", [
    ({"tokens": [[1], [2], [1], [1]]}, "exact sampled focal action"),
    ({"suffixes": [0, 0, 2, 3]}, "every continuation exactly once"),
    ({"groups": ["q", "q", "q", "other"]}, "every continuation exactly once"),
])
def test_aggregation_rejects_corrupted_replica_metadata(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _aggregate([1., 0., 1., 0.], **kwargs)


def test_replica_aggregation_handles_object_arrays_and_permuted_rows():
    order = [7, 2, 4, 0, 6, 1, 5, 3]
    scores = torch.tensor([1., 0., 0., 0., 1., 1., 1., 1.])[order]
    result = aggregate_scoped_c3_actions(
        scores, np.array(["q"] * 8, dtype=object),
        np.array([i // 4 for i in order], dtype=object),
        np.array([i % 4 for i in order], dtype=object),
        [[i // 4 + 100] for i in order], torch.ones(8, dtype=torch.bool),
        torch.ones(8, dtype=torch.bool), continuations_per_action=4,
        positive_only_mask=torch.ones(8, dtype=torch.bool),
    )
    assert result.representative_mask.nonzero().flatten().tolist() == [2, 3]
    assert result.outcome_scores[2] == 1
    assert result.outcome_scores[3] == .25


def test_single_continuation_is_backward_compatible():
    scores = torch.tensor([1., 0., 0.])
    result = aggregate_scoped_c3_actions(
        scores, ["q"] * 3, [0, 1, 2], [0] * 3, [[1], [2], [3]],
        torch.ones(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool),
        continuations_per_action=1, positive_only_mask=torch.ones(3, dtype=torch.bool),
    )
    torch.testing.assert_close(result.outcome_scores, scores)
    assert result.valid_mask.all() and result.update_mask.all()


def _objects(values):
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


@pytest.mark.parametrize("terminal", ["worker_stage_2", "worker_stage_3"])
@pytest.mark.parametrize("duplicate_actions", [False, True])
def test_trainer_masks_copies_after_mean_credit_and_preserves_raw_scores(terminal, duplicate_actions, tmp_path):
    from verl.rema_separated_trainer.ppo.credit_audit import CreditAudit

    trainer = RayReMASeparatedTrainer.__new__(RayReMASeparatedTrainer)
    trainer._credit_audit = CreditAudit(tmp_path / 'credit_audit')
    trainer.scoped_c3_grpo_enabled = True
    trainer.prefix_probe_enabled = True
    trainer.scoped_c3_grpo_config = {
        "branch_turn": 0, "continuations_per_action": 4, "normalize_advantages": False,
        "positive_only_nonterminal_workers": True, "require_eligible_success": True,
    }
    trainer._current_train_agent = "worker_stage_2"
    trainer.global_steps = 1
    trainer.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(
        rollout=SimpleNamespace(max_num_turns=1),
    ))
    trainer._get_hierarchy_config = lambda: {
        "stage_roles": [f"worker_stage_{i}" for i in range(1, 5)],
        "terminal_worker_as_answer": True,
    }
    counts = [4, 3, 2, 1, 0, 4, 3, 1]
    # A terminal action has no sampled suffix: its copied outcomes are identical.
    if terminal == "worker_stage_2":
        counts = [4, 0, 0, 4, 4, 0, 0, 0]
    raw = torch.tensor([float(m < count) for count in counts for m in range(4)])
    batch = DataProto.from_dict(
        tensors={
            "labels": torch.ones(32, 2, dtype=torch.long),
            "step_ids": torch.zeros(32, 2, dtype=torch.long),
            "token_level_rewards": torch.zeros(32, 2),
            "prefix_probe_gate_valid": torch.ones(32, dtype=torch.bool),
            "prefix_probe_collaboration_eligible": torch.ones(32, dtype=torch.bool),
        },
        non_tensors={
            "uid": _objects(["q"] * 32),
            "terminal_stage_role": _objects([terminal] * 32),
            "c3_action_turn": _objects([0] * 32),
            "c3_action_index": _objects([i // 4 for i in range(32)]),
            "c3_suffix_index": _objects([i % 4 for i in range(32)]),
            "worker_stage_2_action_token_ids": _objects([
                [100 if duplicate_actions else i // 4 + 100] for i in range(32)
            ]),
            "worker_stage_2_conversation_history": _objects([
                [{"role": "user", "content": "shared prefix"},
                 {"role": "assistant", "content": f"action {i // 4}"}]
                for i in range(32)
            ]),
        },
    )
    metrics = {}
    trainer._attach_scoped_c3_grpo_signals(batch, {"acc": raw}, metrics)
    torch.testing.assert_close(batch.batch["scoped_c3_raw_outcome_score"], raw)
    assert batch.batch["scoped_c3_causal_valid"].sum() == 8
    torch.testing.assert_close(batch.batch["scoped_c3_outcome_score"][::4], torch.tensor(counts) / 4)
    expected = estimate_scoped_c3_grpo(
        torch.tensor(counts) / 4, ["q"] * 8, torch.ones(8, dtype=torch.bool),
        normalize=False, require_eligible_success=True,
        positive_only_mask=torch.full((8,), terminal != "worker_stage_2"),
    )
    trainer._compute_scoped_c3_grpo_advantage(batch, metrics)
    active = batch.batch["labels"].ne(-100).any(-1)
    assert torch.equal(active[::4], expected.effective_mask)
    copies = torch.arange(32).remainder(4).ne(0)
    assert not active[copies].any()
    assert not batch.batch["advantages"][copies].any()
    torch.testing.assert_close(batch.batch["advantages"][::4, 0],
                               expected.advantage * expected.effective_mask)
    # Zero gradient for every repeated copy, and the same gradient as 8 action rows.
    log_probs = torch.zeros(32, 2, requires_grad=True)
    loss = -(log_probs[active] * batch.batch["advantages"][active]).mean()
    gradient = torch.autograd.grad(loss, log_probs)[0]
    assert not gradient[copies].any()
    assert not gradient[~active].any()
    prefix = "reward/c3/roles/worker_stage_2"
    assert metrics[f"{prefix}/action_count"] == 8
    assert metrics[f"{prefix}/continuation_row_count"] == 32
    assert metrics[f"{prefix}/unique_action_rate"] == (1 / 8 if duplicate_actions else 1)
    audit_metrics = trainer._credit_audit.finish(batch, actor_updated=True, actor_update_step=1)
    assert audit_metrics['train/credit/generated_actions'] == 8
    assert audit_metrics['train/credit/used_positive_actions'] == int(
        (expected.effective_mask & (expected.advantage > 0)).sum()
    )

    # Replay exposes action identity/means even when probe diagnostics are disabled.
    import json

    trainer.prefix_probe_enabled = False
    trainer.config.trainer = SimpleNamespace(default_local_dir=str(tmp_path))
    trainer._get_score_role = lambda: terminal
    trainer._get_rollout_agent_roles = lambda: [f"worker_stage_{i}" for i in range(1, 5)]
    batch.batch[f"{terminal}_turn_level_reward"] = raw[:, None]
    for key, values in {
        "question": ["question"] * 32,
        "reward_model": [{"ground_truth": "1"}] * 32,
        "history": [[] for _ in range(32)],
        "response": [str(score.item()) for score in raw],
        "finish_reason": ["final_boxed_answer"] * 32,
    }.items():
        batch.non_tensor_batch[key] = _objects(values)
    trainer._save_train_generations(batch)
    saved = json.loads((tmp_path / "replay_buffer/train_step_1.jsonl").read_text())
    assert saved["c3_action_index"] == [i // 4 for i in range(32)]
    assert saved["c3_suffix_index"] == list(range(4)) * 8
    assert saved["c3_continuations_per_action"] == 4
    assert saved["score"] == raw.tolist()
    assert saved["c3_action_mean_score"] == batch.batch["scoped_c3_outcome_score"].tolist()
    assert sum(saved["c3_action_representative"]) == 8
    assert saved["c3_action_update_eligible"] == active.tolist()


def test_no_terminal_status_mixing_inside_one_action():
    with pytest.raises(ValueError, match="terminal status"):
        aggregate_scoped_c3_actions(
            torch.tensor([1., 0., 1., 0.]), ["q"] * 4, [0] * 4, list(range(4)),
            [[1]] * 4, torch.ones(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool),
            continuations_per_action=4, positive_only_mask=torch.tensor([True, True, False, True]),
        )

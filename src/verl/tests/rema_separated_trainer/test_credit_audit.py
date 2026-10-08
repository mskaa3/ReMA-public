import json

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.rema_separated_trainer.ppo.actor_batch import pad_scoped_actor_batch
from verl.rema_separated_trainer.ppo.credit_audit import CreditAudit
from verl.rema_separated_trainer.ppo.ray_trainer import split_batch_for_agents
from verl.rema_separated_trainer.ppo.scoped_c3_grpo import (
    aggregate_scoped_c3_actions, estimate_scoped_c3_grpo,
)


def objects(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def sample(uid='q', *, gates=None, positive_only=True, scores=None):
    scores = torch.tensor(scores or [1.] * 4 + [0.] * 4)
    gates = torch.tensor(gates or [True] * 8)
    positive = torch.full((8,), positive_only)
    grouped = aggregate_scoped_c3_actions(
        scores, [uid] * 8, [0] * 4 + [1] * 4, list(range(4)) * 2,
        [[1]] * 4 + [[2]] * 4, torch.ones(8, dtype=torch.bool), gates,
        continuations_per_action=4, positive_only_mask=positive,
    )
    estimate = estimate_scoped_c3_grpo(
        grouped.outcome_scores, [uid] * 8, grouped.valid_mask,
        update_mask=grouped.update_mask, positive_only_mask=positive, normalize=False,
    )
    batch = DataProto.from_dict(tensors={
        'scoped_c3_raw_outcome_score': scores,
        'scoped_c3_outcome_score': grouped.outcome_scores,
        'scoped_c3_positive_only_mask': positive,
        'scoped_c3_action_representative': grouped.representative_mask,
        'prefix_probe_gate_valid': torch.ones(8, dtype=torch.bool),
        'prefix_probe_collaboration_eligible': gates,
        'labels': torch.ones(8, 3, dtype=torch.long),
        'step_ids': torch.zeros(8, 3, dtype=torch.long),
        'advantages': estimate.advantage[:, None].expand(-1, 3).clone(),
    }, non_tensors={
        'uid': objects([uid] * 8),
        'c3_action_index': objects([0] * 4 + [1] * 4),
        'c3_suffix_index': objects(list(range(4)) * 2),
        'question': objects(['Question'] * 8),
        'history': objects([[{'role': 'worker_stage_1', 'content': 'result'}]] * 8),
        'worker_stage_1_conversation_history': objects([
            [{'role': 'assistant', 'content': 'action', 'unused': np.float64('nan')}]
        ] * 8),
        'num_turns': objects([1] * 8),
    }, meta_info={'agent_roles': ['worker_stage_1']})
    batch.batch['labels'][~estimate.effective_mask] = -100
    batch.batch['step_ids'][~estimate.effective_mask] = -100
    batch.batch['advantages'][~estimate.effective_mask] = 0
    return batch, estimate


def events(path):
    return [json.loads(line) for line in (path / 'train_step_1.jsonl').read_text().splitlines()]


def test_tracks_one_action_not_four_suffixes_and_does_not_change_tensors(tmp_path):
    audit = CreditAudit(tmp_path)
    batch, estimate = sample()
    original = {k: v.clone() for k, v in batch.batch.items()}
    audit.capture(batch, estimate, step=1, role='worker_stage_1')
    metrics = audit.finish(batch, actor_updated=True, actor_update_step=1)
    generated = events(tmp_path)[:2]
    final = events(tmp_path)[2:]
    assert generated[0]['suffix_outcomes'] == [1.] * 4
    assert generated[0]['focal_messages'][0]['unused'] is None
    assert final[0]['positive_training_signal'] is True
    assert final[0]['final_trainable_tokens'] == 3
    assert final[1]['status'] == 'nonpositive_advantage_policy'
    assert metrics['train/credit/generated_actions'] == 2
    assert metrics['train/credit/used_positive_actions'] == 1
    assert metrics['train/credit/excluded_actions'] == 1
    assert not audit.pending
    for key, value in original.items():
        torch.testing.assert_close(batch.batch[key], value)


@pytest.mark.parametrize('reason', ['sparse_batch', 'critic_warmup', 'zero_trainable_batch'])
def test_positive_gate_and_advantage_do_not_imply_update(tmp_path, reason):
    batch, estimate = sample()
    audit = CreditAudit(tmp_path)
    audit.capture(batch, estimate, step=1, role='worker_stage_1')
    metrics = audit.finish(batch, actor_updated=False, actor_update_step=0, skip_reason=reason)
    event = events(tmp_path)[2]
    assert event['all_suffix_gates_pass'] and event['advantage'] > 0
    assert event['status'] == reason
    assert not event['positive_training_signal']
    assert metrics['train/credit/used_positive_actions'] == 0


def test_one_failed_suffix_vetoes_action(tmp_path):
    batch, estimate = sample(gates=[True, False, True, True] + [True] * 4)
    audit = CreditAudit(tmp_path)
    audit.capture(batch, estimate, step=1, role='worker_stage_1')
    audit.finish(batch, actor_updated=False, actor_update_step=0)
    event = events(tmp_path)[2]
    assert event['action_mean_score'] == 1
    assert event['status'] == 'suffix_gate_rejected_or_invalid'
    assert not event['positive_training_signal']


def test_equal_means_have_no_signal(tmp_path):
    batch, estimate = sample(scores=[1., 0., 0., 0.] * 2)
    audit = CreditAudit(tmp_path)
    audit.capture(batch, estimate, step=1, role='worker_stage_1')
    audit.finish(batch, actor_updated=False, actor_update_step=0)
    assert {e['status'] for e in events(tmp_path)[2:]} == {'no_effective_c3_contrast'}


def test_accumulates_generation_batches_and_tracks_dropped_actions(tmp_path):
    audit = CreditAudit(tmp_path)
    first, estimate = sample('first')
    audit.capture(first, estimate, step=1, role='worker_stage_1')
    second, estimate = sample('second')
    audit.capture(second, estimate, step=1, role='worker_stage_1')
    metrics = audit.finish(second, actor_updated=True, actor_update_step=1)
    final = {e['audit_id']: e for e in events(tmp_path) if e['event'] == 'disposition'}
    assert final['first:worker_stage_1:0']['status'] == 'not_selected_for_optimizer_batch'
    assert final['second:worker_stage_1:0']['positive_training_signal']
    assert metrics['train/credit/generated_actions'] == 4


def test_identity_survives_role_split_reorder_and_padding(tmp_path):
    batch, estimate = sample(positive_only=False)
    audit = CreditAudit(tmp_path)
    audit.capture(batch, estimate, step=1, role='worker_stage_1')
    split = split_batch_for_agents(batch)['worker_stage_1']
    assert split.non_tensor_batch['credit_audit_id'].dtype == object
    # A transport duplicate of an active action must not overwrite its real mask.
    split.reorder(torch.tensor([0, 4, 1, 2, 3, 5, 6, 7]))
    split = split[:3]
    padded, count = pad_scoped_actor_batch(split, 2)
    assert count == 1
    padded.reorder(torch.tensor([3, 2, 1, 0]))
    metrics = audit.finish(padded, actor_updated=True, actor_update_step=1)
    assert metrics['train/credit/used_positive_actions'] == 1
    assert metrics['train/credit/used_negative_actions'] == 1
    assert sum(e['final_trainable_tokens'] for e in events(tmp_path)[2:]) == 6


def test_actor_failure_is_unknown_not_claimed_excluded(tmp_path):
    batch, estimate = sample()
    audit = CreditAudit(tmp_path)
    audit.capture(batch, estimate, step=1, role='worker_stage_1')
    metrics = audit.finish(batch, actor_updated=False, actor_update_step=0,
                           skip_reason='actor_update_failed_or_partial')
    event = events(tmp_path)[2]
    assert event['positive_training_signal'] is None
    assert event['update_outcome_unknown']
    assert metrics['train/credit/unknown_update_actions'] == 1


def test_single_continuation_without_action_metadata(tmp_path):
    batch, estimate = sample()
    del batch.non_tensor_batch['c3_action_index']
    del batch.non_tensor_batch['c3_suffix_index']
    audit = CreditAudit(tmp_path)
    audit.capture(batch, estimate, step=1, role='worker_stage_1')
    assert [e['action_index'] for e in events(tmp_path)] == list(range(8))


def test_resume_appends_and_missing_disposition_remains_pending(tmp_path):
    audit = CreditAudit(tmp_path)
    batch, estimate = sample('old')
    audit.capture(batch, estimate, step=1, role='worker_stage_1')
    # Simulate a crash before finalization, then resume at the same rollout step.
    audit = CreditAudit(tmp_path)
    batch, estimate = sample('new')
    audit.capture(batch, estimate, step=1, role='worker_stage_1')
    audit.finish(batch, actor_updated=True, actor_update_step=1)
    assert len(events(tmp_path)) == 6
    assert all(e['event'] == 'generated' for e in events(tmp_path) if e['uid'] == 'old')

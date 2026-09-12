from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.rema_separated_trainer.ppo.actor_batch import (
    actor_loss_denominators, pad_scoped_actor_batch,
    padded_microbatch_loss_scale, without_actor_padding,
)
from verl.rema_separated_trainer.ppo.local_results import extract_local_result
from verl.rema_separated_trainer.ppo.multi_agent_rollout import MultiAgentRollout
from verl.rema_separated_trainer.ppo.prefix_probe import compare_worker_final_answers
from verl.rema_separated_trainer.ppo.ray_trainer import (
    RayReMASeparatedTrainer, build_trainable_rank_partitions,
)
from verl.rema_separated_trainer.ppo.scoped_c3_grpo import estimate_scoped_c3_grpo


def _batch(size=32, trainable=28):
    scores = torch.arange(size).remainder(2).float()
    uids = np.array([str(i // 16) for i in range(size)], dtype=object)
    estimate = estimate_scoped_c3_grpo(
        scores, uids, torch.ones(size, dtype=torch.bool),
        update_mask=torch.arange(size) < trainable,
    )
    labels = torch.ones(size, 4, dtype=torch.long)
    labels[~estimate.effective_mask] = -100
    return DataProto.from_dict(
        tensors={
            'input_ids': torch.arange(size * 4).reshape(size, 4),
            'attention_mask': torch.ones_like(labels),
            'position_ids': torch.arange(4).repeat(size, 1),
            'labels': labels,
            'step_ids': torch.zeros_like(labels),
            'advantages': estimate.advantage[:, None].repeat(1, 4),
            'returns': estimate.advantage[:, None].repeat(1, 4),
            'token_level_scores': scores[:, None].repeat(1, 4),
            'token_level_rewards': scores[:, None].repeat(1, 4),
        },
        non_tensors={'uid': uids},
    )


@pytest.mark.parametrize('size,trainable,padded_size', [(16,14,24), (32,28,36), (48,32,48), (80,51,84)])
def test_padding_preserves_real_c3_groups_and_enables_rank_partition(size, trainable, padded_size):
    batch = _batch(size, trainable)
    original_labels = batch.batch['labels'].clone()
    original_advantages = batch.batch['advantages'].clone()
    padded, count = pad_scoped_actor_batch(batch, 12)
    assert len(padded) == padded_size
    assert count == padded_size - size
    torch.testing.assert_close(batch.batch['labels'], original_labels)
    torch.testing.assert_close(padded.batch['advantages'][:size], original_advantages)
    assert padded.batch['labels'].ne(-100).any(-1).sum().item() == trainable
    if count:
        assert not padded.batch['advantages'][size:].any()
        assert (padded.batch['labels'][size:] == -100).all()
        assert (padded.batch['step_ids'][size:] == -100).all()
    partitions = build_trainable_rank_partitions(
        padded.batch['attention_mask'].sum(-1).tolist(), 12,
        padded.batch['labels'].ne(-100).any(-1).tolist(),
    )
    assert partitions is not None
    assert all(len(rank) == padded_size // 12 for rank in partitions)
    assert all(padded.batch['labels'][rank].ne(-100).any() for rank in partitions)
    padded.reorder(torch.tensor([i for rank in partitions for i in rank]))
    restored = without_actor_padding(padded)
    assert len(restored) == size
    assert Counter(restored.non_tensor_batch['uid']) == Counter(batch.non_tensor_batch['uid'])


def test_padding_cannot_manufacture_trainable_rank_coverage():
    batch = _batch(16, 7)
    padded, count = pad_scoped_actor_batch(batch, 12)
    assert padded is batch
    assert count == 0
    assert build_trainable_rank_partitions([4] * 16, 12, [True] * 7 + [False] * 9) is None


def test_transport_padding_does_not_change_mean_loss_gradient():
    batch, _ = pad_scoped_actor_batch(_batch(), 12)
    partitions = build_trainable_rank_partitions(
        [4] * len(batch), 12, batch.batch['labels'].ne(-100).any(-1).tolist(),
    )
    parameter = torch.tensor(2.0, requires_grad=True)
    actual = parameter * 0
    for rank in partitions:
        # Split into microbatches to exercise differing real/padded row counts.
        for micro in (rank[:1], rank[1:]):
            mask = ~batch.batch['actor_padding_mask'][micro]
            values = batch.batch['input_ids'][micro, 0][mask].float()
            if values.numel():
                actual = actual + (parameter * values).mean() * padded_microbatch_loss_scale(
                    torch.tensor(len(values)), torch.tensor(32), 12,
                ) / 12
    expected = (parameter * torch.arange(32).float() * 4).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, parameter)[0],
        torch.autograd.grad(expected, parameter)[0],
    )


@pytest.mark.parametrize('agg_mode', ['token', 'turn', 'trajectory'])
def test_padded_rank_loss_matches_unpadded_loss_with_variable_lengths(agg_mode):
    from verl.rema_trainer.ppo.core_algos import compute_policy_loss
    from verl.utils.torch_functional import masked_mean

    batch = _batch()
    for row in range(len(batch)):
        batch.batch['labels'][row, row % 4 + 1:] = -100
    batch.batch['step_ids'][:] = torch.tensor([0, 0, 1, 1])
    batch.batch['step_ids'][batch.batch['labels'] == -100] = -100
    batch, _ = pad_scoped_actor_batch(batch, 12)
    partitions = build_trainable_rank_partitions(
        [4] * len(batch), 12, batch.batch['labels'].ne(-100).any(-1).tolist(),
    )
    counts = actor_loss_denominators(batch.batch['labels'], batch.batch['step_ids'], agg_mode)
    parameter = torch.tensor(0.01, requires_grad=True)

    def losses(indices):
        data = batch.batch[indices]
        data = data[data['labels'].ne(-100).any(-1)]
        if not len(data):
            return parameter * 0, parameter * 0
        mask = data['labels'].ne(-100)
        log_probs = parameter * data['input_ids'].float() / 128
        pg_loss = compute_policy_loss(
            old_log_prob=torch.zeros_like(log_probs), log_prob=log_probs,
            advantages=data['advantages'], eos_mask=mask, step_id=data['step_ids'],
            cliprange=0.2, agg_mode=agg_mode,
        )[0]
        return pg_loss, masked_mean(log_probs.square(), mask)

    actual = parameter * 0
    for rank in partitions:
        for micro in (rank[:1], rank[1:]):
            data = batch.batch[micro]
            micro_counts = actor_loss_denominators(data['labels'], data['step_ids'], agg_mode)
            pg, regularizer = losses(micro)
            scales = padded_microbatch_loss_scale(micro_counts, counts, 12)
            actual = actual + (pg * scales[0] + regularizer * scales[1]) / 12
    pg, regularizer = losses(list(range(32)))
    expected = pg + regularizer
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, parameter)[0], torch.autograd.grad(expected, parameter)[0],
    )


def test_trainer_balances_padded_rows_for_real_actor_dispatch():
    trainer = _trainer()
    trainer.scoped_c3_grpo_enabled = True
    trainer.actor_rollout_wg = {'worker_stage_1': SimpleNamespace(world_size=12)}
    batch, _ = pad_scoped_actor_batch(_batch(), 12)
    metrics = {}
    assert trainer._balance_batch(batch, metrics)
    assert metrics['global_seqlen/collective_safe'] == 1
    assert metrics['global_seqlen/trainable_sample_count'] == 28
    for shard in batch.chunk(12):
        assert len(shard) == 3
        assert shard.batch['labels'].ne(-100).any()


def test_actor_update_policy_padding_is_neutral_with_entropy_and_kl(monkeypatch):
    import sys
    from types import ModuleType
    from omegaconf import OmegaConf

    # Exercise the real optimizer loop on CPU; GPU attention is not used by
    # the differentiable toy forward below.
    padding_module = ModuleType('flash_attn.bert_padding')
    def unused_attention(*args, **kwargs):
        raise AssertionError('GPU attention must not run in this CPU test')
    for name in ('pad_input', 'unpad_input', 'rearrange', 'index_first_axis'):
        setattr(padding_module, name, unused_attention)
    monkeypatch.setitem(sys.modules, 'flash_attn', ModuleType('flash_attn'))
    monkeypatch.setitem(sys.modules, 'flash_attn.bert_padding', padding_module)
    from verl.workers.actor.dp_rema_actor import DataParallelReMAPPOActor
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))

    batch = _batch(16, 14)
    for row in range(len(batch)):
        batch.batch['labels'][row, row % 4 + 1:] = -100
    batch.batch['step_ids'][batch.batch['labels'] == -100] = -100
    batch.batch['old_log_probs'] = torch.zeros(16, 4)
    batch.batch['ref_log_prob'] = torch.zeros(16, 4)
    batch.meta_info['temperature'] = 1.0
    padded, _ = pad_scoped_actor_batch(batch, 12)

    def train(data, micro_size):
        config = OmegaConf.create({
            'ppo_mini_batch_size': 32, 'ppo_micro_batch_size_per_gpu': micro_size,
            'ppo_epochs': 1, 'use_dynamic_bsz': False, 'use_torch_compile': False,
            'ulysses_sequence_parallel_size': 1, 'use_kl_loss': True,
            'kl_loss_coef': 0.1, 'kl_loss_type': 'low_var_kl',
            'entropy_coeff': 0.01, 'clip_ratio': 0.2, 'clip_ratio_c': 3.0,
            'log_ratio_clip_c': 3.0, 'agg_mode': 'token', 'clip_mode': 'token',
            'grad_clip': 1000.0,
        })
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(0.01)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        actor = DataParallelReMAPPOActor(config, model, optimizer)
        def forward(micro_batch, temperature):
            logits = model.weight.reshape(()) * micro_batch['input_ids'].float() / 64
            return logits.square(), logits
        actor._forward_micro_batch = forward
        metrics = actor.update_policy(data)
        return model.weight.detach(), metrics

    reference, _ = train(batch, 32)
    actual, metrics = train(padded, 3)
    torch.testing.assert_close(actual, reference)
    assert metrics['actor/empty_train_microbatch']


@pytest.mark.parametrize('output,expected', [
    (r'Therefore the blue pens are \boxed{54}.', False),
    (r'The answer is \boxed{98}.', True),
    ('REASONING: \\boxed{98}\nLOCAL_RESULT: \\boxed{54}', False),
    (r'Therefore the blue pens are 54.', None),
    (r'\boxed{54} or \boxed{98}', None),
    (r'\boxed{54} then \boxed{', None),
    ('LOCAL_RESULT: \\boxed{54}\nLOCAL_RESULT: \\boxed{98}', None),
    ('LOCAL_RESULT: \\boxed{54}\nREASONING: \\boxed{98}', False),
    (r'LOCAL_RESULT: \boxed{}', None),
    (r'LOCAL_RESULT: \boxed{UNKNOWN}', None),
])
def test_communication_and_equivalence_share_result_boundary(output, expected):
    routed = MultiAgentRollout._extract_local_result(output)
    assert routed == extract_local_result(output)
    assert compare_worker_final_answers(output, r'\boxed{98}') is expected
    if expected is not None:
        assert routed
        assert compare_worker_final_answers('LOCAL_RESULT: ' + routed, r'\boxed{98}') is expected


def test_result_limit_never_routes_a_truncated_box():
    output = 'LOCAL_RESULT: \\boxed{' + 'x+' * 310 + '1}'
    assert extract_local_result(output) == ''
    assert MultiAgentRollout._extract_local_result(output) == ''
    assert compare_worker_final_answers(output, r'\boxed{1}') is None


def _trainer():
    trainer = RayReMASeparatedTrainer.__new__(RayReMASeparatedTrainer)
    trainer.global_steps = 0
    trainer.actor_update_steps = 0
    trainer.actor_updates_by_role = {}
    trainer.actor_update_count_origin_step = 0
    trainer._current_train_agent = 'worker_stage_1'
    trainer._get_agent12_curriculum_config = lambda: {
        'worker_bootstrap_steps': 2, 'decomposer_transfer_steps': 4,
    }
    return trainer


def test_skipped_attempts_do_not_advance_curriculum_or_periodic_evaluation():
    trainer = _trainer()
    for step in range(1, 21):
        trainer.global_steps = step
        assert trainer._get_schedule_step() == 1
        assert trainer._get_agent12_curriculum_state().phase == 'worker_bootstrap'
        assert not trainer._update_frequency_due(2, actor_updated=False)
    trainer._record_actor_update()
    assert trainer._get_schedule_step() == 2
    assert not trainer._update_frequency_due(2, actor_updated=True)
    trainer._record_actor_update()
    assert trainer._get_agent12_curriculum_state().phase == 'decomposer_transfer'
    assert trainer._update_frequency_due(2, actor_updated=True)
    assert not trainer._update_frequency_due(2, actor_updated=False)
    assert not trainer._update_frequency_due(0, actor_updated=True)
    metrics = trainer._progress_metrics(True)
    assert metrics['train/rollout_step'] == 20
    assert metrics['train/actor_update_step'] == 2
    assert metrics['train/roles/worker_stage_1/actor_update_count'] == 2


def test_skipped_attempts_still_visit_other_worker_roles():
    trainer = _trainer()
    trainer.config = SimpleNamespace(algorithm={'switch_agent': {'freq': 1}})
    trainer._get_train_agent_roles = lambda: ['worker_stage_1', 'worker_stage_2']
    trainer._get_start_agent = lambda: 'worker_stage_1'
    trainer._get_hierarchy_config = lambda: {}
    trainer._get_agent12_curriculum_config = lambda: {
        'enable': True, 'worker_bootstrap_steps': 10, 'train_decomposer': False,
    }
    visited = []
    for step in range(1, 5):
        trainer.global_steps = step
        trainer._update_current_train_agent()
        visited.append(trainer._current_train_agent)
        assert trainer.actor_update_steps == 0
        assert trainer._get_agent12_curriculum_state().phase_step == 0
    assert visited == ['worker_stage_1', 'worker_stage_2'] * 2


def test_update_progress_survives_checkpoint_resume(tmp_path):
    trainer = _trainer()
    trainer.global_steps = 49
    trainer._record_actor_update()
    trainer._current_train_agent = 'worker_stage_2'
    trainer._record_actor_update()
    trainer._save_training_progress(tmp_path)
    resumed = _trainer()
    resumed.global_steps = 49
    resumed._load_training_progress(tmp_path)
    assert resumed.actor_update_steps == 2
    assert resumed.actor_updates_by_role == {'worker_stage_1': 1, 'worker_stage_2': 1}
    assert resumed._get_schedule_step() == 3
    assert not (tmp_path / 'training_progress.json.tmp').exists()


def test_legacy_checkpoint_does_not_guess_actual_updates(tmp_path, capsys):
    trainer = _trainer()
    trainer.global_steps = 340
    trainer._load_training_progress(tmp_path)
    assert trainer.actor_update_steps == 0
    assert trainer.actor_update_count_origin_step == 340
    assert 'Legacy checkpoint' in capsys.readouterr().out


def test_mismatched_progress_checkpoint_fails_closed(tmp_path):
    trainer = _trainer()
    trainer._save_training_progress(tmp_path)
    trainer.global_steps = 1
    with pytest.raises(ValueError, match='does not match'):
        trainer._load_training_progress(tmp_path)

"""CPU lifetime/gradient regression tests; no pretrained models or GPU needed."""
import importlib
import sys
import weakref
from types import ModuleType, SimpleNamespace

import pytest
import torch

try:
    import verl.hierarchical_rema as api
except ModuleNotFoundError:
    import hierarchical_rema as api

offline = importlib.import_module(api.__name__ + '.offline_training')


class LifetimeModel(torch.nn.Module):
    def __init__(self, nonfinite_training=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(10).float() / 100)
        self.references = []
        self.nonfinite_training = nonfinite_training
        self.injected = False

    def forward(self, input_ids, **kwargs):
        assert all(reference() is None for reference in self.references), 'Previous logits still alive'
        logits = input_ids.float().unsqueeze(-1) * self.weight
        if self.nonfinite_training and self.training and torch.is_grad_enabled() and not self.injected:
            logits = logits * float('nan')
            self.injected = True
        self.references.append(weakref.ref(logits))
        return SimpleNamespace(logits=logits)


def rows(count=4):
    result = []
    for index in range(count):
        item = dict(sample_index=index, input_ids=torch.tensor([index+1, 2, 3, 4]),
                    attention_mask=torch.ones(4, dtype=torch.long), position_ids=torch.arange(4),
                    loss_mask=torch.tensor([0, 1, 1, 0]), pad_token_id=0,
                    reward=torch.tensor(1.), advantage=torch.tensor(0.5), group_id='g',
                    role='worker', policy_id='executor')
        for name in ('retained_response_tokens', 'truncated_prompt_tokens', 'truncated_response_tokens',
                     'zero_loss_row', 'zero_loss_due_to_truncation', 'zero_loss_due_to_empty_completion'):
            item[name] = torch.tensor(0)
        result.append(item)
    return result


class Algorithms:
    """Small differentiable objectives to exercise the unchanged PPO call API."""
    @staticmethod
    def compute_policy_loss(*, old_log_prob, log_prob, advantages, eos_mask, cliprange, clip_ratio_c):
        mean = lambda value: (value * eos_mask).sum() / eos_mask.sum()
        ratio = (log_prob - old_log_prob).exp()
        loss = mean(torch.maximum(-advantages * ratio,
                                 -advantages * ratio.clamp(1-cliprange, 1+cliprange)))
        return loss, mean((ratio > 1+cliprange).float()), mean(old_log_prob-log_prob), mean(ratio*0)

    @staticmethod
    def compute_entropy_loss(logits, mask):
        log_prob = logits.log_softmax(-1)
        entropy = -(log_prob.exp() * log_prob).sum(-1)
        return (entropy * mask).sum() / mask.sum()


def old_inline_objective(model, batch, cache, config):
    output = model(input_ids=batch['input_ids'])
    mask = batch['loss_mask'][:, :-1].float()
    log_probs = offline._sequence_log_probs(output.logits[:, :-1, :], batch['input_ids'][:, 1:])
    old = offline._gather_old_log_probs(batch['sample_index'], cache, mask, 'cpu', log_probs.dtype)
    pg, clip, kl, lower = Algorithms.compute_policy_loss(
        old_log_prob=old, log_prob=log_probs,
        advantages=batch['advantage'].unsqueeze(-1).expand_as(log_probs), eos_mask=mask,
        cliprange=config.clip_range, clip_ratio_c=config.clip_ratio_c)
    entropy = Algorithms.compute_entropy_loss(output.logits[:, :-1, :], mask)
    return dict(loss=pg-config.entropy_coeff*entropy, clipfrac=clip, approx_kl=kl,
                clipfrac_lower=lower, entropy=entropy)


@pytest.mark.parametrize('entropy_coeff', [0., 0.001])
def test_scoped_objective_preserves_values_and_gradients(entropy_coeff):
    batch = offline._collate_rows(rows(2))
    cache = {index: torch.tensor([-2.1, -2.2]) for index in range(2)}
    config = offline.OfflineTrainingConfig('mock', 'unused', entropy_coeff=entropy_coeff)
    reference, optimized = LifetimeModel(), LifetimeModel()
    expected = old_inline_objective(reference, batch, cache, config)
    actual, error = offline._grpo_batch_objective(
        optimized, batch, 'cpu', cache, config, Algorithms, offline.DistributedTrainingContext())
    assert error is None
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    expected['loss'].backward()
    actual['loss'].backward()
    torch.testing.assert_close(optimized.weight.grad, reference.weight.grad, rtol=0, atol=0)
    del actual
    assert all(reference() is None for reference in optimized.references)


def test_old_log_prob_cache_releases_each_forward():
    model = LifetimeModel()
    cache = offline._compute_old_log_prob_cache(model, rows(), 2, 'cpu')
    assert len(cache) == 4 and len(model.references) == 2
    assert all(reference() is None for reference in model.references)
    assert all(not tensor.requires_grad and tensor.device.type == 'cpu' for tensor in cache.values())


@pytest.mark.parametrize('bad_batch', [False, True])
def test_validation_releases_finite_and_nonfinite_outputs(bad_batch):
    class ValidationModel(LifetimeModel):
        def forward(self, input_ids, **kwargs):
            output = super().forward(input_ids, **kwargs)
            if bad_batch and len(self.references) == 1:
                output.logits.fill_(float('nan'))
            return output

    model = ValidationModel()
    batches = [offline._collate_rows(rows(2)) for _ in range(3)]
    metrics = offline.evaluate_controller_model(model, batches, torch.device('cpu'))
    assert metrics['val_loss'] > 0 and len(model.references) == 3
    assert all(reference() is None for reference in model.references)


@pytest.mark.parametrize('bad_stage', ['logits', 'batch tensors', 'objective'])
def test_invalid_objective_releases_forward_and_autograd(bad_stage):
    model = LifetimeModel(nonfinite_training=bad_stage == 'logits')
    config = offline.OfflineTrainingConfig('mock', 'unused')
    batch = offline._collate_rows(rows(2))
    cache = {index: torch.tensor([-2.1, -2.2]) for index in range(2)}
    if bad_stage == 'batch tensors':
        batch['advantage'][0] = float('nan')
    if bad_stage == 'objective':
        config.entropy_coeff = float('nan')
    objective, error = offline._grpo_batch_objective(
        model, batch, 'cpu', cache, config, Algorithms, offline.DistributedTrainingContext())
    assert objective is None and error == bad_stage
    assert all(reference() is None for reference in model.references)


@pytest.mark.parametrize('nonfinite_training', [False, True])
def test_training_loop_releases_outputs_before_next_forward_and_eval(tmp_path, monkeypatch, nonfinite_training):
    # Keep external trainer dependencies out of this CPU lifetime test.
    ppo = ModuleType('verl.trainer.ppo')
    ppo.core_algos = Algorithms
    monkeypatch.setitem(sys.modules, 'verl.trainer.ppo', ppo)
    model = LifetimeModel(nonfinite_training=nonfinite_training)
    monkeypatch.setattr(offline, '_load_model_and_tokenizer', lambda *args: (None, model, torch.device('cpu')))
    monkeypatch.setattr(offline, '_init_distributed_training', offline.DistributedTrainingContext)
    monkeypatch.setattr(offline, '_load_scheduler_factory', lambda: (
        lambda optimizer, **kwargs: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)))

    class Dataset:
        def __init__(self, samples, *args):
            self.rows, self.stats = rows(len(samples)), {}

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    monkeypatch.setattr(offline, 'ControllerReplayDataset', Dataset)
    config = offline.OfflineTrainingConfig('mock', str(tmp_path), epochs=2, train_batch_size=2,
        grad_accum_steps=2, entropy_coeff=0.001, eval_every_steps=1, save_final_checkpoint=False,
        logging_steps=1, device='cpu')
    samples = [SimpleNamespace(role='worker', metadata={}) for _ in range(4)]
    tracking = SimpleNamespace(last_step=0, log=lambda *args, **kwargs: None)
    summary = offline.run_offline_policy_training(samples, samples[:2], config, tracking=tracking)
    assert summary['steps'] == 2
    assert summary['skipped_non_finite_batches'] == int(nonfinite_training)
    assert len(model.references) >= 8
    assert all(reference() is None for reference in model.references)

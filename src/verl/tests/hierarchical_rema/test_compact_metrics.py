"""Monitoring-only changes must not change rewards, verification, or selection."""
import copy
import importlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

try:
    import verl.hierarchical_rema as api
except ModuleNotFoundError:
    import hierarchical_rema as api

metrics = importlib.import_module(api.__name__ + '.metrics')
train = importlib.import_module(api.__name__ + '.train')
offline = importlib.import_module(api.__name__ + '.offline_training')


def rollout(outcomes, rewards, subset='math', fallback=False, excluded=False):
    selections = [NS(
        selection=NS(raw_payload={'synthetic_executor_rollout': True}, assignments=[]),
        reward=NS(final_answer_correctness=outcome, total_reward=reward),
        training_excluded=excluded,
        executions=[NS(reward_model_reward=reward, invalid_reason='bad format' if fallback else '')],
    ) for outcome, reward in zip(outcomes, rewards)]
    plan = NS(decomposition=NS(decomposition_id='p', raw_payload={'validation': {
        'fallback_used': fallback}}), selections=selections, decomposition_reward=.4)
    return NS(task=api.TaskExample(task_id='t', prompt='2+2', ground_truth='4',
                                  metadata={'subset': subset}), decompositions=[plan],
              training_batch=NS(worker_grpo_stats={}))


def test_accuracy_is_all_verified_executions_not_best_of_n(capsys):
    value = rollout([1, 0, None], [.8, -.4, None], fallback=True, excluded=True)
    before = copy.deepcopy(value)
    summary = train.epoch_rollout_summary([value], include_subsets=True)
    assert summary['accuracy'] == .5
    assert math.isnan(summary['mean_best_final_correctness'])
    assert train._validation_primary_metric(summary) is None
    displayed = metrics.rollout_tracking_metrics(summary)
    assert displayed == pytest.approx({
        'rollout/accuracy': .5, 'rollout/final_reward_mean': .2,
        'rollout/incorrect_final_reward_mean': -.4, 'rollout/plan_fallback_rate': 1,
        'rollout/invalid_worker_output_rate': 1, 'rollout/unscored_rate': 1/3,
        'rollout/safety_excluded_rate': 1,
    })
    assert metrics.validation_tracking_metrics(summary) == {'val/accuracy': .5, 'val/accuracy/math': .5}
    metrics.warn_verification_coverage(summary, 'validation')
    assert '1/3 rollouts unverified' in capsys.readouterr().out
    assert value == before
    complete = train.epoch_rollout_summary([rollout([0, 1], [-.4, .8])])
    assert complete['accuracy'] == .5
    assert train._validation_primary_metric(complete) == 1


def test_batch_merging_is_count_weighted_and_keeps_negative_sums():
    rows = [rollout([0], [-1]), rollout([0, 1, 0], [-.8, .6, -.4]),
            rollout([None], [None], subset='other')]
    batches = [train.epoch_rollout_summary([row], include_subsets=True, include_tasks=False) for row in rows]
    combined = train.combine_rollout_summaries(batches, include_subsets=True, include_tasks=False)
    direct = train.epoch_rollout_summary(rows, include_subsets=True, include_tasks=False)
    assert combined['monitoring_totals'] == direct['monitoring_totals']
    assert combined['accuracy'] == .25
    assert metrics.rollout_tracking_metrics(combined)['rollout/final_reward_mean'] == pytest.approx(-.4)
    assert combined['subsets']['math']['accuracy'] == .25
    assert math.isnan(combined['subsets']['other']['accuracy'])


def test_absent_evidence_is_not_zero_accuracy_or_zero_reward():
    summary = train.epoch_rollout_summary([rollout([None], [None])])
    displayed = metrics.rollout_tracking_metrics(summary)
    assert math.isnan(displayed['rollout/accuracy'])
    assert math.isnan(displayed['rollout/final_reward_mean'])
    assert math.isnan(displayed['rollout/incorrect_final_reward_mean'])
    value = rollout([1], [.8])
    value.decompositions[0].decomposition.raw_payload = {}
    summary = train.epoch_rollout_summary([value])
    assert math.isnan(metrics.rollout_tracking_metrics(summary)['rollout/plan_fallback_rate'])


@pytest.mark.parametrize('prefix', ['', 'shared_worker/', 'shared_controller/'])
def test_relay_allowlist_preserves_full_files(tmp_path, prefix):
    record = {key: .1 for key in metrics.TRAIN_METRICS}
    record.update(step=3, mean_reward=.9, mean_advantage=0, selector_mean_reward=.7,
                  selector_samples_total=5, overall_mean_reward=.9)
    (tmp_path / 'train_metrics.jsonl').write_text(json.dumps(record) + '\n')
    (tmp_path / 'eval_metrics.jsonl').write_text('{"step": 3, "val_loss": 2}\n')
    (tmp_path / 'summary.json').write_text('{"steps": 3, "val_loss": 2, "train_dataset_diagnostics": {}}')
    originals = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    events = []
    tracker = NS(last_step=10, log=lambda values, step: events.append((step, values)))
    train._relay_offline_metrics_to_tracking(tmp_path, tracker, prefix, 0, 100, 10)
    assert events == [(13, {f'{prefix}train/{key}': .1 for key in metrics.TRAIN_METRICS})]
    assert originals == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


def test_legacy_log_records_do_not_invent_new_statistics():
    assert metrics.learner_tracking_metrics({'loss': 1, 'mean_reward': .7, 'selector_samples_total': 4}) == {
        'train/loss': 1., 'train/reward_mean': .7}
    assert 'shared_worker/optimizer_updates' not in metrics.learner_tracking_metrics(
        {'step': 2000, 'loss': .5}, 'shared_worker/')


def test_optimizer_offset_survives_distributed_config_serialization(tmp_path):
    from dataclasses import asdict
    config = offline.OfflineTrainingConfig('mock', 'unused', optimizer_updates_offset=42)
    filename = tmp_path / 'config.json'
    filename.write_text(json.dumps(asdict(config)))
    assert offline._load_config_from_json(str(filename)).optimizer_updates_offset == 42


def test_integrated_counters_accumulate_per_policy_not_logging_steps(tmp_path, monkeypatch):
    calls, events = [], []
    assigned_steps = [2, 0, 3, 4, 5, 6]
    def learner(train_samples, val_samples, config, tracking=None, tracking_prefix='', log_step_offset=0):
        steps = assigned_steps[len(calls)]
        calls.append((tracking_prefix, config.optimizer_updates_offset, steps))
        final = Path(config.output_dir) / 'final'
        final.mkdir(parents=True)
        return {'steps': steps, 'selected_model_path': str(final), 'selected_model_source': 'final'}
    monkeypatch.setattr(train, 'run_offline_policy_training', learner)
    monkeypatch.setattr(train, '_tracking', lambda *a, **kw: NS(
        last_step=0, log=lambda values, step: events.append(values), finish=lambda: None))
    monkeypatch.setattr(sys, 'argv', ['train.py', '--reward-setting', 'handcrafted', '--backend', 'mock', '--task-source', 'demo',
        '--mode', 'alternating', '--phase', 'executor', '--num-epochs', '3',
        '--parameter-sharing', '--train-worker-model', '--num-decompositions', '3',
        '--num-executor-rollouts', '4', '--rollout-task-batch-size', '1',
        '--update-every-n-rollout-batches', '1', '--disable-rollout-logging',
        '--output-dir', str(tmp_path)])
    train.main()
    assert calls == [('shared_worker/', 0, 2), ('shared_worker/', 2, 0),
                     ('shared_controller/', 0, 3), ('shared_controller/', 3, 4),
                     ('shared_worker/', 2, 5), ('shared_worker/', 7, 6)]
    assert [event['shared_worker/optimizer_updates'] for event in events] == [2, 2, 13]
    assert [event['shared_controller/optimizer_updates'] for event in events] == [0, 7, 7]
    summary = json.loads((tmp_path / 'training_summary.json').read_text())
    expected = [{'shared_worker': 2, 'shared_controller': 0},
                {'shared_worker': 2, 'shared_controller': 7},
                {'shared_worker': 13, 'shared_controller': 7}]
    assert [item['optimizer_updates_by_policy'] for item in summary['epochs']] == expected
    assert summary['optimizer_updates_by_policy'] == expected[-1]
    for index, counters in enumerate(expected, 1):
        saved = json.loads((tmp_path / f'epoch_{index:04d}' / 'epoch_summary.json').read_text())
        assert saved['optimizer_updates_by_policy'] == counters


def test_update_moments_cover_unequal_microbatches_and_ranks(monkeypatch):
    totals = [0.] * 5
    offline._update_sample_moments(totals, torch.tensor([0.]), torch.tensor([1.]))
    offline._update_sample_moments(totals, torch.tensor([1., 1.]), torch.tensor([-1., 0.]))
    local = offline._reduce_sample_moments(totals, 'cpu', offline.DistributedTrainingContext())
    assert local['reward_mean'] == pytest.approx(2/3)
    assert local['reward_std'] == pytest.approx(math.sqrt(2/9))
    assert local['advantage_std'] == pytest.approx(math.sqrt(2/3))
    # Other rank has one row: reward=-1, advantage=2.
    def reduce(value, op):
        value.add_(torch.tensor([1., -1., 1., 2., 4.], dtype=value.dtype))
    monkeypatch.setattr(torch.distributed, 'all_reduce', reduce)
    actual = offline._reduce_sample_moments(totals, 'cpu', offline.DistributedTrainingContext(enabled=True, world_size=2))
    assert actual['reward_mean'] == .25
    assert actual['reward_std'] == pytest.approx(math.sqrt(.6875))
    assert actual['advantage_std'] == pytest.approx(math.sqrt(1.25))
    assert all(isinstance(value, (int, float)) for value in totals)


def test_performance_uses_allocated_not_policy_gpu_count():
    environment = {'SLURM_GPUS_PER_NODE': 'hopper:4', 'SLURM_JOB_NUM_NODES': '2',
                   'RAY_N_GPUS_PER_NODE': '3'}
    assert metrics.allocated_gpu_count(environment) == 8
    assert metrics.allocated_gpu_count({'SLURM_GPUS': '4'}) == 4
    assert metrics.allocated_gpu_count({'RAY_N_GPUS_PER_NODE': '3'}) is None
    result = metrics.performance_metrics(3600, 1000, 8)
    assert result['perf/gpu_hours_per_1000_rollouts'] == 8
    assert result['perf/rollouts_per_second'] == pytest.approx(1000/3600)
    assert 'perf/gpu_hours_per_1000_rollouts' not in metrics.performance_metrics(1, 0, 8)
    assert 'perf/gpu_hours_per_1000_rollouts' not in metrics.performance_metrics(1, 1)

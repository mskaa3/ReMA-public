"""Learning exclusion and lossless rollout logging; no models or S3 credentials."""
from collections import defaultdict
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_graphprm_pilot import api, predictions, rollout, train

safety = importlib.import_module(api.__name__ + '.training_safety')
recording = importlib.import_module(api.__name__ + '.recording')
rewarding = importlib.import_module(api.__name__ + '.rewarding')


def evaluate(scores=(.99, .98), *, outcome=0, status='verified', final_score=-1., enabled=True):
    return safety.SaturatedFailureFilter(enabled=enabled).evaluate(
        task=SimpleNamespace(ground_truth='4'), decomposition=SimpleNamespace(final_node_id='F'),
        executions=[SimpleNamespace(node_id=str(i)) for i in range(len(scores))] + [SimpleNamespace(node_id='F')],
        reward=SimpleNamespace(verification_status=status, final_answer_correctness=outcome),
        compiled_rewards={'workers': {
            **{str(i): {'reward_before_outcome_gate': value, 'reward': .25*value if isinstance(value, float) else None}
               for i, value in enumerate(scores)},
            'F': {'reward_before_outcome_gate': final_score}}})


@pytest.mark.parametrize('scores,expected', [((.99, .98), True), ((.95,), True),
    ((.999, .94), False), ((.99, -.5), False), ((), False), ((None,), False),
    ((float('nan'),), False), ((float('inf'),), False), (('malformed',), False)])
def test_requires_all_intermediate_pre_gate_scores(scores, expected):
    assert evaluate(scores)['excluded'] is expected


@pytest.mark.parametrize('outcome,status,enabled', [(1, 'verified', True),
    (None, 'unverified', True), (0, 'unverified', True), (0, 'verified', False)])
def test_never_excludes_success_unverified_or_disabled(outcome, status, enabled):
    assert not evaluate(outcome=outcome, status=status, enabled=enabled)['excluded']


@pytest.mark.parametrize('value', ['0', '-1', '1.01', 'nan', 'inf', 'bad'])
def test_rejects_invalid_filter_configuration(monkeypatch, value):
    monkeypatch.setenv('GFAM_SATURATED_FAILURE_THRESHOLD', value)
    with pytest.raises(ValueError):
        safety.SaturatedFailureFilter.from_environment()


def test_environment_is_read_once(monkeypatch):
    monkeypatch.setenv('GFAM_SKIP_SATURATED_FAILURES', 'false')
    monkeypatch.setenv('GFAM_SATURATED_FAILURE_THRESHOLD', '.97')
    config = safety.SaturatedFailureFilter.from_environment()
    monkeypatch.setenv('GFAM_SKIP_SATURATED_FAILURES', 'true')
    assert config == safety.SaturatedFailureFilter(False, .97)
    monkeypatch.setenv('GFAM_SKIP_SATURATED_FAILURES', 'typo')
    with pytest.raises(ValueError, match='true/false'):
        safety.SaturatedFailureFilter.from_environment()


def test_does_not_quarantine_an_intermediate_with_a_negative_protocol_penalty():
    result = safety.SaturatedFailureFilter().evaluate(
        task=SimpleNamespace(ground_truth='4'), decomposition=SimpleNamespace(final_node_id='F'),
        executions=[SimpleNamespace(node_id='A'), SimpleNamespace(node_id='F')],
        reward=SimpleNamespace(verification_status='verified', final_answer_correctness=0),
        compiled_rewards={'workers': {'A': {'reward_before_outcome_gate': .999, 'reward': -.1}}})
    assert not result['excluded']
    assert result['not_excluded_reason'] == 'negative_intermediate_reward'


@pytest.mark.parametrize('failed_calls', [{1}, set(range(1, 9))])
def test_exclusion_precedes_grpo_but_keeps_scores_and_all_rollouts(tmp_path, monkeypatch, failed_calls):
    monkeypatch.setenv('GFAM_SKIP_SATURATED_FAILURES', 'true')
    monkeypatch.setenv('GFAM_SATURATED_FAILURE_THRESHOLD', '.95')
    backend = importlib.import_module(api.__name__ + '.backends').MockHierarchicalBackend
    original = backend.execute_worker
    calls = []

    def final_answers(self, task, decomposition, node, *args, **kwargs):
        ex = original(self, task, decomposition, node, *args, **kwargs)
        if node.node_id == decomposition.final_node_id:
            calls.append(node.node_id)
            ex.output_text = ex.raw_output_text = '5' if len(calls) in failed_calls else '4'
        return ex

    def saturated_predictions(executions=(), **kwargs):
        pred = predictions(executions, safe=1.)
        for labels in [pred['final'], *pred['workers'].values()]:
            for payload in labels.values():
                payload['probabilities'] = {'0': .001, '1': .999}
        return pred

    monkeypatch.setattr(backend, 'execute_worker', final_answers)
    monkeypatch.setattr(importlib.import_module('test_graphprm_pilot'), 'predictions', saturated_predictions)
    result, trainer, _ = rollout(tmp_path)
    flagged = [(dec, sel) for dec in result.decompositions for sel in dec.selections if sel.training_excluded]
    assert len(flagged) == len(failed_calls)
    flagged_ids = {(d.decomposition.decomposition_id, s.selection.selection_id) for d, s in flagged}
    for dec, sel in flagged:
        assert not sel.training_eligible and not dec.training_eligible
        assert dec.decomposer_advantage is None
        assert sel.reward.final_answer_correctness == 0.
        assert sel.reward.total_reward == pytest.approx(.2495)
        assert all(ex.reward_model_reward == pytest.approx(.2495) for ex in sel.executions)
        assert dec.decomposition_reward is not None  # Preserve the full repeated-execution mean.
        expected_mean = sum(s.reward_model_outputs['compiled_rewards']['decomposer']['reward'] for s in dec.selections) / 4
        assert dec.decomposition_reward == pytest.approx(expected_mean)
        with pytest.raises(ValueError, match='Excluded'):
            trainer.orchestrator._worker_training_reward(sel, sel.executions[0])
    assert len(result.training_batch.decomposer_samples) == (1 if len(failed_calls) == 1 else 0)
    grouped = defaultdict(list)
    for sample in result.training_batch.worker_samples:
        assert (sample.metadata['decomposition_id'], sample.metadata['selection_id']) not in flagged_ids
        grouped[sample.group_id].append(sample)
    for samples in grouped.values():
        expected = rewarding.group_relative_advantages([sample.reward for sample in samples])
        assert [sample.advantage for sample in samples] == pytest.approx(expected)
        assert len(samples) in {3, 4}
    stats = result.training_batch.worker_grpo_stats
    assert stats['num_worker_samples_skipped_safety'] == sum(len(s.executions) for _, s in flagged)
    assert stats['num_decomposer_samples_skipped_safety'] == (1 if len(failed_calls) == 1 else 2)
    assert not stats['num_worker_samples_skipped_unscored']

    saved = json.loads((tmp_path/'all_rollouts.jsonl').read_text().splitlines()[0])['rollout']
    assert len(saved['decompositions']) == 2
    assert sum(len(d['selections']) for d in saved['decompositions']) == 8
    assert sum(not s['training_eligible'] for d in saved['decompositions'] for s in d['selections']) == len(failed_calls)
    assert len((tmp_path/'training_exclusions.jsonl').read_text().splitlines()) == len(failed_calls)
    # Worker history must not absorb quarantined outcomes either.
    seen = []
    monkeypatch.setattr(trainer.orchestrator.worker_memory, 'record_execution', lambda **kw: seen.append(kw))
    trainer.orchestrator._update_worker_memory(result.task, result.decompositions)
    assert len(seen) == sum(len(s.executions) for d in result.decompositions for s in d.selections if s.training_eligible)

    summary = train.epoch_rollout_summary([result], include_subsets=True)
    assert summary['safety_excluded_rollouts'] == len(failed_calls)
    assert summary['safety_excluded_worker_executions'] == stats['num_worker_samples_skipped_safety']
    assert train.combine_rollout_summaries([summary, summary])['safety_excluded_rollouts'] == 2*len(failed_calls)
    assert sum(s['safety_excluded_rollouts'] for s in summary['subsets'].values()) == len(failed_calls)


@pytest.mark.parametrize('compact', [True, False])
def test_all_logs_keep_original_planner_text_plan_and_repeated_executions(tmp_path, compact):
    result, _, _ = rollout(tmp_path/'source')
    for index, dec in enumerate(result.decompositions):
        dec.decomposition.raw_payload['raw_model_text'] = f'<scratchpad>plan {index}</scratchpad><result>original plan {index}</result>'
    before = copy.deepcopy(result.to_dict())
    recorder = recording.RolloutRecorder(api.RolloutLoggingConfig(
        output_dir=str(tmp_path/'audit'), save_all_rollouts=True, compact_mode=compact))
    recorder.record_task_rollout(result)
    payload = json.loads((tmp_path/'audit/all_rollouts.jsonl').read_text())
    plans = payload['rollout']['decompositions']
    assert len(plans) == 2
    assert payload['rollout']['policy_config'] == before['policy_config']
    for index, plan in enumerate(plans):
        original = before['decompositions'][index]
        assert plan['decomposition']['raw_payload']['raw_model_text'] == original['decomposition']['raw_payload']['raw_model_text']
        assert plan['decomposition']['raw_text'] == original['decomposition']['raw_text']
        assert plan['decomposition']['nodes'] == original['decomposition']['nodes']
        assert len(plan['selections']) == 4
        assert len({s['selection']['selection_id'] for s in plan['selections']}) == 4
        for saved, old in zip(plan['selections'], original['selections']):
            for actual, expected in zip(saved['executions'], old['executions']):
                for key in ('node_id', 'worker_id', 'output_text', 'raw_output_text', 'dependency_outputs'):
                    assert actual[key] == expected[key]
    best = json.loads((tmp_path/'audit/best_selections.jsonl').read_text())
    assert best['decomposition']['raw_payload']['raw_model_text']
    assert result.to_dict() == before  # Logging must not alter learning inputs.


def test_epoch_s3_upload_includes_all_rollout_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[4]
    source = (root/'hierarchical-rema-trainer.sh').read_text()
    upload_function = source.split('upload_epoch_folder_to_s3() {', 1)[1].split('\nsync_epoch_final_models_to_s3()', 1)[0]
    epoch = tmp_path/'epoch_0001'
    logs = epoch/'rollouts/segment_0001'
    logs.mkdir(parents=True)
    for name in ('best_selections.jsonl', 'all_rollouts.jsonl', 'training_exclusions.jsonl'):
        (logs/name).write_text('{}\n')
    # Capture the real launcher's rclone arguments, without remote I/O.
    script = '''set -eu
rclone() { printf '%s\\n' "$@" > "$CALLS"; }
sync_epoch_final_models_to_s3() { :; }
upload_epoch_folder_to_s3() {''' + upload_function + '\nupload_epoch_folder_to_s3 "$EPOCH"\n'
    env = dict(os.environ, CALLS=str(tmp_path/'calls'), EPOCH=str(epoch),
               LOCAL_OUTPUT_DIR=str(tmp_path), S3_OUTPUT_PATH='remote:bucket/run',
               S3_EPOCHS_PATH='remote:bucket/run/epochs')
    subprocess.run(['bash', '-c', script], check=True, env=env, capture_output=True, text=True)
    args = (tmp_path/'calls').read_text().splitlines()
    assert args[:3] == ['copy', str(epoch), 'remote:bucket/run/epochs/epoch_0001']
    assert all(args[i+1].startswith('/train/') for i, arg in enumerate(args) if arg == '--exclude')
    assert '--include' not in args


def test_integrated_logging_defaults_retain_all_rollouts(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['train', '--output-dir', str(tmp_path)])
    args = train.parse_args()
    assert args.rollout_log_mode == 'all'
    assert args.rollout_log_detail == 'full'

"""Live binary-v2 reward safety, without downloads, Ray, Slurm, or GPUs."""
import copy
import importlib
import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace
import pytest

try:
    import verl.hierarchical_rema as api
except ModuleNotFoundError:
    import hierarchical_rema as api

demo = importlib.import_module(api.__name__ + '.demo')
train = importlib.import_module(api.__name__ + '.train')
gpu = importlib.import_module(api.__name__ + '.gpu_partition')
runtime = importlib.import_module(api.__name__ + '.graphprm_v2.runtime')
compile_rewards = importlib.import_module(api.__name__ + '.graphprm_v2.rewards').compile_rewards


def predictions(executions=(), plan=.9, safe=.8, success=.01):
    def item(p):
        return {'probabilities': {'0': 1-p, '1': p}}
    local = {name: item(.8) for name in ('task_fulfillment', 'execution_discipline', 'causal_utility')}
    return {'graph': {'anti_collapse': item(safe), 'global_outcome': item(success)},
            'decomposer': {name: item(plan) for name in ('task_fulfillment', 'execution_discipline')},
            'workers': {ex.node_id: copy.deepcopy(local) for ex in executions}, 'final': local}


class Scorer:
    supports_verified_final_correctness = True

    def __init__(self, failures=()):
        self.failures = set(failures)
        self.calls = []

    def score_rollout(self, *, task, decomposition, selection, executions, final_answer,
                      verified_final_correctness=None):
        self.calls.append((decomposition.decomposition_id, verified_final_correctness))
        if len(self.calls) in self.failures:
            return {'status': 'unscored', 'backend': 'graphprm_binary_v2',
                    'error': {'code': 'context_overflow', 'encoder': 'prm',
                              'total_tokens': 5000, 'max_tokens': 4096}}
        row = runtime.live_record(task, decomposition, executions, final_answer)
        assert 'ground_truth' not in json.dumps(row)
        result = compile_rewards(predictions(executions), row,
                                 verified_final_correctness=verified_final_correctness)
        return {'status': 'scored', 'backend': 'graphprm_binary_v2',
                'compiled_rewards': result['node_rewards'], 'graph_summary': result['graph_summary']}


def rollout(tmp_path, failures=()):
    scorer = Scorer(failures)
    trainer = api.HierarchicalGRPOTrainer(
        gfam_reward_scorer=scorer, train_worker_model=True, min_worker_grpo_group_size=2,
        rollout_logging_config=api.RolloutLoggingConfig(
            output_dir=str(tmp_path), save_best_rollouts=True, save_all_rollouts=True, compact_mode=True))
    result = trainer.run(
        task=api.TaskExample(task_id='pilot', prompt='Solve for x: 2x + 3 = 11.', ground_truth='4',
                             metadata={'skill_focus': 'algebra', 'distractor_answer': '5'}),
        worker_pool=demo.make_worker_pool(base_model_path='mock'),
        policy_config=api.ControllerPolicyConfig(shared_model_path='mock'),
        rollout_config=api.RolloutConfig(num_decompositions=2, num_executor_rollouts_per_decomposition=4),
        schedule=api.TrainingScheduleConfig(mode=api.TrainingMode.JOINT))
    return result, trainer, scorer


def test_verified_outcome_changes_planner_feedback_and_positive_final_credit():
    pred = predictions()
    record = {'trajectory': {'final_answer': '4'}}
    bad = compile_rewards(pred, record, verified_final_correctness=0)
    good = compile_rewards(pred, record, verified_final_correctness=1)
    assert bad['node_rewards']['final']['reward'] == pytest.approx(.25 * good['node_rewards']['final']['reward'])
    assert good['node_rewards']['decomposer']['execution_bonus'] == pytest.approx(.15*.9*.8)
    assert bad['node_rewards']['decomposer']['execution_bonus'] == 0
    assert good['graph_summary']['predicted_success'] == .01
    assert good['graph_summary']['execution_outcome_source'] == 'verified'
    assert good['node_rewards']['decomposer']['reward'] > bad['node_rewards']['decomposer']['reward']
    for pred in (predictions(plan=0), predictions(safe=0)):
        assert compile_rewards(pred, {}, verified_final_correctness=1)['node_rewards']['decomposer']['execution_bonus'] == 0
    with pytest.raises(ValueError, match='binary'):
        compile_rewards(predictions(), {}, verified_final_correctness=float('nan'))


@pytest.mark.parametrize('scale', [0., .1, .25, 1.])
@pytest.mark.parametrize('probability', [.1, .5, .8, 1.])
def test_execution_gate_preserves_negative_credit_and_predictions(scale, probability):
    pred = predictions()
    pred['final'] = {name: {'probabilities': {'0': 1-probability, '1': probability}}
                     for name in ('task_fulfillment', 'execution_discipline', 'causal_utility')}
    pred['workers'] = {node: copy.deepcopy(pred['final']) for node in ('A', 'B')}
    before = copy.deepcopy(pred)
    record = {'trajectory': {'final_answer': '4', 'final_node_id': 'B'},
              'workers': [{'node_id': node, 'output_text': '4'} for node in ('A', 'B')]}
    results = [compile_rewards(pred, record, verified_final_correctness=outcome,
                               verified_failure_positive_scale=scale) for outcome in (0, 1, None)]
    for result, outcome in zip(results, (0, 1, None)):
        factor = scale if outcome == 0 else 1.
        assert result['graph_summary']['execution_positive_credit_scale'] == factor
        local = result['node_rewards']
        for payload in [local['final'], *local['workers'].values()]:
            baseline = .9 * (2*probability-1) if probability > .5 else 2*probability-1
            assert payload['reward_before_outcome_gate'] == pytest.approx(baseline)
            assert payload['reward'] == pytest.approx(min(baseline, 0) + factor*max(baseline, 0))
            assert payload['verified_failure_gate_applied'] == (outcome == 0)
    assert pred == before
    assert compile_rewards(pred, record, verified_final_correctness=0,
                           verified_failure_positive_scale=scale) == results[0]


@pytest.mark.parametrize('probability', [.1, .8])
def test_protocol_penalties_are_applied_after_the_gate_without_discount(probability):
    pred = predictions()
    pred['final'] = {name: {'probabilities': {'0': 1-probability, '1': probability}}
                     for name in ('task_fulfillment', 'execution_discipline', 'causal_utility')}
    pred['workers'] = {'B': copy.deepcopy(pred['final'])}
    record = {'trajectory': {'final_answer': '', 'final_node_id': 'B'},
              'workers': [{'node_id': 'B', 'output_text': '', 'raw_output_text': '4'}]}
    for outcome in (0, 1):
        result = compile_rewards(pred, record, verified_final_correctness=outcome)['node_rewards']
        expected = min(2*probability-1, 0) - .10
        assert result['final']['reward'] == pytest.approx(expected)
        assert result['workers']['B']['reward'] == pytest.approx(expected)


@pytest.mark.parametrize('scale', [-.1, 1.1, float('nan'), float('inf')])
def test_gate_configuration_fails_before_loading_models(monkeypatch, scale):
    with pytest.raises(ValueError, match='verified_failure_positive_scale'):
        compile_rewards(predictions(), {}, verified_failure_positive_scale=scale)
    monkeypatch.setenv('GFAM_VERIFIED_FAILURE_POSITIVE_SCALE', str(scale))
    with pytest.raises(ValueError, match='verified_failure_positive_scale'):
        runtime.GraphPRMV2RewardScorer('never-load.pkl')


@pytest.mark.parametrize('env,override,expected', [('0.1', None, .1), ('0.1', .5, .5), (None, None, .25)])
def test_runtime_reads_gate_setting_once_at_startup(monkeypatch, env, override, expected):
    if env is None:
        monkeypatch.delenv('GFAM_VERIFIED_FAILURE_POSITIVE_SCALE', raising=False)
    else:
        monkeypatch.setenv('GFAM_VERIFIED_FAILURE_POSITIVE_SCALE', env)
    monkeypatch.setattr(runtime.torch, 'load', lambda *a, **kw: {'feature_spec': runtime.DEFAULT_SPEC})
    monkeypatch.setattr(runtime, 'validate_checkpoint', lambda cp: None)
    monkeypatch.setattr(runtime, 'instantiate_model_from_checkpoint', lambda *a: SimpleNamespace(requires_grad_=lambda x: None))
    monkeypatch.setattr(runtime, 'LiveFeatureEngine', lambda *a, **kw: None)
    scorer = runtime.GraphPRMV2RewardScorer('mock.pkl', verified_failure_positive_scale=override)
    assert scorer.verified_failure_positive_scale == expected
    monkeypatch.setenv('GFAM_VERIFIED_FAILURE_POSITIVE_SCALE', '0.99')
    assert scorer.verified_failure_positive_scale == expected


def test_gate_reaches_worker_training_samples_including_final_worker(tmp_path, monkeypatch):
    backend = importlib.import_module(api.__name__ + '.backends').MockHierarchicalBackend
    original = backend.execute_worker
    final_calls = []
    def mixed_final_answers(self, task, decomposition, node, *args, **kwargs):
        execution = original(self, task, decomposition, node, *args, **kwargs)
        if node.node_id == decomposition.final_node_id:
            final_calls.append(node.node_id)
            execution.output_text = execution.raw_output_text = '4' if len(final_calls) % 2 else '5'
        return execution
    monkeypatch.setattr(backend, 'execute_worker', mixed_final_answers)
    result, trainer, scorer = rollout(tmp_path)
    expected_rewards = {}
    outcomes = set()
    final_workers_checked = 0
    for dec in result.decompositions:
        for sel in dec.selections:
            outcome = sel.reward.final_answer_correctness
            outcomes.add(outcome)
            scale = .25 if outcome == 0 else 1.
            compiled = sel.reward_model_outputs['compiled_rewards']
            assert sel.reward.total_reward == compiled['final']['reward']
            assert sel.reward.total_reward == pytest.approx(.54 * scale)
            for ex in sel.executions:
                payload = compiled['workers'][ex.node_id]
                assert ex.reward_model_reward == pytest.approx(.54 * scale)
                assert trainer.orchestrator._worker_training_reward(sel, ex) == payload['reward']
                key = (dec.decomposition.decomposition_id, sel.selection.selection_id, ex.node_id)
                expected_rewards[key] = payload['reward']
                final_workers_checked += ex.node_id == dec.decomposition.final_node_id
    assert outcomes == {0., 1.}
    assert final_workers_checked == 8
    assert result.training_batch.worker_samples
    for sample in result.training_batch.worker_samples:
        key = tuple(sample.metadata[name] for name in ('decomposition_id', 'selection_id', 'node_id'))
        assert sample.reward == expected_rewards[key]


def test_four_repeats_are_averaged(tmp_path):
    result, trainer, scorer = rollout(tmp_path)
    assert len(scorer.calls) == 8
    for dec in result.decompositions:
        values = [s.reward_model_outputs['compiled_rewards']['decomposer']['reward'] for s in dec.selections]
        assert dec.decomposition_reward == pytest.approx(sum(values)/4)
        for sel in dec.selections:
            assert sel.reward_model_outputs['graph_summary']['execution_outcome'] == sel.reward.final_answer_correctness
    assert trainer.decomposer_reward_aggregation == 'mean'
    config = api.RolloutConfig(num_decompositions=16, num_executor_rollouts_per_decomposition=16)
    assert trainer.orchestrator._effective_rollout_counts(config, api.TrainingScheduleConfig(
        mode=api.TrainingMode.ALTERNATING, alternating_phase=api.AlternatingPhase.DECOMPOSER)) == (8, 4)


@pytest.mark.parametrize('failures', [{1}, set(range(1, 9))])
def test_unscored_never_falls_back_or_enters_learning(tmp_path, failures):
    result, trainer, scorer = rollout(tmp_path, failures)
    missing = [s for d in result.decompositions for s in d.selections if s.reward.total_reward is None]
    assert len(missing) == len(failures)
    failed_ids = {(d.decomposition.decomposition_id, s.selection.selection_id)
                  for d in result.decompositions for s in d.selections if s.reward.total_reward is None}
    assert all((sample.metadata['decomposition_id'], sample.metadata['selection_id']) not in failed_ids
               for sample in result.training_batch.worker_samples)
    assert len(result.training_batch.decomposer_samples) == (1 if len(failures) == 1 else 0)
    for sel in missing:
        assert sel.reward_model_outputs['compiled_rewards'] == {}
        assert all(ex.reward_model_reward is None for ex in sel.executions)
        with pytest.raises(ValueError, match='Unscored'):
            trainer.orchestrator._worker_training_reward(sel, sel.executions[0])
    saved = [json.loads(line) for line in (tmp_path/'reward_scoring_failures.jsonl').read_text().splitlines()]
    assert len(saved) == len(failures)
    assert saved[0]['executor_rollout']['reward']['total_reward'] is None
    summary = train.epoch_rollout_summary([result], include_subsets=True)
    assert summary['unscored_rollouts'] == len(failures)
    if len(failures) == 8:
        assert math.isnan(summary['mean_worker_reward'])
        assert summary['reward_metric_task_counts']['mean_worker_reward'] == 0
        assert not (tmp_path/'best_selections.jsonl').exists()
        assert not result.training_batch.worker_samples


def test_metric_aggregation_uses_scored_denominators(tmp_path):
    valid, _, _ = rollout(tmp_path/'valid')
    missing, _, _ = rollout(tmp_path/'missing', range(1, 9))
    good = train.epoch_rollout_summary([valid], include_subsets=True)
    empty = train.epoch_rollout_summary([missing], include_subsets=True)
    combined = train.combine_rollout_summaries([good, empty], include_subsets=True)
    direct = train.epoch_rollout_summary([valid, missing], include_subsets=True)
    for key in good['reward_metric_task_counts']:
        assert combined[key] == pytest.approx(direct[key])
    assert combined['mean_worker_reward'] == good['mean_worker_reward']
    assert combined['unscored_rollouts'] == 8
    reloaded = json.loads(json.dumps(train._json_safe([good, empty]), allow_nan=False))
    assert train.combine_rollout_summaries(reloaded)['mean_worker_reward'] == good['mean_worker_reward']


@pytest.mark.parametrize('visible', ['0,1,2,3', '3,5,6,7', 'GPU-a,GPU-b', 'MIG-a,MIG-b'])
def test_gpu_masks_are_disjoint(visible):
    policy = gpu.partition(visible, 1, 'policy').split(',')
    reward = gpu.partition(visible, 1, 'reward').split(',')
    assert not set(policy) & set(reward)
    assert policy + reward == visible.split(',')


@pytest.mark.parametrize('visible', ['', '0', '0,0', '-1,0'])
def test_gpu_isolation_fails_closed(visible):
    with pytest.raises(ValueError):
        gpu.partition(visible, 1, 'reward')


def test_gpu_wrapper_sets_mask_before_child_starts():
    result = subprocess.run([sys.executable, gpu.__file__, '--role', 'reward', '--',
                             sys.executable, '-c', 'import os; print(os.environ["CUDA_VISIBLE_DEVICES"])'],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='2,4,6,7'),
                            capture_output=True, text=True, check=True)
    assert result.stdout.splitlines()[-1] == '7'


def test_entire_training_batch_can_be_unscored(tmp_path, monkeypatch):
    gfam = importlib.import_module(api.__name__ + '.gfam_reward')
    monkeypatch.setattr(gfam, 'load_cached_reward_scorer', lambda **kw: Scorer(range(1, 1000)))
    monkeypatch.setattr(train, 'run_offline_policy_training', lambda **kw: pytest.fail('No unscored training'))
    monkeypatch.setattr(sys, 'argv', ['train.py', '--task-source', 'demo', '--backend', 'mock',
                                    '--num-epochs', '1', '--gfam-reward-model-pkl', 'mock.pkl',
                                    '--rollout-progress-every', '1', '--output-dir', str(tmp_path)])
    train.main()
    payload = json.loads((tmp_path/'epoch_0001'/'rollout_summary.json').read_text())
    assert payload['mean_worker_reward'] is None
    assert payload['mean_best_decomposition_reward'] is None
    assert payload['unscored_rollouts'] > 0

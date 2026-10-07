"""Portable Graph+PRM runtime tests; no model downloads or GPU required."""
import copy
from types import SimpleNamespace

import pytest
import torch

try:
    from verl.hierarchical_rema import graphprm_core as core, graphprm_model as model, graphprm_reward as runtime, gfam_reward
    from verl.hierarchical_rema.graphprm_compiler import compile_rewards_from_scores
except ModuleNotFoundError:
    from hierarchical_rema import graphprm_core as core, graphprm_model as model, graphprm_reward as runtime, gfam_reward
    from hierarchical_rema.graphprm_compiler import compile_rewards_from_scores

import importlib
inputs = importlib.import_module(gfam_reward.__package__ + '.graphprm_inputs')


def record():
    tasks = [dict(node_id='A', instruction='Compute 2 + 2', dependencies=[], required_skills=[]),
             dict(node_id='B', instruction='Multiply A by 3', dependencies=['A'], required_skills=[])]
    return dict(trajectory_id='t::plan::executor-1', task=dict(prompt='Compute (2+2)*3', ground_truth='SECRET'),
                decomposition=dict(subtasks=tasks, final_node_id='B'),
                trajectory=dict(final_node_id='B', final_answer='12'),
                workers=[dict(node_id='A', output_text='<worker_scratchpad>PRIVATE</worker_scratchpad><worker_result>4</worker_result>', dependencies=[], is_final_node=False),
                         dict(node_id='B', output_text='12', dependencies=['A'], is_final_node=True)],
                graph=dict(used_dependency_edges=[dict(source='A', target='B')]))


class FakeEncoder:
    def encode_texts(self, texts):
        assert not any('SECRET' in text or 'PRIVATE' in text for text in texts)
        return torch.ones((len(texts), 4))


@pytest.mark.parametrize('probability', [0., .25, .5, 2/3, .7, 1.])
def test_binary_utility_threshold_and_default(probability):
    import importlib
    compiler = importlib.import_module(gfam_reward.__package__ + '.graphprm_compiler')
    centered = 2 * probability - 1
    assert compiler.binary_label_utility(centered) == centered
    assert compiler.binary_label_utility(centered, 2.) == pytest.approx(3 * probability - 2)


@pytest.mark.parametrize('penalty', [-1., float('nan'), float('inf')])
def test_invalid_penalty_fails_before_model_loading(penalty):
    with pytest.raises(ValueError, match='finite and non-negative'):
        runtime.GraphPRMRewardScorer('unused.pkl', bad_class_penalty=penalty)


class FakeExtractor:
    calls = 0
    def __init__(self, *args, **kwargs):
        pass
    def render(self, row):
        return model.render_graphprm_sequence(row, tokenizer=None, edge_mode='prefer_used',
            max_problem_chars=3000, max_step_chars=1400, max_dependency_chars=450, include_final_marker=True)
    def extract_batch(self, prompts):
        type(self).calls += 1
        assert not any('SECRET' in p or 'PRIVATE' in p for p in prompts)
        return [dict(marker_count=3, marker_positions=[0, 1, 2], scores=[.3, .7, .8], hidden_states=torch.ones((3, 8)))]
    def alignment(self, render):
        return dict(marker_positions=list(range(min(3, len(render['targets'])))),
                    target_keys=[[t['scope'], t.get('node_id', '')] for t in render['targets']],
                    total_tokens=1000, retained_tokens=512)


@pytest.fixture
def checkpoint(tmp_path):
    layout = model.GraphPRMFeatureLayout('bge_prm_hidden', 4, 8, 2, ('has_prm_marker', 'is_final_prm_marker'))
    model.set_graphprm_layout(layout)
    config = dict(hidden_dim=16, message_passing_layers=1, dropout=0., graph_architecture='gat', gat_heads=2, node_head_context='ego')
    network = model.GraphPRMHybridModel(text_dim=14, metadata_dim=8, **config)
    ckpt = dict(config=config, model_state_dict=network.state_dict(), node_types=core.NODE_TYPES,
                edge_types=core.EDGE_TYPES, label_schema=dict(graph=core.GRAPH_LABELS,
                decomposer=core.DECOMPOSER_LABELS, worker=core.WORKER_LABELS, final=core.FINAL_LABELS),
                encoder_spec=dict(backend='graphprm_hybrid', feature_layout=layout.to_json(),
                bge_encoder=dict(backend='sentence-transformers', model_name='test'),
                prm_features=dict(model='Qwen/Qwen2.5-Math-PRM-7B')))
    path = tmp_path / 'best_model.pkl'
    torch.save(ckpt, path)
    return path


def test_binary_auto_dispatch_cache_and_no_reference_leak(checkpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(gfam_reward, 'build_sentence_encoder', lambda *a: FakeEncoder())
    monkeypatch.setattr(runtime, 'FrozenQwenMarkerFeatureExtractor', FakeExtractor)
    FakeExtractor.calls = 0
    scorer = gfam_reward.GFAMRewardScorer(str(checkpoint), device='cpu', feature_cache_dir=str(tmp_path/'cache'))
    result = scorer.graphprm.score_record(record())
    changed = record()
    changed['task']['ground_truth'] = 'CHANGED SECRET'
    changed['trajectory_id'] = 't::plan::executor-2'
    result2 = scorer.graphprm.score_record(changed)
    assert FakeExtractor.calls == 1
    assert result['model_predictions'] == result2['model_predictions']
    assert 'selector' not in result['compiled_rewards']
    assert set(result['compiled_rewards']['workers']) == {'A', 'B'}
    for payload in result['model_predictions']['workers']['A'].values():
        assert set(payload['probabilities']) == {'0', '1'}
        assert payload['centered_score'] == pytest.approx(2*payload['probabilities']['1'] - 1, abs=1e-6)
    assert result['graph_summary']['prm_max_length'] == 512


def test_encoders_reused_across_rollout_segments(checkpoint, monkeypatch):
    monkeypatch.setattr(gfam_reward, 'build_sentence_encoder', lambda *a: FakeEncoder())
    gfam_reward.load_cached_reward_scorer.cache_clear()
    try:
        first = gfam_reward.load_cached_reward_scorer(checkpoint_path=str(checkpoint), device='cpu')
        second = gfam_reward.load_cached_reward_scorer(checkpoint_path=str(checkpoint), device='cpu')
        assert first is second
    finally:
        gfam_reward.load_cached_reward_scorer.cache_clear()


def test_penalty_changes_rewards_not_predictions_or_feature_cache(checkpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(gfam_reward, 'build_sentence_encoder', lambda *a: FakeEncoder())
    monkeypatch.setattr(runtime, 'FrozenQwenMarkerFeatureExtractor', FakeExtractor)
    FakeExtractor.calls = 0
    results = []
    for penalty in (1., 2.):
        scorer = gfam_reward.GFAMRewardScorer(str(checkpoint), device='cpu',
            feature_cache_dir=str(tmp_path/'cache'), bad_class_penalty=penalty)
        results.append(scorer.graphprm.score_record(record()))
    symmetric, asymmetric = results
    assert FakeExtractor.calls == 1
    assert symmetric['model_predictions'] == asymmetric['model_predictions']
    assert asymmetric['graph_summary']['bad_class_penalty'] == 2.
    assert asymmetric['graph_summary']['label_utility_positive_threshold'] == pytest.approx(2/3)
    for scope in ('decomposer', 'final'):
        before = symmetric['compiled_rewards'][scope]
        after = asymmetric['compiled_rewards'][scope]
        assert after['quality'] == pytest.approx(1.5 * before['quality'] - .5)
        assert after['reward'] <= before['reward']
        assert after['positive_cap'] == before['positive_cap']
    for node in ('A', 'B'):
        before = symmetric['compiled_rewards']['workers'][node]
        after = asymmetric['compiled_rewards']['workers'][node]
        assert after['quality'] == pytest.approx(1.5 * before['quality'] - .5)
        assert after['badness'] == before['badness']
        assert after['reward'] <= before['reward']


def test_features_map_by_execution_node_not_worker_identity():
    row = runtime.normalize_record(record())
    blueprint = core.prepare_graph_blueprint(row)
    assert len(blueprint.worker_indices_by_node_id) == 2
    assert all(core.NODE_TYPES[t] != 'selector' for t in blueprint.node_type_ids)
    layout = model.GraphPRMFeatureLayout('bge_prm_hidden', 4, 8, 2, ('has_prm_marker', 'is_final_prm_marker'))
    payload = dict(targets=[dict(scope='worker', node_id='A'), dict(scope='worker', node_id='B'), dict(scope='final')],
                   marker_count=1, hidden_states=torch.full((1, 8), 3.), scores=[.5])
    features = model.build_fused_feature_tensor(blueprint=blueprint, bge_embeddings=torch.zeros((7, 4)), prm_payload=payload, layout=layout)
    assert features[blueprint.worker_indices_by_node_id['A'], 4:12].tolist() == [3.]*8
    assert features[blueprint.worker_indices_by_node_id['B'], 4:12].count_nonzero() == 0
    assert features[blueprint.final_index, -1] == 1


def test_decomposer_reward_responds_to_execution_and_collapse():
    row = runtime.normalize_record(record())
    good = {name: 1. for name in core.WORKER_LABELS}
    inputs = dict(example=SimpleNamespace(source_record=row), graph_scores=dict(global_outcome=1., anti_collapse=1.),
                  decomposer_scores=dict(task_fulfillment=1., execution_discipline=1.),
                  worker_scores=dict(A=good, B=good), final_scores=good, final_anchor_score=1.)
    healthy = compile_rewards_from_scores(inputs)['node_rewards']['decomposer']['reward']
    failed = copy.deepcopy(inputs)
    failed['final_anchor_score'] = -1.
    failed['graph_scores']['anti_collapse'] = -1.
    assert compile_rewards_from_scores(failed)['node_rewards']['decomposer']['reward'] < healthy


def test_nonfinite_prm_rejected(checkpoint, monkeypatch):
    monkeypatch.setattr(gfam_reward, 'build_sentence_encoder', lambda *a: FakeEncoder())
    class Broken(FakeExtractor):
        def extract_batch(self, prompts):
            return [dict(marker_count=3, hidden_states=torch.full((3, 8), float('nan')), scores=[.5]*3)]
    monkeypatch.setattr(runtime, 'FrozenQwenMarkerFeatureExtractor', Broken)
    scorer = runtime.GraphPRMRewardScorer(checkpoint)
    with pytest.raises(RuntimeError, match='Non-finite Qwen'):
        scorer.score_record(record())


@pytest.mark.parametrize('phase', ['joint', 'executor', 'decomposer'])
def test_single_executor_rollouts_route_binary_rewards(checkpoint, monkeypatch, phase):
    import importlib
    package = gfam_reward.__package__
    api = importlib.import_module(package)
    demo = importlib.import_module(package + '.demo')
    monkeypatch.setattr(gfam_reward, 'build_sentence_encoder', lambda *a: FakeEncoder())
    monkeypatch.setattr(runtime, 'FrozenQwenMarkerFeatureExtractor', FakeExtractor)
    scorer = gfam_reward.GFAMRewardScorer(str(checkpoint), device='cpu')
    trainer = api.HierarchicalGRPOTrainer(gfam_reward_scorer=scorer, train_worker_model=True, min_worker_grpo_group_size=2)
    schedule = api.TrainingScheduleConfig(mode=api.TrainingMode.JOINT) if phase == 'joint' else api.TrainingScheduleConfig(
        mode=api.TrainingMode.ALTERNATING, alternating_phase=api.AlternatingPhase(phase))
    rollout = trainer.run(task=api.TaskExample(task_id='test', prompt='Solve for x: 2x + 3 = 11.', ground_truth='4',
                        metadata={'skill_focus': 'algebra', 'distractor_answer': '5'}),
        worker_pool=demo.make_worker_pool(base_model_path='mock'),
        policy_config=api.ControllerPolicyConfig(parameter_sharing=True, shared_model_path='mock'),
        rollout_config=api.RolloutConfig(num_decompositions=1, num_executor_rollouts_per_decomposition=2), schedule=schedule)
    for decomposition in rollout.decompositions:
        assert len(decomposition.selections) >= 2
        for execution_rollout in decomposition.selections:
            assert execution_rollout.reward_model_outputs['backend'] == 'graphprm_binary'
            compiled = execution_rollout.reward_model_outputs['compiled_rewards']
            assert execution_rollout.reward.total_reward == compiled['final']['reward']
            for execution in execution_rollout.executions:
                assert execution.worker_id == 'executor_worker'
                assert execution.reward_model_reward == compiled['workers'][execution.node_id]['reward']
            assert all(a.reward_model_reward is None for a in execution_rollout.selection.assignments)
    batch = rollout.training_batch
    assert not batch.selector_samples
    assert bool(batch.decomposer_samples) == (phase != 'executor')
    assert bool(batch.worker_samples) == (phase != 'decomposer')


@pytest.mark.parametrize('text', [
    'x < 3 and y > 2', 'a<b<=c and a+b>c', 'x <= 4\ny >= 2',
    'Map <A,B> to <C,D>', '<unknown>not protocol</unknown>',
])
def test_cleaner_preserves_math_and_is_idempotent(text):
    for wrapped in (text, '<worker_result>' + text + '</worker_result>'):
        assert core.result_text(wrapped) == text
        assert core.result_text(core.result_text(wrapped)) == text


def test_unclosed_scratchpad_is_not_exposed():
    assert core.result_text('<worker_scratchpad>PRIVATE') == ''
    assert core.result_text('<scratchpad>PRIVATE</scratchpad><result>0</result>') == '0'
    assert inputs.delivered_result({'output_text': 0}) == '0'


def test_rejected_artifact_is_never_restored_and_raw_is_preserved():
    row = record()
    row['workers'][0].update(output_text='', raw_output_text='<worker_result>SECRET42</worker_result>', invalid_reason='missing_worker_result')
    normalized = runtime.normalize_record(row)
    assert normalized['workers'][0]['output_text'] == ''
    assert normalized['workers'][0]['raw_output_text'] == row['workers'][0]['raw_output_text']
    assert not any('SECRET42' in text for text in core.prepare_graph_blueprint(normalized).node_texts)
    render = FakeExtractor().render(normalized)
    assert 'SECRET42' not in render['prompt']
    assert inputs.delivered_result({'raw_output_text': 'SECRET42'}) == ''


def test_protocol_guard_is_separate_and_reaches_rewards(checkpoint, monkeypatch):
    monkeypatch.setattr(gfam_reward, 'build_sentence_encoder', lambda *a: FakeEncoder())
    monkeypatch.setattr(runtime, 'FrozenQwenMarkerFeatureExtractor', FakeExtractor)
    row = record()
    row['workers'][1].update(output_text='', raw_output_text='12', invalid_reason='missing_worker_result')
    row['trajectory']['final_answer'] = ''
    scorer = runtime.GraphPRMRewardScorer(checkpoint)
    result = scorer.score_record(row)
    for payload in (result['compiled_rewards']['workers']['B'], result['compiled_rewards']['final']):
        assert payload['reward'] <= -.10
        assert payload['protocol_invalid_reason'] == 'missing_worker_result'
        assert payload['reward'] == min(payload['reward_before_protocol_guard'], 0) - .10
    assert result['record']['workers'][1]['raw_output_text'] == '12'


def test_live_adapter_uses_original_delivered_fields(monkeypatch):
    # A legacy cleaner can remove an unclosed scratchpad tag before we see it.
    legacy = runtime.normalize_record(record())
    legacy['workers'][0]['upstream_context'] = [dict(node_id='X', used_value='LEAK')]
    monkeypatch.setattr(gfam_reward, '_build_inference_record', lambda *a: legacy)
    scorer = runtime.GraphPRMRewardScorer.__new__(runtime.GraphPRMRewardScorer)
    scorer.score_record = runtime.normalize_record
    a = SimpleNamespace(node_id='A', output_text='<scratchpad>LEAK',
        dependency_outputs={'X': '<scratchpad>LEAK'}, invalid_reason='missing_worker_result',
        raw_output_text='<worker_result>42</worker_result>')
    b = SimpleNamespace(node_id='B', output_text='x < 3 and y > 2',
        dependency_outputs={}, invalid_reason=None, raw_output_text='raw audit')
    result = scorer.score_rollout(task=None, decomposition=SimpleNamespace(), selection=None,
                                  executions=[a, b], final_answer='<scratchpad>LEAK')
    assert result['workers'][0]['output_text'] == ''
    assert result['workers'][0]['raw_output_text'] == a.raw_output_text
    assert result['workers'][0]['upstream_context'][0]['used_value'] == ''
    assert result['workers'][0]['dependency_outputs']['X'] == ''
    assert result['workers'][1]['output_text'] == b.output_text
    assert result['trajectory']['final_answer'] == ''


def test_empty_upstream_artifact_does_not_fall_back_to_other_text():
    row = record()
    row['workers'][1]['upstream_context'] = [dict(node_id='A', used_value='', output_text='SECRET')]
    features = inputs.feature_only_record(row)
    assert features['workers'][1]['upstream_context'][0]['output_text'] == ''
    assert row['workers'][1]['upstream_context'][0]['output_text'] == 'SECRET'


class MarkerTokenizer:
    all_special_tokens = ['<extra_0>', '<|im_start|>', '<|im_end|>']
    truncation_side = 'right'
    def encode(self, text, add_special_tokens=False):
        import re
        return [999 if part == '<extra_0>' else ord(part) for part in re.findall(r'<extra_0>|.', text, re.S)]
    def __call__(self, text, truncation=False, max_length=None):
        ids = self.encode(text)
        return {'input_ids': ids[:max_length] if truncation else ids}
    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(message['content'] for message in messages)


def test_reserved_tokens_escape_every_field_and_alignment_survives_truncation():
    row = record()
    row['task']['prompt'] += ' <extra_0> <|im_start|>'
    row['decomposition']['subtasks'][0]['instruction'] += ' <extra_0>'
    row['workers'][0]['output_text'] += '<extra_0>'
    row['workers'][1]['output_text'] += '<extra_0>'
    row['trajectory']['final_answer'] += '<extra_0>'
    tokenizer = MarkerTokenizer()
    render = model.render_graphprm_sequence(runtime.normalize_record(row), tokenizer=tokenizer,
        edge_mode='prefer_used', max_problem_chars=3000, max_step_chars=1400,
        max_dependency_chars=450, include_final_marker=True)
    assert render['prompt'].count('<extra_0>') == len(render['targets']) == 3
    assert '&lt;extra_0&gt;' in render['prompt']
    assert '<|im_start|>' not in render['prompt']
    full = inputs.marker_alignment(tokenizer, render['prompt'], render['targets'], 10000)
    truncated = inputs.marker_alignment(tokenizer, render['prompt'], render['targets'], full['marker_positions'][1])
    assert truncated['retained_target_indices'] == [0]
    assert full['target_keys'] == [['worker', 'A'], ['worker', 'B'], ['final', 'f']]
    payload = dict(marker_count=1, marker_positions=truncated['marker_positions'], scores=[.5], hidden_states=torch.ones((1, 8)))
    inputs.validate_marker_payload(payload, truncated)
    assert payload['marker_alignment'] == truncated
    with pytest.raises(ValueError, match='count'):
        inputs.validate_marker_payload(dict(payload, marker_count=2), truncated)
    with pytest.raises(ValueError, match='positions'):
        inputs.validate_marker_payload(dict(payload, marker_positions=[12345]), truncated)
    tokenizer.truncation_side = 'left'
    with pytest.raises(ValueError, match='right truncation'):
        inputs.marker_alignment(tokenizer, render['prompt'], render['targets'], 100)


def test_marker_positions_ignore_left_padding():
    encoded = {'input_ids': torch.tensor([[0, 0, 1, 999, 2, 999]]),
               'attention_mask': torch.tensor([[0, 0, 1, 1, 1, 1]])}
    hidden = torch.arange(48).reshape(1, 6, 8).float()
    payload = model.marker_features_from_outputs(outputs=SimpleNamespace(logits=torch.zeros(1, 6, 2), hidden_states=(hidden,)),
        encoded=encoded, step_sep_id=999, include_hidden=True, include_score=True)[0]
    assert payload['marker_positions'] == [1, 3]
    torch.testing.assert_close(payload['hidden_states'].float(), hidden[0, [3, 5]])


@pytest.mark.parametrize('batched', [False, True])
def test_live_execution_and_learned_rewards_are_reference_invariant(checkpoint, tmp_path, monkeypatch, batched):
    package = gfam_reward.__package__
    schema = importlib.import_module(package + '.schema')
    backends = importlib.import_module(package + '.backends')
    orchestration = importlib.import_module(package + '.orchestrator')
    monkeypatch.setattr(gfam_reward, 'build_sentence_encoder', lambda *a: FakeEncoder())
    monkeypatch.setattr(runtime, 'FrozenQwenMarkerFeatureExtractor', FakeExtractor)
    blueprints = []
    original_builder = core.prepare_graph_blueprint
    def capture_blueprint(row):
        blueprint = original_builder(row)
        blueprints.append(copy.deepcopy((blueprint.node_texts, blueprint.metadata_rows, blueprint.edge_rows)))
        return blueprint
    monkeypatch.setattr(core, 'prepare_graph_blueprint', capture_blueprint)
    FakeExtractor.calls = 0
    scorer = runtime.GraphPRMRewardScorer(checkpoint, cache_dir=tmp_path / 'reference-invariance')
    backend = backends.TransformersHierarchicalBackend()
    completion = '<worker_result>42</worker_result>'
    monkeypatch.setattr(backend, '_generate_text', lambda **kw: (completion, .1))
    monkeypatch.setattr(backend, '_generate_text_batch',
                        lambda **kw: [(completion, .1) for _ in kw['prompt_texts']])
    orchestrator = orchestration.HierarchicalReMAOrchestrator(
        backend=backend, reward_weights=schema.RewardWeights(), worker_memory=None, gfam_reward_scorer=scorer)
    worker = schema.WorkerSpec('executor_worker', 'Math executor', [], '', base_model_path='stub')
    pool = schema.WorkerPoolConfig(base_model_path='stub', workers=[worker])
    plan = schema.DecompositionCandidate('plan', '', 'product', 'integer', [
        schema.SubtaskNode('A', 'Compute 6 * 7'),
        schema.SubtaskNode('B', 'Format the product as an integer', ['A']),
    ], 'B')
    rollouts = []
    for truth in ('42', '43', ''):
        task = schema.TaskExample('same-task', 'Compute 6 * 7', truth)
        selection = schema.SelectionCandidate('same-execution', [
            schema.WorkerAssignment(node.node_id, worker.worker_id, rationale='Shared executor', compatibility=1.) for node in plan.nodes
        ], raw_payload={'synthetic_executor_rollout': True})
        if batched:
            state = orchestration._SelectionExecutionState(0, 0, 0, task, plan, selection, ['A', 'B'])
            rollout = orchestrator._execute_selections_batch([state], pool)[(0, 0, 0)]
        else:
            rollout = orchestrator._execute_selection(task, plan, selection, pool)
        rollouts.append(rollout)
        assert rollout.final_answer == '42'
        assert rollout.executions[1].dependency_outputs == {'A': '42'}
        assert all(e.success and not e.invalid_reason and not e.final_answer_leak for e in rollout.executions)
    assert [r.executions[0].reference_answer_match for r in rollouts] == [True, False, False]
    assert [r.reward.final_answer_correctness for r in rollouts] == [1., 0., 0.]
    assert FakeExtractor.calls == 1
    assert blueprints[0] == blueprints[1] == blueprints[2]
    for other in rollouts[1:]:
        assert [e.worker_prompt for e in other.executions] == [e.worker_prompt for e in rollouts[0].executions]
        assert other.reward_model_outputs == rollouts[0].reward_model_outputs
        assert [e.reward_model_reward for e in other.executions] == [e.reward_model_reward for e in rollouts[0].executions]

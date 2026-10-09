"""Batching/placement parity tests with deterministic encoders, no downloads."""
import hashlib
import importlib
import re
import sys
from datetime import timedelta
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import pytest
import torch

try:
    import verl.hierarchical_rema as api
except ModuleNotFoundError:
    import hierarchical_rema as api

pkg = api.__name__
live = importlib.import_module(pkg + '.graphprm_v2.live_features')
features = importlib.import_module(pkg + '.graphprm_v2.features')
gm = importlib.import_module(pkg + '.graphprm_v2.graphprm_model')
core = importlib.import_module(pkg + '.graphprm_v2.graphprm_core')
runtime = importlib.import_module(pkg + '.graphprm_v2.runtime')
offline = importlib.import_module(pkg + '.offline_training')
train = importlib.import_module(pkg + '.train')
demo = importlib.import_module(pkg + '.demo')


class Batch(dict):
    def to(self, device):
        return Batch({k: v.to(device) for k, v in self.items()})


class Tokenizer:
    padding_side = 'left'
    all_special_tokens = ['<extra_0>']

    def encode(self, text, **kwargs):
        return [99 if token == '<extra_0>' else 1 + sum(token.encode()) % 80
                for token in re.findall(r'<extra_0>|\S+', text)]

    def __call__(self, text, truncation=False, max_length=None, **kwargs):
        if isinstance(text, str):
            ids = self.encode(text)
            return {'input_ids': ids[:max_length] if truncation else ids}
        ids = [self.encode(item) for item in text]
        size = max(map(len, ids))
        return Batch(input_ids=torch.tensor([[0] * (size-len(row)) + row for row in ids]),
                     attention_mask=torch.tensor([[0] * (size-len(row)) + [1] * len(row) for row in ids]))


class Encoder:
    def __init__(self, *args, **kwargs):
        self.tokenizer = self.tokenize
        self.calls = []

    def tokenize(self, text, **kwargs):
        return {'input_ids': [1] * (9000 if 'BGE_OVERFLOW' in text else len(text.split()))}

    def eval(self):
        return self

    def requires_grad_(self, value):
        return self

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        values = torch.tensor([list(hashlib.sha256(text.encode()).digest()[:4]) for text in texts]).float()
        return torch.nn.functional.normalize(values, dim=1)


class PRM(torch.nn.Module):
    def __init__(self, max_batch=100):
        super().__init__()
        self.calls = []
        self.max_batch = max_batch

    def forward(self, input_ids, attention_mask, **kwargs):
        self.calls.append(len(input_ids))
        if len(input_ids) > self.max_batch:
            raise torch.cuda.OutOfMemoryError('simulated capacity limit')
        values = input_ids.cumsum(dim=1).float() / 1000
        hidden = values.unsqueeze(-1).expand(-1, -1, 8).clone()
        return SimpleNamespace(logits=torch.stack([-values, values], dim=-1), hidden_states=(hidden,))


def row(index=0, extra=''):
    return runtime.canonical_record({
        'task': {'prompt': f'Calculate 2 + {index}. ' + extra},
        'decomposition': {'subtasks': [
            {'node_id': 'A', 'instruction': 'Compute the partial sum', 'dependencies': []},
            {'node_id': 'B', 'instruction': 'Report the sum', 'dependencies': ['A']}], 'final_node_id': 'B'},
        'workers': [{'node_id': 'A', 'output_text': str(index+2), 'dependency_outputs': {}},
                    {'node_id': 'B', 'output_text': str(index+2), 'dependency_outputs': {'A': str(index+2)}}],
        'trajectory': {'final_answer': str(index+2), 'final_node_id': 'B'},
    })


@pytest.fixture
def spec(monkeypatch):
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=Encoder))
    monkeypatch.setattr(features, 'load_tokenizer', lambda *a: Tokenizer())
    monkeypatch.setattr(features, 'load_qwen_prm', lambda *a: (PRM(), 'cpu'))
    monkeypatch.setattr(live, 'load_qwen_prm', lambda *a: (PRM(), 'cpu'))
    return dict(features.DEFAULT_SPEC, bge_dim=4, prm_hidden_dim=8)


def network(spec):
    gm.set_graphprm_layout(features.layout_from_spec(spec))
    torch.manual_seed(17)
    return gm.GraphPRMHybridModel(text_dim=14, metadata_dim=len(core.METADATA_NAMES), hidden_dim=16,
                                  message_passing_layers=1, dropout=0., graph_architecture='relational',
                                  gat_heads=2, node_head_context='ego').eval()


def scorer(engine, spec, batch_size=64):
    value = runtime.GraphPRMV2RewardScorer.__new__(runtime.GraphPRMV2RewardScorer)
    value.engine, value.device, value.lock = engine, torch.device('cpu'), RLock()
    value.model, value.bad_class_penalty = network(spec), 1.
    value.rollout_batch_size = batch_size
    return value


@pytest.mark.parametrize('padding', ['left', 'right'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_marker_only_transfer_matches_reference(padding, dtype):
    ids = torch.tensor([[1, 99, 2, 99], [0, 0, 3, 99]] if padding == 'left'
                       else [[1, 99, 2, 99], [3, 99, 0, 0]])
    encoded = Batch(input_ids=ids, attention_mask=(ids != 0).long())
    torch.manual_seed(4)
    outputs = SimpleNamespace(logits=torch.randn(2, 4, 2).to(dtype),
                              hidden_states=(torch.randn(2, 4, 8).to(dtype),))
    original = gm.marker_features_from_outputs(outputs=outputs, encoded=encoded, step_sep_id=99,
                                               include_hidden=True, include_score=True)
    optimized = live.marker_features_on_device(outputs=outputs, encoded=encoded, step_sep_id=99)
    for before, after in zip(original, optimized):
        assert before['scores'] == after['scores']
        assert before['marker_positions'] == after['marker_positions']
        assert torch.equal(before['hidden_states'], after['hidden_states'])


def test_live_features_equal_offline_and_batch_rewards_equal_single(spec, tmp_path):
    rows = [row(i, 'extra ' * i) for i in (8, 0, 5, 2)]
    original = features.FeatureEngine(spec, tmp_path/'reference', device='cpu', prm_batch_size=1)
    original.prepare(rows)
    engine = live.LiveFeatureEngine(spec, tmp_path/'live', device='cpu', prm_batch_size=16)
    batch_scorer = scorer(engine, spec)
    batch = batch_scorer.score_records(rows, [1, 0, 1, 0])
    assert engine.last_stats['prm_max_batch'] == 4
    assert engine.last_stats['prm_padded_tokens'] >= engine.last_stats['prm_input_tokens']
    for index, source in enumerate(rows):
        before, after = original.example(source), engine.example(source)
        for key, value in vars(before).items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, getattr(after, key)), key
            else:
                assert value == getattr(after, key), key
        single = batch_scorer.score_record(source, [1, 0, 1, 0][index])
        assert batch[index] == single
    assert engine.last_stats['cached_graphs'] == 1
    assert features.DEFAULT_SPEC['feature_code_sha256'] == spec['feature_code_sha256']
    # The live adapter accepts existing offline caches without encoder work.
    reused = live.LiveFeatureEngine(spec, tmp_path/'reference', device='cpu')
    assert reused.prepare_safe(rows) == {}
    assert reused.last_stats['cached_graphs'] == len(rows)
    assert reused.encoder is None and reused.prm_model is None


def test_one_overflow_does_not_shift_other_results(spec, tmp_path):
    engine = live.LiveFeatureEngine(spec, tmp_path, device='cpu')
    reward = scorer(engine, spec)
    rows = [row(1), row(2, 'word ' * 5000), row(3), row(4, 'BGE_OVERFLOW'), row(5)]
    results = reward.score_records(rows, [0, 1, 1, 1, 0])
    assert [item['status'] for item in results] == ['scored', 'unscored', 'scored', 'unscored', 'scored']
    assert results[1]['error']['encoder'] == 'prm'
    assert results[3]['error']['encoder'] == 'bge'
    for index in (0, 2, 4):
        assert results[index]['record'] == rows[index]
        assert results[index] == reward.score_record(rows[index], [0, 1, 1, 1, 0][index])


def test_oom_backoff_and_bounded_ram_cache(spec, tmp_path, monkeypatch):
    engine = live.LiveFeatureEngine(spec, tmp_path, device='cpu', prm_batch_size=16,
                                    bge_cache_size=2, graph_cache_size=2)
    engine.prm_model = PRM(max_batch=2)
    cleared = []
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: cleared.append(True))
    rows = [row(i) for i in range(5)]
    assert engine.prepare_safe(rows) == {}
    assert engine.prm_batch_size == 2
    assert engine.last_stats['oom_retries'] == 1
    assert engine.last_stats['prm_max_batch'] == 2
    assert len(engine._texts.items) <= 2
    assert len(engine._graphs.items) <= 2
    assert len(cleared) == 1
    assert engine.prepare_safe(rows) == {}
    assert engine.last_stats['prm_batches'] == 0
    assert len(cleared) == 1


def test_ram_cache_reuses_bge_without_disk_reads(spec, tmp_path, monkeypatch):
    engine = live.LiveFeatureEngine(spec, tmp_path, device='cpu')
    first = row(1)
    engine.prepare_safe([first])
    original_loader = live.load_cache
    loaded_texts = []

    def track(path, identity):
        if path.parent.name == 'bge':
            loaded_texts.append(identity)
        return original_loader(path, identity)

    monkeypatch.setattr(live, 'load_cache', track)
    engine.prepare_safe([row(2)])
    assert engine.last_stats['bge_ram_hits'] > 0
    assert len(loaded_texts) < len(engine._texts.items)


def test_single_row_oom_and_alignment_errors_remain_fatal(spec, tmp_path):
    engine = live.LiveFeatureEngine(spec, tmp_path, device='cpu')
    engine.prm_model = PRM(max_batch=0)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        engine.prepare_safe([row(1)])
    engine.prm_model = PRM()
    engine.render = lambda record: (_ for _ in ()).throw(ValueError('Unexpected reserved PRM markers'))
    with pytest.raises(ValueError, match='reserved PRM markers'):
        engine.prepare_safe([row(2)])


def test_orchestrator_batches_without_changing_node_rewards(tmp_path):
    pilot = importlib.import_module('test_graphprm_pilot')

    class Batched(pilot.Scorer):
        rollout_batch_size = 4
        batches = []

        def score_rollouts(self, requests):
            self.batches.append(len(requests))
            return [self.score_rollout(**request) for request in requests]

    outputs = []
    for reward_scorer in (pilot.Scorer({2}), Batched({2})):
        trainer = api.HierarchicalGRPOTrainer(gfam_reward_scorer=reward_scorer,
                                             train_worker_model=True, min_worker_grpo_group_size=2)
        result = trainer.run(task=api.TaskExample(task_id='t', prompt='Solve for x: 2x + 3 = 11.',
                             ground_truth='4', metadata={'skill_focus': 'algebra', 'distractor_answer': '5'}),
            worker_pool=demo.make_worker_pool(base_model_path='mock'),
            policy_config=api.ControllerPolicyConfig(shared_model_path='mock'),
            rollout_config=api.RolloutConfig(num_decompositions=2, num_executor_rollouts_per_decomposition=4),
            schedule=api.TrainingScheduleConfig(mode=api.TrainingMode.JOINT))
        outputs.append([(sample.group_id, sample.reward, sample.advantage)
                        for sample in result.training_batch.worker_samples])
    assert Batched.batches == [4, 4]
    assert outputs[0] == outputs[1]


def test_learner_nodes_are_exact_not_spread():
    nodes = [{'NodeID': name, 'Alive': True, 'Resources': {'GPU': 3}} for name in ('b', 'a', 'c')]
    assert train._offline_rank_node_ids(nodes, nnodes=1, gpus_per_node=3, driver_node_id='b') == ['b']*3
    assert train._offline_rank_node_ids(nodes, nnodes=2, gpus_per_node=3, driver_node_id='b') == ['b']*3 + ['a']*3
    with pytest.raises(RuntimeError, match='eligible'):
        train._offline_rank_node_ids(nodes, nnodes=4, gpus_per_node=3)
    with pytest.raises(RuntimeError, match='capacity'):
        train._offline_rank_node_ids(nodes, nnodes=1, gpus_per_node=4, driver_node_id='b')


def test_old_log_probs_compute_shards_but_preserve_complete_cache(monkeypatch):
    rows = [{'sample_index': i, 'input_ids': torch.tensor([i+1, 2, 3]),
             'attention_mask': torch.ones(3, dtype=torch.long), 'position_ids': torch.arange(3),
             'loss_mask': torch.tensor([0, 1, 1])} for i in range(7)]
    monkeypatch.setattr(offline, '_collate_rows', lambda batch: {
        key: [item[key] for item in batch] if key == 'sample_index' else torch.stack([item[key] for item in batch])
        for key in rows[0]})

    class Model(torch.nn.Module):
        count = 0

        def forward(self, input_ids, **kwargs):
            self.count += len(input_ids)
            return SimpleNamespace(logits=input_ids.float().unsqueeze(-1) * torch.arange(10).float() / 100)

    baseline = offline._compute_old_log_prob_cache(Model(), rows, 2, 'cpu')
    total = 0
    for rank in range(3):
        model = Model()
        context = offline.DistributedTrainingContext(enabled=True, rank=rank, world_size=3, backend='gloo')

        def gather(shards, local):
            for key, value in local.items():
                torch.testing.assert_close(value, baseline[key], rtol=0, atol=0)
            for index in range(3):
                shards[index] = dict(baseline) if index != rank else local

        monkeypatch.setattr(torch.distributed, 'all_gather_object', gather)
        actual = offline._compute_old_log_prob_cache(model, rows, 2, 'cpu', context)
        assert set(actual) == set(baseline)
        for index in baseline:
            assert torch.equal(actual[index], baseline[index])
        total += model.count
    assert total == 9  # Seven unique examples + two sampler padding rows, not 21.


class DistributedProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.arange(10).float() / 100)
        self.register_buffer('bias', torch.zeros(10))
        self.count = 0

    def forward(self, input_ids, **kwargs):
        self.count += len(input_ids)
        return SimpleNamespace(logits=input_ids.float().unsqueeze(-1) * self.scale + self.bias)


def probe_rows():
    result = []
    for index in range(7):
        size = 3 + index % 3
        item = dict(sample_index=index, input_ids=torch.tensor([index+1] + [2]*(size-1)),
                    attention_mask=torch.ones(size, dtype=torch.long), position_ids=torch.arange(size),
                    loss_mask=torch.tensor([1]*(size-1) + [0]), pad_token_id=0,
                    reward=torch.tensor(1.), advantage=torch.tensor(1.), group_id='g',
                    role='worker', policy_id='executor')
        for field in ('retained_response_tokens', 'truncated_prompt_tokens', 'truncated_response_tokens',
                      'zero_loss_row', 'zero_loss_due_to_truncation', 'zero_loss_due_to_empty_completion'):
            item[field] = torch.tensor(0)
        result.append(item)
    return result


def distributed_probe(rank, world_size, directory):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(Path(directory)/'rendezvous').as_uri(),
                            rank=rank, world_size=world_size, timeout=timedelta(seconds=45))
    try:
        model = torch.nn.parallel.DistributedDataParallel(DistributedProbeModel())
        context = offline.DistributedTrainingContext(enabled=True, rank=rank, world_size=world_size, backend='gloo')
        values = offline._compute_old_log_prob_cache(model, probe_rows(), 2, 'cpu', context)
        torch.save({'values': values, 'computed_rows': model.module.count}, Path(directory)/f'rank_{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason='Requires CPU Gloo')
def test_real_three_rank_old_log_prob_cache(tmp_path):
    import torch.multiprocessing as mp
    baseline = offline._compute_old_log_prob_cache(DistributedProbeModel(), probe_rows(), 2, 'cpu')
    mp.spawn(distributed_probe, args=(3, str(tmp_path)), nprocs=3, join=True)
    computed = 0
    for rank in range(3):
        saved = torch.load(tmp_path/f'rank_{rank}.pt', weights_only=True)
        computed += saved['computed_rows']
        assert set(saved['values']) == set(baseline)
        for index, value in baseline.items():
            torch.testing.assert_close(saved['values'][index], value, rtol=0, atol=0)
    assert computed == 9

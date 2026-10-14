"""CPU tests for overlap, bounded queues, and DDP-safe length buckets."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib
from threading import Event
from types import SimpleNamespace

import pytest

try:
    import verl.hierarchical_rema as api
except ModuleNotFoundError:
    import hierarchical_rema as api

helpers = importlib.import_module(api.__name__ + ".throughput")
orchestration = importlib.import_module(api.__name__ + ".orchestrator")
demo = importlib.import_module(api.__name__ + ".demo")


def test_scoring_runs_while_producer_continues_and_drains():
    started, generation = Event(), Event()
    def score(rows):
        started.set()
        assert generation.wait(2)
        return {row: row * 2 for row in reversed(rows)}
    with helpers.RewardPipeline(score, batch_size=2) as pipeline:
        pipeline.add(1, 1)
        pipeline.add(2, 2)
        assert started.wait(2)
        generation.set()
        pipeline.add(3, 3)
        assert pipeline.finish() == {1: 2, 2: 4, 3: 6}


def test_queue_applies_backpressure():
    started, release = Event(), Event()
    def score(rows):
        started.set()
        assert release.wait(2)
        return dict.fromkeys(rows, 1)
    with helpers.RewardPipeline(score, batch_size=1, max_pending=1) as pipeline:
        pipeline.add(1, 1)
        assert started.wait(2)
        with ThreadPoolExecutor(1) as producer:
            future = producer.submit(pipeline.add, 2, 2)
            assert not future.done()
            release.set()
            future.result(timeout=2)
        assert len(pipeline.futures) <= 1
        assert len(pipeline.finish()) == 2


@pytest.mark.parametrize("overlap", [True, False])
def test_reward_failure_and_bad_identity_are_not_silently_dropped(overlap):
    with pytest.raises(ValueError, match="Misaligned"):
        with helpers.RewardPipeline(lambda rows: {"wrong": 0}, batch_size=1, overlap=overlap) as pipeline:
            pipeline.add("a", 1)
            pipeline.finish()
    def fail(rows):
        raise RuntimeError("scorer failed")
    with pytest.raises(RuntimeError, match="scorer failed"):
        with helpers.RewardPipeline(fail, batch_size=1, overlap=overlap) as pipeline:
            pipeline.add("a", 1)
            pipeline.finish()


def test_duplicate_submission_is_rejected():
    with helpers.RewardPipeline(lambda rows: {x: x for x in rows}, batch_size=5) as pipeline:
        pipeline.add(1, 1)
        with pytest.raises(ValueError, match="twice"):
            pipeline.add(1, 1)
        assert pipeline.finish() == {1: 1}


@pytest.mark.parametrize("size", [0, 1, 2, 5, 17, 101])
@pytest.mark.parametrize("world", [1, 3, 6])
def test_length_buckets_cover_all_rows_and_balance_ranks(size, world):
    lengths = [(i * 113) % 1000 + 1 for i in range(size)]
    ranks = [helpers.LengthBucketBatchSampler(lengths, 4, rank=r, world_size=world, seed=19)
             for r in range(world)]
    batches = [list(sampler) for sampler in ranks]
    assert len(set(map(len, batches))) == 1
    assert all(len(b) == len(ranks[i]) for i, b in enumerate(batches))
    for step in zip(*batches):
        assert len(set(map(len, step))) == 1
    counts = Counter(i for rank in batches for batch in rank for i in batch)
    assert set(counts) == set(range(size))
    assert sum(counts.values()) - size <= world - 1
    assert batches[0] == list(ranks[0])
    ranks[0].set_epoch(1)
    if len(batches[0]) > 3:
        assert list(ranks[0]) != batches[0]


def test_bucketing_reduces_padding_without_truncation():
    lengths = [8, 1024] * 64
    batches = list(helpers.LengthBucketBatchSampler(lengths, 8, shuffle=False))
    padded = sum(max(lengths[i] for i in b) * len(b) for b in batches)
    assert padded == sum(lengths)
    assert padded < len(lengths) * 1024


def test_orchestrator_scores_before_later_generation_and_preserves_rewards(monkeypatch):
    pilot = importlib.import_module("test_graphprm_pilot")
    outputs = []
    for overlap in (False, True):
        monkeypatch.setenv("GFAM_OVERLAP_REWARDS", str(int(overlap)))
        monkeypatch.setenv("REMA_VERIFIER_WORKERS", "2")
        started, next_generation = Event(), Event()
        class Scorer(pilot.Scorer):
            rollout_batch_size = 2
            def score_rollouts(self, requests):
                started.set()
                if overlap:
                    assert next_generation.wait(3)
                return [self.score_rollout(**request) for request in requests]
        trainer = api.HierarchicalGRPOTrainer(gfam_reward_scorer=Scorer(),
                                              train_worker_model=True, min_worker_grpo_group_size=2)
        original = trainer.backend.execute_workers_batch
        trainer.backend.config = SimpleNamespace(worker_batch_size=2)
        generation_calls = []
        def generate(requests):
            generation_calls.append(1)
            if started.is_set():
                next_generation.set()
            result = original(requests)
            # When the first complete pair is submitted, give the background
            # consumer a chance to start before the next producer chunk.
            if len(generation_calls) > 1 and overlap:
                started.wait(.02)
                if started.is_set():
                    next_generation.set()
            return result
        trainer.backend.execute_workers_batch = generate
        result = trainer.run(
            task=api.TaskExample(task_id="t", prompt="Solve for x: 2x + 3 = 11.", ground_truth="4",
                                 metadata={"skill_focus": "algebra", "distractor_answer": "5"}),
            worker_pool=demo.make_worker_pool(base_model_path="mock"),
            policy_config=api.ControllerPolicyConfig(shared_model_path="mock"),
            rollout_config=api.RolloutConfig(num_decompositions=2, num_executor_rollouts_per_decomposition=4),
            schedule=api.TrainingScheduleConfig(mode=api.TrainingMode.JOINT))
        if overlap:
            assert next_generation.is_set()
        outputs.append(result.to_dict())
    assert outputs[0] == outputs[1]

"""Bounded scheduling helpers; no policy or reward semantics live here."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import math
import os
import random
import time


def positive_env(name, default):
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def enabled(name, default="1"):
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}


class RewardPipeline:
    """One GPU consumer, bounded in-flight batches, results keyed by identity."""
    def __init__(self, score, *, batch_size, max_pending=2, overlap=True):
        if batch_size < 1 or max_pending < 1:
            raise ValueError("Reward pipeline limits must be positive")
        self.score, self.batch_size, self.max_pending = score, batch_size, max_pending
        self.overlap = overlap
        self.waiting, self.futures, self.results, self.seen = [], deque(), {}, set()
        self.executor = None
        self.wait_seconds = 0.0

    def __enter__(self):
        if self.overlap:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rema-reward")
        return self

    def _collect(self):
        future, expected = self.futures.popleft()
        started = time.monotonic()
        self._merge(future.result(), expected)
        self.wait_seconds += time.monotonic() - started

    def _merge(self, result, expected):
        if set(result) != set(expected) or set(result) & self.results.keys():
            raise ValueError("Misaligned or duplicate reward results")
        self.results.update(result)

    def add(self, key, item):
        if key in self.seen:
            raise ValueError(f"Trajectory submitted twice: {key}")
        self.seen.add(key)
        self.waiting.append((key, item))
        if len(self.waiting) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.waiting:
            return
        batch, self.waiting = self.waiting, []
        expected = [key for key, _ in batch]
        if self.executor is None:
            self._merge(self.score([item for _, item in batch]), expected)
            return
        while len(self.futures) >= self.max_pending:
            self._collect()
        self.futures.append((self.executor.submit(self.score, [item for _, item in batch]), expected))
        while self.futures and self.futures[0][0].done():
            self._collect()

    def finish(self):
        self.flush()
        while self.futures:
            self._collect()
        if set(self.results) != self.seen:
            raise ValueError("Incomplete reward pipeline")
        print(f"[hierarchical-rema][reward-pipeline] trajectories={len(self.results)} "
              f"overlap={self.overlap} max_pending_batches={self.max_pending} "
              f"consumer_wait_s={self.wait_seconds:.3f}", flush=True)
        return self.results

    def __exit__(self, exc_type, exc, tb):
        if self.executor is not None:
            # Do not release policy/resources while a reward batch is still active.
            self.executor.shutdown(wait=True, cancel_futures=True)


class LengthBucketBatchSampler:
    """Equal DDP steps and per-step row counts, with at most world_size-1 repeats.

    Sort shuffled megabatches, then stripe each global batch across ranks. This
    balances long sequences across ranks rather than assigning one rank a tail.
    """
    def __init__(self, lengths, batch_size, *, rank=0, world_size=1, seed=0,
                 shuffle=True, bucket_batches=50):
        if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size or bucket_batches < 1:
            raise ValueError("Invalid length-bucket sampler configuration")
        self.lengths = list(lengths)
        self.batch_size, self.rank, self.world_size = batch_size, rank, world_size
        self.seed, self.shuffle, self.bucket_batches = seed, shuffle, bucket_batches
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return math.ceil(math.ceil(len(self.lengths) / self.world_size) / self.batch_size)

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if not indices:
            return
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)
        total = math.ceil(len(indices) / self.world_size) * self.world_size
        indices += (indices * math.ceil((total - len(indices)) / len(indices)))[:total - len(indices)]
        width = self.batch_size * self.world_size
        pool_width = width * self.bucket_batches
        batches = []
        for start in range(0, total, pool_width):
            pool = sorted(indices[start:start + pool_width], key=self.lengths.__getitem__)
            batches.extend(pool[offset:offset + width] for offset in range(0, len(pool), width))
        if self.shuffle:
            rng.shuffle(batches)
        for batch in batches:
            yield batch[self.rank::self.world_size]

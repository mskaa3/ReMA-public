"""Bounded frozen-probe prefetch inside one on-policy rollout batch."""

from concurrent.futures import ThreadPoolExecutor
import logging
from time import monotonic


def pipeline_config(config):
    config = dict(config or {})
    enabled = config.get('enable', False)
    size = config.get('groups_per_chunk', 12)
    if type(enabled) is not bool:
        raise ValueError('rollout_probe_pipeline.enable must be a boolean')
    if type(size) is not int or size < 1:
        raise ValueError('rollout_probe_pipeline.groups_per_chunk must be a positive integer')
    return {'enable': enabled, 'groups_per_chunk': size}


def complete_group_slices(group_ids, rollouts_per_group, groups_per_chunk):
    """Validate the entire input before scheduling; never split a C3 group."""
    for value in (rollouts_per_group, groups_per_chunk):
        if type(value) is not int or value < 1:
            raise ValueError('Rollout and chunk group sizes must be positive integers')
    ids = list(group_ids)
    if not ids or len(ids) % rollouts_per_group:
        raise ValueError('Pipeline requires complete rollout groups')
    seen = set()
    for start in range(0, len(ids), rollouts_per_group):
        group = ids[start:start + rollouts_per_group]
        if group[0] in seen or any(uid != group[0] for uid in group):
            raise ValueError('Pipeline requires distinct, contiguous, complete UID groups')
        seen.add(group[0])
    width = groups_per_chunk * rollouts_per_group
    return [slice(start, min(start + width, len(ids))) for start in range(0, len(ids), width)]


def cached_probe_responses(items, cache, key_fn):
    """Missing prefetch results are errors, not zeros or a second inference."""
    keys = [key_fn(*item) for item in items]
    if any(key not in cache for key in keys):
        raise RuntimeError('Incomplete pipeline probe cache; refusing to change gate measurements')
    values = [cache[key] for key in keys]
    return tuple([value[column] for value in values] for column in range(3))


def run_probe_pipeline(chunks, generate, prepare, probe, cancel_probe):
    """Overlap only frozen inference; all trainer/scoring mutations stay on caller.

    ``prepare`` returns ordered {context_key: message} requests. The single
    background worker only calls the independent frozen probe. Results are
    drained in input order, with a barrier before returning to the trainer.
    """
    outputs, cache = [], {}
    generation_intervals, probe_intervals = [], []
    wait_seconds = 0.0
    pending = None
    pending_keys = []
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='frozen-probe')
    started = monotonic()

    def infer(messages):
        start = monotonic()
        responses = probe(messages)
        if len(responses) != 3 or any(len(column) != len(messages) for column in responses):
            raise RuntimeError('Pipeline probe returned an incomplete generation batch')
        return list(zip(*responses)), (start, monotonic())

    def drain():
        nonlocal pending, wait_seconds
        if pending is None:
            return
        start = monotonic()
        values, interval = pending.result()
        wait_seconds += monotonic() - start
        cache.update(zip(pending_keys, values))
        probe_intervals.append(interval)
        pending = None

    try:
        for chunk in chunks:
            # Surface an already-failed probe before generating more work.
            if pending is not None and pending.done():
                drain()
            start = monotonic()
            output = generate(chunk)
            generation_intervals.append((start, monotonic()))
            outputs.append(output)
            # One completed chunk may wait for scoring; no unbounded queue.
            drain()
            requests = prepare(chunk, output)
            fresh = {key: message for key, message in requests.items() if key not in cache}
            if fresh:
                pending_keys = list(fresh)
                pending = executor.submit(infer, list(fresh.values()))
        drain()
    except BaseException:
        if pending is not None:
            pending.cancel()
        try:
            cancel_probe()
        except Exception:
            logging.getLogger(__name__).warning('Could not cancel frozen probe', exc_info=True)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    stats = {
        'chunks': len(outputs),
        'unique_probe_requests': len(cache),
        'generation_seconds': sum(end - start for start, end in generation_intervals),
        'probe_seconds': sum(end - start for start, end in probe_intervals),
        'probe_wait_seconds': wait_seconds,
        'wall_seconds': monotonic() - started,
        'overlap_seconds': sum(
            max(0.0, min(g_end, p_end) - max(g_start, p_start))
            for g_start, g_end in generation_intervals
            for p_start, p_end in probe_intervals
        ),
    }
    return outputs, cache, stats

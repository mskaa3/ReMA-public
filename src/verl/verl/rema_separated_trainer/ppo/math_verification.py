"""Process-isolated symbolic checks with parent-enforced deadlines."""

from concurrent.futures import TimeoutError
from multiprocessing import get_context


def boxed_answer_is_parseable(response):
    from .prefix_probe import parse_boxed_math_answer

    return parse_boxed_math_answer(response) is not None


def worker_answer_matches_terminal(pair):
    from .prefix_probe import compare_worker_final_answers

    return compare_worker_final_answers(*pair)


def bounded_verification_map(function, inputs, *, timeout_seconds=10, max_workers=1):
    """Return unknown for killed checks, without dropping/reordering batch rows.

    A SIGALRM raised inside a Python weakref callback can be ignored. An outer
    process deadline remains effective even then. Spawn avoids inheriting the
    trainer's Ray/CUDA state; only strings and Boolean results cross processes.
    """
    from pebble import ProcessExpired, ProcessPool

    inputs = list(inputs)
    if timeout_seconds <= 0 or max_workers < 1:
        raise ValueError('Verification timeout and worker count must be positive')
    stats = {'checks': len(inputs), 'timeout_count': 0, 'worker_crash_count': 0,
             'unknown_count': 0}
    if not inputs:
        return [], stats
    results = []
    with ProcessPool(max_workers=min(max_workers, len(inputs)), context=get_context('spawn')) as pool:
        iterator = pool.map(function, inputs, timeout=timeout_seconds, chunksize=1).result()
        for _ in inputs:
            try:
                result = next(iterator)
            except TimeoutError:
                stats['timeout_count'] += 1
                result = None
            except ProcessExpired:
                stats['worker_crash_count'] += 1
                result = None
            except StopIteration as exc:
                raise RuntimeError('Symbolic verifier returned an incomplete batch') from exc
            # Unexpected infrastructure/programming errors still propagate.
            if result is not None and not isinstance(result, bool):
                raise TypeError('Symbolic checks must return bool or None')
            stats['unknown_count'] += int(result is None)
            results.append(result)
    return results, stats

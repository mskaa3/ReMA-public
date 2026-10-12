"""CPU-only watchdog tests. Run directly with Python; requires Pebble, not Ray."""

import importlib.util
import os
from pathlib import Path
import signal
import time
import unittest


# Load only the watchdog so these tests do not require the GPU training stack.
SOURCE = Path(__file__).resolve().parents[2] / 'verl/rema_separated_trainer/ppo/math_verification.py'
SPEC = importlib.util.spec_from_file_location('math_verification_watchdog_test', SOURCE)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def fake_check(value):
    if value == 'hang':
        # Reproduce the important failure mode: the inner alarm cannot stop it.
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, signal.SIG_IGN)
            signal.alarm(1)
        time.sleep(30)
    if value == 'crash':
        os._exit(23)
    if value == 'bug':
        raise ValueError('unexpected verifier bug')
    return value


def isolated_check(parent_pid):
    return os.getpid() != parent_pid


class MathVerificationTests(unittest.TestCase):
    def test_preserves_order_and_unknown_is_not_false(self):
        values, stats = VERIFIER.bounded_verification_map(fake_check, [True, None, False])
        self.assertEqual(values, [True, None, False])
        self.assertEqual(stats, {'checks': 3, 'timeout_count': 0,
                                 'worker_crash_count': 0, 'unknown_count': 1})

    def test_hang_is_killed_and_remaining_examples_continue(self):
        before = time.monotonic()
        values, stats = VERIFIER.bounded_verification_map(
            fake_check, [True, 'hang', False, 'hang', True], timeout_seconds=0.5,
        )
        self.assertLess(time.monotonic() - before, 15)
        self.assertEqual(values, [True, None, False, None, True])
        self.assertEqual(stats['timeout_count'], 2)
        self.assertEqual(stats['unknown_count'], 2)
        self.assertEqual(stats['worker_crash_count'], 0)

    def test_crashed_worker_is_replaced_and_does_not_reject_answer(self):
        values, stats = VERIFIER.bounded_verification_map(fake_check, [True, 'crash', False])
        self.assertEqual(values, [True, None, False])
        self.assertEqual(stats['worker_crash_count'], 1)
        self.assertEqual(stats['unknown_count'], 1)
        self.assertEqual(stats['timeout_count'], 0)

    def test_check_is_not_run_in_parent(self):
        handler = signal.getsignal(signal.SIGALRM) if hasattr(signal, 'SIGALRM') else None
        values, _ = VERIFIER.bounded_verification_map(isolated_check, [os.getpid()])
        self.assertEqual(values, [True])
        if hasattr(signal, 'SIGALRM'):
            self.assertEqual(signal.getsignal(signal.SIGALRM), handler)

    def test_unexpected_errors_are_not_silently_skipped(self):
        with self.assertRaisesRegex(ValueError, 'unexpected verifier bug'):
            VERIFIER.bounded_verification_map(fake_check, ['bug'])
        with self.assertRaisesRegex(TypeError, 'bool or None'):
            VERIFIER.bounded_verification_map(fake_check, ['not a Boolean'])

    def test_empty_input_and_invalid_limits(self):
        values, stats = VERIFIER.bounded_verification_map(fake_check, [])
        self.assertEqual(values, [])
        self.assertEqual(stats['checks'], 0)
        for limits in ({'timeout_seconds': 0}, {'max_workers': 0}):
            with self.assertRaises(ValueError):
                VERIFIER.bounded_verification_map(fake_check, [True], **limits)


if __name__ == '__main__':
    unittest.main()

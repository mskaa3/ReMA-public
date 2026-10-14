"""Parallel CPU verifiers preserve cache affinity and unknown/error semantics."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import sys
from threading import Barrier

import pytest

try:
    import verl.hierarchical_rema as api
except ModuleNotFoundError:
    import hierarchical_rema as api

backend = importlib.import_module(api.__name__ + ".rewarding")._math_verifier_module()
isolated = importlib.import_module(backend.__package__ + ".isolated_verifier")


def index(operation, *arguments):
    return int.from_bytes(hashlib.sha256(json.dumps([operation, arguments], ensure_ascii=True).encode()).digest()[:8], "big") % 2


def test_distinct_shards_can_execute_concurrently():
    barrier = Barrier(2)
    class Client:
        def call(self, operation, *args):
            barrier.wait(timeout=2)
            return {"status": "ok", "value": args[0]}
    pool = isolated.VerifierPool(2, Client)
    values = [next(i for i in range(100) if index("compare", i) == shard) for shard in range(2)]
    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(lambda i: pool.call("compare", i), values))
    assert [r["value"] for r in results] == values


def test_pool_timeout_remains_unknown_is_cached_and_other_shard_survives(monkeypatch):
    code = '''
import json, sys
print('{"status":"ready"}', flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request['operation'] == 'hang':
        while True: pass
    print('{"status":"ok","value":1.0}', flush=True)
'''
    monkeypatch.setenv("REMA_VERIFIER_TIMEOUT_SECONDS", ".1")
    pool = isolated.VerifierPool(2, lambda: isolated.VerifierProcess([sys.executable, "-u", "-c", code]))
    bad = index("hang")
    other = next(i for i in range(100) if index("ok", i) != bad)
    try:
        with ThreadPoolExecutor(2) as executor:
            future = executor.submit(pool.call, "hang")
            assert pool.call("ok", other)["value"] == 1.
            result = future.result(timeout=3)
        assert result["status"] == "unverified"
        assert result["code"] == "verification_timeout"
        assert pool.clients[bad].process is None
        assert pool.call("hang") == result
        assert pool.clients[bad].process is None
        assert pool.clients[1-bad].process.poll() is None
    finally:
        pool.close()


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_real_pool_preserves_mathematical_decisions(workers):
    pool = isolated.VerifierPool(workers)
    requests = [("compare", ("0.5", r"\frac{1}{2}")), ("compare", ("0.6", r"\frac{1}{2}")),
                ("validate_reference", (r"\frac{}{2}",))]
    try:
        with ThreadPoolExecutor(workers) as executor:
            result = list(executor.map(lambda r: pool.call(r[0], *r[1]), requests))
        assert result[0] == {"status": "ok", "value": 1.}
        assert result[1] == {"status": "ok", "value": 0.}
        assert result[2]["status"] == "unverified"
    finally:
        pool.close()


@pytest.mark.parametrize("workers", [0, -1, 17])
def test_pool_capacity_is_bounded(workers):
    with pytest.raises(ValueError):
        isolated.VerifierPool(workers)

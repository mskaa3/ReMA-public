"""Persistent, CPU-only mathematical verifier with a parent-enforced deadline.

Launched as a file, not via the training entry point: no Ray/CUDA state is copied
or initialized. Only JSON strings/scalars cross this boundary, never SymPy objects.
"""

import sys

if __name__ == "__main__":
    # Avoid shadowing stdlib math and the installed math_verify package with
    # neighboring adapter files, before importing anything that uses them.
    sys.path.pop(0)

import atexit
from collections import OrderedDict
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import select
import subprocess
import threading
import time
import traceback


def _seconds(name, default):
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite, positive number")
    return value


class VerifierProcess:
    def __init__(self, command=None, cache_size=4096):
        self.command = command or [sys.executable, "-u", str(Path(__file__).resolve())]
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.process = None
        self.buffer = b""
        self.owner_pid = os.getpid()
        atexit.register(self.close)

    def close(self):
        if os.getpid() != self.owner_pid:
            return
        process, self.process = self.process, None
        self.buffer = b""
        if process is not None:
            # SIGKILL cannot be swallowed by SymPy's exception handlers.
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            process.stdin.close()
            process.stdout.close()

    def _read(self, deadline):
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError("Mathematical verifier exceeded its deadline")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("Mathematical verifier exited without a response; inspect stderr")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def _write(self, data, deadline):
        view = memoryview(data)
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [self.process.stdin], [], remaining)[1]:
                raise TimeoutError("Mathematical verifier request exceeded its deadline")
            try:
                size = os.write(self.process.stdin.fileno(), view)
                view = view[size:]
            except BlockingIOError:
                continue

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, bufsize=0, env=env,
        )
        os.set_blocking(self.process.stdin.fileno(), False)
        try:
            ready = self._read(time.monotonic() + _seconds("REMA_VERIFIER_STARTUP_TIMEOUT_SECONDS", 30))
            if ready.get("status") != "ready":
                raise RuntimeError(f"Cannot initialize mathematical verifier: {ready}")
        except Exception as exc:
            self.close()
            raise RuntimeError(f"Cannot start mathematical verifier: {exc}") from exc

    def call(self, operation, *arguments):
        timeout = _seconds("REMA_VERIFIER_TIMEOUT_SECONDS", 10)
        request = json.dumps({"operation": operation, "arguments": arguments}, ensure_ascii=True)
        key = (timeout, request)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return dict(self.cache[key])
            self._start()
            deadline = time.monotonic() + timeout
            try:
                self._write((request + "\n").encode(), deadline)
                response = self._read(deadline)
            except TimeoutError:
                self.close()
                response = {"status": "unverified", "code": "verification_timeout",
                            "message": f"Mathematical {operation} exceeded {timeout:g}s; verifier was killed"}
                print(f"[answer-verifier] {response['message']}", file=sys.stderr, flush=True)
            except Exception:
                self.close()
                raise
            if response.get("status") == "fatal":
                self.close()
                raise RuntimeError(f"Mathematical verifier failed: {response.get('message')}")
            if response.get("status") not in {"ok", "unverified"}:
                self.close()
                raise RuntimeError(f"Invalid mathematical verifier response: {response!r}")
            # Cache unknowns too: repeated rollouts must not repeatedly stall on
            # the same pathological expression. The timeout is part of the key.
            self.cache[key] = dict(response)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            return response


def _serve():
    wire = sys.stdout

    def send(response):
        wire.write(json.dumps(response, ensure_ascii=True, allow_nan=False) + "\n")
        wire.flush()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            spec = importlib.util.spec_from_file_location(
                "_rema_math_backend", Path(__file__).with_name("math_verify.py"))
            backend = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(backend)
            backend._verifier()
    except Exception:
        send({"status": "fatal", "message": traceback.format_exc()})
        return
    send({"status": "ready"})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                value = backend._dispatch_verification(request["operation"], request["arguments"])
            response = {"status": "ok", "value": value}
        except backend.AnswerVerificationError as exc:
            response = {"status": "unverified", "code": exc.code, "message": str(exc)}
        except Exception:
            send({"status": "fatal", "message": traceback.format_exc()})
            return
        send(response)


_client = None
_client_pid = None
_client_lock = threading.Lock()


class VerifierPool:
    """Independent killable processes; identical requests share a cache/lock."""
    def __init__(self, workers=1, process_factory=VerifierProcess):
        if not 1 <= workers <= 16:
            raise ValueError("REMA_VERIFIER_WORKERS must be between 1 and 16")
        self.clients = [process_factory() for _ in range(workers)]

    def call(self, operation, *arguments):
        key = json.dumps([operation, arguments], ensure_ascii=True).encode()
        index = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(self.clients)
        return self.clients[index].call(operation, *arguments)

    def close(self):
        for client in self.clients:
            client.close()


def verify_in_process(operation, *arguments):
    global _client, _client_pid
    if _client is None or _client_pid != os.getpid():
        with _client_lock:
            if _client is None or _client_pid != os.getpid():
                _client = VerifierPool(int(os.environ.get("REMA_VERIFIER_WORKERS", "1")))
                _client_pid = os.getpid()
    return _client.call(operation, *arguments)


if __name__ == "__main__":
    _serve()

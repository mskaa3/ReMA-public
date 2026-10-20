"""Supervise a frozen, loopback-only vLLM coach beside the ReMA driver.

Invoke AFTER gpu_partition.py. The server inherits only the reserved GPU, never
joins the policy Ray cluster, and is torn down when the driver exits.
"""
import argparse
from dataclasses import asdict, dataclass
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


@dataclass(frozen=True)
class LocalCoachConfig:
    model: str = "Qwen/Qwen3-32B"
    revision: str = "main"
    max_model_len: int = 8192
    max_tokens: int = 4096
    max_num_seqs: int = 1
    gpu_memory_utilization: float = .90
    startup_timeout: float = 1800.
    shutdown_timeout: float = 30.
    seed: int = 42

    @classmethod
    def from_env(cls, env):
        config = cls(
            model=env.get("REMA_COACH_MODEL", "Qwen/Qwen3-32B"),
            revision=env.get("REMA_COACH_REVISION", "main"),
            max_model_len=int(env.get("REMA_COACH_MAX_MODEL_LEN", 8192)),
            max_tokens=int(env.get("REMA_COACH_MAX_TOKENS", 4096)),
            max_num_seqs=int(env.get("REMA_COACH_MAX_NUM_SEQS", 1)),
            gpu_memory_utilization=float(env.get("REMA_COACH_GPU_MEMORY_UTILIZATION", .90)),
            startup_timeout=float(env.get("REMA_COACH_STARTUP_TIMEOUT", 1800)),
            shutdown_timeout=float(env.get("REMA_COACH_SHUTDOWN_TIMEOUT", 30)),
            seed=int(env.get("REMA_COACH_SEED", 42)))
        if not config.model.strip() or not config.revision.strip():
            raise ValueError("Coach model and revision cannot be empty")
        if not 1 <= config.max_tokens < config.max_model_len <= 32768:
            raise ValueError("Require 1 <= REMA_COACH_MAX_TOKENS < REMA_COACH_MAX_MODEL_LEN <= 32768")
        if not 1 <= config.max_num_seqs <= 32 or not 0 <= config.seed < 2**32:
            raise ValueError("Invalid coach sequence count or seed")
        if not math.isfinite(config.gpu_memory_utilization) or not 0 < config.gpu_memory_utilization < 1:
            raise ValueError("REMA_COACH_GPU_MEMORY_UTILIZATION must be in (0, 1)")
        for value in (config.startup_timeout, config.shutdown_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Coach startup/shutdown timeouts must be positive and finite")
        return config


def validate_gpu_isolation(env):
    devices = [d.strip() for d in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if env.get("REMA_GPU_ROLE") != "reward" or len(devices) != 1 or devices[0] == "-1":
        raise ValueError("Local coach requires exactly one isolated reward GPU; use the Slurm launcher "
                         "with GFAM_REWARD_GPU_RESERVATION=1")
    return devices[0]


def resolve_model(config):
    """Pin weights and tokenizer to the same Hub snapshot before starting vLLM."""
    from packaging.version import Version
    versions = {name: metadata.version(name) for name in ("vllm", "transformers")}
    if Version(versions["vllm"]) < Version("0.8.5") or Version(versions["transformers"]) < Version("4.51.0"):
        raise RuntimeError("Qwen3 coach requires vLLM >=0.8.5 and Transformers >=4.51.0 inside the SIF. "
                           "Rebuild the container; do not upgrade the running policy environment.")
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(config.model, revision=config.revision,
                                              trust_remote_code=False)
    revision = getattr(model_config, "_commit_hash", None)
    if not revision or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Coach must resolve to an immutable Hugging Face Hub commit")
    context = getattr(model_config, "max_position_embeddings", None)
    if context and config.max_model_len > context:
        raise ValueError(f"Coach context limit exceeds model context {context}")
    return revision, versions


def server_command(config, revision, port):
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", config.model, "--served-model-name", config.model,
            "--revision", revision, "--tokenizer-revision", revision,
            "--host", "127.0.0.1", "--port", str(port),
            "--tensor-parallel-size", "1", "--distributed-executor-backend", "mp",
            "--dtype", "bfloat16", "--max-model-len", str(config.max_model_len),
            "--max-num-seqs", str(config.max_num_seqs),
            "--max-num-batched-tokens", str(config.max_model_len),
            "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            "--seed", str(config.seed), "--generation-config", "vllm",
            "--enforce-eager", "--disable-log-requests"]


def server_environment(env, token):
    result = dict(env)
    # Policy engines use their own Ray/SPMD settings. The coach must not join them.
    for key in list(result):
        if key.startswith(("RAY_", "VLLM_")) or key in ("MASTER_ADDR", "MASTER_PORT", "RANK",
                                                     "WORLD_SIZE", "LOCAL_RANK"):
            result.pop(key)
    result["VLLM_API_KEY"] = token
    result["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    result["VLLM_USE_V1"] = "0"
    return result


def _json_request(url, token, payload=None, timeout=10):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    # Loopback must never be sent through a site HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        body = response.read(1_000_001)
    if len(body) > 1_000_000:
        raise ValueError("Oversized coach readiness response")
    return json.loads(body)


def wait_ready(server, endpoint, token, model, timeout):
    deadline, next_report = time.monotonic() + timeout, 0.
    while time.monotonic() < deadline:
        code = server.poll()
        if code is not None:
            raise RuntimeError(f"Local coach exited during startup (code {code}); inspect coach_server.log")
        try:
            info = _json_request(endpoint + "/models", token)
            if any(item.get("id") == model for item in info.get("data", [])):
                return
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            pass
        if time.monotonic() >= next_report:
            print("[rema-coach] Waiting for local vLLM; see coach_server.log", flush=True)
            next_report = time.monotonic() + 60
        time.sleep(2)
    raise TimeoutError("Local coach startup timed out; inspect coach_server.log")


def smoke_check(endpoint, token, model):
    result = _json_request(endpoint + "/chat/completions", token, {
        "model": model, "messages": [{"role": "user", "content":
            'Respond with exactly {"ready":true}, without explanation or markdown.'}],
        "max_tokens": 64, "temperature": 0, "top_p": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "guided_json": {"type": "object", "properties": {"ready": {"type": "boolean"}},
                        "required": ["ready"], "additionalProperties": False}}, timeout=120)
    choice = result["choices"][0]
    if choice.get("finish_reason") != "stop" or json.loads(choice["message"]["content"]) != {"ready": True}:
        raise RuntimeError("Local coach failed the non-thinking JSON smoke check")


def stop_process_group(process, timeout):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    # Also clean up engine workers if their parent has already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def supervise(config, output_dir, command):
    env = os.environ.copy()
    device = validate_gpu_isolation(env)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rema-coach] Resolving frozen {config.model}@{config.revision}", flush=True)
    revision, versions = resolve_model(config)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    endpoint, token = f"http://127.0.0.1:{port}/v1", secrets.token_urlsafe(32)
    model_identity = {"model": config.model, "revision": revision, "versions": versions,
                      "dtype": "bfloat16", "thinking": False, "seed": config.seed,
                      "max_model_len": config.max_model_len,
                      "guided_json": True, "engine": "vllm_v0",
                      "tensor_parallel_size": 1, "enforce_eager": True,
                      "max_num_seqs": config.max_num_seqs}
    env.update(REMA_COACH_BACKEND="local_vllm", REMA_COACH_MODEL=config.model,
               REMA_COACH_BASE_URL=endpoint, REMA_COACH_API_KEY_ENV="REMA_LOCAL_COACH_KEY",
               REMA_LOCAL_COACH_KEY=token, REMA_COACH_RESOLVED_REVISION=revision,
               REMA_COACH_LOCAL_IDENTITY=json.dumps(model_identity, sort_keys=True))
    receipt = {**asdict(config), **model_identity, "endpoint": endpoint, "reward_gpu": device,
               "status": "starting", "tensor_parallel_size": 1, "enforce_eager": True}
    receipt_path = output_dir / "coach_server.json"

    def save_receipt():
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    server = driver = None
    save_receipt()
    try:
        with (output_dir / "coach_server.log").open("a", buffering=1) as log:
            server = subprocess.Popen(server_command(config, revision, port),
                                      env=server_environment(env, token),
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            wait_ready(server, endpoint, token, config.model, config.startup_timeout)
            smoke_check(endpoint, token, config.model)
            receipt.update(status="ready", server_pid=server.pid)
            save_receipt()
            print(f"[rema-coach] Ready: {config.model}@{revision} GPU={device} endpoint={endpoint}", flush=True)
            driver = subprocess.Popen(command, env=env, start_new_session=True)
            while True:
                code = driver.poll()
                if code is not None:
                    receipt.update(status="driver_finished", driver_exit_code=code)
                    return code if code >= 0 else 128 - code
                if server.poll() is not None:
                    raise RuntimeError("Local coach exited while training; stopping driver, not changing rewards")
                time.sleep(1)
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        # Ignore additional termination signals while bounded cleanup runs.
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            stop_process_group(driver, config.shutdown_timeout)
            stop_process_group(server, config.shutdown_timeout)
            save_receipt()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A ReMA driver command is required")
    return supervise(LocalCoachConfig.from_env(os.environ), args.output_dir, command)


if __name__ == "__main__":
    sys.exit(main())

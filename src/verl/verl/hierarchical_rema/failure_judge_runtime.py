"""Evaluation-scoped vLLM process and explicit handoff of the reward GPU."""
from contextlib import contextmanager
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from urllib.parse import urlsplit


class ResourceRestoreError(RuntimeError):
    pass


def coach_request(base, token, path, *, post=False):
    request = urllib.request.Request(base + path, data=b"" if post else None,
                                     headers={"Authorization": "Bearer " + token})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=60) as response:
        body = response.read(1_000_001)
    if len(body) > 1_000_000:
        raise ValueError("Oversized coach handoff response")
    return json.loads(body) if body else {}


def wait_coach_state(base, token, sleeping, request=coach_request):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if request(base, token, "/is_sleeping").get("is_sleeping") is sleeping:
            return
        time.sleep(.5)
    raise TimeoutError("Coach GPU handoff timed out")


@contextmanager
def pause_local_coach(env, request=coach_request):
    if env.get("REMA_COACH_BACKEND") != "local_vllm":
        yield
        return
    endpoint = env.get("REMA_COACH_BASE_URL", "")
    parsed = urlsplit(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost") or parsed.path != "/v1":
        raise ValueError("Refusing GPU handoff to an unknown coach endpoint")
    base = endpoint[:-3]
    token = env.get(env.get("REMA_COACH_API_KEY_ENV", "REMA_LOCAL_COACH_KEY"), "")
    # Check support before changing state. Never kill the supervised coach.
    before = request(base, token, "/is_sleeping")
    if type(before.get("is_sleeping")) is not bool:
        raise ValueError("Coach does not expose a usable sleep-state endpoint")
    if before["is_sleeping"]:
        raise ValueError("Coach was unexpectedly asleep before validation handoff")
    try:
        request(base, token, "/sleep?level=1", post=True)
        wait_coach_state(base, token, True, request)
        yield
    finally:
        # This also runs if sleep timed out after the request reached the server.
        try:
            request(base, token, "/wake_up", post=True)
            wait_coach_state(base, token, False, request)
        except Exception as exc:
            raise ResourceRestoreError("Could not restore coach after failure-judge GPU handoff; "
                                       "stopping instead of continuing with unavailable rewards") from exc


def run_local_judge(config, documents, output_dir):
    from .reward_providers.local_coach import validate_gpu_isolation, server_environment, stop_process_group
    from .gfam_reward import load_cached_reward_scorer
    from .evaluation_exports import _atomic
    env = os.environ.copy()
    validate_gpu_isolation(env)
    output_dir.mkdir(parents=True, exist_ok=True)
    # No rollout trainer still owns this scorer at this validation boundary.
    clear = getattr(load_cached_reward_scorer, "cache_clear", None)
    if clear:
        clear()
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    job_path = output_dir / "failure_judge_input.json"
    results_path = output_dir / "failure_judge_results.jsonl"
    identity_path = output_dir / "failure_judge_runtime.json"
    job = {"config": asdict(config), "documents": documents,
           "results_path": str(results_path), "identity_path": str(identity_path),
           "cache_dir": str(output_dir.parent / ".failure_judge_cache")}
    _atomic(job_path, lambda handle: json.dump(job, handle, ensure_ascii=False))
    _atomic(results_path, lambda handle: None)
    _atomic(identity_path, lambda handle: json.dump({}, handle))
    worker = Path(__file__).with_name("failure_judge_worker.py")
    child_env = server_environment(env, "")
    child_env.pop("VLLM_USE_V1", None)  # Gemma uses the current engine, not the policy's pinned v0.
    child_env.pop("VLLM_API_KEY", None)
    child_env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    child_env.pop("VLLM_SERVER_DEV_MODE", None)
    error = None
    with pause_local_coach(env):
        process = None
        try:
            with (output_dir / "failure_judge.log").open("a", buffering=1) as log:
                process = subprocess.Popen([config.python, str(worker), str(job_path)],
                    env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                started = time.monotonic()
                while True:
                    try:
                        code = process.wait(timeout=min(30, max(1, config.timeout - (time.monotonic() - started))))
                        if code != 0:
                            error = f"Judge process exited with code {code}; inspect failure_judge.log"
                        break
                    except subprocess.TimeoutExpired:
                        elapsed = time.monotonic() - started
                        print(f"[failure-judge] running elapsed_s={elapsed:.0f}; see {output_dir / 'failure_judge.log'}",
                              flush=True)
                        if elapsed >= config.timeout:
                            error = "Judge process timeout; completed judgments retained"
                            break
        finally:
            # Gemma and all its engine workers must exit BEFORE waking the coach.
            stop_process_group(process, 30)
    metadata = json.loads(identity_path.read_text())
    if error:
        metadata["runtime_error"] = error
    results = {}
    with results_path.open() as handle:
        for line in handle:
            try:
                result = json.loads(line)
                if result.get("input_id") in documents:
                    results[result["input_id"]] = result
            except (ValueError, TypeError):
                pass  # A killed process may leave only its last line incomplete.
    for key in documents:
        results.setdefault(key, {"status": "unavailable", "error": error or "No judgment returned"})
    return {"metadata": metadata, "results": results}

import importlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
import urllib.error

import pytest

from test_reward_providers import base, coach, factory, rema, request, response

local = importlib.import_module(rema.__name__ + ".reward_providers.local_coach")
REVISION = "a" * 40


def identity(revision=REVISION):
    return {"model": "Qwen/Qwen3-32B", "revision": revision, "seed": 42, "thinking": False,
            "versions": {"vllm": "0.8.5.post1", "transformers": "4.51.3"}}


def local_provider(tmp_path, transport=None, port=1234, revision=REVISION):
    return coach.CoachRewardProvider(
        model="Qwen/Qwen3-32B", base_url=f"http://127.0.0.1:{port}/v1", cache_dir=tmp_path,
        backend="local_vllm", local_identity=json.dumps(identity(revision)),
        transport=transport, retries=2)


def test_default_shell_coach_needs_no_external_endpoint():
    root = Path(__file__).resolve().parents[4]
    env = {k: v for k, v in os.environ.items() if not k.startswith("REMA_COACH")}
    env.update(REWARD_SETTING="llm_coach", REMA_COACH_BASE_URL="https://must-not-reuse.invalid")
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s\\n" "$REMA_COACH_BACKEND" "$REMA_COACH_MODEL" '
         '"$REMA_COACH_MAX_NUM_SEQS" "${REMA_COACH_BASE_URL:-unset}"', "_",
         str(root / "scripts/reward-setting.sh")], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["local_vllm", "Qwen/Qwen3-32B", "1", "unset"]
    launcher = (root / "hierarchical-rema-trainer.sh").read_text()
    assert launcher.count('${GPU_PARTITION} --role reward -- ${COACH_SUPERVISOR}') == 2
    assert 'Local vLLM coach requires GFAM_REWARD_GPU_RESERVATION=1' in launcher


@pytest.mark.parametrize("mask,role", [("", "reward"), ("0,1", "reward"), ("-1", "reward"), ("0", "policy")])
def test_refuses_unisolated_gpu(mask, role):
    with pytest.raises(ValueError, match="isolated"):
        local.validate_gpu_isolation({"CUDA_VISIBLE_DEVICES": mask, "REMA_GPU_ROLE": role})


def test_server_is_frozen_single_gpu_and_detached_from_policy_ray():
    config = local.LocalCoachConfig.from_env({})
    assert local.validate_gpu_isolation({"CUDA_VISIBLE_DEVICES": "GPU-uuid", "REMA_GPU_ROLE": "reward"}) == "GPU-uuid"
    args = local.server_command(config, REVISION, 1234)
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--tensor-parallel-size") + 1] == "1"
    assert args[args.index("--distributed-executor-backend") + 1] == "mp"
    assert args[args.index("--tokenizer-revision") + 1] == REVISION
    assert "--trust-remote-code" not in args and "--enable-lora" not in args
    assert "--api-key" not in args and "--enforce-eager" in args
    env = local.server_environment({"CUDA_VISIBLE_DEVICES": "3", "RAY_ADDRESS": "policy",
                                   "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True", "VLLM_USE_RAY_SPMD_WORKER": "1",
                                   "HF_HOME": "/scratch/hf", "RANK": "0"}, "secret")
    assert env["CUDA_VISIBLE_DEVICES"] == "3" and env["HF_HOME"] == "/scratch/hf"
    assert "RAY_ADDRESS" not in env and "RANK" not in env
    assert "VLLM_USE_RAY_SPMD_WORKER" not in env and "VLLM_ALLOW_RUNTIME_LORA_UPDATING" not in env
    assert env["VLLM_API_KEY"] == "secret"


@pytest.mark.parametrize("overrides", [
    {"REMA_COACH_MAX_MODEL_LEN": "2048"}, {"REMA_COACH_MAX_NUM_SEQS": "0"},
    {"REMA_COACH_GPU_MEMORY_UTILIZATION": "nan"}, {"REMA_COACH_STARTUP_TIMEOUT": "inf"},
    {"REMA_COACH_SEED": "-1"}])
def test_local_config_rejects_invalid_resource_limits(overrides):
    with pytest.raises(ValueError):
        local.LocalCoachConfig.from_env(overrides)


def test_container_version_preflight_without_downloading(monkeypatch):
    monkeypatch.setattr(local.metadata, "version", lambda name: "0.8.4" if name == "vllm" else "4.51.3")
    with pytest.raises(RuntimeError, match="Rebuild the container"):
        local.resolve_model(local.LocalCoachConfig())


def test_resolves_same_commit_for_weights_and_tokenizer(monkeypatch):
    seen = []
    monkeypatch.setattr(local.metadata, "version", lambda name: "0.8.5.post1" if name == "vllm" else "4.51.3")
    class AutoConfig:
        @staticmethod
        def from_pretrained(model, **kwargs):
            seen.append((model, kwargs))
            return SimpleNamespace(_commit_hash=REVISION, max_position_embeddings=40960)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoConfig=AutoConfig))
    revision, _ = local.resolve_model(local.LocalCoachConfig())
    assert revision == REVISION
    assert seen[0][1] == {"revision": "main", "trust_remote_code": False}
    args = local.server_command(local.LocalCoachConfig(), revision, 1234)
    assert args[args.index("--revision") + 1] == args[args.index("--tokenizer-revision") + 1] == REVISION


def test_local_payload_and_port_independent_cache(tmp_path):
    req, seen = request(), []
    def transport(payload):
        seen.append(payload)
        return json.dumps(response(req))
    first = local_provider(tmp_path, transport).score_rollout(**req)
    second = local_provider(tmp_path, transport, port=6789).score_rollout(**req)
    assert first["status"] == "scored"
    assert second["graph_summary"]["coach_cache_hit"] and len(seen) == 1
    payload = seen[0]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["guided_json"]["properties"]["workers"]["required"] == ["A", "F"]
    assert "SECRET_REFERENCE" not in json.dumps(payload)
    assert "REJECTED_AUDIT" not in json.dumps(payload)
    local_provider(tmp_path, transport, revision="b" * 40).score_rollout(**req)
    assert len(seen) == 2


def test_local_manifest_is_stable_across_port_and_key_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("REMA_COACH_BACKEND", "local_vllm")
    monkeypatch.setenv("REMA_COACH_MODEL", "Qwen/Qwen3-32B")
    monkeypatch.setenv("REMA_COACH_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("REMA_COACH_LOCAL_IDENTITY", json.dumps(identity()))
    monkeypatch.setenv("REMA_LOCAL_COACH_KEY", "secret")
    args = SimpleNamespace(reward_setting="llm_coach")
    factory.validate_reward_setting(args)
    factory.write_reward_manifest(args, tmp_path)
    monkeypatch.setenv("REMA_COACH_BASE_URL", "http://127.0.0.1:5678/v1")
    monkeypatch.setenv("REMA_LOCAL_COACH_KEY", "another-secret")
    factory.write_reward_manifest(args, tmp_path)
    assert "secret" not in (tmp_path / "reward_setting.json").read_text()
    monkeypatch.setenv("REMA_COACH_LOCAL_IDENTITY", json.dumps(identity("b" * 40)))
    with pytest.raises(ValueError, match="different reward"):
        factory.write_reward_manifest(args, tmp_path)


def test_local_provider_requires_supervisor_identity(monkeypatch):
    monkeypatch.setenv("REMA_COACH_BACKEND", "local_vllm")
    monkeypatch.setenv("REMA_COACH_MODEL", "Qwen/Qwen3-32B")
    monkeypatch.setenv("REMA_COACH_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.delenv("REMA_COACH_LOCAL_IDENTITY", raising=False)
    with pytest.raises(ValueError, match="supervisor"):
        factory.validate_reward_setting(SimpleNamespace(reward_setting="llm_coach"))


@pytest.mark.parametrize("status,message,expected", [
    (400, "This model's maximum context length is 8192 tokens", "coach_context_overflow"),
    (401, "wrong api key", "coach_http_401")])
def test_nonretryable_request_failure_is_unscored(tmp_path, monkeypatch, status, message, expected):
    calls = []
    class Opener:
        def open(self, *args, **kwargs):
            calls.append(1)
            raise urllib.error.HTTPError("http://127.0.0.1", status, "", {},
                                         io.BytesIO(message.encode()))
    monkeypatch.setattr(coach.urllib.request, "build_opener", lambda *args: Opener())
    result = local_provider(tmp_path).score_rollout(**request())
    assert result["status"] == "unscored" and result["compiled_rewards"] == {}
    assert result["error"]["code"] == expected
    assert result["error"]["attempts"] == 1 and len(calls) == 1
    assert not list(tmp_path.glob("*.json"))


FAKE_SERVER = '''
from http.server import BaseHTTPRequestHandler, HTTPServer
import json, os, sys
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def answer(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def authorized(self):
        if self.headers.get("Authorization") != "Bearer " + os.environ["VLLM_API_KEY"]:
            self.send_error(401)
            return False
        return True
    def do_GET(self):
        if self.authorized():
            self.answer({"data":[{"id":sys.argv[2]}]})
    def do_POST(self):
        if not self.authorized():
            return
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert payload["chat_template_kwargs"] == {"enable_thinking":False}
        assert "guided_json" in payload
        self.answer({"choices":[{"finish_reason":"stop","message":{"content":'{"ready":true}'}}]})
HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
'''


@pytest.fixture
def supervised(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("REMA_GPU_ROLE", "reward")
    monkeypatch.setattr(local, "resolve_model", lambda config: (REVISION, {"vllm": "test"}))
    server_file = tmp_path / "fake_server.py"
    server_file.write_text(FAKE_SERVER)
    monkeypatch.setattr(local, "server_command",
                        lambda config, revision, port: [sys.executable, str(server_file), str(port), config.model])
    return local.LocalCoachConfig(startup_timeout=10, shutdown_timeout=1)


def assert_gone(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_supervisor_lifecycle_preserves_driver_status_and_cleans_server(tmp_path, supervised):
    command = [sys.executable, "-c", (
        "import os,sys; assert os.environ['CUDA_VISIBLE_DEVICES']=='3'; "
        "assert os.environ['REMA_COACH_RESOLVED_REVISION']=='" + REVISION + "'; "
        "assert os.environ['REMA_LOCAL_COACH_KEY']; sys.exit(7)")]
    assert local.supervise(supervised, tmp_path, command) == 7
    receipt = json.loads((tmp_path / "coach_server.json").read_text())
    assert receipt["status"] == "driver_finished" and receipt["driver_exit_code"] == 7
    assert receipt["revision"] == REVISION
    assert "REMA_LOCAL_COACH_KEY" not in json.dumps(receipt)
    assert_gone(receipt["server_pid"])


def test_startup_failure_never_starts_driver(tmp_path, supervised, monkeypatch):
    monkeypatch.setattr(local, "server_command", lambda *args: [sys.executable, "-c", "raise SystemExit(3)"])
    marker = tmp_path / "driver-started"
    with pytest.raises(RuntimeError, match="startup"):
        local.supervise(supervised, tmp_path,
                        [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"])
    assert not marker.exists()
    assert json.loads((tmp_path / "coach_server.json").read_text())["status"] == "failed"


def test_server_death_stops_driver_not_reward_fallback(tmp_path, supervised):
    receipt = str(tmp_path / "coach_server.json")
    driver_pid = tmp_path / "driver_pid"
    command = [sys.executable, "-c",
               f"import os,signal,time,json; from pathlib import Path; "
               f"Path({str(driver_pid)!r}).write_text(str(os.getpid())); "
               f"os.kill(json.loads(Path({receipt!r}).read_text())['server_pid'], signal.SIGTERM); time.sleep(30)"]
    with pytest.raises(RuntimeError, match="stopping driver"):
        local.supervise(supervised, tmp_path, command)
    assert_gone(int(driver_pid.read_text()))


def test_signal_during_readiness_cleans_server(tmp_path, supervised, monkeypatch):
    processes = []
    original = local.subprocess.Popen
    def start(*args, **kwargs):
        proc = original(*args, **kwargs)
        processes.append(proc)
        return proc
    monkeypatch.setattr(local.subprocess, "Popen", start)
    def interrupted(*args):
        os.kill(os.getpid(), signal.SIGTERM)
    monkeypatch.setattr(local, "wait_ready", interrupted)
    with pytest.raises(SystemExit) as exc:
        local.supervise(supervised, tmp_path, [sys.executable, "-c", "pass"])
    assert exc.value.code == 143
    assert processes[0].poll() is not None


def test_readiness_deadline_is_bounded(monkeypatch):
    ticks = iter([0, 0, 0, 0, 2])
    monkeypatch.setattr(local.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(local.time, "sleep", lambda _: None)
    monkeypatch.setattr(local, "_json_request", lambda *args: {})
    with pytest.raises(TimeoutError, match="startup timed out"):
        local.wait_ready(SimpleNamespace(poll=lambda: None), "url", "key", "model", 1)

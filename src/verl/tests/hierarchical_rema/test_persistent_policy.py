"""Persistent engine control-plane tests, without Ray/CUDA downloads."""
import importlib
from types import SimpleNamespace

import pytest

try:
    import verl.hierarchical_rema as api
except ModuleNotFoundError:
    import hierarchical_rema as api

policy = importlib.import_module(api.__name__ + ".persistent_policy")
generation = importlib.import_module(api.__name__ + ".ray_generation")


def test_checkpoint_identity_detects_replacement_at_same_path(tmp_path):
    file = tmp_path / "model.safetensors"
    file.write_bytes(b"first")
    first = policy.checkpoint_stamp(tmp_path)
    file.write_bytes(b"second")
    assert policy.checkpoint_stamp(tmp_path) != first
    assert policy.checkpoint_stamp("org/model") == "hub:org/model"


def test_gpu_lease_requires_sleep_unique_devices_and_exact_nodes():
    slots = [{"node_id": node, "cuda_visible_devices": str(gpu), "sleeping": True}
             for node in ("a", "b") for gpu in range(3)]
    selected = policy.choose_borrowed_slots(slots, ["b"] * 3)
    assert [s["cuda_visible_devices"] for s in selected] == ["0", "1", "2"]
    assert len(slots) == 6
    with pytest.raises(RuntimeError, match="sleep"):
        policy.choose_borrowed_slots([dict(slots[0], sleeping=False)], ["a"])
    with pytest.raises(RuntimeError, match="unique"):
        policy.choose_borrowed_slots([slots[0], slots[0]], ["a"])
    with pytest.raises(RuntimeError, match="No sleeping"):
        policy.choose_borrowed_slots(slots, ["c"])


class Workers:
    world_size = 2
    def __init__(self):
        self.refreshes = []
        self.sleeps = 0
    def refresh_rollout(self, path, version):
        self.refreshes.append((path, version))
        return [version] * 2
    def suspend_for_learning(self):
        self.sleeps += 1
        return [{"node_id": "a", "cuda_visible_devices": str(i), "sleeping": True} for i in range(2)]


def test_persistent_bundle_reused_and_generation_blocked_during_lease(monkeypatch):
    monkeypatch.setenv("REMA_PERSISTENT_ROLLOUT", "1")
    manager = generation.RayVLLMGenerationManager(api.VLLMBackendConfig())
    workers = Workers()
    builds = []
    def build(key):
        builds.append(key)
        return generation._RayBundle(key, key.model_path, object(), workers, None)
    monkeypatch.setattr(manager, "_build_bundle", build)
    first = manager._get_bundle(model_path="org/model", response_length=128)
    assert manager._get_bundle(model_path="org/model", response_length=128) is first
    slots = manager.suspend_for_learning()
    assert len(slots) == 2
    assert manager.suspend_for_learning() == slots
    assert workers.sleeps == 1
    with pytest.raises(RuntimeError, match="leased"):
        manager._get_bundle(model_path="org/model", response_length=128)
    manager.finish_learning()
    assert manager._get_bundle(model_path="org/model", response_length=128) is first
    assert len(builds) == 1


def test_stale_worker_acknowledgement_is_fatal(monkeypatch):
    monkeypatch.setenv("REMA_PERSISTENT_ROLLOUT", "1")
    manager = generation.RayVLLMGenerationManager(api.VLLMBackendConfig())
    key = generation.RayGenerationKey("org/model", manager._required_bundle_response_length(128), "hub:org/model")
    workers = SimpleNamespace(world_size=1, refresh_rollout=lambda *a: ["stale"])
    manager._bundle_cache[key] = generation._RayBundle(key, "org/model", None, workers, None)
    with pytest.raises(RuntimeError, match="stale"):
        manager._get_bundle(model_path="org/model", response_length=128)


def test_tensor_parallel_persistent_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("REMA_PERSISTENT_ROLLOUT", "1")
    with pytest.raises(ValueError, match="TP=1"):
        generation.RayVLLMGenerationManager(api.VLLMBackendConfig(tensor_model_parallel_size=2))


def test_checkpoint_cannot_change_inside_segment(monkeypatch, tmp_path):
    monkeypatch.setenv("REMA_PERSISTENT_ROLLOUT", "1")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"old")
    manager = generation.RayVLLMGenerationManager(api.VLLMBackendConfig())
    workers = Workers()
    monkeypatch.setattr(manager, "_build_bundle", lambda key:
        generation._RayBundle(key, key.model_path, object(), workers, None))
    manager._get_bundle(model_path=str(tmp_path), response_length=128)
    checkpoint.write_bytes(b"new checkpoint")
    with pytest.raises(RuntimeError, match="inside an active rollout segment"):
        manager._get_bundle(model_path=str(tmp_path), response_length=128)
    manager.suspend_for_learning()
    manager.finish_learning()
    assert manager._pinned_versions == {}


@pytest.fixture
def worker_module(monkeypatch):
    import sys
    from pathlib import Path
    # Exercise the CUDA worker's refresh state machine without importing Ray/vLLM.
    monkeypatch.setitem(sys.modules, "verl.single_controller.base", SimpleNamespace(Worker=object))
    monkeypatch.setitem(sys.modules, "verl.single_controller.base.decorator", SimpleNamespace(
        register=lambda **kw: lambda fn: fn,
        Dispatch=SimpleNamespace(ONE_TO_ALL=1, DP_COMPUTE_PROTO=2)))
    spec = importlib.util.spec_from_file_location(api.__name__ + ".persistent_rollout_worker",
        Path(api.__file__).parent / "persistent_rollout_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(module, "checkpoint_stamp", lambda path: "new")
    return module


def make_refresh_worker(module):
    import torch
    worker = module.PersistentRolloutWorker.__new__(module.PersistentRolloutWorker)
    worker.failed, worker.sleeping = False, True
    worker.path, worker.version = "old-path", "old"
    worker.architecture = {"hidden_size": 4}
    worker.config = SimpleNamespace(model=SimpleNamespace(trust_remote_code=False))
    worker.events = []
    worker.rollout = SimpleNamespace(inference_engine=SimpleNamespace(
        wake_up=lambda: worker.events.append("wake")))
    worker.runner = SimpleNamespace(model=SimpleNamespace(
        parameters=lambda: iter([torch.zeros(1)]), named_parameters=lambda: [("weight", None)],
        load_weights=lambda values: worker.events.append(("load", dict(values))) or {"weight"}))
    worker.reset_cache = lambda: worker.events.append("reset") or True
    return worker


def install_fake_hf(monkeypatch, info=None):
    import sys
    config = SimpleNamespace(to_dict=lambda: {"hidden_size": 4})
    cpu_model = SimpleNamespace(state_dict=lambda: {"weight": 7})
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: config),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: (cpu_model, info or {}))))
    monkeypatch.setitem(sys.modules, "verl.utils.fs", SimpleNamespace(copy_to_local=lambda path: path))


def test_refresh_loads_complete_weights_then_invalidates_prefix_cache(worker_module, monkeypatch):
    install_fake_hf(monkeypatch)
    worker = make_refresh_worker(worker_module)
    assert worker.refresh_rollout("new-path", "new") == "new"
    assert worker.events == ["wake", ("load", {"weight": 7}), "reset"]
    assert (worker.path, worker.version, worker.failed) == ("new-path", "new", False)
    worker.refresh_rollout("new-path", "new")
    assert len(worker.events) == 3  # No second load/reset for an unchanged checkpoint.


@pytest.mark.parametrize("failure", ["hf_missing", "vllm_missing", "cache_reset", "changed_during_load"])
def test_partial_refresh_never_allows_generation(worker_module, monkeypatch, failure):
    install_fake_hf(monkeypatch, {"missing_keys": ["weight"]} if failure == "hf_missing" else None)
    worker = make_refresh_worker(worker_module)
    if failure == "vllm_missing":
        worker.runner.model.load_weights = lambda values: set()
    elif failure == "cache_reset":
        worker.reset_cache = lambda: False
    elif failure == "changed_during_load":
        stamps = iter(["new", "changed"])
        monkeypatch.setattr(worker_module, "checkpoint_stamp", lambda path: next(stamps))
    with pytest.raises(RuntimeError):
        worker.refresh_rollout("new-path", "new")
    assert worker.failed and worker.version == "old"
    with pytest.raises(RuntimeError, match="invalid, or stale"):
        worker.generate_sequences(SimpleNamespace(meta_info={"rema_weight_version": "old"}))

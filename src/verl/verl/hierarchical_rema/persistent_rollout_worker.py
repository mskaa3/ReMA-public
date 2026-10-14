"""TP=1 vLLM-only workers with explicit weight refresh and learner handoff.

Unlike ActorRolloutRefWorker this does not construct a duplicate FSDP policy or
resynchronize unchanged policy weights for every generation chunk.
"""
import os

import torch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch

from .persistent_policy import architecture_signature, checkpoint_stamp


class PersistentRolloutWorker(Worker):
    def __init__(self, config, role="rollout"):
        super().__init__()
        if role != "rollout" or config.rollout.tensor_model_parallel_size != 1:
            raise ValueError("Persistent rollout currently supports rollout-only TP=1")
        self.config = config
        torch.cuda.set_device(0)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.sleeping = True
        self.version = None
        self.failed = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from transformers import AutoConfig
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

        path = copy_to_local(self.config.model.path)
        self.tokenizer = hf_tokenizer(path, trust_remote_code=self.config.model.trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_config = AutoConfig.from_pretrained(path, trust_remote_code=self.config.model.trust_remote_code)
        self.architecture = architecture_signature(model_config)
        self.config.rollout.enable_sleep_mode = True
        self.config.rollout.seed = int(self.config.rollout.get("seed", 0)) + 1000 + self.rank
        self.rollout = vLLMRollout(path, self.config.rollout, self.tokenizer, model_config)
        engine = self.rollout.inference_engine
        # This adapter intentionally targets the installed vLLM V0/SPMD API.
        # Do not silently use a different engine with unvalidated refresh semantics.
        self.runner = engine.llm_engine.model_executor.driver_worker.worker.model_runner
        self.reset_cache = getattr(engine, "reset_prefix_cache", None) or getattr(engine.llm_engine, "reset_prefix_cache", None)
        if not callable(self.reset_cache) or not callable(getattr(engine, "wake_up", None)):
            raise RuntimeError("Installed vLLM lacks persistent-engine sleep/cache-reset support")
        self.version = self.config.model.rema_weight_version
        self.path = self.config.model.path
        if checkpoint_stamp(self.path) != self.version:
            raise RuntimeError("Checkpoint changed during initial rollout-engine loading")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def refresh_rollout(self, model_path, version):
        if self.failed:
            raise RuntimeError("Persistent engine invalid after a previous failed refresh")
        if self.path == model_path and self.version == version:
            if checkpoint_stamp(model_path) != version:
                raise RuntimeError("Checkpoint was replaced before generation")
            self._wake()
            return version
        from transformers import AutoConfig, AutoModelForCausalLM
        from verl.utils.fs import copy_to_local

        if checkpoint_stamp(model_path) != version:
            raise RuntimeError("Checkpoint changed before rollout weight refresh")
        path = copy_to_local(model_path)
        config = AutoConfig.from_pretrained(path, trust_remote_code=self.config.model.trust_remote_code)
        if architecture_signature(config) != self.architecture:
            raise ValueError("Persistent rollout requires identical model architecture/config")
        try:
            self._wake()
            # No new GPU policy copy: stream the CPU HF tensors into the existing
            # vLLM parameters. HF loader validates the checkpoint before transfer.
            cpu_model, info = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=next(self.runner.model.parameters()).dtype, device_map="cpu",
                trust_remote_code=self.config.model.trust_remote_code, output_loading_info=True)
            if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise RuntimeError(f"Incomplete policy checkpoint during refresh: {info}")
            loaded = self.runner.model.load_weights(cpu_model.state_dict().items())
            missing = set(dict(self.runner.model.named_parameters())) - set(loaded or ())
            if missing:
                raise RuntimeError(f"vLLM did not refresh all parameters: {sorted(missing)}")
            del cpu_model
            torch.cuda.synchronize()
            if self.reset_cache() is False:
                raise RuntimeError("vLLM refused to invalidate the old-policy prefix cache")
            if checkpoint_stamp(model_path) != version:
                raise RuntimeError("Checkpoint changed during rollout weight refresh")
            self.path, self.version = model_path, version
            return version
        except Exception:
            self.failed = True
            raise

    def _wake(self):
        if self.sleeping:
            self.rollout.inference_engine.wake_up()
            self.sleeping = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def suspend_for_learning(self):
        import ray
        if not self.sleeping:
            torch.cuda.synchronize()
            self.rollout.inference_engine.sleep(level=1)
            torch.cuda.empty_cache()
            self.sleeping = True
        return {"node_id": ray.get_runtime_context().get_node_id(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "sleeping": self.sleeping, "version": self.version}

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts):
        if self.failed or self.sleeping or prompts.meta_info.get("rema_weight_version") != self.version:
            raise RuntimeError("Rollout attempted with sleeping, invalid, or stale policy weights")
        prompts.meta_info.update(eos_token_id=self.tokenizer.eos_token_id,
                                 pad_token_id=self.tokenizer.pad_token_id)
        return self.rollout.generate_sequences(prompts.to("cuda:0")).to("cpu")

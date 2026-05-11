from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schema import VLLMBackendConfig


@dataclass(frozen=True)
class RayGenerationKey:
    model_path: str
    response_length: int
    temperature: float
    top_p: float
    do_sample: bool


@dataclass
class RayGenerationResult:
    text: str
    entropy: float
    stop_reason: Optional[str]
    response_length: int


@dataclass
class _RayBundle:
    key: RayGenerationKey
    local_model_path: str
    tokenizer: object
    worker_group: object
    resource_pool: object


def build_vllm_rollout_config_dict(
    *,
    model_path: str,
    response_length: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    config: VLLMBackendConfig,
) -> Dict:
    max_model_len = config.max_model_len
    if max_model_len is None:
        max_model_len = config.prompt_length + response_length

    return {
        "trainer": {
            "nnodes": config.nnodes,
            "n_gpus_per_node": config.n_gpus_per_node,
        },
        "model": {
            "path": model_path,
            "external_lib": None,
            "override_config": {},
            "use_remove_padding": False,
            "enable_gradient_checkpointing": False,
            "trust_remote_code": config.trust_remote_code,
            "use_liger": False,
        },
        "rollout": {
            "name": "vllm",
            "temperature": temperature,
            "top_k": -1,
            "top_p": top_p,
            "prompt_length": config.prompt_length,
            "response_length": response_length,
            "dtype": config.dtype,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "ignore_eos": False,
            "enforce_eager": config.enforce_eager,
            "free_cache_engine": config.free_cache_engine,
            "load_format": config.load_format,
            "tensor_model_parallel_size": config.tensor_model_parallel_size,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "max_model_len": max_model_len,
            "max_num_seqs": config.max_num_seqs,
            "log_prob_micro_batch_size": None,
            "log_prob_micro_batch_size_per_gpu": 8,
            "use_fire_sampling": False,
            "do_sample": do_sample,
            "disable_log_stats": config.disable_log_stats,
            "enable_chunked_prefill": config.enable_chunked_prefill,
            "detokenize": config.detokenize,
            "n": 1,
        },
        "actor": {
            "strategy": "fsdp",
            "ulysses_sequence_parallel_size": 1,
            "fsdp_config": {
                "fsdp_size": -1,
            },
        },
    }


def proxy_entropy_from_generation(
    *,
    stop_reason: Optional[str],
    response_length: int,
    max_new_tokens: int,
) -> float:
    bounded_length = max(response_length, 0)
    safe_max_tokens = max(max_new_tokens, 1)
    length_ratio = min(bounded_length / safe_max_tokens, 1.0)
    stop_reason = (stop_reason or "").lower()

    if stop_reason in {"stop", "eos"}:
        base = 0.45
    elif stop_reason in {"length", "max_tokens"}:
        base = 1.15
    elif stop_reason:
        base = 0.85
    else:
        base = 0.75

    return min(2.0, base + 0.75 * length_ratio)


class RayVLLMGenerationManager:
    def __init__(self, config: VLLMBackendConfig) -> None:
        self.config = config
        self._active_bundle: Optional[_RayBundle] = None

    @staticmethod
    def _require_runtime() -> None:
        import importlib.util

        missing = [
            package
            for package in ("ray", "torch", "omegaconf", "transformers")
            if importlib.util.find_spec(package) is None
        ]
        if missing:
            raise ImportError(
                "RayVLLMGenerationManager requires missing packages: "
                + ", ".join(missing)
            )

    @staticmethod
    def _chunk_range(total_size: int, chunk_size: int) -> Sequence[Tuple[int, int]]:
        safe_chunk_size = max(int(chunk_size), 1)
        return [
            (start, min(start + safe_chunk_size, total_size))
            for start in range(0, total_size, safe_chunk_size)
        ]

    def _ensure_ray_initialized(self) -> None:
        import ray

        if ray.is_initialized():
            return
        ray_address = os.environ.get("RAY_ADDRESS", "").strip()
        ray_namespace = os.environ.get("RAY_NAMESPACE", "").strip()
        init_kwargs: Dict[str, Any] = {
            "runtime_env": {
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            }
        }
        if ray_address:
            init_kwargs["address"] = ray_address
        if ray_namespace:
            init_kwargs["namespace"] = ray_namespace
        ray.init(**init_kwargs)

    @staticmethod
    def _build_chat_messages(
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _encode_prompt_batch(
        self,
        *,
        tokenizer: object,
        prompt_texts: Sequence[str],
        system_prompt: str | None,
        do_sample: bool,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ):
        import torch

        try:
            from verl import DataProto
            from verl.utils.model import compute_position_id_with_mask
        except ModuleNotFoundError:
            from protocol import DataProto
            from utils.model import compute_position_id_with_mask

        chat_batch = [
            self._build_chat_messages(user_prompt=prompt_text, system_prompt=system_prompt)
            for prompt_text in prompt_texts
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                inputs = tokenizer.apply_chat_template(
                    chat_batch,
                    add_generation_prompt=True,
                    padding=True,
                    truncation=True,
                    max_length=self.config.prompt_length,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                )
            except TypeError:
                prompt_strings = [
                    tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    for messages in chat_batch
                ]
                inputs = tokenizer(
                    prompt_strings,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.prompt_length,
                )
        else:
            prompt_strings = []
            for prompt_text in prompt_texts:
                if system_prompt:
                    prompt_strings.append(
                        f"System:\n{system_prompt}\n\nUser:\n{prompt_text}\n\nAssistant:\n"
                    )
                else:
                    prompt_strings.append(prompt_text)
            inputs = tokenizer(
                prompt_strings,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.prompt_length,
            )

        if not isinstance(inputs["input_ids"], torch.Tensor):
            raise RuntimeError("Tokenizer returned unsupported prompt tensors for Ray/vLLM generation")

        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        meta_info: Dict[str, Any] = {
            "do_sample": do_sample,
        }
        if sampling_overrides:
            meta_info["sampling_overrides"] = dict(sampling_overrides)
        return DataProto.from_dict(
            tensors={
                "input_ids": inputs["input_ids"],
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            meta_info=meta_info,
        )

    def _build_bundle(
        self,
        key: RayGenerationKey,
    ) -> _RayBundle:
        self._require_runtime()
        self._ensure_ray_initialized()

        import ray
        from omegaconf import OmegaConf

        try:
            from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
            from verl.utils import hf_tokenizer
            from verl.utils.fs import copy_to_local
            from verl.workers.fsdp_workers import ActorRolloutRefWorker
        except ModuleNotFoundError:
            from single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
            from utils import hf_tokenizer
            from utils.fs import copy_to_local
            from workers.fsdp_workers import ActorRolloutRefWorker

        local_model_path = copy_to_local(key.model_path)
        tokenizer = hf_tokenizer(
            local_model_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        worker_config = OmegaConf.create(
            build_vllm_rollout_config_dict(
                model_path=key.model_path,
                response_length=key.response_length,
                temperature=key.temperature,
                top_p=key.top_p,
                do_sample=key.do_sample,
                config=self.config,
            )
        )
        cpus_per_node = self.config.cpus_per_node
        max_collocate_count = 5
        if cpus_per_node is not None and self.config.n_gpus_per_node > 0:
            max_collocate_count = max(1, int(cpus_per_node) // int(self.config.n_gpus_per_node))
        resource_pool = RayResourcePool(
            process_on_nodes=[self.config.n_gpus_per_node] * self.config.nnodes,
            max_colocate_count=max_collocate_count,
        )
        ray_cls_with_init = RayClassWithInitArgs(
            cls=ray.remote(ActorRolloutRefWorker),
            config=worker_config,
            role="rollout",
        )
        worker_group = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=ray_cls_with_init,
        )
        worker_group.init_model()
        return _RayBundle(
            key=key,
            local_model_path=local_model_path,
            tokenizer=tokenizer,
            worker_group=worker_group,
            resource_pool=resource_pool,
        )

    def _close_bundle(self, bundle: _RayBundle) -> None:
        import ray
        from ray.util import remove_placement_group

        for worker in bundle.worker_group.workers:
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                pass
        for placement_group in bundle.resource_pool.get_placement_groups():
            try:
                remove_placement_group(placement_group)
            except Exception:
                pass

    def close(self) -> None:
        if self._active_bundle is None:
            return
        self._close_bundle(self._active_bundle)
        self._active_bundle = None

    def _get_bundle(
        self,
        *,
        model_path: str,
        response_length: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> _RayBundle:
        key = RayGenerationKey(
            model_path=model_path,
            response_length=response_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        if self._active_bundle is not None and self._active_bundle.key == key:
            return self._active_bundle
        if self._active_bundle is not None:
            self._close_bundle(self._active_bundle)
            self._active_bundle = None
        self._active_bundle = self._build_bundle(key)
        return self._active_bundle

    def generate_batch(
        self,
        *,
        model_path: str,
        prompt_texts: Sequence[str],
        max_new_tokens: int,
        batch_size: int,
        system_prompt: str | None = None,
        temperature: float | None = None,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[RayGenerationResult]:
        if not prompt_texts:
            return []

        resolved_temperature = self.config.temperature if temperature is None else float(temperature)
        resolved_do_sample = resolved_temperature > 0.0
        bundle = self._get_bundle(
            model_path=model_path,
            response_length=max_new_tokens,
            temperature=resolved_temperature,
            top_p=self.config.top_p,
            do_sample=resolved_do_sample,
        )
        tokenizer = bundle.tokenizer

        try:
            from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        except ModuleNotFoundError:
            from protocol import pad_dataproto_to_divisor, unpad_dataproto

        results: List[RayGenerationResult] = []
        for start, end in self._chunk_range(len(prompt_texts), batch_size):
            prompt_chunk = prompt_texts[start:end]
            prompt_proto = self._encode_prompt_batch(
                tokenizer=tokenizer,
                prompt_texts=prompt_chunk,
                system_prompt=system_prompt,
                do_sample=resolved_do_sample,
                sampling_overrides=sampling_overrides,
            )
            padded_prompt_proto, pad_size = pad_dataproto_to_divisor(
                prompt_proto,
                bundle.worker_group.world_size,
            )
            output = bundle.worker_group.generate_sequences(padded_prompt_proto)
            output = unpad_dataproto(output, pad_size=pad_size)

            output_text = output.non_tensor_batch.get("text")
            stop_reasons = output.non_tensor_batch.get("stop_reasons")
            response_lengths = output.non_tensor_batch.get("gen_response_lengths")

            if output_text is None:
                decoded = tokenizer.batch_decode(
                    output.batch["responses"],
                    skip_special_tokens=True,
                )
                output_text = np.array(decoded, dtype=object)
            if stop_reasons is None:
                stop_reasons = np.array([None] * len(output_text), dtype=object)
            if response_lengths is None:
                response_lengths = np.array(
                    [len(text.split()) for text in output_text.tolist()],
                    dtype=object,
                )

            for text, stop_reason, response_length in zip(
                output_text.tolist(),
                stop_reasons.tolist(),
                response_lengths.tolist(),
            ):
                text = str(text).strip()
                response_length_int = int(response_length) if response_length is not None else 0
                results.append(
                    RayGenerationResult(
                        text=text,
                        entropy=proxy_entropy_from_generation(
                            stop_reason=stop_reason,
                            response_length=response_length_int,
                            max_new_tokens=max_new_tokens,
                        ),
                        stop_reason=None if stop_reason is None else str(stop_reason),
                        response_length=response_length_int,
                    )
                )

        return results

    def generate_one(
        self,
        *,
        model_path: str,
        prompt_text: str,
        max_new_tokens: int,
        system_prompt: str | None = None,
        temperature: float | None = None,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> RayGenerationResult:
        return self.generate_batch(
            model_path=model_path,
            prompt_texts=[prompt_text],
            max_new_tokens=max_new_tokens,
            batch_size=1,
            system_prompt=system_prompt,
            temperature=temperature,
            sampling_overrides=sampling_overrides,
        )[0]

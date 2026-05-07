try:
    from verl.hierarchical_rema.ray_generation import (
        build_vllm_rollout_config_dict,
        proxy_entropy_from_generation,
    )
    from verl.hierarchical_rema.schema import VLLMBackendConfig
except ModuleNotFoundError:
    from hierarchical_rema.ray_generation import (
        build_vllm_rollout_config_dict,
        proxy_entropy_from_generation,
    )
    from hierarchical_rema.schema import VLLMBackendConfig


def test_build_vllm_rollout_config_dict_uses_expected_repo_shape() -> None:
    config = VLLMBackendConfig(
        prompt_length=1536,
        controller_max_new_tokens=384,
        worker_max_new_tokens=192,
        nnodes=2,
        n_gpus_per_node=4,
        tensor_model_parallel_size=2,
        gpu_memory_utilization=0.6,
        max_num_batched_tokens=16384,
        max_num_seqs=512,
        dtype="bfloat16",
    )

    payload = build_vllm_rollout_config_dict(
        model_path="Qwen/Qwen2.5-1.5B-Instruct",
        response_length=256,
        config=config,
    )

    assert payload["trainer"] == {"nnodes": 2, "n_gpus_per_node": 4}
    assert payload["model"]["path"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert payload["rollout"]["name"] == "vllm"
    assert payload["rollout"]["prompt_length"] == 1536
    assert payload["rollout"]["response_length"] == 256
    assert payload["rollout"]["max_model_len"] == 1792
    assert payload["rollout"]["tensor_model_parallel_size"] == 2
    assert payload["rollout"]["max_num_batched_tokens"] == 16384
    assert payload["rollout"]["max_num_seqs"] == 512
    assert payload["rollout"]["detokenize"] is True
    assert payload["actor"]["strategy"] == "fsdp"
    assert config.controller_max_new_tokens == 384
    assert config.worker_max_new_tokens == 192
    assert config.max_format_retries == 2


def test_proxy_entropy_penalizes_truncation_more_than_clean_stop() -> None:
    clean_stop = proxy_entropy_from_generation(
        stop_reason="stop",
        response_length=32,
        max_new_tokens=128,
    )
    length_stop = proxy_entropy_from_generation(
        stop_reason="length",
        response_length=128,
        max_new_tokens=128,
    )

    assert clean_stop < length_stop
    assert 0.0 <= clean_stop <= 2.0
    assert 0.0 <= length_stop <= 2.0

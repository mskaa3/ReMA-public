"""Exercise the real vLLM adapter and C3 sharing with a CPU sampling backend."""

import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.rema_separated_trainer.ppo.multi_agent_rollout import MultiAgentRollout


class _SamplingParams:
    # vLLM distinguishes an advancing engine RNG from a per-request seed.
    seed = None
    n = 1
    temperature = 1.0
    max_tokens = 4
    logprobs = None
    detokenize = True
    top_p = 1.0
    top_k = -1
    min_p = 0.0
    best_of = 1
    include_stop_str_in_output = False

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _SamplingEngine:
    def __init__(self, **kwargs):
        self.seed = kwargs["seed"]
        self.rng = random.Random(self.seed)
        self.calls = []

    def sleep(self, level):
        assert level == 1

    def generate(self, prompts, sampling_params, use_tqdm):
        self.calls.append(SimpleNamespace(
            size=len(prompts), seed=sampling_params.seed,
            temperature=sampling_params.temperature,
            max_tokens=sampling_params.max_tokens, n=sampling_params.n,
        ))
        outputs = []
        for _ in prompts:
            rng = self.rng if sampling_params.seed is None else random.Random(sampling_params.seed)
            token = 2 if sampling_params.temperature == 0 else rng.randrange(10, 2**30)
            outputs.append(SimpleNamespace(outputs=[SimpleNamespace(
                token_ids=[token, 1], text=str(token), finish_reason="stop",
            )]))
        return outputs


@pytest.fixture
def adapter_factory(monkeypatch):
    # Load the real adapter without importing the CUDA-only vllm package initializer.
    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = _SamplingEngine
    fake_vllm.SamplingParams = _SamplingParams
    distributed = ModuleType("vllm.distributed")
    distributed.parallel_state = SimpleNamespace()
    third_party = ModuleType("verl.third_party.vllm")
    third_party.vllm_version = "0.8.5"
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    monkeypatch.setitem(sys.modules, "verl.third_party.vllm", third_party)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    path = Path(__file__).parents[2] / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    spec = importlib.util.spec_from_file_location("sampling_adapter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def build(seed=1):
        config = OmegaConf.create({
            "seed": seed, "n": 32, "temperature": 1.0,
            "prompt_length": 16, "response_length": 4,
            "tensor_model_parallel_size": 1, "enforce_eager": True,
            "free_cache_engine": False, "dtype": "bfloat16",
            "gpu_memory_utilization": 0.5, "disable_log_stats": True,
            "enable_chunked_prefill": False,
            "val_kwargs": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
        })
        return module.vLLMRollout(
            "test-model", config, SimpleNamespace(pad_token_id=0),
            SimpleNamespace(max_position_embeddings=64),
        )

    return build


def _prompts(size, **meta):
    # Identical contexts deliberately expose the fixed-per-request-seed bug.
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[0, 5, 6]]).repeat(size, 1),
            "attention_mask": torch.tensor([[0, 1, 1]]).repeat(size, 1),
            "position_ids": torch.tensor([[0, 0, 1]]).repeat(size, 1),
        },
        meta_info={"eos_token_id": 1, "is_multi_turn": True, **meta},
    )


def _sample(adapter, size=8, **meta):
    return adapter.generate_sequences(_prompts(size, **meta)).non_tensor_batch["text"].tolist()


def test_engine_seed_does_not_become_a_shared_request_seed(adapter_factory):
    adapter = adapter_factory(seed=17)
    assert adapter.inference_engine.seed == 17
    assert adapter.sampling_params.seed is None
    assert adapter.sampling_params.temperature == 1.0
    assert len(set(_sample(adapter))) == 8
    assert adapter.inference_engine.calls[-1].n == 1
    assert adapter.sampling_params.n == 32


def test_reproduces_fixed_request_seed_collapse_and_verifies_fix(adapter_factory):
    adapter = adapter_factory()
    adapter.sampling_params.seed = 1
    collapsed = _sample(adapter)
    assert len(set(collapsed)) == 1
    assert _sample(adapter) == collapsed
    adapter.sampling_params.seed = None
    first = _sample(adapter)
    second = _sample(adapter)
    assert len(set(first + second)) == 16


def test_fresh_seeded_engine_reproduces_stream_without_restarting_each_call(adapter_factory):
    first, second = adapter_factory(), adapter_factory()
    first_calls = [_sample(first), _sample(first)]
    assert first_calls == [_sample(second), _sample(second)]
    assert first_calls[0] != first_calls[1]
    assert first_calls[0] != _sample(adapter_factory(seed=2))


@pytest.mark.parametrize("meta", [{"do_sample": False}, {"validate": True}])
def test_greedy_validation_and_probes_stay_greedy_without_changing_training(adapter_factory, meta):
    adapter = adapter_factory()
    assert len(set(_sample(adapter, max_new_tokens=2, **meta))) == 1
    call = adapter.inference_engine.calls[-1]
    assert call.temperature == 0
    assert call.max_tokens == 2
    assert adapter.sampling_params.temperature == 1.0
    assert adapter.sampling_params.max_tokens == 4
    assert adapter.sampling_params.seed is None
    assert _sample(adapter) == _sample(adapter_factory())


@pytest.mark.parametrize("world_size", [1, 12])
def test_eight_actions_four_suffixes_share_only_the_intended_records(adapter_factory, world_size):
    adapter = adapter_factory()
    rollout = MultiAgentRollout.__new__(MultiAgentRollout)
    rollout.rollout_wg_dict = {"worker": SimpleNamespace(
        world_size=world_size, raw_generate_sequences=adapter.generate_sequences,
    )}
    rollout._prepare_chat_prompts = lambda role, chats, tokenizers: _prompts(len(chats))
    indices = list(range(32))
    chats = {i: [{"role": "user", "content": "identical context"}] for i in indices}
    groups = ["q"] * 32
    action_groups = [("q", i // 4) for i in indices]

    def generate(group_ids, coupled):
        return rollout._generate_from_hierarchical_chat_map(
            "worker", indices, chats, {"worker": None},
            {"do_sample": True, "validate": False}, 4, None,
            group_ids, coupled,
        )

    prefix = generate(groups, {"q"})
    assert len({record[0] for record in prefix.values()}) == 1
    actions = generate(action_groups, set(action_groups))
    assert len({record[0] for record in actions.values()}) == 8
    for start in range(0, 32, 4):
        assert len({tuple(actions[i][3]) for i in range(start, start + 4)}) == 1
    suffixes = generate(groups, set())
    assert len({record[0] for record in suffixes.values()}) == 32
    assert all(call.seed is None for call in adapter.inference_engine.calls)

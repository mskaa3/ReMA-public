from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Sequence

if TYPE_CHECKING:
    import torch

from .controller_data import ControllerReplaySample, load_samples_from_jsonl


@dataclass
class OfflineTrainingConfig:
    model_name_or_path: str
    output_dir: str
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    train_batch_size: int = 1
    grad_accum_steps: int = 8
    epochs: int = 1
    max_length: int = 4096
    truncation: str = "left"
    clip_range: float = 0.2
    clip_ratio_c: float = 3.0
    entropy_coeff: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.01
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 200
    eval_every_steps: int = 0
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    gradient_checkpointing: bool = True
    project_name: str = "hierarchical-rema"
    experiment_name: str = "hierarchical-rema-train"
    enable_wandb: bool = False
    save_final_checkpoint: bool = True
    save_best_checkpoint: bool = False
    save_intermediate_checkpoints: bool = False
    prune_unselected_checkpoints: bool = False


@dataclass
class DistributedTrainingContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "nccl"

def _ensure_repo_root_on_path() -> None:
    import sys

    package_roots = (
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[2],
    )
    for package_root in reversed(package_roots):
        package_root_str = str(package_root)
        if package_root_str not in sys.path:
            sys.path.insert(0, package_root_str)


class RayOfflineGRPOWorker:
    def __init__(self) -> None:
        _ensure_repo_root_on_path()

    def get_node_ip(self) -> str:
        import ray

        return ray.util.get_node_ip_address()

    def run(
        self,
        *,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        train_samples_jsonl: str,
        val_samples_jsonl: str,
        config_json: str,
    ):
        _ensure_repo_root_on_path()
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["PYTHONUNBUFFERED"] = "1"

        summary = run_offline_policy_training_from_jsonl(
            train_samples_jsonl=train_samples_jsonl,
            val_samples_jsonl=val_samples_jsonl,
            config_json=config_json,
        )
        return {
            "rank": rank,
            "node_ip": master_addr if rank == 0 else self.get_node_ip(),
            "summary": summary,
        }


def _lazy_torch():
    import torch

    return torch


def _resolve_dtype(torch, dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    torch = _lazy_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _distributed_context_from_env() -> DistributedTrainingContext:
    world_size = int(os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS") or 1)
    rank = int(os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or 0)
    local_rank = int(os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID") or 0)
    torch = _lazy_torch()
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    return DistributedTrainingContext(
        enabled=world_size > 1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
    )


def _init_distributed_training() -> DistributedTrainingContext:
    context = _distributed_context_from_env()
    if not context.enabled:
        return context

    torch = _lazy_torch()
    import torch.distributed as dist

    if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
        raise RuntimeError(
            "Distributed offline GRPO requires MASTER_ADDR and MASTER_PORT to be set."
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(context.local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend=context.backend,
            init_method="env://",
            timeout=timedelta(minutes=30),
        )
    return context


def _destroy_distributed_training(context: DistributedTrainingContext) -> None:
    if not context.enabled:
        return
    torch = _lazy_torch()
    import torch.distributed as dist

    if dist.is_initialized():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()
        dist.destroy_process_group()


def _is_primary_process(context: DistributedTrainingContext) -> bool:
    return (not context.enabled) or context.rank == 0


def _all_reduce_mean(value: float, device, context: DistributedTrainingContext) -> float:
    if not context.enabled:
        return float(value)
    torch = _lazy_torch()
    import torch.distributed as dist

    tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= max(context.world_size, 1)
    return float(tensor.item())


def _all_reduce_sum(value: float, device, context: DistributedTrainingContext) -> float:
    if not context.enabled:
        return float(value)
    torch = _lazy_torch()
    import torch.distributed as dist

    tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _load_scheduler_factory():
    _ensure_repo_root_on_path()
    try:
        from verl.utils.torch_functional import get_cosine_schedule_with_warmup

        return get_cosine_schedule_with_warmup
    except Exception:
        from transformers import get_cosine_schedule_with_warmup

        return get_cosine_schedule_with_warmup


def _init_tracking(config: OfflineTrainingConfig):
    _ensure_repo_root_on_path()
    try:
        try:
            from verl.utils.tracking import Tracking
        except ModuleNotFoundError:
            from utils.tracking import Tracking
    except Exception as exc:
        print(f"[hierarchical-rema][tracking] offline tracking disabled due to import/init error: {exc}")
        return None

    backends = ["console"]
    if config.enable_wandb:
        backends.append("wandb")
    print(
        f"[hierarchical-rema][tracking] offline tracking backends={','.join(backends)} "
        f"project={config.project_name} experiment={config.experiment_name}"
    )
    try:
        return Tracking(
            project_name=config.project_name,
            experiment_name=config.experiment_name,
            default_backend=backends,
            config=asdict(config),
        )
    except Exception as exc:
        print(f"[hierarchical-rema][tracking] offline tracking initialization failed: {exc}")
        return None


def _finish_tracking(tracking) -> None:
    if tracking is None:
        return
    try:
        finish_fn = getattr(tracking, "finish", None)
        if callable(finish_fn):
            finish_fn()
        else:
            tracking.__del__()
    except Exception:
        pass


def _load_model_and_tokenizer(
    config: OfflineTrainingConfig,
    distributed_context: DistributedTrainingContext,
):
    torch = _lazy_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = _resolve_dtype(torch, config.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=config.trust_remote_code,
    )
    if config.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if torch.cuda.is_available():
        device_name = config.device
        if distributed_context.enabled:
            device_name = f"cuda:{distributed_context.local_rank}"
        device = torch.device(device_name)
    else:
        device = torch.device("cpu")
    model.to(device)
    return tokenizer, model, device


def _format_prompt(tokenizer, prompt_text: str) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            pass
    return f"User:\n{prompt_text}\n\nAssistant:\n"


class ControllerReplayDataset:
    def __init__(self, samples: Sequence[ControllerReplaySample], tokenizer, max_length: int, truncation: str):
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.rows = [self._tokenize_sample(sample_index, sample) for sample_index, sample in enumerate(self.samples)]

    def _tokenize_sample(self, sample_index: int, sample: ControllerReplaySample) -> Dict:
        torch = _lazy_torch()

        prompt_text = _format_prompt(self.tokenizer, sample.prompt_text)
        response_text = sample.completion_text + (self.tokenizer.eos_token or "")

        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        response_ids = self.tokenizer(response_text, return_tensors="pt", add_special_tokens=False)

        prompt_input_ids = prompt_ids["input_ids"][0]
        prompt_attention_mask = prompt_ids["attention_mask"][0]
        response_input_ids = response_ids["input_ids"][0]
        response_attention_mask = response_ids["attention_mask"][0]

        input_ids = torch.cat((prompt_input_ids, response_input_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        prompt_length = prompt_input_ids.shape[0]
        response_length = response_input_ids.shape[0]

        if input_ids.shape[0] > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                prompt_length = min(prompt_length, self.max_length)
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            else:
                raise ValueError(
                    f"Sequence length {input_ids.shape[0]} exceeds max_length={self.max_length}"
                )

        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        last_idx = min(prompt_length + response_length, loss_mask.size(0)) - 1
        if last_idx >= 0:
            loss_mask[last_idx] = 0

        return {
            "sample_index": sample_index,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "reward": torch.tensor(sample.reward, dtype=torch.float32),
            "advantage": torch.tensor(sample.advantage, dtype=torch.float32),
            "group_id": sample.group_id,
            "role": sample.role,
            "policy_id": sample.policy_id,
            "pad_token_id": self.pad_token_id,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict:
        return self.rows[index]


def _collate_rows(rows: Sequence[Dict]) -> Dict:
    torch = _lazy_torch()

    max_length = max(row["input_ids"].shape[0] for row in rows)
    batch = {
        "sample_index": [],
        "input_ids": [],
        "attention_mask": [],
        "position_ids": [],
        "loss_mask": [],
        "reward": [],
        "advantage": [],
        "group_id": [],
        "role": [],
        "policy_id": [],
    }

    for row in rows:
        seq_len = row["input_ids"].shape[0]
        pad_len = max_length - seq_len
        pad_token_id = row["pad_token_id"]
        if pad_len > 0:
            batch["input_ids"].append(
                torch.cat(
                    [
                        row["input_ids"],
                        torch.full((pad_len,), pad_token_id, dtype=row["input_ids"].dtype),
                    ]
                )
            )
            batch["attention_mask"].append(
                torch.cat(
                    [row["attention_mask"], torch.zeros((pad_len,), dtype=row["attention_mask"].dtype)]
                )
            )
            batch["position_ids"].append(
                torch.cat(
                    [row["position_ids"], torch.zeros((pad_len,), dtype=row["position_ids"].dtype)]
                )
            )
            batch["loss_mask"].append(
                torch.cat([row["loss_mask"], torch.zeros((pad_len,), dtype=row["loss_mask"].dtype)])
            )
        else:
            batch["input_ids"].append(row["input_ids"])
            batch["attention_mask"].append(row["attention_mask"])
            batch["position_ids"].append(row["position_ids"])
            batch["loss_mask"].append(row["loss_mask"])

        batch["sample_index"].append(row["sample_index"])
        batch["reward"].append(row["reward"])
        batch["advantage"].append(row["advantage"])
        batch["group_id"].append(row["group_id"])
        batch["role"].append(row["role"])
        batch["policy_id"].append(row["policy_id"])

    return {
        "sample_index": list(batch["sample_index"]),
        "input_ids": torch.stack(batch["input_ids"], dim=0),
        "attention_mask": torch.stack(batch["attention_mask"], dim=0),
        "position_ids": torch.stack(batch["position_ids"], dim=0),
        "loss_mask": torch.stack(batch["loss_mask"], dim=0),
        "reward": torch.stack(batch["reward"], dim=0),
        "advantage": torch.stack(batch["advantage"], dim=0),
        "group_id": list(batch["group_id"]),
        "role": list(batch["role"]),
        "policy_id": list(batch["policy_id"]),
    }


def _sequence_log_probs(logits, labels):
    torch = _lazy_torch()
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return gathered


def _compute_old_log_prob_cache(model, dataset: ControllerReplayDataset, batch_size: int, device) -> Dict[int, torch.Tensor]:
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_rows,
    )
    cached: Dict[int, torch.Tensor] = {}
    model.eval()
    with _lazy_torch().no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            loss_mask = batch["loss_mask"][:, :-1]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            token_log_probs = _sequence_log_probs(outputs.logits[:, :-1, :], input_ids[:, 1:])
            for row_idx, sample_index in enumerate(batch["sample_index"]):
                cached[sample_index] = token_log_probs[row_idx][loss_mask[row_idx].bool()].detach().cpu()
    return cached


def _gather_old_log_probs(
    sample_indices,
    old_log_prob_cache,
    loss_mask,
    device,
    dtype,
):
    torch = _lazy_torch()
    padded = []
    target_length = loss_mask.shape[1]
    for row_idx, sample_index in enumerate(sample_indices):
        old_log_probs = old_log_prob_cache[sample_index]
        row = torch.zeros((target_length,), dtype=dtype)
        valid_length = int(loss_mask[row_idx].sum().item())
        take = min(valid_length, old_log_probs.shape[0], target_length)
        if take > 0:
            positions = torch.nonzero(loss_mask[row_idx], as_tuple=False).squeeze(-1)[:take]
            row[positions] = old_log_probs[:take].to(dtype)
        padded.append(row)
    return torch.stack(padded, dim=0).to(device)


def _count_parameters(model) -> Dict[str, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "num_parameters": total,
        "num_trainable_parameters": trainable,
    }


def _filter_batch_rows(batch: Dict, row_mask) -> Dict:
    filtered = {}
    keep_rows = row_mask.tolist()
    for key, value in batch.items():
        if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == len(keep_rows):
            filtered[key] = value[row_mask]
        elif isinstance(value, list) and len(value) == len(keep_rows):
            filtered[key] = [item for item, keep in zip(value, keep_rows) if keep]
        else:
            filtered[key] = value
    return filtered


def _selector_format_bucket(sample: ControllerReplaySample) -> str | None:
    if sample.role != "selector":
        return None
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    validation = metadata.get("format_validation")
    if not isinstance(validation, dict):
        return "clean"
    if bool(validation.get("fallback_used")):
        return "hard_fallback"

    attempt_raw = validation.get("attempt", 0)
    try:
        attempt_count = max(int(attempt_raw), 0)
    except (TypeError, ValueError):
        attempt_count = 0

    errors_before_success = validation.get("errors_before_success")
    if isinstance(errors_before_success, list):
        attempt_count = max(attempt_count, len(errors_before_success))

    if attempt_count > 0:
        return "model_repaired"
    if bool(validation.get("partial_completion_used")):
        return "locally_repaired"
    return "clean"


def _empty_selector_format_counts() -> Dict[str, int]:
    return {
        "clean": 0,
        "locally_repaired": 0,
        "model_repaired": 0,
        "hard_fallback": 0,
    }


def _update_selector_format_counts(
    counts: Dict[str, int],
    *,
    sample_indices,
    samples: Sequence[ControllerReplaySample],
) -> None:
    for sample_index in sample_indices:
        bucket = _selector_format_bucket(samples[int(sample_index)])
        if bucket is not None:
            counts[bucket] += 1


def _empty_role_reward_stats() -> Dict[str, Dict[str, float]]:
    return {
        "selector": {"sum": 0.0, "count": 0.0},
        "decomposer": {"sum": 0.0, "count": 0.0},
    }


def _update_role_reward_stats(
    stats: Dict[str, Dict[str, float]],
    *,
    rewards,
    roles,
) -> None:
    for reward, role in zip(rewards, roles):
        bucket = stats.get(str(role))
        if bucket is None:
            continue
        bucket["sum"] += float(reward.item())
        bucket["count"] += 1.0


def _all_finite(*tensors) -> bool:
    torch = _lazy_torch()
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def run_offline_policy_training(
    train_samples: Sequence[ControllerReplaySample],
    val_samples: Sequence[ControllerReplaySample],
    config: OfflineTrainingConfig,
    tracking=None,
    tracking_prefix: str = "",
    log_step_offset: int = 0,
) -> Dict:
    _ensure_repo_root_on_path()
    from torch.utils.data import DataLoader

    torch = _lazy_torch()
    tracking_last_step = getattr(tracking, "last_step", None)
    if tracking_last_step is not None:
        log_step_offset = max(int(log_step_offset), int(tracking_last_step))
    distributed_context = _init_distributed_training()
    is_primary = _is_primary_process(distributed_context)
    try:
        from verl.trainer.ppo import core_algos
    except Exception as exc:
        raise ImportError("GRPO training requires the repo PPO core_algos module to be importable") from exc

    _set_random_seeds(config.seed)
    scheduler_factory = _load_scheduler_factory()
    tokenizer, model, device = _load_model_and_tokenizer(config, distributed_context)
    if distributed_context.enabled:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            model,
            device_ids=[distributed_context.local_rank] if device.type == "cuda" else None,
            output_device=distributed_context.local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    train_dataset = ControllerReplayDataset(train_samples, tokenizer, config.max_length, config.truncation)
    val_dataset = ControllerReplayDataset(val_samples, tokenizer, config.max_length, config.truncation)

    generator = torch.Generator().manual_seed(config.seed)
    train_sampler = None
    val_sampler = None
    if distributed_context.enabled:
        from torch.utils.data import DistributedSampler

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=distributed_context.world_size,
            rank=distributed_context.rank,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
        )
        if len(val_dataset):
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=distributed_context.world_size,
                rank=distributed_context.rank,
                shuffle=False,
                drop_last=False,
            )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=_collate_rows,
        generator=generator if train_sampler is None else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=_collate_rows,
    ) if len(val_dataset) else None

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_update_steps = max(
        1,
        math.ceil(len(train_loader) * config.epochs / max(config.grad_accum_steps, 1)),
    )
    warmup_steps = int(total_update_steps * config.warmup_ratio)
    scheduler = scheduler_factory(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    old_log_prob_cache = _compute_old_log_prob_cache(
        model=model,
        dataset=train_dataset,
        batch_size=config.train_batch_size,
        device=device,
    )

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    owns_tracking = tracking is None
    if is_primary:
        tracking = tracking or _init_tracking(config)
    else:
        tracking = None
    metrics_log_path = output_dir / "train_metrics.jsonl"
    eval_log_path = output_dir / "eval_metrics.jsonl"
    config_path = output_dir / "train_config.json"
    if is_primary:
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, indent=2, sort_keys=True)

    training_data_stats = {
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "objective": "grpo",
        **_count_parameters(model),
    }
    if is_primary:
        with (output_dir / "training_data_stats.json").open("w", encoding="utf-8") as handle:
            json.dump(training_data_stats, handle, indent=2, sort_keys=True)
        print(
            f"[hierarchical-rema][grpo] experiment={config.experiment_name} "
            f"train_samples={len(train_samples)} val_samples={len(val_samples)} "
            f"updates={total_update_steps} world_size={distributed_context.world_size}"
        )
        if tracking is not None:
            tracking.log(
                {
                    "train/num_train_samples": len(train_samples),
                    "train/num_val_samples": len(val_samples),
                    "train/total_update_steps": total_update_steps,
                },
                step=log_step_offset,
            )

    global_step = 0
    skipped_empty_batches = 0
    skipped_non_finite_batches = 0
    optimizer.zero_grad(set_to_none=True)
    best_val_loss = None
    best_val_step = 0
    step_selector_format_counts = _empty_selector_format_counts()
    step_role_reward_stats = _empty_role_reward_stats()

    for epoch in range(config.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        if is_primary:
            print(f"[hierarchical-rema][grpo] epoch {epoch + 1}/{config.epochs}")
        for batch_idx, batch in enumerate(train_loader):
            valid_row_mask = batch["loss_mask"][:, :-1].sum(dim=1) > 0
            if not bool(valid_row_mask.any().item()):
                skipped_empty_batches += 1
                optimizer.zero_grad(set_to_none=True)
                step_selector_format_counts = _empty_selector_format_counts()
                step_role_reward_stats = _empty_role_reward_stats()
                if is_primary:
                    print(
                        f"[hierarchical-rema][grpo] skipping empty batch "
                        f"epoch={epoch + 1}/{config.epochs} batch={batch_idx + 1}/{len(train_loader)}"
                    )
                continue
            if not bool(valid_row_mask.all().item()):
                batch = _filter_batch_rows(batch, valid_row_mask)
                skipped_empty_batches += 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            loss_mask = batch["loss_mask"][:, :-1].to(device).float()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            if not _all_finite(outputs.logits):
                skipped_non_finite_batches += 1
                optimizer.zero_grad(set_to_none=True)
                step_selector_format_counts = _empty_selector_format_counts()
                step_role_reward_stats = _empty_role_reward_stats()
                if is_primary:
                    print(
                        f"[hierarchical-rema][grpo] skipping non-finite logits "
                        f"epoch={epoch + 1}/{config.epochs} batch={batch_idx + 1}/{len(train_loader)}"
                    )
                continue
            token_log_probs = _sequence_log_probs(outputs.logits[:, :-1, :], input_ids[:, 1:])
            old_log_probs = _gather_old_log_probs(
                sample_indices=batch["sample_index"],
                old_log_prob_cache=old_log_prob_cache,
                loss_mask=loss_mask,
                device=device,
                dtype=token_log_probs.dtype,
            )
            advantages = batch["advantage"].to(device).unsqueeze(-1).expand_as(token_log_probs)
            if not _all_finite(token_log_probs, old_log_probs, advantages, loss_mask):
                skipped_non_finite_batches += 1
                optimizer.zero_grad(set_to_none=True)
                step_selector_format_counts = _empty_selector_format_counts()
                step_role_reward_stats = _empty_role_reward_stats()
                if is_primary:
                    print(
                        f"[hierarchical-rema][grpo] skipping non-finite batch tensors "
                        f"epoch={epoch + 1}/{config.epochs} batch={batch_idx + 1}/{len(train_loader)}"
                    )
                continue
            pg_loss, clipfrac, approx_kl, clipfrac_lower = core_algos.compute_policy_loss(
                old_log_prob=old_log_probs,
                log_prob=token_log_probs,
                advantages=advantages,
                eos_mask=loss_mask,
                cliprange=config.clip_range,
                clip_ratio_c=config.clip_ratio_c,
            )
            entropy = core_algos.compute_entropy_loss(outputs.logits[:, :-1, :], loss_mask)
            loss = pg_loss - config.entropy_coeff * entropy
            if not _all_finite(pg_loss, clipfrac, approx_kl, clipfrac_lower, entropy, loss):
                skipped_non_finite_batches += 1
                optimizer.zero_grad(set_to_none=True)
                step_selector_format_counts = _empty_selector_format_counts()
                step_role_reward_stats = _empty_role_reward_stats()
                if is_primary:
                    print(
                        f"[hierarchical-rema][grpo] skipping non-finite objective "
                        f"epoch={epoch + 1}/{config.epochs} batch={batch_idx + 1}/{len(train_loader)} "
                        f"mean_reward={float(batch['reward'].mean().item()):.4f} "
                        f"mean_advantage={float(batch['advantage'].mean().item()):.4f}"
                    )
                continue

            _update_selector_format_counts(
                step_selector_format_counts,
                sample_indices=batch["sample_index"],
                samples=train_samples,
            )
            _update_role_reward_stats(
                step_role_reward_stats,
                rewards=batch["reward"],
                roles=batch["role"],
            )
            loss = loss / max(config.grad_accum_steps, 1)
            loss.backward()

            should_step = (batch_idx + 1) % max(config.grad_accum_steps, 1) == 0 or (batch_idx + 1) == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                selector_total_local = sum(step_selector_format_counts.values())
                selector_reward_count = int(
                    round(_all_reduce_sum(step_role_reward_stats["selector"]["count"], device, distributed_context))
                )
                selector_reward_sum = _all_reduce_sum(
                    step_role_reward_stats["selector"]["sum"],
                    device,
                    distributed_context,
                )
                decomposer_reward_count = int(
                    round(_all_reduce_sum(step_role_reward_stats["decomposer"]["count"], device, distributed_context))
                )
                decomposer_reward_sum = _all_reduce_sum(
                    step_role_reward_stats["decomposer"]["sum"],
                    device,
                    distributed_context,
                )
                metrics = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": _all_reduce_mean(
                        float(loss.detach().item() * max(config.grad_accum_steps, 1)),
                        device,
                        distributed_context,
                    ),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "objective": "grpo",
                    "mean_reward": _all_reduce_mean(float(batch["reward"].mean().item()), device, distributed_context),
                    "mean_advantage": _all_reduce_mean(float(batch["advantage"].mean().item()), device, distributed_context),
                    "approx_kl": _all_reduce_mean(float(approx_kl.detach().item()), device, distributed_context),
                    "entropy": _all_reduce_mean(float(entropy.detach().item()), device, distributed_context),
                    "clipfrac": _all_reduce_mean(float(clipfrac.detach().item()), device, distributed_context),
                    "clipfrac_lower": _all_reduce_mean(float(clipfrac_lower.detach().item()), device, distributed_context),
                    "selector_samples_total": int(
                        round(_all_reduce_sum(selector_total_local, device, distributed_context))
                    ),
                    "selector_samples_clean": int(
                        round(_all_reduce_sum(step_selector_format_counts["clean"], device, distributed_context))
                    ),
                    "selector_samples_locally_repaired": int(
                        round(_all_reduce_sum(step_selector_format_counts["locally_repaired"], device, distributed_context))
                    ),
                    "selector_samples_model_repaired": int(
                        round(_all_reduce_sum(step_selector_format_counts["model_repaired"], device, distributed_context))
                    ),
                    "selector_samples_hard_fallback": int(
                        round(_all_reduce_sum(step_selector_format_counts["hard_fallback"], device, distributed_context))
                    ),
                }
                if selector_reward_count > 0:
                    metrics["selector_mean_reward"] = selector_reward_sum / selector_reward_count
                    metrics["selector_reward_count"] = selector_reward_count
                if decomposer_reward_count > 0:
                    metrics["decomposer_mean_reward"] = decomposer_reward_sum / decomposer_reward_count
                    metrics["decomposer_reward_count"] = decomposer_reward_count
                step_selector_format_counts = _empty_selector_format_counts()
                step_role_reward_stats = _empty_role_reward_stats()
                if is_primary:
                    with metrics_log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                    if (
                        global_step == 1
                        or global_step % max(config.logging_steps, 1) == 0
                        or global_step == total_update_steps
                    ):
                        concise_metrics = {
                            f"{tracking_prefix}train/loss": metrics["loss"],
                            f"{tracking_prefix}train/lr": metrics["lr"],
                            f"{tracking_prefix}train/mean_reward": metrics["mean_reward"],
                            f"{tracking_prefix}train/overall_mean_reward": metrics["mean_reward"],
                            f"{tracking_prefix}train/mean_advantage": metrics["mean_advantage"],
                            f"{tracking_prefix}train/overall_mean_advantage": metrics["mean_advantage"],
                            f"{tracking_prefix}train/approx_kl": metrics["approx_kl"],
                            f"{tracking_prefix}train/entropy": metrics["entropy"],
                            f"{tracking_prefix}train/clipfrac": metrics["clipfrac"],
                            f"{tracking_prefix}train/selector_samples_total": metrics["selector_samples_total"],
                            f"{tracking_prefix}train/selector_samples_clean": metrics["selector_samples_clean"],
                            f"{tracking_prefix}train/selector_samples_locally_repaired": metrics["selector_samples_locally_repaired"],
                            f"{tracking_prefix}train/selector_samples_model_repaired": metrics["selector_samples_model_repaired"],
                            f"{tracking_prefix}train/selector_samples_hard_fallback": metrics["selector_samples_hard_fallback"],
                        }
                        if "selector_mean_reward" in metrics:
                            concise_metrics[f"{tracking_prefix}train/selector_mean_reward"] = metrics["selector_mean_reward"]
                        if "decomposer_mean_reward" in metrics:
                            concise_metrics[f"{tracking_prefix}train/decomposer_mean_reward"] = metrics["decomposer_mean_reward"]
                        selector_format_log = ""
                        if metrics["selector_samples_total"] > 0:
                            selector_format_log = (
                                f" selector_samples={metrics['selector_samples_total']} "
                                f"clean={metrics['selector_samples_clean']} "
                                f"locally_repaired={metrics['selector_samples_locally_repaired']} "
                                f"model_repaired={metrics['selector_samples_model_repaired']} "
                                f"hard_fallback={metrics['selector_samples_hard_fallback']}"
                            )
                        role_reward_log = ""
                        if "selector_mean_reward" in metrics:
                            role_reward_log += f" selector_mean_reward={metrics['selector_mean_reward']:.4f}"
                        if "decomposer_mean_reward" in metrics:
                            role_reward_log += f" decomposer_mean_reward={metrics['decomposer_mean_reward']:.4f}"
                        print(
                            f"[hierarchical-rema][grpo] step={global_step}/{total_update_steps} "
                            f"epoch={epoch + 1}/{config.epochs} "
                            f"loss={metrics['loss']:.6f} "
                            f"mean_reward={metrics['mean_reward']:.4f} "
                            f"mean_advantage={metrics['mean_advantage']:.4f} "
                            f"approx_kl={metrics['approx_kl']:.6f}"
                            f"{selector_format_log}"
                            f"{role_reward_log}"
                        )
                        if tracking is not None:
                            tracking.log(concise_metrics, step=log_step_offset + global_step)

                if config.eval_every_steps > 0 and val_loader is not None and global_step % config.eval_every_steps == 0:
                    val_metrics = evaluate_controller_model(
                        model=model,
                        dataloader=val_loader,
                        device=device,
                        distributed_context=distributed_context,
                    )
                    if (
                        config.save_best_checkpoint
                        and (best_val_loss is None or val_metrics["val_loss"] < best_val_loss)
                    ):
                        best_val_loss = val_metrics["val_loss"]
                        best_val_step = global_step
                        if is_primary:
                            _save_model_checkpoint(model, tokenizer, output_dir / "best")
                    if is_primary:
                        with eval_log_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps({"step": global_step, **val_metrics}, sort_keys=True) + "\n")
                        print(
                            f"[hierarchical-rema][grpo] eval step={global_step} "
                            f"val_loss={val_metrics['val_loss']:.6f}"
                        )
                        if tracking is not None:
                            tracking.log(
                                {f"{tracking_prefix}val/val_loss": val_metrics["val_loss"]},
                                step=log_step_offset + global_step,
                            )

                if (
                    config.save_intermediate_checkpoints
                    and config.save_steps > 0
                    and global_step % config.save_steps == 0
                ):
                    if is_primary:
                        _save_model_checkpoint(model, tokenizer, output_dir / f"checkpoint-{global_step}")
                        print(
                            f"[hierarchical-rema][grpo] saved checkpoint step={global_step} "
                            f"path={output_dir / f'checkpoint-{global_step}'}"
                        )

    if config.save_final_checkpoint and is_primary:
        _save_model_checkpoint(model, tokenizer, output_dir / "final")
    final_val_metrics = {}
    if val_loader is not None:
        final_val_metrics = evaluate_controller_model(
            model=model,
            dataloader=val_loader,
            device=device,
            distributed_context=distributed_context,
        )
        if (
            config.save_best_checkpoint
            and (best_val_loss is None or final_val_metrics["val_loss"] <= best_val_loss)
        ):
            best_val_loss = final_val_metrics["val_loss"]
            best_val_step = global_step
            if is_primary:
                _save_model_checkpoint(model, tokenizer, output_dir / "best")
    final_checkpoint_path = output_dir / "final"
    best_checkpoint_path = output_dir / "best"
    selected_model_path = (
        str(best_checkpoint_path)
        if config.save_best_checkpoint and best_checkpoint_path.exists()
        else (
            str(final_checkpoint_path)
            if config.save_final_checkpoint and final_checkpoint_path.exists()
            else str(output_dir)
        )
    )
    selected_model_source = (
        "best"
        if config.save_best_checkpoint and best_checkpoint_path.exists()
        else (
            "final"
            if config.save_final_checkpoint and final_checkpoint_path.exists()
            else "output_dir"
        )
    )

    if is_primary and config.prune_unselected_checkpoints:
        removable_checkpoint_dirs = []
        if selected_model_source == "best" and final_checkpoint_path.exists():
            removable_checkpoint_dirs.append(final_checkpoint_path)
        elif selected_model_source == "final" and best_checkpoint_path.exists():
            removable_checkpoint_dirs.append(best_checkpoint_path)
        for checkpoint_dir in removable_checkpoint_dirs:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    summary = {
        "output_dir": str(output_dir),
        "steps": global_step,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "objective": "grpo",
        "skipped_empty_batches": skipped_empty_batches,
        "skipped_non_finite_batches": skipped_non_finite_batches,
        "save_final_checkpoint": config.save_final_checkpoint,
        "save_best_checkpoint": config.save_best_checkpoint,
        "save_intermediate_checkpoints": config.save_intermediate_checkpoints,
        "prune_unselected_checkpoints": config.prune_unselected_checkpoints,
        "final_checkpoint_path": str(final_checkpoint_path) if final_checkpoint_path.exists() else "",
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() else "",
        "selected_model_path": selected_model_path,
        "selected_model_source": selected_model_source,
        "best_val_loss": best_val_loss,
        "best_val_step": best_val_step,
        **_count_parameters(model),
    }
    if final_val_metrics:
        summary.update(final_val_metrics)
    if is_primary:
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(
            f"[hierarchical-rema][grpo] finished experiment={config.experiment_name} "
            f"steps={global_step} output_dir={output_dir} "
            f"skipped_empty_batches={skipped_empty_batches} "
            f"skipped_non_finite_batches={skipped_non_finite_batches} "
            f"selected_model_source={selected_model_source}"
        )
        if tracking is not None:
            final_metrics = {
                f"{tracking_prefix}train/final_steps": global_step,
                f"{tracking_prefix}train/num_train_samples": len(train_samples),
                f"{tracking_prefix}train/num_val_samples": len(val_samples),
                f"{tracking_prefix}train/skipped_empty_batches": skipped_empty_batches,
                f"{tracking_prefix}train/skipped_non_finite_batches": skipped_non_finite_batches,
            }
            if "val_loss" in summary:
                final_metrics[f"{tracking_prefix}val/final_loss"] = summary["val_loss"]
            if best_val_loss is not None:
                final_metrics[f"{tracking_prefix}val/best_loss"] = float(best_val_loss)
            tracking.log(final_metrics, step=log_step_offset + max(global_step, 1))
            if owns_tracking:
                _finish_tracking(tracking)
    _destroy_distributed_training(distributed_context)
    return summary


def evaluate_controller_model(
    model,
    dataloader,
    device,
    distributed_context: DistributedTrainingContext | None = None,
) -> Dict[str, float]:
    torch = _lazy_torch()
    distributed_context = distributed_context or DistributedTrainingContext()
    model.eval()
    total_loss_sum = torch.tensor(0.0, device=device)
    total_weight_sum = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            loss_mask = batch["loss_mask"][:, :-1].to(device).float()
            if not bool((loss_mask.sum(dim=1) > 0).any().item()):
                continue
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            if not bool(torch.isfinite(outputs.logits).all().item()):
                continue
            labels = input_ids[:, 1:]
            per_token_loss = torch.nn.functional.cross_entropy(
                outputs.logits[:, :-1, :].reshape(-1, outputs.logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).view_as(loss_mask)
            masked_loss = per_token_loss * loss_mask
            total_loss_sum += masked_loss.sum()
            total_weight_sum += torch.clamp(loss_mask.sum(), min=0.0)
    if distributed_context.enabled:
        import torch.distributed as dist

        dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_weight_sum, op=dist.ReduceOp.SUM)
    model.train()
    if float(total_weight_sum.item()) <= 0.0:
        return {"val_loss": 0.0}
    loss = total_loss_sum / torch.clamp(total_weight_sum, min=1.0)
    if not bool(torch.isfinite(loss).item()):
        return {"val_loss": 0.0}
    return {"val_loss": float(loss.detach().cpu().item())}


def _save_model_checkpoint(model, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model = model.module if hasattr(model, "module") else model
    original_use_cache = getattr(base_model.config, "use_cache", None)
    generation_config = getattr(base_model, "generation_config", None)
    original_generation_use_cache = (
        getattr(generation_config, "use_cache", None) if generation_config is not None else None
    )
    try:
        # Training disables KV cache for gradient checkpointing, but rollout/inference
        # checkpoints must restore cache usage or subsequent hierarchical generation
        # becomes dramatically slower.
        if original_use_cache is not None:
            base_model.config.use_cache = True
        if generation_config is not None and original_generation_use_cache is not None:
            generation_config.use_cache = True
        base_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    finally:
        if original_use_cache is not None:
            base_model.config.use_cache = original_use_cache
        if generation_config is not None and original_generation_use_cache is not None:
            generation_config.use_cache = original_generation_use_cache


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline hierarchical ReMA GRPO training from saved replay samples")
    parser.add_argument("--train-samples-jsonl", required=True)
    parser.add_argument("--val-samples-jsonl", default="")
    parser.add_argument("--config-json", required=True)
    return parser.parse_args()


def _load_config_from_json(config_path: str) -> OfflineTrainingConfig:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return OfflineTrainingConfig(**payload)


def run_offline_policy_training_from_jsonl(
    *,
    train_samples_jsonl: str,
    val_samples_jsonl: str = "",
    config_json: str,
):
    train_samples = load_samples_from_jsonl(train_samples_jsonl)
    val_samples = load_samples_from_jsonl(val_samples_jsonl) if val_samples_jsonl else []
    config = _load_config_from_json(config_json)
    return run_offline_policy_training(
        train_samples=train_samples,
        val_samples=val_samples,
        config=config,
    )


def main() -> None:
    args = _parse_cli_args()
    run_offline_policy_training_from_jsonl(
        train_samples_jsonl=args.train_samples_jsonl,
        val_samples_jsonl=args.val_samples_jsonl,
        config_json=args.config_json,
    )


if __name__ == "__main__":
    main()

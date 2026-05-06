from __future__ import annotations


def train_worker_role_lora_adapters(*_args, **_kwargs):
    raise NotImplementedError(
        "Worker-role LoRA training is intentionally left as a placeholder. "
        "The current hierarchical training path only optimizes the controller policies."
    )

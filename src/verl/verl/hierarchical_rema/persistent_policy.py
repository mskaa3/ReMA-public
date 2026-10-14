"""Fail-closed control plane for opt-in persistent rollout engines."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def checkpoint_stamp(model_path):
    path = Path(model_path).expanduser()
    if not path.is_dir():
        if path.is_absolute():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return "hub:" + str(model_path)
    files = sorted(p for p in path.iterdir() if p.is_file() and
                   (p.suffix in {".json", ".safetensors", ".bin", ".model"}))
    if not files:
        raise ValueError(f"Checkpoint has no model files: {path}")
    # Runtime checkpoint mirrors are atomically replaced. Stat identity avoids
    # hashing gigabytes on every generation chunk while detecting replacements.
    return hashlib.sha256(json.dumps([(p.name, p.stat().st_size, p.stat().st_mtime_ns,
                                      p.stat().st_ino) for p in files]).encode()).hexdigest()


def engine_signature(config):
    fields = asdict(config)
    for name in ("temperature", "top_p", "do_sample", "controller_temperature", "worker_temperature",
                 "controller_batch_size", "worker_batch_size"):
        fields.pop(name, None)
    return fields


def tokenizer_signature(tokenizer):
    return hashlib.sha256(json.dumps({"vocab": tokenizer.get_vocab(),
        "special_tokens": tokenizer.special_tokens_map, "chat_template": tokenizer.chat_template},
        sort_keys=True, default=str).encode()).hexdigest()


def architecture_signature(config):
    fields = config.to_dict()
    for key in ("_name_or_path", "transformers_version", "torch_dtype", "dtype", "_attn_implementation_autoset"):
        fields.pop(key, None)
    return fields


def choose_borrowed_slots(slots, rank_nodes):
    if any(not slot.get("sleeping") for slot in slots):
        raise RuntimeError("All rollout engines must acknowledge sleep before learning")
    identities = [(s["node_id"], s["cuda_visible_devices"]) for s in slots]
    if len(set(identities)) != len(slots) or any(not gpu or "," in gpu for _, gpu in identities):
        raise RuntimeError("Rollout GPU lease identities are not unique single devices")
    available = list(slots)
    selected = []
    for node in rank_nodes:
        index = next((i for i, slot in enumerate(available) if slot["node_id"] == node), None)
        if index is None:
            raise RuntimeError(f"No sleeping rollout GPU available on learner node {node}")
        selected.append(available.pop(index))
    return selected

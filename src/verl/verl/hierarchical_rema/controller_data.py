from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence

from .schema import TaskRollout


@dataclass
class ControllerReplaySample:
    role: str
    policy_id: str
    group_id: str
    prompt_text: str
    completion_text: str
    reward: float
    advantage: float
    metadata: Dict
    task_id: str
    source_path: str
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict) -> "ControllerReplaySample":
        return cls(
            role=str(payload["role"]),
            policy_id=str(payload["policy_id"]),
            group_id=str(payload["group_id"]),
            prompt_text=str(payload["prompt_text"]),
            completion_text=str(payload["completion_text"]),
            reward=float(payload["reward"]),
            advantage=float(payload["advantage"]),
            metadata=dict(payload.get("metadata", {})),
            task_id=str(payload["task_id"]),
            source_path=str(payload.get("source_path", "")),
            timestamp=payload.get("timestamp"),
        )


def discover_rollout_files(
    inputs: Sequence[str],
    default_filename: str = "all_rollouts.jsonl",
) -> List[Path]:
    files: List[Path] = []
    for raw_input in inputs:
        path = Path(raw_input).expanduser().resolve()
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            candidate = path / default_filename
            if candidate.exists():
                files.append(candidate)
                continue
            nested = sorted(path.rglob(default_filename))
            files.extend(nested)
            continue
        raise FileNotFoundError(f"Could not find rollout input: {raw_input}")
    if not files:
        raise FileNotFoundError("No rollout JSONL files were found")
    return sorted(set(files))


def load_controller_samples_from_rollouts(
    rollout_paths: Sequence[str],
    roles: Optional[Sequence[str]] = None,
    min_reward: Optional[float] = None,
    min_advantage: Optional[float] = None,
) -> List[ControllerReplaySample]:
    allowed_roles = set(roles) if roles else None
    samples: List[ControllerReplaySample] = []
    for rollout_path in discover_rollout_files(rollout_paths):
        with rollout_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rollout = record["rollout"]
                training_batch = rollout["training_batch"]
                task_id = rollout["task"]["task_id"]
                timestamp = record.get("timestamp")
                for batch_key in ("decomposer_samples", "selector_samples", "worker_samples"):
                    for sample in training_batch.get(batch_key, []):
                        role = sample["role"]
                        reward = float(sample["reward"])
                        advantage = float(sample["advantage"])
                        if allowed_roles is not None and role not in allowed_roles:
                            continue
                        if min_reward is not None and reward < min_reward:
                            continue
                        if min_advantage is not None and advantage < min_advantage:
                            continue
                        samples.append(
                            ControllerReplaySample(
                                role=role,
                                policy_id=sample["policy_id"],
                                group_id=sample["group_id"],
                                prompt_text=sample["prompt_text"],
                                completion_text=sample["completion_text"],
                                reward=reward,
                                advantage=advantage,
                                metadata=dict(sample.get("metadata", {})),
                                task_id=task_id,
                                source_path=str(rollout_path),
                                timestamp=timestamp,
                            )
                        )
    if not samples:
        raise ValueError("No controller replay samples matched the provided filters")
    return samples


def controller_samples_from_task_rollouts(
    task_rollouts: Sequence[TaskRollout],
    roles: Optional[Sequence[str]] = None,
    min_reward: Optional[float] = None,
    min_advantage: Optional[float] = None,
    source_path: str = "",
) -> List[ControllerReplaySample]:
    allowed_roles = set(roles) if roles else None
    samples: List[ControllerReplaySample] = []
    for rollout in task_rollouts:
        training_batch = rollout.training_batch
        for batch_key in ("decomposer_samples", "selector_samples", "worker_samples"):
            for sample in getattr(training_batch, batch_key):
                role = sample.role
                reward = float(sample.reward)
                advantage = float(sample.advantage)
                if allowed_roles is not None and role not in allowed_roles:
                    continue
                if min_reward is not None and reward < min_reward:
                    continue
                if min_advantage is not None and advantage < min_advantage:
                    continue
                samples.append(
                    ControllerReplaySample(
                        role=role,
                        policy_id=sample.policy_id,
                        group_id=sample.group_id,
                        prompt_text=sample.prompt_text,
                        completion_text=sample.completion_text,
                        reward=reward,
                        advantage=advantage,
                        metadata=dict(sample.metadata),
                        task_id=rollout.task.task_id,
                        source_path=source_path,
                        timestamp=None,
                    )
                )
    if not samples:
        raise ValueError("No controller replay samples matched the provided filters")
    return samples


def group_samples_by_policy(
    samples: Sequence[ControllerReplaySample],
) -> Dict[str, List[ControllerReplaySample]]:
    grouped: Dict[str, List[ControllerReplaySample]] = {}
    for sample in samples:
        grouped.setdefault(sample.policy_id, []).append(sample)
    return grouped


def train_val_split(
    samples: Sequence[ControllerReplaySample],
    val_ratio: float,
    seed: int,
) -> tuple[List[ControllerReplaySample], List[ControllerReplaySample]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    sample_list = list(samples)
    rng = random.Random(seed)
    rng.shuffle(sample_list)
    if not sample_list:
        return [], []
    val_size = int(len(sample_list) * val_ratio)
    if val_ratio > 0 and val_size == 0 and len(sample_list) > 1:
        val_size = 1
    if val_size >= len(sample_list):
        val_size = max(len(sample_list) - 1, 0)
    val_samples = sample_list[:val_size]
    train_samples = sample_list[val_size:]
    return train_samples, val_samples


def write_samples_to_jsonl(
    samples: Sequence[ControllerReplaySample],
    output_path: str,
) -> str:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), sort_keys=True) + "\n")
    return str(output)


def load_samples_from_jsonl(input_path: str) -> List[ControllerReplaySample]:
    path = Path(input_path).expanduser().resolve()
    samples: List[ControllerReplaySample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(ControllerReplaySample.from_dict(json.loads(line)))
    return samples


def summarize_samples_by_policy(
    samples: Sequence[ControllerReplaySample],
) -> Dict[str, Dict]:
    grouped = group_samples_by_policy(samples)
    summaries: Dict[str, Dict] = {}
    for policy_id, policy_samples in grouped.items():
        rewards = [sample.reward for sample in policy_samples]
        advantages = [sample.advantage for sample in policy_samples]
        summaries[policy_id] = {
            "policy_id": policy_id,
            "num_samples": len(policy_samples),
            "roles": sorted({sample.role for sample in policy_samples}),
            "num_tasks": len({sample.task_id for sample in policy_samples}),
            "num_groups": len({sample.group_id for sample in policy_samples}),
            "reward_mean": mean(rewards),
            "reward_min": min(rewards),
            "reward_max": max(rewards),
            "advantage_mean": mean(advantages),
            "advantage_min": min(advantages),
            "advantage_max": max(advantages),
            "model_paths": sorted(
                {
                    model_path
                    for sample in policy_samples
                    for model_path in [sample.metadata.get("model_path")]
                    if model_path
                }
            ),
            "source_paths": sorted({sample.source_path for sample in policy_samples}),
        }
    return summaries

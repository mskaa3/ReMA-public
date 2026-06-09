from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .controller_data import (
    ControllerReplaySample,
    group_samples_by_policy,
    load_controller_samples_from_rollouts,
    summarize_samples_by_policy,
    train_val_split,
    write_samples_to_jsonl,
)
from .offline_training import OfflineTrainingConfig, run_offline_policy_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hierarchical controller policies from rollout JSONL logs with offline GRPO")
    parser.add_argument("--input", nargs="+", required=True, help="Rollout JSONL files or directories containing all_rollouts.jsonl")
    parser.add_argument("--policy-id", action="append", default=None, help="Only train these policy ids")
    parser.add_argument("--role", choices=["decomposer", "selector", "worker", "both", "all"], default="both")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-reward",
        type=float,
        default=None,
        help=(
            "Optional minimum replay reward required for a replay sample to enter training. "
            "Disabled by default."
        ),
    )
    parser.add_argument("--min-advantage", type=float, default=None)
    parser.add_argument("--save-replay-copy", action="store_true", help="Save train/val/all sample JSONL files per policy for inspection")

    parser.add_argument("--model-path", default=None)
    parser.add_argument("--decomposer-model-path", default=None)
    parser.add_argument("--selector-model-path", default=None)
    parser.add_argument("--worker-model-path", default=None)

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--truncation", choices=["left", "right", "error"], default="left")
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--clip-ratio-c", type=float, default=3.0)
    parser.add_argument("--entropy-coeff", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-every-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--project-name", default="hierarchical-rema")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--enable-wandb", action="store_true")
    return parser.parse_args()


def _selected_roles(role: str) -> List[str] | None:
    if role == "all":
        return None
    if role == "both":
        return ["decomposer", "selector"]
    return [role]


def filter_policy_ids(
    grouped_samples: Dict[str, List[ControllerReplaySample]],
    policy_ids: List[str] | None,
) -> Dict[str, List[ControllerReplaySample]]:
    if not policy_ids:
        return grouped_samples
    selected = set(policy_ids)
    filtered = {policy_id: samples for policy_id, samples in grouped_samples.items() if policy_id in selected}
    if not filtered:
        raise ValueError(f"No policy ids matched the requested filter: {sorted(selected)}")
    return filtered


def model_path_for_policy(policy_id: str, samples: List[ControllerReplaySample], args: argparse.Namespace) -> str:
    sample_role = samples[0].role
    if args.model_path:
        return args.model_path
    if sample_role == "decomposer" and args.decomposer_model_path:
        return args.decomposer_model_path
    if sample_role == "selector" and args.selector_model_path:
        return args.selector_model_path
    if sample_role == "worker" and args.worker_model_path:
        return args.worker_model_path

    model_paths = {sample.metadata.get("model_path") for sample in samples if sample.metadata.get("model_path")}
    model_paths.discard(None)
    if len(model_paths) == 1:
        return next(iter(model_paths))
    if len(model_paths) > 1 and policy_id == "shared_controller":
        return sorted(model_paths)[0]
    raise ValueError(
        f"Could not infer a single model path for policy '{policy_id}'. "
        "Pass --model-path or role-specific model path overrides."
    )


def prepare_policy_splits(
    samples: List[ControllerReplaySample],
    policy_ids: List[str] | None,
    val_ratio: float,
    seed: int,
) -> Dict[str, Dict[str, List[ControllerReplaySample]]]:
    grouped = group_samples_by_policy(samples)
    grouped = filter_policy_ids(grouped, policy_ids)
    policy_splits = {}
    for policy_id, policy_samples in grouped.items():
        train_samples, val_samples = train_val_split(
            policy_samples,
            val_ratio=val_ratio,
            seed=seed,
        )
        if not train_samples:
            raise ValueError(f"Policy '{policy_id}' has no train samples after the split")
        policy_splits[policy_id] = {
            "all": policy_samples,
            "train": train_samples,
            "val": val_samples,
        }
    return policy_splits


def write_policy_manifest(
    output_dir: Path,
    policy_splits: Dict[str, Dict[str, List[ControllerReplaySample]]],
    model_path_resolver,
) -> None:
    manifest = {
        "objective": "grpo",
        "policies": {},
    }
    for policy_id, split in policy_splits.items():
        all_samples = split["all"]
        train_samples = split["train"]
        val_samples = split["val"]
        manifest["policies"][policy_id] = {
            "model_path": model_path_resolver(policy_id, train_samples or all_samples),
            "all_stats": summarize_samples_by_policy(all_samples)[policy_id],
            "train_stats": summarize_samples_by_policy(train_samples)[policy_id],
            "val_stats": summarize_samples_by_policy(val_samples)[policy_id] if val_samples else None,
        }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def maybe_save_replay_copy(
    policy_dir: Path,
    split: Dict[str, List[ControllerReplaySample]],
    enabled: bool,
) -> Dict[str, str]:
    if not enabled:
        return {}

    train_jsonl = write_samples_to_jsonl(split["train"], policy_dir / "train_samples.jsonl")
    val_jsonl = write_samples_to_jsonl(split["val"], policy_dir / "val_samples.jsonl") if split["val"] else train_jsonl
    all_jsonl = write_samples_to_jsonl(split["all"], policy_dir / "all_samples.jsonl")
    return {
        "train_jsonl": train_jsonl,
        "val_jsonl": val_jsonl,
        "all_jsonl": all_jsonl,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_controller_samples_from_rollouts(
        rollout_paths=args.input,
        roles=_selected_roles(args.role),
        min_reward=args.min_reward,
        min_advantage=args.min_advantage,
    )
    policy_splits = prepare_policy_splits(
        samples=samples,
        policy_ids=args.policy_id,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    write_policy_manifest(
        output_dir=output_dir,
        policy_splits=policy_splits,
        model_path_resolver=lambda policy_id, sample_subset: model_path_for_policy(policy_id, sample_subset, args),
    )

    summaries = {}
    for policy_id, split in policy_splits.items():
        policy_dir = output_dir / policy_id
        policy_dir.mkdir(parents=True, exist_ok=True)
        replay_exports = maybe_save_replay_copy(policy_dir, split, enabled=args.save_replay_copy)

        model_path = model_path_for_policy(policy_id, split["train"] or split["all"], args)
        experiment_name = args.experiment_name or output_dir.name
        if len(policy_splits) > 1:
            experiment_name = f"{experiment_name}-{policy_id}"
        print(
            f"[hierarchical-rema][replay-train] policy={policy_id} "
            f"train_samples={len(split['train'])} val_samples={len(split['val'])} "
            f"model={model_path}"
        )
        training_config = OfflineTrainingConfig(
            model_name_or_path=model_path,
            output_dir=str(policy_dir),
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            train_batch_size=args.train_batch_size,
            grad_accum_steps=args.grad_accum_steps,
            epochs=args.epochs,
            max_length=args.max_length,
            truncation=args.truncation,
            clip_range=args.clip_range,
            clip_ratio_c=args.clip_ratio_c,
            entropy_coeff=args.entropy_coeff,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_every_steps=args.eval_every_steps,
            device=args.device,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
            gradient_checkpointing=args.gradient_checkpointing,
            project_name=args.project_name,
            experiment_name=experiment_name,
            enable_wandb=args.enable_wandb,
            save_best_checkpoint=args.eval_every_steps > 0,
        )
        summaries[policy_id] = run_offline_policy_training(
            train_samples=split["train"],
            val_samples=split["val"],
            config=training_config,
        )
        summaries[policy_id]["policy_id"] = policy_id
        summaries[policy_id]["model_path"] = model_path
        summaries[policy_id]["objective"] = "grpo"
        summaries[policy_id].update(replay_exports)

    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)
    print(f"[hierarchical-rema][replay-train] wrote summary to {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()

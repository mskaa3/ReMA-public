from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .controller_data import controller_samples_from_task_rollouts
from .demo import make_demo_tasks, make_worker_pool
from .offline_training import OfflineTrainingConfig, run_offline_policy_training
from .orchestrator import HierarchicalGRPOTrainer
from .replay_train import (
    maybe_save_replay_copy,
    model_path_for_policy,
    prepare_policy_splits,
    write_policy_manifest,
)
from .schema import (
    AlternatingPhase,
    ControllerPolicyConfig,
    HFBackendConfig,
    RolloutLoggingConfig,
    RolloutConfig,
    TaskExample,
    TaskRollout,
    TrainingMode,
    TrainingScheduleConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated hierarchical ReMA training: tasks -> rollouts -> GRPO update")
    parser.add_argument("--task-source", default="demo", help="Task dataset path or the special value 'demo'")
    parser.add_argument("--task-format", choices=["auto", "demo", "jsonl", "json", "parquet"], default="auto")
    parser.add_argument("--prompt-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--task-id-key", default="idx")
    parser.add_argument("--max-tasks", type=int, default=0, help="Limit the total number of loaded tasks; 0 means all")
    parser.add_argument("--tasks-per-epoch", type=int, default=0, help="How many tasks to rollout per epoch; 0 means all loaded tasks")
    parser.add_argument("--shuffle-tasks", action="store_true")

    parser.add_argument("--backend", choices=["mock", "hf"], default="mock")
    parser.add_argument("--mode", choices=["joint", "alternating"], default="joint")
    parser.add_argument("--phase", choices=["selector", "decomposer"], default="selector", help="Starting phase when mode=alternating")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--parameter-sharing", action="store_true")
    parser.add_argument("--shared-model-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--decomposer-model-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--selector-model-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--worker-base-model-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-decompositions", type=int, default=3)
    parser.add_argument("--num-selections", type=int, default=2)
    parser.add_argument("--soft-max-hops", type=int, default=None)
    parser.add_argument("--hard-max-hops", type=int, default=None)
    parser.add_argument("--soft-hop-penalty", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--controller-max-new-tokens", type=int, default=768)
    parser.add_argument("--worker-max-new-tokens", type=int, default=256)
    parser.add_argument("--disable-rollout-logging", action="store_true")
    parser.add_argument("--best-k", type=int, default=10)

    parser.add_argument("--policy-id", action="append", default=None, help="Only train these policy ids after rollout generation")
    parser.add_argument("--role", choices=["decomposer", "selector", "both"], default="both", help="Only train these controller roles after rollout generation")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-reward", type=float, default=None)
    parser.add_argument("--min-advantage", type=float, default=None)
    parser.add_argument("--save-replay-copy", action="store_true", help="Save train/val/all sample JSONL files per policy for inspection")

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1, help="Number of GRPO update epochs per outer training epoch")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--truncation", choices=["left", "right", "error"], default="left")
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--clip-ratio-c", type=float, default=3.0)
    parser.add_argument("--entropy-coeff", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--project-name", default="hierarchical-rema")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--enable-wandb", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _selected_roles(role: str) -> List[str] | None:
    if role == "both":
        return None
    return [role]


def _release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _tracking(args: argparse.Namespace, config_payload: Dict[str, Any]):
    if not args.enable_wandb:
        return None
    try:
        from verl.utils.tracking import Tracking
    except Exception:
        return None

    return Tracking(
        project_name=args.project_name,
        experiment_name=args.experiment_name or Path(args.output_dir).name,
        default_backend=["wandb"],
        config=config_payload,
    )


def _finish_tracking(tracking) -> None:
    if tracking is None:
        return
    try:
        tracking.__del__()
    except Exception:
        pass


def _infer_task_format(task_source: str, task_format: str) -> str:
    if task_format != "auto":
        return task_format
    if task_source == "demo":
        return "demo"
    suffix = Path(task_source).suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".parquet":
        return "parquet"
    raise ValueError(f"Could not infer task format from source: {task_source}")


def _resolve_task_source_path(task_source: str) -> str:
    if task_source == "demo":
        return task_source

    path = Path(task_source).expanduser().resolve()
    if path.is_file():
        return str(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Task source does not exist: {task_source}")

    preferred_names = (
        "all_test_data.jsonl",
        "test.jsonl",
        "test.json",
        "test.parquet",
        "data.jsonl",
        "data.json",
        "data.parquet",
    )
    for name in preferred_names:
        candidate = path / name
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())

    candidates = sorted(
        item.resolve()
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in {".jsonl", ".json", ".parquet"}
    )
    if len(candidates) == 1:
        return str(candidates[0])
    if candidates:
        return str(candidates[0])

    raise ValueError(
        f"Could not find a supported task file inside directory: {task_source}. "
        "Expected a .jsonl, .json, or .parquet file."
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


def _nested_get(row: Dict[str, Any], key_path: str) -> Any:
    current: Any = row
    for part in key_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(key_path)
    return current


def _get_first_available(row: Dict[str, Any], candidates: Sequence[str]) -> Any:
    for candidate in candidates:
        try:
            value = _nested_get(row, candidate)
        except KeyError:
            continue
        if value is not None:
            return value
    raise KeyError(", ".join(candidates))


def _load_task_rows(task_source: str, task_format: str) -> List[Dict[str, Any]]:
    path = Path(task_source).expanduser().resolve()
    if task_format == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if task_format == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "items", "records"):
                if key in payload and isinstance(payload[key], list):
                    return payload[key]
        raise ValueError(f"Unsupported JSON task payload in {task_source}")
    if task_format == "parquet":
        import pandas as pd

        dataframe = pd.read_parquet(path)
        return dataframe.to_dict(orient="records")
    raise ValueError(f"Unsupported task format: {task_format}")


def load_tasks(
    task_source: str,
    task_format: str,
    prompt_key: str,
    answer_key: str,
    task_id_key: str,
    max_tasks: int,
) -> List[TaskExample]:
    resolved_source = _resolve_task_source_path(task_source)
    resolved_format = _infer_task_format(resolved_source, task_format)
    if resolved_format == "demo":
        tasks = make_demo_tasks()
        return tasks[:max_tasks] if max_tasks and max_tasks > 0 else tasks

    rows = _load_task_rows(resolved_source, resolved_format)
    tasks: List[TaskExample] = []
    for row_index, row in enumerate(rows):
        prompt = _get_first_available(row, [prompt_key, "question", "instruction", "prompt"])
        answer = _get_first_available(row, [answer_key, "answer", "reward_model.ground_truth", "ground_truth"])
        try:
            task_id = _get_first_available(row, [task_id_key, "unique_id", "idx", "id"])
        except KeyError:
            task_id = f"task-{row_index}"

        metadata = {
            key: _json_safe(value)
            for key, value in row.items()
            if key not in {prompt_key, answer_key, task_id_key, "question", "instruction", "prompt", "answer"}
        }
        tasks.append(
            TaskExample(
                task_id=str(task_id),
                prompt=str(prompt),
                ground_truth=str(answer),
                metadata=metadata,
            )
        )

    if max_tasks and max_tasks > 0:
        tasks = tasks[:max_tasks]
    if not tasks:
        raise ValueError("No tasks were loaded from the task source")
    return tasks


def select_epoch_tasks(
    tasks: Sequence[TaskExample],
    epoch_index: int,
    tasks_per_epoch: int,
    shuffle_tasks: bool,
    seed: int,
) -> List[TaskExample]:
    task_list = list(tasks)
    if tasks_per_epoch <= 0 or tasks_per_epoch >= len(task_list):
        if shuffle_tasks:
            rng = random.Random(seed + epoch_index)
            rng.shuffle(task_list)
        return task_list

    if shuffle_tasks:
        rng = random.Random(seed + epoch_index)
        return rng.sample(task_list, tasks_per_epoch)

    start = (epoch_index * tasks_per_epoch) % len(task_list)
    selected = [task_list[(start + offset) % len(task_list)] for offset in range(tasks_per_epoch)]
    return selected


def build_schedule(mode: str, phase: AlternatingPhase) -> TrainingScheduleConfig:
    return TrainingScheduleConfig(
        mode=TrainingMode(mode),
        alternating_phase=phase,
    )


def next_phase(phase: AlternatingPhase) -> AlternatingPhase:
    return AlternatingPhase.DECOMPOSER if phase == AlternatingPhase.SELECTOR else AlternatingPhase.SELECTOR


def epoch_rollout_summary(rollouts: Sequence[TaskRollout]) -> Dict[str, Any]:
    task_summaries = []
    best_decomposition_rewards = []
    best_selection_rewards = []
    mean_selection_rewards = []
    best_correctness = []
    for rollout in rollouts:
        best_decomposition = max(rollout.decompositions, key=lambda item: item.decomposition_reward)
        selection_rewards = [selection.reward.total_reward for decomposition in rollout.decompositions for selection in decomposition.selections]
        selection_correctness = [selection.reward.final_answer_correctness for decomposition in rollout.decompositions for selection in decomposition.selections]
        best_decomposition_rewards.append(best_decomposition.decomposition_reward)
        best_selection_rewards.append(max(selection_rewards))
        mean_selection_rewards.append(sum(selection_rewards) / max(len(selection_rewards), 1))
        best_correctness.append(max(selection_correctness) if selection_correctness else 0.0)
        task_summaries.append(
            {
                "task_id": rollout.task.task_id,
                "best_decomposition_id": best_decomposition.decomposition.decomposition_id,
                "best_decomposition_reward": best_decomposition.decomposition_reward,
                "best_selection_reward": max(selection_rewards),
                "mean_selection_reward": mean_selection_rewards[-1],
                "best_final_correctness": best_correctness[-1],
            }
        )

    return {
        "num_tasks": len(rollouts),
        "mean_best_decomposition_reward": sum(best_decomposition_rewards) / max(len(best_decomposition_rewards), 1),
        "mean_best_selection_reward": sum(best_selection_rewards) / max(len(best_selection_rewards), 1),
        "mean_selection_reward": sum(mean_selection_rewards) / max(len(mean_selection_rewards), 1),
        "mean_best_final_correctness": sum(best_correctness) / max(len(best_correctness), 1),
        "tasks": task_summaries,
    }


def _current_policy_config(args: argparse.Namespace, current_paths: Dict[str, str]) -> ControllerPolicyConfig:
    if args.parameter_sharing:
        shared = current_paths["shared_controller"]
        return ControllerPolicyConfig(
            parameter_sharing=True,
            shared_model_path=shared,
            decomposer_model_path=shared,
            selector_model_path=shared,
        )
    return ControllerPolicyConfig(
        parameter_sharing=False,
        shared_model_path=None,
        decomposer_model_path=current_paths["decomposer_controller"],
        selector_model_path=current_paths["selector_controller"],
    )


def _update_current_paths(current_paths: Dict[str, str], policy_id: str, final_path: str) -> None:
    current_paths[policy_id] = final_path
    if policy_id == "shared_controller":
        current_paths["decomposer_controller"] = final_path
        current_paths["selector_controller"] = final_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_task_source = _resolve_task_source_path(args.task_source)

    tasks = load_tasks(
        task_source=resolved_task_source,
        task_format=args.task_format,
        prompt_key=args.prompt_key,
        answer_key=args.answer_key,
        task_id_key=args.task_id_key,
        max_tasks=args.max_tasks,
    )
    worker_pool = make_worker_pool(base_model_path=args.worker_base_model_path)
    rollout_config = RolloutConfig(
        num_decompositions=args.num_decompositions,
        num_selections_per_decomposition=args.num_selections,
        soft_max_hops=args.soft_max_hops,
        hard_max_hops=args.hard_max_hops,
        soft_hop_penalty=args.soft_hop_penalty,
    )

    current_paths = {
        "shared_controller": args.model_path or args.shared_model_path,
        "decomposer_controller": args.model_path or args.decomposer_model_path,
        "selector_controller": args.model_path or args.selector_model_path,
    }
    current_phase = AlternatingPhase(args.phase)
    tracking = _tracking(args, config_payload=vars(args))
    tracking_step_offset = 0
    job_summary: Dict[str, Any] = {
        "task_source": args.task_source,
        "resolved_task_source": resolved_task_source,
        "num_loaded_tasks": len(tasks),
        "num_epochs": args.num_epochs,
        "epochs": [],
    }

    print(
        f"[hierarchical-rema][integrated] loaded_tasks={len(tasks)} "
        f"task_source={resolved_task_source} mode={args.mode} backend={args.backend}"
    )

    for epoch_index in range(args.num_epochs):
        epoch_number = epoch_index + 1
        epoch_dir = output_dir / f"epoch_{epoch_number:04d}"
        rollout_dir = epoch_dir / "rollouts"
        train_dir = epoch_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        epoch_tasks = select_epoch_tasks(
            tasks=tasks,
            epoch_index=epoch_index,
            tasks_per_epoch=args.tasks_per_epoch,
            shuffle_tasks=args.shuffle_tasks,
            seed=args.seed,
        )
        policy_config = _current_policy_config(args, current_paths)
        schedule = build_schedule(args.mode, current_phase)
        logging_config = None
        if not args.disable_rollout_logging:
            logging_config = RolloutLoggingConfig(
                output_dir=str(rollout_dir),
                best_k=args.best_k,
            )

        print(
            f"[hierarchical-rema][integrated] epoch={epoch_number}/{args.num_epochs} "
            f"phase={schedule.alternating_phase.value} tasks={len(epoch_tasks)}"
        )
        rollout_trainer = HierarchicalGRPOTrainer(
            backend_type=args.backend,
            hf_backend_config=HFBackendConfig(
                temperature=args.temperature,
                top_p=args.top_p,
                controller_max_new_tokens=args.controller_max_new_tokens,
                worker_max_new_tokens=args.worker_max_new_tokens,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=args.torch_dtype,
            ),
            rollout_logging_config=logging_config,
        )
        rollouts = [
            rollout_trainer.run(
                task=task,
                worker_pool=worker_pool,
                policy_config=policy_config,
                rollout_config=rollout_config,
                schedule=schedule,
            )
            for task in epoch_tasks
        ]
        rollout_summary = epoch_rollout_summary(rollouts)
        print(
            f"[hierarchical-rema][integrated] epoch={epoch_number} "
            f"mean_best_selection_reward={rollout_summary['mean_best_selection_reward']:.4f} "
            f"mean_best_decomposition_reward={rollout_summary['mean_best_decomposition_reward']:.4f} "
            f"mean_best_final_correctness={rollout_summary['mean_best_final_correctness']:.4f}"
        )
        if tracking is not None:
            tracking.log(
                {
                    "rollout/mean_best_selection_reward": rollout_summary["mean_best_selection_reward"],
                    "rollout/mean_best_decomposition_reward": rollout_summary["mean_best_decomposition_reward"],
                    "rollout/mean_best_final_correctness": rollout_summary["mean_best_final_correctness"],
                    "rollout/num_tasks": rollout_summary["num_tasks"],
                },
                step=tracking_step_offset,
            )

        with (epoch_dir / "rollout_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(rollout_summary, handle, indent=2, sort_keys=True)

        del rollout_trainer
        _release_memory()

        try:
            samples = controller_samples_from_task_rollouts(
                task_rollouts=rollouts,
                roles=_selected_roles(args.role),
                min_reward=args.min_reward,
                min_advantage=args.min_advantage,
                source_path=str(epoch_dir / "rollouts"),
            )
            policy_splits = prepare_policy_splits(
                samples=samples,
                policy_ids=args.policy_id,
                val_ratio=args.val_ratio,
                seed=args.seed + epoch_index,
            )
        except ValueError as exc:
            epoch_summary = {
                "epoch": epoch_number,
                "schedule": {
                    "mode": schedule.mode.value,
                    "phase": schedule.alternating_phase.value,
                },
                "rollout_summary": rollout_summary,
                "training_skipped": str(exc),
            }
            with (epoch_dir / "epoch_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(epoch_summary, handle, indent=2, sort_keys=True)
            job_summary["epochs"].append(epoch_summary)
            if args.mode == "alternating":
                current_phase = next_phase(current_phase)
            continue

        replay_like_args = argparse.Namespace(
            model_path=args.model_path,
            decomposer_model_path=policy_config.decomposer_model_path,
            selector_model_path=policy_config.selector_model_path,
        )
        write_policy_manifest(
            output_dir=train_dir,
            policy_splits=policy_splits,
            model_path_resolver=lambda policy_id, sample_subset: model_path_for_policy(policy_id, sample_subset, replay_like_args),
        )

        training_summaries: Dict[str, Any] = {}
        for policy_id, split in policy_splits.items():
            policy_dir = train_dir / policy_id
            policy_dir.mkdir(parents=True, exist_ok=True)
            replay_exports = maybe_save_replay_copy(policy_dir, split, enabled=args.save_replay_copy)
            model_path = model_path_for_policy(policy_id, split["train"] or split["all"], replay_like_args)
            experiment_name = args.experiment_name or output_dir.name
            experiment_name = f"{experiment_name}-epoch{epoch_number:04d}-{policy_id}"
            print(
                f"[hierarchical-rema][integrated] training policy={policy_id} "
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
                seed=args.seed + epoch_index,
                logging_steps=args.logging_steps,
                save_steps=args.save_steps,
                eval_every_steps=args.eval_every_steps,
                device=args.device,
                torch_dtype=args.torch_dtype,
                trust_remote_code=args.trust_remote_code,
                gradient_checkpointing=args.gradient_checkpointing,
                project_name=args.project_name,
                experiment_name=experiment_name,
                enable_wandb=False,
            )
            summary = run_offline_policy_training(
                train_samples=split["train"],
                val_samples=split["val"],
                config=training_config,
                tracking=tracking,
                tracking_prefix=f"{policy_id}/",
                log_step_offset=tracking_step_offset,
            )
            tracking_step_offset += max(int(summary["steps"]), 1)
            final_model_path = str(policy_dir / "final")
            _update_current_paths(current_paths, policy_id, final_model_path)
            summary["policy_id"] = policy_id
            summary["model_path"] = model_path
            summary["final_model_path"] = final_model_path
            summary.update(replay_exports)
            training_summaries[policy_id] = summary
            print(
                f"[hierarchical-rema][integrated] finished policy={policy_id} "
                f"steps={summary['steps']} final_model_path={final_model_path}"
            )
            _release_memory()

        epoch_summary = {
            "epoch": epoch_number,
            "schedule": {
                "mode": schedule.mode.value,
                "phase": schedule.alternating_phase.value,
            },
            "rollout_summary": rollout_summary,
            "training_summaries": training_summaries,
            "current_policy_paths": dict(current_paths),
        }
        with (epoch_dir / "epoch_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(epoch_summary, handle, indent=2, sort_keys=True)
        job_summary["epochs"].append(epoch_summary)

        if args.mode == "alternating":
            current_phase = next_phase(current_phase)

    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(job_summary, handle, indent=2, sort_keys=True)
    print(f"[hierarchical-rema][integrated] wrote job summary to {output_dir / 'training_summary.json'}")
    _finish_tracking(tracking)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .backends import MockHierarchicalBackend, RayVLLMHierarchicalBackend, TransformersHierarchicalBackend
from .controller_data import controller_samples_from_task_rollouts, write_samples_to_jsonl
from .demo import make_demo_tasks, make_worker_pool
from .offline_training import (
    OfflineTrainingConfig,
    RayOfflineGRPOWorker,
    run_offline_policy_training,
)
from .orchestrator import HierarchicalGRPOTrainer
from .rewarding import WorkerPerformanceMemory
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
    RewardWeights,
    RolloutLoggingConfig,
    RolloutConfig,
    TaskExample,
    TaskRollout,
    TrainingMode,
    TrainingScheduleConfig,
    VLLMBackendConfig,
    WorkerRewardMode,
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
    parser.add_argument("--val-task-source", default="", help="Validation task dataset path; empty disables external benchmark validation")
    parser.add_argument("--val-task-format", choices=["auto", "demo", "jsonl", "json", "parquet"], default="auto")
    parser.add_argument("--val-prompt-key", default="question")
    parser.add_argument("--val-answer-key", default="answer")
    parser.add_argument("--val-task-id-key", default="idx")
    parser.add_argument("--max-val-tasks", type=int, default=0, help="Limit the total number of loaded validation tasks; 0 means all")
    parser.add_argument(
        "--val-tasks-per-epoch",
        type=int,
        default=128,
        help="How many validation tasks to run per epoch; 0 means all loaded validation tasks",
    )
    parser.add_argument(
        "--val-tasks-per-subset",
        type=int,
        default=0,
        help=(
            "How many validation tasks to run per subset/dataset when external validation is triggered; "
            "0 disables subset-stratified capping."
        ),
    )
    parser.add_argument(
        "--external-validation-every-n-epochs",
        type=int,
        default=10,
        help=(
            "How often to run the full external benchmark validation. "
            "1 = every outer epoch, 5 = every fifth outer epoch, "
            "0 = final outer epoch only, negative values disable it entirely."
        ),
    )
    parser.add_argument("--disable-external-validation", action="store_true")

    parser.add_argument("--backend", choices=["mock", "hf", "vllm"], default="mock")
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
    parser.add_argument("--max-nodes-per-decomposition", type=int, default=None)
    parser.add_argument("--soft-max-hops", type=int, default=None)
    parser.add_argument("--hard-max-hops", type=int, default=None)
    parser.add_argument("--soft-hop-penalty", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--controller-temperature", type=float, default=None)
    parser.add_argument("--worker-temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--final-answer-correctness-reward-only",
        action="store_true",
        help=(
            "Make selection/worker reward depend only on final-answer correctness. "
            "Controller-specific penalties still apply on top."
        ),
    )
    parser.add_argument(
        "--final-answer-reward-weight",
        type=float,
        default=1.5,
        help="Weight assigned to exact final-answer correctness in selection reward",
    )
    parser.add_argument(
        "--confidence-reward-weight",
        type=float,
        default=0.05,
        help="Weight assigned to low-entropy worker responses in selection reward",
    )
    parser.add_argument(
        "--compatibility-reward-weight",
        type=float,
        default=0.1,
        help="Weight assigned to worker-node compatibility in selection reward",
    )
    # parser.add_argument(
    #     "--worker-success-weight",
    #     type=float,
    #     default=0.25,
    #     help="Relative weight of local worker execution success in worker performance memory updates",
    # )
    parser.add_argument(
        "--worker-final-correctness-weight",
        type=float,
        default=0.75,
        help="Relative weight of final-answer correctness in worker performance memory updates",
    )
    parser.add_argument(
        "--controller-constrained-decoding",
        dest="controller_constrained_decoding",
        action="store_true",
        help="Enable constrained decoding hints for decomposer/selector controllers",
    )
    parser.add_argument(
        "--disable-controller-constrained-decoding",
        dest="controller_constrained_decoding",
        action="store_false",
        help="Disable constrained decoding hints for controllers",
    )
    parser.set_defaults(controller_constrained_decoding=True)
    parser.add_argument("--controller-max-new-tokens", type=int, default=768)
    parser.add_argument("--decomposer-max-new-tokens", type=int, default=0, help="0 reuses --controller-max-new-tokens")
    parser.add_argument("--selector-max-new-tokens", type=int, default=0, help="0 reuses --controller-max-new-tokens")
    parser.add_argument("--worker-max-new-tokens", type=int, default=256)
    parser.add_argument("--val-num-decompositions", type=int, default=1)
    parser.add_argument("--val-num-selections", type=int, default=1)
    parser.add_argument("--val-temperature", type=float, default=0.0)
    parser.add_argument("--val-controller-temperature", type=float, default=None)
    parser.add_argument("--val-worker-temperature", type=float, default=None)
    parser.add_argument("--val-top-p", type=float, default=1.0)
    parser.add_argument("--val-controller-max-new-tokens", type=int, default=0, help="0 reuses --controller-max-new-tokens")
    parser.add_argument("--val-decomposer-max-new-tokens", type=int, default=0, help="0 reuses the training decomposer/controller setting")
    parser.add_argument("--val-selector-max-new-tokens", type=int, default=0, help="0 reuses the training selector/controller setting")
    parser.add_argument("--val-worker-max-new-tokens", type=int, default=0, help="0 reuses --worker-max-new-tokens")
    parser.add_argument("--controller-batch-size", type=int, default=8)
    parser.add_argument("--worker-batch-size", type=int, default=16)
    parser.add_argument("--val-controller-batch-size", type=int, default=0, help="0 reuses --controller-batch-size during external validation")
    parser.add_argument("--val-worker-batch-size", type=int, default=0, help="0 reuses --worker-batch-size during external validation")
    parser.add_argument("--rollout-prompt-length", type=int, default=2048)
    parser.add_argument("--ray-nnodes", type=int, default=1)
    parser.add_argument("--ray-n-gpus-per-node", type=int, default=1)
    parser.add_argument("--ray-cpus-per-node", type=int, default=0, help="0 reuses the backend default; otherwise shapes Ray rollout bundle CPU reservations")
    parser.add_argument("--offline-grpo-distributed", action="store_true")
    parser.add_argument("--offline-grpo-nnodes", type=int, default=1)
    parser.add_argument("--offline-grpo-gpus-per-node", type=int, default=1)
    parser.add_argument("--offline-grpo-master-port", type=int, default=29501)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=2048)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--disable-rollout-logging", action="store_true")
    parser.add_argument("--rollout-log-mode", choices=["best", "all"], default="best")
    parser.add_argument("--rollout-log-detail", choices=["compact", "full"], default="compact")
    parser.add_argument("--best-k", type=int, default=10)
    parser.add_argument(
        "--rollout-task-batch-size",
        type=int,
        default=32,
        help="How many tasks to rollout together in one batched hierarchical pass; 0 means all epoch tasks",
    )
    parser.add_argument(
        "--update-every-n-rollout-batches",
        type=int,
        default=4,
        help=(
            "How many rollout task batches to collect before running an offline GRPO update. "
            "1 = update after every rollout batch, 4 = update every four rollout batches."
        ),
    )
    parser.add_argument(
        "--rollout-progress-every",
        type=int,
        default=10,
        help="Print rollout progress every N tasks during integrated training; 0 disables progress logs",
    )
    parser.add_argument(
        "--val-rollout-task-batch-size",
        type=int,
        default=64,
        help="How many validation tasks to rollout together per batch; 0 reuses --rollout-task-batch-size",
    )

    parser.add_argument("--policy-id", action="append", default=None, help="Only train these policy ids after rollout generation")
    parser.add_argument("--role", choices=["decomposer", "selector", "both"], default="both", help="Only train these controller roles after rollout generation")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-reward",
        type=float,
        default=0.21,
        help=(
            "Minimum replay reward required for a controller sample to enter training. "
            "The default filters out most incorrect-but-confident selector samples."
        ),
    )
    parser.add_argument("--min-advantage", type=float, default=None)
    parser.add_argument(
        "--decomposer-reward-aggregation",
        choices=["mean", "best"],
        default="best",
        help=(
            "How to aggregate selection rewards into a decomposer reward. "
            "`best` uses the strongest selection under a decomposition; `mean` averages all selections."
        ),
    )
    parser.add_argument(
        "--decomposer-no-correct-selection-scale",
        type=float,
        default=0.25,
        help=(
            "Scale applied to the positive part of a decomposition reward when none of its "
            "selections reach the correct final answer. 0.0 zeros positive spillover; 1.0 disables the gate."
        ),
    )
    parser.add_argument(
        "--controller-format-retry-penalty",
        type=float,
        default=0.05,
        help="Penalty subtracted from controller sample reward/advantage when output required format repair",
    )
    parser.add_argument(
        "--controller-format-fallback-penalty",
        type=float,
        default=0.25,
        help="Penalty subtracted from controller sample reward/advantage when fallback output was used",
    )
    parser.add_argument(
        "--selector-partial-completion-penalty",
        type=float,
        default=0.10,
        help=(
            "Extra penalty subtracted from selector sample reward/advantage when "
            "a malformed selector output was locally repaired into a valid assignment"
        ),
    )
    parser.add_argument("--save-replay-copy", action="store_true", help="Save train/val/all sample JSONL files per policy for inspection")

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1, help="Number of GRPO update epochs per outer training epoch")
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
    parser.add_argument("--checkpoint-mode", choices=["final", "all"], default="final")
    parser.add_argument("--prune-stale-policy-models", action="store_true")
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


def _relay_offline_metrics_to_tracking(
    output_dir: Path,
    tracking,
    tracking_prefix: str,
    log_step_offset: int,
    num_train_samples: int,
    num_val_samples: int,
) -> None:
    if tracking is None:
        return

    tracking.log(
        {
            f"{tracking_prefix}train/num_train_samples": num_train_samples,
            f"{tracking_prefix}train/num_val_samples": num_val_samples,
        },
        step=log_step_offset,
    )

    metrics_log_path = output_dir / "train_metrics.jsonl"
    if metrics_log_path.exists():
        with metrics_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                step = log_step_offset + int(record["step"])
                metrics = {
                    f"{tracking_prefix}train/loss": float(record["loss"]),
                    f"{tracking_prefix}train/lr": float(record["lr"]),
                    f"{tracking_prefix}train/mean_reward": float(record["mean_reward"]),
                    f"{tracking_prefix}train/overall_mean_reward": float(record["mean_reward"]),
                    f"{tracking_prefix}train/mean_advantage": float(record["mean_advantage"]),
                    f"{tracking_prefix}train/overall_mean_advantage": float(record["mean_advantage"]),
                    f"{tracking_prefix}train/approx_kl": float(record["approx_kl"]),
                    f"{tracking_prefix}train/entropy": float(record["entropy"]),
                    f"{tracking_prefix}train/clipfrac": float(record["clipfrac"]),
                }
                optional_float_fields = {
                    "selector_mean_reward": "selector_mean_reward",
                    "decomposer_mean_reward": "decomposer_mean_reward",
                }
                optional_int_fields = {
                    "selector_samples_total": "selector_samples_total",
                    "selector_samples_clean": "selector_samples_clean",
                    "selector_samples_locally_repaired": "selector_samples_locally_repaired",
                    "selector_samples_model_repaired": "selector_samples_model_repaired",
                    "selector_samples_hard_fallback": "selector_samples_hard_fallback",
                }
                for record_key, metric_suffix in optional_float_fields.items():
                    if record_key in record:
                        metrics[f"{tracking_prefix}train/{metric_suffix}"] = float(record[record_key])
                for record_key, metric_suffix in optional_int_fields.items():
                    if record_key in record:
                        metrics[f"{tracking_prefix}train/{metric_suffix}"] = int(record[record_key])
                tracking.log(metrics, step=step)

    eval_log_path = output_dir / "eval_metrics.jsonl"
    if eval_log_path.exists():
        with eval_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                step = log_step_offset + int(record["step"])
                tracking.log(
                    {f"{tracking_prefix}val/val_loss": float(record["val_loss"])},
                    step=step,
                )

    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        final_metrics = {
            f"{tracking_prefix}train/final_steps": int(summary.get("steps", 0)),
            f"{tracking_prefix}train/skipped_empty_batches": int(summary.get("skipped_empty_batches", 0)),
            f"{tracking_prefix}train/skipped_non_finite_batches": int(summary.get("skipped_non_finite_batches", 0)),
        }
        if "val_loss" in summary:
            final_metrics[f"{tracking_prefix}val/final_loss"] = float(summary["val_loss"])
        if summary.get("best_val_loss") is not None:
            final_metrics[f"{tracking_prefix}val/best_loss"] = float(summary["best_val_loss"])
        tracking.log(final_metrics, step=log_step_offset + max(int(summary.get("steps", 0)), 1))


def _offline_training_ray_runtime_env() -> Dict[str, Any]:
    repo_pkg_root = str(Path(__file__).resolve().parents[1])
    pythonpath_entries = [
        entry
        for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    ]
    if repo_pkg_root not in pythonpath_entries:
        pythonpath_entries.insert(0, repo_pkg_root)
    env_vars = {
        "PYTHONPATH": os.pathsep.join(pythonpath_entries),
        "PYTHONUNBUFFERED": "1",
    }
    offline_alloc_conf = os.environ.get("OFFLINE_GRPO_PYTORCH_CUDA_ALLOC_CONF")
    if offline_alloc_conf is not None:
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = offline_alloc_conf
    for env_name in (
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "WANDB_API_KEY",
        "WANDB_BASE_URL",
        "WANDB_MODE",
        "TOKENIZERS_PARALLELISM",
        "NCCL_DEBUG",
    ):
        env_value = os.environ.get(env_name)
        if env_value:
            env_vars[env_name] = env_value
    return {"env_vars": env_vars}


def _offline_training_alloc_conf() -> str | None:
    offline_alloc_conf = os.environ.get("OFFLINE_GRPO_PYTORCH_CUDA_ALLOC_CONF")
    if offline_alloc_conf is None:
        return None
    stripped = offline_alloc_conf.strip()
    return stripped or None


def _ensure_ray_initialized_for_offline_training() -> None:
    import ray

    if ray.is_initialized():
        return
    ray_address = os.environ.get("RAY_ADDRESS", "").strip()
    ray_namespace = os.environ.get("RAY_NAMESPACE", "").strip()
    init_kwargs: Dict[str, Any] = {}
    init_kwargs["runtime_env"] = _offline_training_ray_runtime_env()
    if ray_address:
        init_kwargs["address"] = ray_address
    if ray_namespace:
        init_kwargs["namespace"] = ray_namespace
    ray.init(**init_kwargs)


def _run_distributed_offline_policy_training(
    train_samples,
    val_samples,
    config: OfflineTrainingConfig,
    tracking,
    tracking_prefix: str,
    log_step_offset: int,
    nnodes: int,
    gpus_per_node: int,
    master_port: int,
) -> Dict[str, Any]:
    _ensure_ray_initialized_for_offline_training()
    import ray

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = write_samples_to_jsonl(train_samples, output_dir / "train_samples.jsonl")
    val_jsonl = write_samples_to_jsonl(val_samples, output_dir / "val_samples.jsonl") if val_samples else ""
    config_json = output_dir / "distributed_train_config.json"
    with config_json.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True)

    world_size = nnodes * gpus_per_node
    cluster_resources = ray.cluster_resources()
    cluster_gpus = float(cluster_resources.get("GPU", 0.0))
    if cluster_gpus + 1e-6 < world_size:
        raise RuntimeError(
            f"offline distributed GRPO requested world_size={world_size}, "
            f"but Ray cluster only reports GPU={cluster_gpus}"
        )

    worker_cls = ray.remote(
        num_gpus=1,
        num_cpus=1,
        max_restarts=0,
        runtime_env=_offline_training_ray_runtime_env(),
    )(RayOfflineGRPOWorker)
    workers = [
        worker_cls.options(scheduling_strategy="SPREAD").remote()
        for _ in range(world_size)
    ]
    master_addr = ray.get(workers[0].get_node_ip.remote())
    print(
        f"[hierarchical-rema][grpo] launching distributed offline learner "
        f"nnodes={nnodes} gpus_per_node={gpus_per_node} world_size={world_size} "
        f"policy_output_dir={output_dir} backend=ray master_addr={master_addr} "
        f"cluster_gpu={cluster_gpus:.0f}"
    )
    try:
        results = ray.get(
            [
                worker.run.remote(
                    rank=rank,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    train_samples_jsonl=train_jsonl,
                    val_samples_jsonl=val_jsonl,
                    config_json=str(config_json),
                )
                for rank, worker in enumerate(workers)
            ]
        )
    finally:
        for worker in workers:
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                pass

    rank0_result = next((item for item in results if int(item.get("rank", -1)) == 0), None)
    if rank0_result is None:
        raise RuntimeError("Distributed offline GRPO finished without a rank-0 result")

    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Distributed offline GRPO completed without writing {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    _relay_offline_metrics_to_tracking(
        output_dir=output_dir,
        tracking=tracking,
        tracking_prefix=tracking_prefix,
        log_step_offset=log_step_offset,
        num_train_samples=len(train_samples),
        num_val_samples=len(val_samples),
    )
    return summary


def _prune_policy_artifacts(policy_root: Path) -> None:
    for subdir_name in ("final", "best"):
        subdir = policy_root / subdir_name
        if subdir.exists():
            shutil.rmtree(subdir, ignore_errors=True)
    for checkpoint_dir in policy_root.glob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


def _tracking(args: argparse.Namespace, config_payload: Dict[str, Any]):
    if not args.enable_wandb:
        print("[hierarchical-rema][tracking] wandb disabled")
        return None
    try:
        try:
            from verl.utils.tracking import Tracking
        except ModuleNotFoundError:
            from utils.tracking import Tracking
    except Exception as exc:
        print(f"[hierarchical-rema][tracking] wandb disabled due to import/init error: {exc}")
        return None

    experiment_name = _default_experiment_name(args)
    print(
        f"[hierarchical-rema][tracking] initializing wandb "
        f"project={args.project_name} experiment={experiment_name}"
    )
    try:
        return Tracking(
            project_name=args.project_name,
            experiment_name=experiment_name,
            default_backend=["wandb"],
            config=config_payload,
        )
    except Exception as exc:
        print(f"[hierarchical-rema][tracking] wandb initialization failed: {exc}")
        return None


def _sanitize_experiment_component(value: str) -> str:
    sanitized = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            sanitized.append(char)
        elif char in {"/", " ", ":", ",", "=", "+", "(", ")"}:
            sanitized.append("-")
        else:
            sanitized.append("-")
    result = "".join(sanitized).strip("-")
    while "--" in result:
        result = result.replace("--", "-")
    return result or "unknown"


def _default_experiment_name(args: argparse.Namespace) -> str:
    if args.experiment_name:
        return args.experiment_name

    job_id = os.environ.get("SLURM_JOB_ID")
    mode = getattr(args, "mode", None) or os.environ.get("MODE") or "unknown"
    parameter_sharing_raw = os.environ.get("PARAMETER_SHARING", "false").lower()
    parameter_sharing = parameter_sharing_raw in {"1", "true", "yes"}

    model_path = (
        getattr(args, "model_path", None)
        or getattr(args, "shared_model_path", None)
        or getattr(args, "decomposer_model_path", None)
        or os.environ.get("MODEL_PATH")
        or os.environ.get("DECOMPOSER_MODEL_PATH")
    )
    model_name = Path(model_path).name if model_path else "model"

    components = [
        "train",
        f"mode-{_sanitize_experiment_component(str(mode))}",
        f"ps-{'true' if parameter_sharing else 'false'}",
        _sanitize_experiment_component(model_name),
    ]
    if job_id:
        components.append(str(job_id))
    return "-".join(components)


def _finish_tracking(tracking) -> None:
    if tracking is None:
        return
    try:
        tracking.__del__()
    except Exception:
        pass


def _rollout_config_for_schedule(
    base_rollout_config: RolloutConfig,
    schedule: TrainingScheduleConfig,
) -> RolloutConfig:
    num_decompositions = base_rollout_config.num_decompositions
    num_selections = base_rollout_config.num_selections_per_decomposition
    if schedule.mode == TrainingMode.ALTERNATING:
        if schedule.alternating_phase == AlternatingPhase.SELECTOR:
            num_selections = max(num_selections, 2)
        elif schedule.alternating_phase == AlternatingPhase.DECOMPOSER:
            num_decompositions = max(num_decompositions, 2)
    return RolloutConfig(
        num_decompositions=num_decompositions,
        num_selections_per_decomposition=num_selections,
        max_nodes_per_decomposition=base_rollout_config.max_nodes_per_decomposition,
        soft_max_hops=base_rollout_config.soft_max_hops,
        hard_max_hops=base_rollout_config.hard_max_hops,
        soft_hop_penalty=base_rollout_config.soft_hop_penalty,
        soft_hop_penalty_power=base_rollout_config.soft_hop_penalty_power,
    )


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


def _task_subset_name(task: TaskExample) -> str:
    metadata = task.metadata or {}
    for key in ("subset", "dataset", "source_subset", "benchmark_subset"):
        value = metadata.get(key)
        if value is not None and value != "":
            return str(value)
    return "unknown"


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


def select_epoch_tasks_by_subset(
    tasks: Sequence[TaskExample],
    epoch_index: int,
    tasks_per_subset: int,
    shuffle_tasks: bool,
    seed: int,
) -> List[TaskExample]:
    if tasks_per_subset <= 0:
        return list(tasks)

    tasks_by_subset: Dict[str, List[TaskExample]] = {}
    for task in tasks:
        tasks_by_subset.setdefault(_task_subset_name(task), []).append(task)

    selected: List[TaskExample] = []
    for subset_name in sorted(tasks_by_subset):
        subset_tasks = tasks_by_subset[subset_name]
        subset_seed = seed + epoch_index + sum(ord(ch) for ch in subset_name)
        selected.extend(
            select_epoch_tasks(
                tasks=subset_tasks,
                epoch_index=epoch_index,
                tasks_per_epoch=tasks_per_subset,
                shuffle_tasks=shuffle_tasks,
                seed=subset_seed,
            )
        )
    return selected


def chunk_tasks(
    tasks: Sequence[TaskExample],
    batch_size: int,
) -> List[List[TaskExample]]:
    task_list = list(tasks)
    if batch_size <= 0 or batch_size >= len(task_list):
        return [task_list]
    return [
        task_list[start:start + batch_size]
        for start in range(0, len(task_list), batch_size)
    ]


def build_schedule(mode: str, phase: AlternatingPhase) -> TrainingScheduleConfig:
    return TrainingScheduleConfig(
        mode=TrainingMode(mode),
        alternating_phase=phase,
    )


def next_phase(phase: AlternatingPhase) -> AlternatingPhase:
    return AlternatingPhase.DECOMPOSER if phase == AlternatingPhase.SELECTOR else AlternatingPhase.SELECTOR


def rollout_workload_estimate(
    num_tasks: int,
    rollout_config: RolloutConfig,
    schedule: TrainingScheduleConfig,
) -> Dict[str, int]:
    num_decompositions = rollout_config.num_decompositions
    num_selections = rollout_config.num_selections_per_decomposition
    if schedule.mode == TrainingMode.ALTERNATING:
        if schedule.alternating_phase == AlternatingPhase.SELECTOR:
            num_decompositions = 1
        else:
            num_selections = 1

    max_nodes = max(int(rollout_config.max_nodes_per_decomposition), 1)
    controller_generations_per_task = num_decompositions + (num_decompositions * num_selections)
    worker_generations_per_task_upper_bound = num_decompositions * num_selections * max_nodes
    return {
        "num_decompositions": num_decompositions,
        "num_selections": num_selections,
        "controller_generations_per_task": controller_generations_per_task,
        "worker_generations_per_task_upper_bound": worker_generations_per_task_upper_bound,
        "controller_generations_total": controller_generations_per_task * num_tasks,
        "worker_generations_total_upper_bound": worker_generations_per_task_upper_bound * num_tasks,
    }


def _best_rollout_metrics(rollout: TaskRollout) -> Dict[str, Any]:
    best_decomposition = max(rollout.decompositions, key=lambda item: item.decomposition_reward)
    selection_rewards = [
        selection.reward.total_reward
        for decomposition in rollout.decompositions
        for selection in decomposition.selections
    ]
    selection_correctness = [
        selection.reward.final_answer_correctness
        for decomposition in rollout.decompositions
        for selection in decomposition.selections
    ]
    return {
        "best_decomposition_id": best_decomposition.decomposition.decomposition_id,
        "best_decomposition_reward": best_decomposition.decomposition_reward,
        "best_selection_reward": max(selection_rewards) if selection_rewards else 0.0,
        "mean_selection_reward": sum(selection_rewards) / max(len(selection_rewards), 1),
        "best_final_correctness": max(selection_correctness) if selection_correctness else 0.0,
    }


def epoch_rollout_summary(
    rollouts: Sequence[TaskRollout],
    include_subsets: bool = False,
    include_tasks: bool = True,
) -> Dict[str, Any]:
    task_summaries = []
    best_decomposition_rewards = []
    best_selection_rewards = []
    mean_selection_rewards = []
    best_correctness = []
    subset_metrics: Dict[str, Dict[str, Any]] = {}
    for rollout in rollouts:
        metrics = _best_rollout_metrics(rollout)
        best_decomposition_rewards.append(metrics["best_decomposition_reward"])
        best_selection_rewards.append(metrics["best_selection_reward"])
        mean_selection_rewards.append(metrics["mean_selection_reward"])
        best_correctness.append(metrics["best_final_correctness"])
        subset_name = _task_subset_name(rollout.task)
        if include_tasks:
            task_summaries.append(
                {
                    "task_id": rollout.task.task_id,
                    "subset": subset_name,
                    **metrics,
                }
            )
        if include_subsets:
            bucket = subset_metrics.setdefault(
                subset_name,
                {
                    "num_tasks": 0,
                    "num_correct": 0.0,
                    "best_decomposition_rewards": [],
                    "best_selection_rewards": [],
                    "mean_selection_rewards": [],
                    "best_correctness": [],
                },
            )
            bucket["num_tasks"] += 1
            bucket["num_correct"] += metrics["best_final_correctness"]
            bucket["best_decomposition_rewards"].append(metrics["best_decomposition_reward"])
            bucket["best_selection_rewards"].append(metrics["best_selection_reward"])
            bucket["mean_selection_rewards"].append(metrics["mean_selection_reward"])
            bucket["best_correctness"].append(metrics["best_final_correctness"])

    summary = {
        "num_tasks": len(rollouts),
        "mean_best_decomposition_reward": sum(best_decomposition_rewards) / max(len(best_decomposition_rewards), 1),
        "mean_best_selection_reward": sum(best_selection_rewards) / max(len(best_selection_rewards), 1),
        "mean_selection_reward": sum(mean_selection_rewards) / max(len(mean_selection_rewards), 1),
        "mean_best_final_correctness": sum(best_correctness) / max(len(best_correctness), 1),
        "tasks": task_summaries if include_tasks else [],
    }
    if include_subsets:
        summary["subsets"] = {
            subset_name: {
                "num_tasks": bucket["num_tasks"],
                "num_correct": bucket["num_correct"],
                "mean_best_decomposition_reward": sum(bucket["best_decomposition_rewards"]) / max(bucket["num_tasks"], 1),
                "mean_best_selection_reward": sum(bucket["best_selection_rewards"]) / max(bucket["num_tasks"], 1),
                "mean_selection_reward": sum(bucket["mean_selection_rewards"]) / max(bucket["num_tasks"], 1),
                "mean_best_final_correctness": sum(bucket["best_correctness"]) / max(bucket["num_tasks"], 1),
            }
            for subset_name, bucket in subset_metrics.items()
        }
    return summary


def combine_rollout_summaries(
    summaries: Sequence[Dict[str, Any]],
    *,
    include_subsets: bool = False,
    include_tasks: bool = True,
) -> Dict[str, Any]:
    total_tasks = 0
    total_best_decomposition_reward = 0.0
    total_best_selection_reward = 0.0
    total_mean_selection_reward = 0.0
    total_best_final_correctness = 0.0
    combined_tasks: List[Dict[str, Any]] = []
    subset_accumulators: Dict[str, Dict[str, float]] = {}

    for summary in summaries:
        num_tasks = int(summary.get("num_tasks", 0))
        total_tasks += num_tasks
        total_best_decomposition_reward += float(summary.get("mean_best_decomposition_reward", 0.0)) * num_tasks
        total_best_selection_reward += float(summary.get("mean_best_selection_reward", 0.0)) * num_tasks
        total_mean_selection_reward += float(summary.get("mean_selection_reward", 0.0)) * num_tasks
        total_best_final_correctness += float(summary.get("mean_best_final_correctness", 0.0)) * num_tasks

        if include_tasks:
            combined_tasks.extend(summary.get("tasks", []))

        if not include_subsets:
            continue
        for subset_name, subset_summary in summary.get("subsets", {}).items():
            bucket = subset_accumulators.setdefault(
                subset_name,
                {
                    "num_tasks": 0.0,
                    "num_correct": 0.0,
                    "best_decomposition_reward_sum": 0.0,
                    "best_selection_reward_sum": 0.0,
                    "mean_selection_reward_sum": 0.0,
                    "best_final_correctness_sum": 0.0,
                },
            )
            subset_tasks = int(subset_summary.get("num_tasks", 0))
            bucket["num_tasks"] += subset_tasks
            bucket["num_correct"] += float(subset_summary.get("num_correct", 0.0))
            bucket["best_decomposition_reward_sum"] += (
                float(subset_summary.get("mean_best_decomposition_reward", 0.0)) * subset_tasks
            )
            bucket["best_selection_reward_sum"] += (
                float(subset_summary.get("mean_best_selection_reward", 0.0)) * subset_tasks
            )
            bucket["mean_selection_reward_sum"] += (
                float(subset_summary.get("mean_selection_reward", 0.0)) * subset_tasks
            )
            bucket["best_final_correctness_sum"] += (
                float(subset_summary.get("mean_best_final_correctness", 0.0)) * subset_tasks
            )

    summary = {
        "num_tasks": total_tasks,
        "mean_best_decomposition_reward": total_best_decomposition_reward / max(total_tasks, 1),
        "mean_best_selection_reward": total_best_selection_reward / max(total_tasks, 1),
        "mean_selection_reward": total_mean_selection_reward / max(total_tasks, 1),
        "mean_best_final_correctness": total_best_final_correctness / max(total_tasks, 1),
        "tasks": combined_tasks if include_tasks else [],
    }
    if include_subsets:
        summary["subsets"] = {
            subset_name: {
                "num_tasks": int(bucket["num_tasks"]),
                "num_correct": bucket["num_correct"],
                "mean_best_decomposition_reward": bucket["best_decomposition_reward_sum"] / max(bucket["num_tasks"], 1.0),
                "mean_best_selection_reward": bucket["best_selection_reward_sum"] / max(bucket["num_tasks"], 1.0),
                "mean_selection_reward": bucket["mean_selection_reward_sum"] / max(bucket["num_tasks"], 1.0),
                "mean_best_final_correctness": bucket["best_final_correctness_sum"] / max(bucket["num_tasks"], 1.0),
            }
            for subset_name, bucket in subset_accumulators.items()
        }
    return summary


def _build_rollout_trainer(
    args: argparse.Namespace,
    *,
    backend_temperature: float,
    controller_temperature: float | None,
    worker_temperature: float | None,
    backend_top_p: float,
    controller_max_new_tokens: int,
    decomposer_max_new_tokens: int | None,
    selector_max_new_tokens: int | None,
    worker_max_new_tokens: int,
    rollout_logging_config: RolloutLoggingConfig | None,
    controller_batch_size: int | None = None,
    worker_batch_size: int | None = None,
    backend: Any | None = None,
) -> HierarchicalGRPOTrainer:
    effective_controller_batch_size = (
        controller_batch_size if controller_batch_size is not None else args.controller_batch_size
    )
    effective_worker_batch_size = (
        worker_batch_size if worker_batch_size is not None else args.worker_batch_size
    )
    worker_reward_mode = (
        WorkerRewardMode.FINAL_ANSWER_CORRECTNESS_ONLY
        if args.final_answer_correctness_reward_only
        else WorkerRewardMode.CURRENT
    )
    return HierarchicalGRPOTrainer(
        reward_weights=RewardWeights(
            final_answer=args.final_answer_reward_weight,
            confidence=args.confidence_reward_weight,
            compatibility=args.compatibility_reward_weight,
            worker_reward_mode=worker_reward_mode,
            worker_success_weight=getattr(args, "worker_success_weight", 0.0),
            worker_final_correctness_weight=args.worker_final_correctness_weight,
        ),
        backend_type=args.backend,
        hf_backend_config=HFBackendConfig(
            temperature=backend_temperature,
            controller_temperature=controller_temperature,
            worker_temperature=worker_temperature,
            top_p=backend_top_p,
            do_sample=backend_temperature > 0.0,
            controller_max_new_tokens=controller_max_new_tokens,
            decomposer_max_new_tokens=decomposer_max_new_tokens,
            selector_max_new_tokens=selector_max_new_tokens,
            worker_max_new_tokens=worker_max_new_tokens,
            controller_batch_size=effective_controller_batch_size,
            worker_batch_size=effective_worker_batch_size,
            controller_constrained_decoding=args.controller_constrained_decoding,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
        ),
        vllm_backend_config=VLLMBackendConfig(
            temperature=backend_temperature,
            controller_temperature=controller_temperature,
            worker_temperature=worker_temperature,
            top_p=backend_top_p,
            do_sample=backend_temperature > 0.0,
            prompt_length=args.rollout_prompt_length,
            controller_max_new_tokens=controller_max_new_tokens,
            decomposer_max_new_tokens=decomposer_max_new_tokens,
            selector_max_new_tokens=selector_max_new_tokens,
            worker_max_new_tokens=worker_max_new_tokens,
            controller_batch_size=effective_controller_batch_size,
            worker_batch_size=effective_worker_batch_size,
            controller_constrained_decoding=args.controller_constrained_decoding,
            nnodes=args.ray_nnodes,
            n_gpus_per_node=args.ray_n_gpus_per_node,
            cpus_per_node=(args.ray_cpus_per_node if args.ray_cpus_per_node > 0 else None),
            tensor_model_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            max_num_seqs=args.vllm_max_num_seqs,
            max_model_len=args.vllm_max_model_len,
            dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
        ),
        rollout_logging_config=rollout_logging_config,
        backend=backend,
        controller_format_retry_penalty=args.controller_format_retry_penalty,
        controller_format_fallback_penalty=args.controller_format_fallback_penalty,
        selector_partial_completion_penalty=args.selector_partial_completion_penalty,
        decomposer_reward_aggregation=args.decomposer_reward_aggregation,
        decomposer_no_correct_selection_scale=args.decomposer_no_correct_selection_scale,
    )


def _build_rollout_backend(
    args: argparse.Namespace,
    *,
    backend_temperature: float,
    controller_temperature: float | None,
    worker_temperature: float | None,
    backend_top_p: float,
    controller_max_new_tokens: int,
    decomposer_max_new_tokens: int | None,
    selector_max_new_tokens: int | None,
    worker_max_new_tokens: int,
    controller_batch_size: int | None = None,
    worker_batch_size: int | None = None,
):
    effective_controller_batch_size = (
        controller_batch_size if controller_batch_size is not None else args.controller_batch_size
    )
    effective_worker_batch_size = (
        worker_batch_size if worker_batch_size is not None else args.worker_batch_size
    )
    hf_config = HFBackendConfig(
        temperature=backend_temperature,
        controller_temperature=controller_temperature,
        worker_temperature=worker_temperature,
        top_p=backend_top_p,
        do_sample=backend_temperature > 0.0,
        controller_max_new_tokens=controller_max_new_tokens,
        decomposer_max_new_tokens=decomposer_max_new_tokens,
        selector_max_new_tokens=selector_max_new_tokens,
        worker_max_new_tokens=worker_max_new_tokens,
        controller_batch_size=effective_controller_batch_size,
        worker_batch_size=effective_worker_batch_size,
        controller_constrained_decoding=args.controller_constrained_decoding,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
    )
    if args.backend == "mock":
        return MockHierarchicalBackend(worker_memory=WorkerPerformanceMemory())
    if args.backend == "hf":
        return TransformersHierarchicalBackend(config=hf_config)

    vllm_config = VLLMBackendConfig(
        temperature=backend_temperature,
        controller_temperature=controller_temperature,
        worker_temperature=worker_temperature,
        top_p=backend_top_p,
        do_sample=backend_temperature > 0.0,
        prompt_length=args.rollout_prompt_length,
        controller_max_new_tokens=controller_max_new_tokens,
        decomposer_max_new_tokens=decomposer_max_new_tokens,
        selector_max_new_tokens=selector_max_new_tokens,
        worker_max_new_tokens=worker_max_new_tokens,
        controller_batch_size=effective_controller_batch_size,
        worker_batch_size=effective_worker_batch_size,
        controller_constrained_decoding=args.controller_constrained_decoding,
        nnodes=args.ray_nnodes,
        n_gpus_per_node=args.ray_n_gpus_per_node,
        cpus_per_node=(args.ray_cpus_per_node if args.ray_cpus_per_node > 0 else None),
        tensor_model_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        max_num_seqs=args.vllm_max_num_seqs,
        max_model_len=args.vllm_max_model_len,
        dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    return RayVLLMHierarchicalBackend(config=vllm_config)


def run_external_validation(
    *,
    args: argparse.Namespace,
    epoch_number: int,
    val_tasks: Sequence[TaskExample],
    worker_pool,
    policy_config: ControllerPolicyConfig,
    output_dir: Path,
    tracking,
    tracking_step: int,
    base_rollout_config: RolloutConfig,
) -> Dict[str, Any]:
    controller_max_new_tokens = (
        args.val_controller_max_new_tokens
        if args.val_controller_max_new_tokens > 0
        else args.controller_max_new_tokens
    )
    decomposer_max_new_tokens = (
        args.val_decomposer_max_new_tokens
        if args.val_decomposer_max_new_tokens > 0
        else (
            args.decomposer_max_new_tokens
            if args.decomposer_max_new_tokens > 0
            else controller_max_new_tokens
        )
    )
    selector_max_new_tokens = (
        args.val_selector_max_new_tokens
        if args.val_selector_max_new_tokens > 0
        else (
            args.selector_max_new_tokens
            if args.selector_max_new_tokens > 0
            else controller_max_new_tokens
        )
    )
    controller_temperature = (
        args.val_controller_temperature
        if args.val_controller_temperature is not None
        else (
            args.controller_temperature
            if args.controller_temperature is not None
            else args.val_temperature
        )
    )
    worker_max_new_tokens = (
        args.val_worker_max_new_tokens
        if args.val_worker_max_new_tokens > 0
        else args.worker_max_new_tokens
    )
    worker_temperature = (
        args.val_worker_temperature
        if args.val_worker_temperature is not None
        else (
            args.worker_temperature
            if args.worker_temperature is not None
            else args.val_temperature
        )
    )
    rollout_task_batch_size = (
        args.val_rollout_task_batch_size
        if args.val_rollout_task_batch_size > 0
        else args.rollout_task_batch_size
    )
    controller_batch_size = (
        args.val_controller_batch_size
        if args.val_controller_batch_size > 0
        else args.controller_batch_size
    )
    worker_batch_size = (
        args.val_worker_batch_size
        if args.val_worker_batch_size > 0
        else args.worker_batch_size
    )

    validation_rollout_config = RolloutConfig(
        num_decompositions=args.val_num_decompositions,
        num_selections_per_decomposition=args.val_num_selections,
        max_nodes_per_decomposition=base_rollout_config.max_nodes_per_decomposition,
        soft_max_hops=base_rollout_config.soft_max_hops,
        hard_max_hops=base_rollout_config.hard_max_hops,
        soft_hop_penalty=base_rollout_config.soft_hop_penalty,
        soft_hop_penalty_power=base_rollout_config.soft_hop_penalty_power,
    )
    validation_schedule = TrainingScheduleConfig(mode=TrainingMode.JOINT)
    workload = rollout_workload_estimate(
        num_tasks=len(val_tasks),
        rollout_config=validation_rollout_config,
        schedule=validation_schedule,
    )
    print(
        f"[hierarchical-rema][validation] epoch={epoch_number} tasks={len(val_tasks)} "
        f"controller_total={workload['controller_generations_total']} "
        f"worker_total_upper_bound={workload['worker_generations_total_upper_bound']} "
        f"task_batch_size={rollout_task_batch_size} "
        f"controller_batch_size={controller_batch_size} "
        f"worker_batch_size={worker_batch_size}"
    )

    trainer = _build_rollout_trainer(
        args,
        backend_temperature=args.val_temperature,
        controller_temperature=controller_temperature,
        worker_temperature=worker_temperature,
        backend_top_p=args.val_top_p,
        controller_max_new_tokens=controller_max_new_tokens,
        decomposer_max_new_tokens=decomposer_max_new_tokens,
        selector_max_new_tokens=selector_max_new_tokens,
        worker_max_new_tokens=worker_max_new_tokens,
        controller_batch_size=controller_batch_size,
        worker_batch_size=worker_batch_size,
        rollout_logging_config=None,
    )

    try:
        batch_summaries: List[Dict[str, Any]] = []
        val_batches = chunk_tasks(val_tasks, rollout_task_batch_size)
        validation_start_time = time.time()
        validation_tasks_completed = 0
        for batch_index, task_batch in enumerate(val_batches, start=1):
            print(
                f"[hierarchical-rema][validation] batch_start "
                f"epoch={epoch_number} batch={batch_index}/{len(val_batches)} "
                f"tasks_in_batch={len(task_batch)}"
            )
            batch_rollouts = trainer.run_many(
                tasks=task_batch,
                worker_pool=worker_pool,
                policy_config=policy_config,
                rollout_config=validation_rollout_config,
                schedule=validation_schedule,
                update_worker_memory=False,
                progress_label=(
                    f"{validation_tasks_completed + 1}-"
                    f"{validation_tasks_completed + len(task_batch)}/{len(val_tasks)}"
                ),
            )
            batch_summaries.append(
                epoch_rollout_summary(
                    batch_rollouts,
                    include_subsets=True,
                    include_tasks=False,
                )
            )
            validation_tasks_completed += len(batch_rollouts)
            elapsed = time.time() - validation_start_time
            avg_seconds_per_task = elapsed / max(validation_tasks_completed, 1)
            remaining_tasks = len(val_tasks) - validation_tasks_completed
            eta_seconds = avg_seconds_per_task * remaining_tasks
            print(
                f"[hierarchical-rema][validation] progress "
                f"epoch={epoch_number} task={validation_tasks_completed}/{len(val_tasks)} "
                f"elapsed_s={elapsed:.1f} eta_s={eta_seconds:.1f}"
            )
            del batch_rollouts
    finally:
        trainer.close()
        del trainer
        _release_memory()

    summary = combine_rollout_summaries(
        batch_summaries,
        include_subsets=True,
        include_tasks=False,
    )
    with (output_dir / "validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(
        f"[hierarchical-rema][validation] epoch={epoch_number} "
        f"mean_best_selection_reward={summary['mean_best_selection_reward']:.4f} "
        f"mean_best_decomposition_reward={summary['mean_best_decomposition_reward']:.4f} "
        f"mean_best_final_correctness={summary['mean_best_final_correctness']:.4f}"
    )
    if tracking is not None:
        metrics = {
            "val/mean_best_selection_reward": summary["mean_best_selection_reward"],
            "val/mean_best_decomposition_reward": summary["mean_best_decomposition_reward"],
            "val/mean_best_final_correctness": summary["mean_best_final_correctness"],
            "val/num_tasks": summary["num_tasks"],
        }
        for subset_name, subset_summary in summary.get("subsets", {}).items():
            metrics[f"val/test_score/{subset_name}"] = subset_summary["mean_best_selection_reward"]
            metrics[f"val/acc/{subset_name}"] = subset_summary["mean_best_final_correctness"]
        tracking.log(metrics, step=tracking_step)
    return summary


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


def _replay_model_args(current_paths: Dict[str, str]) -> argparse.Namespace:
    return argparse.Namespace(
        model_path=None,
        decomposer_model_path=current_paths["decomposer_controller"],
        selector_model_path=current_paths["selector_controller"],
    )


def _update_current_paths(current_paths: Dict[str, str], policy_id: str, model_path: str) -> None:
    current_paths[policy_id] = model_path
    if policy_id == "shared_controller":
        current_paths["decomposer_controller"] = model_path
        current_paths["selector_controller"] = model_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_task_source = _resolve_task_source_path(args.task_source)
    resolved_val_task_source = None if args.disable_external_validation or not args.val_task_source else _resolve_task_source_path(args.val_task_source)

    tasks = load_tasks(
        task_source=resolved_task_source,
        task_format=args.task_format,
        prompt_key=args.prompt_key,
        answer_key=args.answer_key,
        task_id_key=args.task_id_key,
        max_tasks=args.max_tasks,
    )
    val_tasks: List[TaskExample] = []
    if resolved_val_task_source is not None:
        val_tasks = load_tasks(
            task_source=resolved_val_task_source,
            task_format=args.val_task_format,
            prompt_key=args.val_prompt_key,
            answer_key=args.val_answer_key,
            task_id_key=args.val_task_id_key,
            max_tasks=args.max_val_tasks,
        )
    worker_pool = make_worker_pool(base_model_path=args.worker_base_model_path)
    max_nodes_per_decomposition = (
        args.max_nodes_per_decomposition
        if args.max_nodes_per_decomposition is not None
        else (args.hard_max_hops if args.hard_max_hops is not None else 4)
    )
    base_rollout_config = RolloutConfig(
        num_decompositions=args.num_decompositions,
        num_selections_per_decomposition=args.num_selections,
        max_nodes_per_decomposition=max_nodes_per_decomposition,
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
        "val_task_source": args.val_task_source,
        "resolved_val_task_source": resolved_val_task_source,
        "num_loaded_tasks": len(tasks),
        "num_loaded_val_tasks": len(val_tasks),
        "num_epochs": args.num_epochs,
        "epochs": [],
    }

    print(
        f"[hierarchical-rema][integrated] loaded_tasks={len(tasks)} "
        f"task_source={resolved_task_source} mode={args.mode} backend={args.backend}"
    )
    if resolved_val_task_source is not None:
        print(
            f"[hierarchical-rema][integrated] loaded_val_tasks={len(val_tasks)} "
            f"val_task_source={resolved_val_task_source}"
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
        schedule = build_schedule(args.mode, current_phase)

        print(
            f"[hierarchical-rema][integrated] epoch={epoch_number}/{args.num_epochs} "
            f"phase={schedule.alternating_phase.value} tasks={len(epoch_tasks)}"
        )
        rollout_config = _rollout_config_for_schedule(base_rollout_config, schedule)
        if (
            rollout_config.num_decompositions != base_rollout_config.num_decompositions
            or rollout_config.num_selections_per_decomposition
            != base_rollout_config.num_selections_per_decomposition
        ):
            print(
                f"[hierarchical-rema][integrated] adjusted_rollout_counts "
                f"decompositions={rollout_config.num_decompositions} "
                f"selections={rollout_config.num_selections_per_decomposition}"
            )
        workload = rollout_workload_estimate(
            num_tasks=len(epoch_tasks),
            rollout_config=rollout_config,
            schedule=schedule,
        )
        print(
            f"[hierarchical-rema][integrated] rollout_budget "
            f"controller_total={workload['controller_generations_total']} "
            f"worker_total_upper_bound={workload['worker_generations_total_upper_bound']} "
            f"controller_per_task={workload['controller_generations_per_task']} "
            f"worker_per_task_upper_bound={workload['worker_generations_per_task_upper_bound']} "
            f"task_batch_size={args.rollout_task_batch_size} "
            f"controller_batch_size={args.controller_batch_size} "
            f"worker_batch_size={args.worker_batch_size}"
        )
        rollouts: List[TaskRollout] = []
        training_updates: List[Dict[str, Any]] = []
        training_summaries: Dict[str, Any] = {}
        segment_summaries: List[Dict[str, Any]] = []
        training_skipped_messages: List[str] = []
        rollout_start_time = time.time()
        running_best_selection_reward = 0.0
        running_best_decomposition_reward = 0.0
        running_best_correctness = 0.0
        tasks_completed = 0
        rollout_tracking_step = tracking_step_offset
        rollout_progress_path = epoch_dir / "rollout_progress.json"
        task_batches = chunk_tasks(epoch_tasks, args.rollout_task_batch_size)
        update_every_n_rollout_batches = max(int(args.update_every_n_rollout_batches), 1)
        batch_cursor = 0
        segment_index = 0
        while batch_cursor < len(task_batches):
            segment_index += 1
            policy_config = _current_policy_config(args, current_paths)
            replay_like_args = _replay_model_args(current_paths)
            segment_batches = task_batches[batch_cursor:batch_cursor + update_every_n_rollout_batches]
            segment_rollout_dir = rollout_dir / f"segment_{segment_index:04d}"
            segment_train_dir = train_dir / f"segment_{segment_index:04d}"
            segment_train_dir.mkdir(parents=True, exist_ok=True)
            segment_logging_config = None
            if not args.disable_rollout_logging:
                segment_logging_config = RolloutLoggingConfig(
                    output_dir=str(segment_rollout_dir),
                    save_all_rollouts=args.rollout_log_mode == "all",
                    save_best_rollouts=True,
                    best_k=args.best_k,
                    compact_mode=args.rollout_log_detail == "compact",
                )

            print(
                f"[hierarchical-rema][integrated] rollout_update_segment "
                f"epoch={epoch_number} segment={segment_index} "
                f"batches={batch_cursor + 1}-{batch_cursor + len(segment_batches)}/{len(task_batches)}"
            )

            segment_backend = _build_rollout_backend(
                args,
                backend_temperature=args.temperature,
                controller_temperature=(
                    args.controller_temperature
                    if args.controller_temperature is not None
                    else args.temperature
                ),
                worker_temperature=(
                    args.worker_temperature
                    if args.worker_temperature is not None
                    else args.temperature
                ),
                backend_top_p=args.top_p,
                controller_max_new_tokens=args.controller_max_new_tokens,
                decomposer_max_new_tokens=(
                    args.decomposer_max_new_tokens
                    if args.decomposer_max_new_tokens > 0
                    else args.controller_max_new_tokens
                ),
                selector_max_new_tokens=(
                    args.selector_max_new_tokens
                    if args.selector_max_new_tokens > 0
                    else args.controller_max_new_tokens
                ),
                worker_max_new_tokens=args.worker_max_new_tokens,
            )
            rollout_trainer = _build_rollout_trainer(
                args,
                backend_temperature=args.temperature,
                controller_temperature=(
                    args.controller_temperature
                    if args.controller_temperature is not None
                    else args.temperature
                ),
                worker_temperature=(
                    args.worker_temperature
                    if args.worker_temperature is not None
                    else args.temperature
                ),
                backend_top_p=args.top_p,
                controller_max_new_tokens=args.controller_max_new_tokens,
                decomposer_max_new_tokens=(
                    args.decomposer_max_new_tokens
                    if args.decomposer_max_new_tokens > 0
                    else args.controller_max_new_tokens
                ),
                selector_max_new_tokens=(
                    args.selector_max_new_tokens
                    if args.selector_max_new_tokens > 0
                    else args.controller_max_new_tokens
                ),
                worker_max_new_tokens=args.worker_max_new_tokens,
                rollout_logging_config=segment_logging_config,
                backend=segment_backend,
            )
            segment_rollouts: List[TaskRollout] = []
            try:
                for segment_offset, task_batch in enumerate(segment_batches):
                    batch_index = batch_cursor + segment_offset + 1
                    print(
                        f"[hierarchical-rema][integrated] rollout_batch_start "
                        f"epoch={epoch_number} batch={batch_index}/{len(task_batches)} "
                        f"tasks_in_batch={len(task_batch)} "
                        f"completed_before_batch={tasks_completed}"
                    )
                    batch_rollouts = rollout_trainer.run_many(
                        tasks=task_batch,
                        worker_pool=worker_pool,
                        policy_config=policy_config,
                        rollout_config=rollout_config,
                        schedule=schedule,
                        progress_label=f"{tasks_completed + 1}-{tasks_completed + len(task_batch)}/{len(epoch_tasks)}",
                    )
                    segment_rollouts.extend(batch_rollouts)
                    rollouts.extend(batch_rollouts)

                    for rollout in batch_rollouts:
                        best_decomposition = max(rollout.decompositions, key=lambda item: item.decomposition_reward)
                        selection_rewards = [
                            selection.reward.total_reward
                            for decomposition in rollout.decompositions
                            for selection in decomposition.selections
                        ]
                        selection_correctness = [
                            selection.reward.final_answer_correctness
                            for decomposition in rollout.decompositions
                            for selection in decomposition.selections
                        ]
                        running_best_selection_reward += max(selection_rewards)
                        running_best_decomposition_reward += best_decomposition.decomposition_reward
                        running_best_correctness += max(selection_correctness) if selection_correctness else 0.0

                    tasks_completed += len(batch_rollouts)
                    should_log_progress = (
                        args.rollout_progress_every > 0
                        and (
                            tasks_completed == len(epoch_tasks)
                            or tasks_completed == len(batch_rollouts)
                            or tasks_completed % args.rollout_progress_every == 0
                        )
                    )
                    if should_log_progress:
                        elapsed = time.time() - rollout_start_time
                        avg_seconds_per_task = elapsed / max(tasks_completed, 1)
                        remaining_tasks = len(epoch_tasks) - tasks_completed
                        eta_seconds = avg_seconds_per_task * remaining_tasks
                        progress_metrics = {
                            "epoch": epoch_number,
                            "batch": batch_index,
                            "num_batches": len(task_batches),
                            "tasks_completed": tasks_completed,
                            "num_epoch_tasks": len(epoch_tasks),
                            "progress_fraction": tasks_completed / max(len(epoch_tasks), 1),
                            "elapsed_s": elapsed,
                            "eta_s": eta_seconds,
                            "avg_best_selection_reward": running_best_selection_reward / tasks_completed,
                            "avg_best_decomposition_reward": running_best_decomposition_reward / tasks_completed,
                            "avg_best_final_correctness": running_best_correctness / tasks_completed,
                        }
                        print(
                            f"[hierarchical-rema][integrated] rollout_progress "
                            f"epoch={epoch_number} batch={batch_index}/{len(task_batches)} "
                            f"task={tasks_completed}/{len(epoch_tasks)} "
                            f"elapsed_s={elapsed:.1f} eta_s={eta_seconds:.1f} "
                            f"avg_best_selection_reward={progress_metrics['avg_best_selection_reward']:.4f} "
                            f"avg_best_decomposition_reward={progress_metrics['avg_best_decomposition_reward']:.4f} "
                            f"avg_best_final_correctness={progress_metrics['avg_best_final_correctness']:.4f}"
                        )
                        with rollout_progress_path.open("w", encoding="utf-8") as handle:
                            json.dump(progress_metrics, handle, indent=2, sort_keys=True)
                        if tracking is not None:
                            rollout_tracking_step += 1
                            tracking.log(
                                {
                                    "rollout_progress/tasks_completed": progress_metrics["tasks_completed"],
                                    "rollout_progress/num_tasks": progress_metrics["num_epoch_tasks"],
                                    "rollout_progress/fraction": progress_metrics["progress_fraction"],
                                    "rollout_progress/elapsed_s": progress_metrics["elapsed_s"],
                                    "rollout_progress/eta_s": progress_metrics["eta_s"],
                                    "rollout_progress/avg_best_selection_reward": progress_metrics["avg_best_selection_reward"],
                                    "rollout_progress/avg_best_decomposition_reward": progress_metrics["avg_best_decomposition_reward"],
                                    "rollout_progress/avg_best_final_correctness": progress_metrics["avg_best_final_correctness"],
                                },
                                step=rollout_tracking_step,
                            )
                segment_rollout_summary = epoch_rollout_summary(segment_rollouts)
                with (segment_train_dir / "rollout_summary.json").open("w", encoding="utf-8") as handle:
                    json.dump(segment_rollout_summary, handle, indent=2, sort_keys=True)
            finally:
                rollout_trainer.close()
                del rollout_trainer
                _release_memory()

            segment_training_updates: List[Dict[str, Any]] = []
            segment_training_skipped = None
            try:
                samples = controller_samples_from_task_rollouts(
                    task_rollouts=segment_rollouts,
                    roles=_selected_roles(args.role),
                    min_reward=args.min_reward,
                    min_advantage=args.min_advantage,
                    source_path=str(segment_rollout_dir),
                )
                policy_splits = prepare_policy_splits(
                    samples=samples,
                    policy_ids=args.policy_id,
                    val_ratio=args.val_ratio,
                    seed=args.seed + epoch_index + segment_index,
                )
            except ValueError as exc:
                segment_training_skipped = str(exc)
                training_skipped_messages.append(
                    f"segment={segment_index} batches={batch_cursor + 1}-{batch_cursor + len(segment_batches)}: {exc}"
                )
            else:
                write_policy_manifest(
                    output_dir=segment_train_dir,
                    policy_splits=policy_splits,
                    model_path_resolver=lambda policy_id, sample_subset: model_path_for_policy(policy_id, sample_subset, replay_like_args),
                )

                for policy_id, split in policy_splits.items():
                    policy_dir = segment_train_dir / policy_id
                    policy_dir.mkdir(parents=True, exist_ok=True)
                    replay_exports = maybe_save_replay_copy(policy_dir, split, enabled=args.save_replay_copy)
                    model_path = model_path_for_policy(policy_id, split["train"] or split["all"], replay_like_args)
                    experiment_name = _default_experiment_name(args)
                    experiment_name = (
                        f"{experiment_name}-epoch{epoch_number:04d}-segment{segment_index:04d}-{policy_id}"
                    )
                    print(
                        f"[hierarchical-rema][integrated] training policy={policy_id} "
                        f"segment={segment_index} train_samples={len(split['train'])} "
                        f"val_samples={len(split['val'])} model={model_path}"
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
                        seed=args.seed + epoch_index + segment_index,
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
                        save_final_checkpoint=True,
                        save_best_checkpoint=args.eval_every_steps > 0,
                        save_intermediate_checkpoints=args.checkpoint_mode == "all",
                    )
                    previous_model_path = current_paths.get(policy_id)
                    if args.offline_grpo_distributed:
                        summary = _run_distributed_offline_policy_training(
                            train_samples=split["train"],
                            val_samples=split["val"],
                            config=training_config,
                            tracking=tracking,
                            tracking_prefix=f"{policy_id}/",
                            log_step_offset=tracking_step_offset,
                            nnodes=args.offline_grpo_nnodes,
                            gpus_per_node=args.offline_grpo_gpus_per_node,
                            master_port=args.offline_grpo_master_port,
                        )
                    else:
                        previous_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
                        offline_alloc_conf = _offline_training_alloc_conf()
                        try:
                            if offline_alloc_conf is not None:
                                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = offline_alloc_conf
                            summary = run_offline_policy_training(
                                train_samples=split["train"],
                                val_samples=split["val"],
                                config=training_config,
                                tracking=tracking,
                                tracking_prefix=f"{policy_id}/",
                                log_step_offset=tracking_step_offset,
                            )
                        finally:
                            if previous_alloc_conf is None:
                                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
                            else:
                                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = previous_alloc_conf
                    tracking_step_offset += max(int(summary["steps"]), 1)
                    final_model_path = str(policy_dir / "final")
                    selected_model_path = str(summary.get("selected_model_path") or final_model_path)
                    selected_model_source = str(summary.get("selected_model_source") or "final")
                    _update_current_paths(current_paths, policy_id, selected_model_path)
                    if args.prune_stale_policy_models and previous_model_path and previous_model_path != selected_model_path:
                        previous_policy_root = Path(previous_model_path).expanduser().resolve().parent
                        if previous_policy_root != policy_dir.resolve():
                            _prune_policy_artifacts(previous_policy_root)
                    summary["policy_id"] = policy_id
                    summary["model_path"] = model_path
                    summary["final_model_path"] = final_model_path
                    summary["selected_model_path"] = selected_model_path
                    summary["selected_model_source"] = selected_model_source
                    summary["segment"] = segment_index
                    summary["batch_start"] = batch_cursor + 1
                    summary["batch_end"] = batch_cursor + len(segment_batches)
                    summary.update(replay_exports)
                    training_summaries[policy_id] = summary
                    training_updates.append(summary)
                    segment_training_updates.append(summary)
                    print(
                        f"[hierarchical-rema][integrated] finished policy={policy_id} "
                        f"segment={segment_index} steps={summary['steps']} "
                        f"selected_model_source={selected_model_source} "
                        f"selected_model_path={selected_model_path}"
                    )
                    _release_memory()

            segment_summaries.append(
                {
                    "segment": segment_index,
                    "batch_start": batch_cursor + 1,
                    "batch_end": batch_cursor + len(segment_batches),
                    "num_tasks": sum(len(batch) for batch in segment_batches),
                    "rollout_summary": segment_rollout_summary,
                    "training_updates": segment_training_updates,
                    "training_skipped": segment_training_skipped,
                }
            )
            batch_cursor += len(segment_batches)

        rollout_summary = epoch_rollout_summary(rollouts)
        tracking_step_offset = max(tracking_step_offset, rollout_tracking_step)
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

        validation_summary = None
        should_run_external_validation = False
        if val_tasks:
            validation_interval = int(args.external_validation_every_n_epochs)
            should_run_external_validation = epoch_number == args.num_epochs
            if validation_interval < 0:
                should_run_external_validation = False
            elif validation_interval > 0 and epoch_number % validation_interval == 0:
                should_run_external_validation = True
        if should_run_external_validation:
            epoch_val_tasks = val_tasks
            if args.val_tasks_per_subset > 0:
                epoch_val_tasks = select_epoch_tasks_by_subset(
                    tasks=epoch_val_tasks,
                    epoch_index=epoch_index,
                    tasks_per_subset=args.val_tasks_per_subset,
                    shuffle_tasks=False,
                    seed=args.seed,
                )
            epoch_val_tasks = select_epoch_tasks(
                tasks=epoch_val_tasks,
                epoch_index=epoch_index,
                tasks_per_epoch=args.val_tasks_per_epoch,
                shuffle_tasks=False,
                seed=args.seed,
            )
            validation_summary = run_external_validation(
                args=args,
                epoch_number=epoch_number,
                val_tasks=epoch_val_tasks,
                worker_pool=worker_pool,
                policy_config=_current_policy_config(args, current_paths),
                output_dir=epoch_dir,
                tracking=tracking,
                tracking_step=tracking_step_offset,
                base_rollout_config=base_rollout_config,
            )
        elif val_tasks:
            print(
                f"[hierarchical-rema][validation] skipped epoch={epoch_number} "
                f"interval={args.external_validation_every_n_epochs}"
            )

        epoch_summary = {
            "epoch": epoch_number,
            "schedule": {
                "mode": schedule.mode.value,
                "phase": schedule.alternating_phase.value,
            },
            "rollout_summary": rollout_summary,
            "segment_summaries": segment_summaries,
            "training_updates": training_updates,
            "training_summaries": training_summaries,
            "validation_summary": validation_summary,
            "external_validation_ran": should_run_external_validation,
            "current_policy_paths": dict(current_paths),
        }
        if training_skipped_messages:
            epoch_summary["training_skipped"] = training_skipped_messages
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

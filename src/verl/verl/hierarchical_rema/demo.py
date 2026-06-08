from __future__ import annotations

import argparse
import json

from .orchestrator import HierarchicalGRPOTrainer
from .prompts import (
    DEFAULT_ARITHMETIC_PREALGEBRA_WORKER_PROMPT,
    DEFAULT_ALGEBRA_SYMBOLIC_WORKER_PROMPT,
    DEFAULT_CALCULUS_ANALYSIS_WORKER_PROMPT,
    DEFAULT_DISCRETE_NUMBER_THEORY_WORKER_PROMPT,
    DEFAULT_GEOMETRY_TRIGONOMETRY_WORKER_PROMPT,
)
from .schema import (
    AlternatingPhase,
    ControllerPolicyConfig,
    HFBackendConfig,
    RewardWeights,
    RolloutLoggingConfig,
    RolloutConfig,
    TaskExample,
    TrainingMode,
    TrainingScheduleConfig,
    VLLMBackendConfig,
    WorkerRewardMode,
    WorkerPoolConfig,
    WorkerSpec,
)


def make_demo_tasks() -> list[TaskExample]:
    return [
        TaskExample(
            task_id="algebra_linear",
            prompt="Solve for x: 2x + 3 = 11.",
            ground_truth="4",
            metadata={
                "skill_focus": "algebra",
                "distractor_answer": "5",
            },
        ),
        TaskExample(
            task_id="analysis_derivative",
            prompt="Differentiate sin(x).",
            ground_truth="cos(x)",
            metadata={
                "skill_focus": "calculus",
                "distractor_answer": "sin(x)",
            },
        ),
    ]


def make_worker_pool(base_model_path: str | None) -> WorkerPoolConfig:
    return WorkerPoolConfig(
        base_model_path=base_model_path,
        enable_role_lora=False,
        workers=[
            WorkerSpec(
                worker_id="arithmetic_prealgebra_worker",
                description="Exact arithmetic, fractions, ratios, and simplification specialist.",
                skills=["arithmetic", "prealgebra"],
                system_prompt=DEFAULT_ARITHMETIC_PREALGEBRA_WORKER_PROMPT,
                base_model_path=base_model_path,
            ),
            WorkerSpec(
                worker_id="algebra_symbolic_worker",
                description="Equation solving and symbolic algebra specialist.",
                skills=["algebra", "equations", "polynomials", "symbolic_manipulation", "simplification"],
                system_prompt=DEFAULT_ALGEBRA_SYMBOLIC_WORKER_PROMPT,
                base_model_path=base_model_path,
            ),
            WorkerSpec(
                worker_id="geometry_trigonometry_worker",
                description="Geometry, trigonometry, and coordinate methods specialist.",
                skills=["geometry", "trigonometry", "coordinate_geometry"],
                system_prompt=DEFAULT_GEOMETRY_TRIGONOMETRY_WORKER_PROMPT,
                base_model_path=base_model_path,
            ),
            WorkerSpec(
                worker_id="calculus_analysis_worker",
                description="Calculus, limits, and function analysis specialist.",
                skills=["calculus", "analysis", "functions", "limits"],
                system_prompt=DEFAULT_CALCULUS_ANALYSIS_WORKER_PROMPT,
                base_model_path=base_model_path,
            ),
            WorkerSpec(
                worker_id="discrete_number_theory_worker",
                description="Counting, probability, and number theory specialist.",
                skills=["combinatorics", "probability", "number_theory", "discrete_math"],
                system_prompt=DEFAULT_DISCRETE_NUMBER_THEORY_WORKER_PROMPT,
                base_model_path=base_model_path,
            ),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchical ReMA MVP demo")
    parser.add_argument("--backend", choices=["mock", "hf", "vllm"], default="mock")
    parser.add_argument("--task", choices=["all", "algebra", "analysis", "calculus_analysis"], default="all")
    parser.add_argument("--mode", choices=["joint", "alternating"], default="joint")
    parser.add_argument("--phase", choices=["selector", "decomposer"], default="selector")
    parser.add_argument("--parameter-sharing", action="store_true")
    parser.add_argument("--shared-model-path", default="Qwen/Qwen2.5-1.5B-Instruct")
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
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--final-answer-correctness-reward-only",
        action="store_true",
        help="Make worker/selection reward depend only on final-answer correctness.",
    )
    parser.add_argument("--controller-max-new-tokens", type=int, default=768)
    parser.add_argument("--worker-max-new-tokens", type=int, default=256)
    parser.add_argument("--rollout-prompt-length", type=int, default=2048)
    parser.add_argument("--ray-nnodes", type=int, default=1)
    parser.add_argument("--ray-n-gpus-per-node", type=int, default=1)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=2048)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/hierarchical_rema")
    parser.add_argument("--rollout-log-mode", choices=["best", "all"], default="all")
    parser.add_argument("--rollout-log-detail", choices=["compact", "full"], default="full")
    parser.add_argument("--best-k", type=int, default=10)
    parser.add_argument("--print-mode", choices=["summary", "full", "none"], default="summary")
    parser.add_argument("--disable-rollout-logging", action="store_true")
    return parser.parse_args()


def _rollout_summary(rollout) -> dict:
    best_decomposition = max(rollout.decompositions, key=lambda item: item.decomposition_reward)
    best_selection_reward = max(
        selection.reward.total_reward
        for decomposition in rollout.decompositions
        for selection in decomposition.selections
    )
    return {
        "task_id": rollout.task.task_id,
        "num_decompositions": len(rollout.decompositions),
        "num_total_selections": sum(len(decomposition.selections) for decomposition in rollout.decompositions),
        "best_decomposition_id": best_decomposition.decomposition.decomposition_id,
        "best_decomposition_reward": best_decomposition.decomposition_reward,
        "mean_decomposition_reward": sum(
            decomposition.decomposition_reward for decomposition in rollout.decompositions
        ) / max(len(rollout.decompositions), 1),
        "best_selection_reward": best_selection_reward,
    }


def main() -> None:
    args = parse_args()
    tasks = make_demo_tasks()
    if args.task == "algebra":
        tasks = [task for task in tasks if task.metadata["skill_focus"] == "algebra"]
    elif args.task in {"analysis", "calculus_analysis"}:
        tasks = [task for task in tasks if task.metadata["skill_focus"] == "calculus"]

    schedule = TrainingScheduleConfig(
        mode=TrainingMode(args.mode),
        alternating_phase=AlternatingPhase(args.phase),
    )
    policy_config = ControllerPolicyConfig(
        parameter_sharing=args.parameter_sharing,
        shared_model_path=args.shared_model_path,
        decomposer_model_path=args.decomposer_model_path,
        selector_model_path=args.selector_model_path,
    )
    max_nodes_per_decomposition = (
        args.max_nodes_per_decomposition
        if args.max_nodes_per_decomposition is not None
        else (args.hard_max_hops if args.hard_max_hops is not None else 4)
    )
    rollout_config = RolloutConfig(
        num_decompositions=args.num_decompositions,
        num_selections_per_decomposition=args.num_selections,
        max_nodes_per_decomposition=max_nodes_per_decomposition,
        soft_max_hops=args.soft_max_hops,
        hard_max_hops=args.hard_max_hops,
        soft_hop_penalty=args.soft_hop_penalty,
    )
    worker_pool = make_worker_pool(base_model_path=args.worker_base_model_path)

    logging_config = None
    if not args.disable_rollout_logging:
        logging_config = RolloutLoggingConfig(
            output_dir=args.output_dir,
            save_all_rollouts=args.rollout_log_mode == "all",
            save_best_rollouts=True,
            best_k=args.best_k,
            compact_mode=args.rollout_log_detail == "compact",
        )

    trainer = HierarchicalGRPOTrainer(
        reward_weights=RewardWeights(
            worker_reward_mode=(
                WorkerRewardMode.FINAL_ANSWER_CORRECTNESS_ONLY
                if args.final_answer_correctness_reward_only
                else WorkerRewardMode.CURRENT
            )
        ),
        backend_type=args.backend,
        hf_backend_config=HFBackendConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            controller_max_new_tokens=args.controller_max_new_tokens,
            worker_max_new_tokens=args.worker_max_new_tokens,
        ),
        vllm_backend_config=VLLMBackendConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0.0,
            prompt_length=args.rollout_prompt_length,
            controller_max_new_tokens=args.controller_max_new_tokens,
            worker_max_new_tokens=args.worker_max_new_tokens,
            nnodes=args.ray_nnodes,
            n_gpus_per_node=args.ray_n_gpus_per_node,
            tensor_model_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            max_num_seqs=args.vllm_max_num_seqs,
            max_model_len=args.vllm_max_model_len,
            trust_remote_code=True,
        ),
        rollout_logging_config=logging_config,
    )
    try:
        rollouts = [
            trainer.run(
                task=task,
                worker_pool=worker_pool,
                policy_config=policy_config,
                rollout_config=rollout_config,
                schedule=schedule,
            )
            for task in tasks
        ]
        if args.print_mode == "none":
            return

        if args.print_mode == "full":
            print(json.dumps([rollout.to_dict() for rollout in rollouts], indent=2, sort_keys=True))
            return

        summary = {
            "backend": args.backend,
            "mode": args.mode,
            "phase": args.phase,
            "num_tasks": len(rollouts),
            "tasks": [_rollout_summary(rollout) for rollout in rollouts],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        trainer.close()


if __name__ == "__main__":
    main()

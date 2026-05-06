from __future__ import annotations

import argparse
import json

from .orchestrator import HierarchicalGRPOTrainer
from .prompts import DEFAULT_ALGEBRA_WORKER_PROMPT, DEFAULT_ANALYSIS_WORKER_PROMPT
from .schema import (
    AlternatingPhase,
    ControllerPolicyConfig,
    HFBackendConfig,
    RolloutLoggingConfig,
    RolloutConfig,
    TaskExample,
    TrainingMode,
    TrainingScheduleConfig,
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
                "skill_focus": "analysis",
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
                worker_id="algebra_worker",
                description="Exact symbolic manipulation specialist.",
                skills=["algebra", "symbolic_manipulation"],
                system_prompt=DEFAULT_ALGEBRA_WORKER_PROMPT,
                base_model_path=base_model_path,
            ),
            WorkerSpec(
                worker_id="analysis_worker",
                description="Calculus and theorem-driven analysis specialist.",
                skills=["analysis", "calculus"],
                system_prompt=DEFAULT_ANALYSIS_WORKER_PROMPT,
                base_model_path=base_model_path,
            ),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchical ReMA MVP demo")
    parser.add_argument("--backend", choices=["mock", "hf"], default="mock")
    parser.add_argument("--task", choices=["all", "algebra", "analysis"], default="all")
    parser.add_argument("--mode", choices=["joint", "alternating"], default="joint")
    parser.add_argument("--phase", choices=["selector", "decomposer"], default="selector")
    parser.add_argument("--parameter-sharing", action="store_true")
    parser.add_argument("--shared-model-path", default="Qwen/Qwen2.5-1.5B-Instruct")
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
    parser.add_argument("--output-dir", default="outputs/hierarchical_rema")
    parser.add_argument("--best-k", type=int, default=10)
    parser.add_argument("--disable-rollout-logging", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = make_demo_tasks()
    if args.task == "algebra":
        tasks = [task for task in tasks if task.metadata["skill_focus"] == "algebra"]
    elif args.task == "analysis":
        tasks = [task for task in tasks if task.metadata["skill_focus"] == "analysis"]

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
    rollout_config = RolloutConfig(
        num_decompositions=args.num_decompositions,
        num_selections_per_decomposition=args.num_selections,
        soft_max_hops=args.soft_max_hops,
        hard_max_hops=args.hard_max_hops,
        soft_hop_penalty=args.soft_hop_penalty,
    )
    worker_pool = make_worker_pool(base_model_path=args.worker_base_model_path)

    logging_config = None
    if not args.disable_rollout_logging:
        logging_config = RolloutLoggingConfig(
            output_dir=args.output_dir,
            best_k=args.best_k,
        )

    trainer = HierarchicalGRPOTrainer(
        backend_type=args.backend,
        hf_backend_config=HFBackendConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            controller_max_new_tokens=args.controller_max_new_tokens,
            worker_max_new_tokens=args.worker_max_new_tokens,
        ),
        rollout_logging_config=logging_config,
    )
    results = [
        trainer.run(
            task=task,
            worker_pool=worker_pool,
            policy_config=policy_config,
            rollout_config=rollout_config,
            schedule=schedule,
        ).to_dict()
        for task in tasks
    ]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

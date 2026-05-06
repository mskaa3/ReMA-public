import json

try:
    from verl.hierarchical_rema import (
        AlternatingPhase,
        ControllerPolicyConfig,
        HierarchicalGRPOTrainer,
        RolloutLoggingConfig,
        RolloutConfig,
        TaskExample,
        TrainingMode,
        TrainingScheduleConfig,
        WorkerPoolConfig,
        WorkerSpec,
    )
    from verl.hierarchical_rema.prompts import (
        DEFAULT_ALGEBRA_WORKER_PROMPT,
        DEFAULT_ANALYSIS_WORKER_PROMPT,
    )
except ModuleNotFoundError:
    from hierarchical_rema import (
        AlternatingPhase,
        ControllerPolicyConfig,
        HierarchicalGRPOTrainer,
        RolloutLoggingConfig,
        RolloutConfig,
        TaskExample,
        TrainingMode,
        TrainingScheduleConfig,
        WorkerPoolConfig,
        WorkerSpec,
    )
    from hierarchical_rema.prompts import (
        DEFAULT_ALGEBRA_WORKER_PROMPT,
        DEFAULT_ANALYSIS_WORKER_PROMPT,
    )


def make_worker_pool() -> WorkerPoolConfig:
    return WorkerPoolConfig(
        base_model_path="mock-model",
        enable_role_lora=False,
        workers=[
            WorkerSpec(
                worker_id="algebra_worker",
                description="Exact symbolic manipulation specialist.",
                skills=["algebra", "symbolic_manipulation"],
                system_prompt=DEFAULT_ALGEBRA_WORKER_PROMPT,
            ),
            WorkerSpec(
                worker_id="analysis_worker",
                description="Calculus and theorem-driven analysis specialist.",
                skills=["analysis", "calculus"],
                system_prompt=DEFAULT_ANALYSIS_WORKER_PROMPT,
            ),
        ],
    )


def make_task(skill_focus: str, prompt: str, ground_truth: str, distractor: str) -> TaskExample:
    return TaskExample(
        task_id=f"{skill_focus}-task",
        prompt=prompt,
        ground_truth=ground_truth,
        metadata={
            "skill_focus": skill_focus,
            "distractor_answer": distractor,
        },
    )


def test_joint_rollout_builds_selector_and_decomposer_groups() -> None:
    trainer = HierarchicalGRPOTrainer()
    task = make_task("algebra", "Solve for x: 2x + 3 = 11.", "4", "5")
    rollout = trainer.run(
        task=task,
        worker_pool=make_worker_pool(),
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        rollout_config=RolloutConfig(num_decompositions=3, num_selections_per_decomposition=2),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
    )

    assert len(rollout.decompositions) == 3
    assert all(len(decomposition.selections) == 2 for decomposition in rollout.decompositions)
    assert len(rollout.training_batch.decomposer_samples) == 3
    assert len(rollout.training_batch.selector_samples) == 6
    assert rollout.training_batch.frozen_roles == []


def test_dag_execution_uses_selected_worker_and_tracks_final_answer() -> None:
    trainer = HierarchicalGRPOTrainer()
    task = make_task("analysis", "Differentiate sin(x).", "cos(x)", "sin(x)")
    rollout = trainer.run(
        task=task,
        worker_pool=make_worker_pool(),
        policy_config=ControllerPolicyConfig(parameter_sharing=True, shared_model_path="mock-shared"),
        rollout_config=RolloutConfig(num_decompositions=1, num_selections_per_decomposition=2),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
    )

    decomposition = rollout.decompositions[0]
    strong_selection = decomposition.selections[0]
    weak_selection = decomposition.selections[1]

    assert strong_selection.final_answer == "cos(x)"
    assert strong_selection.reward.final_answer_correctness == 1.0
    assert weak_selection.reward.total_reward <= strong_selection.reward.total_reward
    assert all(execution.worker_id for execution in strong_selection.executions)


def test_alternating_selector_phase_freezes_decomposer() -> None:
    trainer = HierarchicalGRPOTrainer()
    task = make_task("algebra", "Solve for x: 2x + 3 = 11.", "4", "5")
    rollout = trainer.run(
        task=task,
        worker_pool=make_worker_pool(),
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        rollout_config=RolloutConfig(num_decompositions=4, num_selections_per_decomposition=3),
        schedule=TrainingScheduleConfig(
            mode=TrainingMode.ALTERNATING,
            alternating_phase=AlternatingPhase.SELECTOR,
        ),
    )

    assert len(rollout.decompositions) == 1
    assert len(rollout.decompositions[0].selections) == 3
    assert len(rollout.training_batch.decomposer_samples) == 0
    assert len(rollout.training_batch.selector_samples) == 3
    assert rollout.training_batch.frozen_roles == ["decomposer"]


def test_alternating_decomposer_phase_freezes_selector() -> None:
    trainer = HierarchicalGRPOTrainer()
    task = make_task("analysis", "Differentiate sin(x).", "cos(x)", "sin(x)")
    rollout = trainer.run(
        task=task,
        worker_pool=make_worker_pool(),
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        rollout_config=RolloutConfig(num_decompositions=4, num_selections_per_decomposition=3),
        schedule=TrainingScheduleConfig(
            mode=TrainingMode.ALTERNATING,
            alternating_phase=AlternatingPhase.DECOMPOSER,
        ),
    )

    assert len(rollout.decompositions) == 4
    assert all(len(decomposition.selections) == 1 for decomposition in rollout.decompositions)
    assert len(rollout.training_batch.decomposer_samples) == 4
    assert len(rollout.training_batch.selector_samples) == 0
    assert rollout.training_batch.frozen_roles == ["selector"]


def test_controller_prompts_include_worker_performance_history() -> None:
    trainer = HierarchicalGRPOTrainer()
    worker_pool = make_worker_pool()
    first_task = make_task("algebra", "Solve for x: 2x + 3 = 11.", "4", "5")
    second_task = make_task("analysis", "Differentiate sin(x).", "cos(x)", "sin(x)")

    trainer.run(
        task=first_task,
        worker_pool=worker_pool,
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        rollout_config=RolloutConfig(num_decompositions=1, num_selections_per_decomposition=1),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
    )
    second_rollout = trainer.run(
        task=second_task,
        worker_pool=worker_pool,
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        rollout_config=RolloutConfig(num_decompositions=1, num_selections_per_decomposition=1),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
    )

    prompt_text = second_rollout.training_batch.decomposer_samples[0].prompt_text
    prompt_payload = json.loads(prompt_text[prompt_text.index("{"):])
    worker_summaries = {
        worker["worker_id"]: worker["performance_summary"]
        for worker in prompt_payload["available_workers"]
    }

    assert "completion_rate" in worker_summaries["algebra_worker"]
    assert worker_summaries["algebra_worker"]["num_assignments"] > 0
    assert "recent_history" in worker_summaries["algebra_worker"]


def test_soft_hop_penalty_and_hard_hop_truncation_are_applied() -> None:
    trainer = HierarchicalGRPOTrainer()
    task = make_task("algebra", "Solve for x: 2x + 3 = 11.", "4", "5")
    rollout = trainer.run(
        task=task,
        worker_pool=make_worker_pool(),
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        rollout_config=RolloutConfig(
            num_decompositions=2,
            num_selections_per_decomposition=1,
            soft_max_hops=1,
            hard_max_hops=2,
            soft_hop_penalty=0.2,
        ),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
    )

    truncated_decomposition = rollout.decompositions[1]
    assert truncated_decomposition.decomposition.soft_penalty > 0.0
    assert truncated_decomposition.decomposition.was_hard_truncated is True
    assert truncated_decomposition.decomposition.effective_num_hops <= 2
    assert truncated_decomposition.decomposition_reward == (
        truncated_decomposition.base_decomposition_reward
        - truncated_decomposition.decomposition.soft_penalty
    )


def test_rollout_recorder_writes_jsonl_files(tmp_path) -> None:
    trainer = HierarchicalGRPOTrainer(
        rollout_logging_config=RolloutLoggingConfig(
            output_dir=str(tmp_path),
            save_all_rollouts=True,
            save_best_rollouts=True,
            best_k=2,
        )
    )
    task = make_task("analysis", "Differentiate sin(x).", "cos(x)", "sin(x)")
    trainer.run(
        task=task,
        worker_pool=make_worker_pool(),
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        rollout_config=RolloutConfig(num_decompositions=2, num_selections_per_decomposition=2),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
    )

    assert (tmp_path / "all_rollouts.jsonl").exists()
    assert (tmp_path / "best_decompositions.jsonl").exists()
    assert (tmp_path / "best_selections.jsonl").exists()

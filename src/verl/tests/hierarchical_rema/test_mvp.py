import json
import pytest

try:
    from verl.hierarchical_rema.backends import _postprocess_worker_output
    from verl.hierarchical_rema import (
        AlternatingPhase,
        ControllerPolicyConfig,
        ControllerTrainingSample,
        DecompositionCandidate,
        DecompositionRollout,
        HierarchicalTrainingBatch,
        HierarchicalGRPOTrainer,
        RewardWeights,
        RolloutLoggingConfig,
        RolloutConfig,
        SelectionCandidate,
        SelectionRewardBreakdown,
        SelectionRollout,
        SubtaskNode,
        TaskExample,
        TaskRollout,
        TrainingMode,
        TrainingScheduleConfig,
        REDACTED_FINAL_ANSWER_LEAK_OUTPUT,
        WorkerExecution,
        WorkerRewardMode,
        WorkerPoolConfig,
        WorkerSpec,
    )
    from verl.hierarchical_rema.controller_data import controller_samples_from_task_rollouts
    from verl.hierarchical_rema.demo import make_worker_pool as make_default_worker_pool
    from verl.hierarchical_rema.prompts import (
        render_decomposer_prompt,
        render_selector_prompt,
        render_worker_prompt,
    )
    from verl.hierarchical_rema.rewarding import build_selection_reward
    from verl.hierarchical_rema.structured import (
        StructuredOutputError,
        apply_decomposition_limits,
        build_fallback_decomposition,
        extract_decomposition_payload,
        extract_json_dict,
        extract_worker_result_text,
        extract_selection_payload,
        validate_decomposition_payload,
    )
except ModuleNotFoundError:
    from hierarchical_rema.backends import _postprocess_worker_output
    from hierarchical_rema import (
        AlternatingPhase,
        ControllerPolicyConfig,
        ControllerTrainingSample,
        DecompositionCandidate,
        DecompositionRollout,
        HierarchicalTrainingBatch,
        HierarchicalGRPOTrainer,
        RewardWeights,
        RolloutLoggingConfig,
        RolloutConfig,
        SelectionCandidate,
        SelectionRewardBreakdown,
        SelectionRollout,
        SubtaskNode,
        TaskExample,
        TaskRollout,
        TrainingMode,
        TrainingScheduleConfig,
        REDACTED_FINAL_ANSWER_LEAK_OUTPUT,
        WorkerExecution,
        WorkerRewardMode,
        WorkerPoolConfig,
        WorkerSpec,
    )
    from hierarchical_rema.controller_data import controller_samples_from_task_rollouts
    from hierarchical_rema.demo import make_worker_pool as make_default_worker_pool
    from hierarchical_rema.prompts import (
        render_decomposer_prompt,
        render_selector_prompt,
        render_worker_prompt,
    )
    from hierarchical_rema.rewarding import build_selection_reward
    from hierarchical_rema.structured import (
        StructuredOutputError,
        apply_decomposition_limits,
        build_fallback_decomposition,
        extract_decomposition_payload,
        extract_json_dict,
        extract_worker_result_text,
        extract_selection_payload,
        validate_decomposition_payload,
    )


def make_worker_pool() -> WorkerPoolConfig:
    return make_default_worker_pool(base_model_path="mock-model")


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


def test_run_many_builds_rollouts_for_multiple_tasks() -> None:
    trainer = HierarchicalGRPOTrainer()
    tasks = [
        make_task("algebra", "Solve for x: 2x + 3 = 11.", "4", "5"),
        make_task("analysis", "Differentiate sin(x).", "cos(x)", "sin(x)"),
    ]
    rollouts = trainer.run_many(
        tasks=tasks,
        worker_pool=make_worker_pool(),
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        rollout_config=RolloutConfig(num_decompositions=2, num_selections_per_decomposition=2),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
    )

    assert len(rollouts) == 2
    assert [rollout.task.task_id for rollout in rollouts] == [task.task_id for task in tasks]
    assert all(len(rollout.decompositions) == 2 for rollout in rollouts)
    assert all(len(rollout.training_batch.decomposer_samples) == 2 for rollout in rollouts)
    assert all(len(rollout.training_batch.selector_samples) == 4 for rollout in rollouts)


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


def test_final_answer_correctness_only_reward_mode_clamps_worker_reward() -> None:
    trainer = HierarchicalGRPOTrainer(
        reward_weights=RewardWeights(
            worker_reward_mode=WorkerRewardMode.FINAL_ANSWER_CORRECTNESS_ONLY
        )
    )
    task = make_task("analysis", "Differentiate sin(x).", "cos(x)", "sin(x)")
    rollout = trainer.run(
        task=task,
        worker_pool=make_worker_pool(),
        policy_config=ControllerPolicyConfig(parameter_sharing=True, shared_model_path="mock-shared"),
        rollout_config=RolloutConfig(num_decompositions=1, num_selections_per_decomposition=2),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
    )

    strong_selection = rollout.decompositions[0].selections[0]
    weak_selection = rollout.decompositions[0].selections[1]

    assert strong_selection.reward.total_reward == 1.0
    assert strong_selection.reward.total_reward == strong_selection.reward.final_answer_correctness
    assert weak_selection.reward.total_reward == weak_selection.reward.final_answer_correctness


def test_fallback_paths_do_not_keep_positive_controller_reward_by_default() -> None:
    trainer = HierarchicalGRPOTrainer(
        controller_format_fallback_penalty=0.25,
        controller_fallback_positive_reward_scale=0.0,
    )
    fallback_payload = {"validation": {"fallback_used": True}}
    clean_selector_payload = {"validation": {"fallback_used": False}}

    assert trainer.orchestrator._format_adjusted_reward(
        1.0,
        fallback_payload,
        role="decomposer",
    ) == -0.25
    assert trainer.orchestrator._format_adjusted_reward(
        1.0,
        clean_selector_payload,
        role="selector",
        upstream_payloads=(fallback_payload,),
    ) == 0.0
    assert trainer.orchestrator._format_adjusted_reward(
        -0.4,
        clean_selector_payload,
        role="selector",
        upstream_payloads=(fallback_payload,),
    ) == -0.4


def test_worker_training_skips_samples_from_fallback_decompositions() -> None:
    trainer = HierarchicalGRPOTrainer(train_worker_model=True)
    task = make_task("algebra", "Solve for x: 2x + 3 = 11.", "4", "5")
    worker_pool = make_worker_pool()
    rollout_config = RolloutConfig()
    fallback_decomposition = build_fallback_decomposition(
        task_id=task.task_id,
        task_prompt=task.prompt,
        raw_text="bad decomposition",
        error_message="forced fallback for test",
        rollout_config=rollout_config,
    )
    fallback_decomposition.raw_payload["controller_prompt"] = "Decompose the task."

    selection_rollout = SelectionRollout(
        selection=SelectionCandidate(selection_id="sel-1", assignments=[]),
        executions=[
            WorkerExecution(
                node_id="1",
                worker_id=worker_pool.workers[0].worker_id,
                output_text="4",
                raw_output_text="4",
                worker_prompt="Solve the task directly.",
                entropy=0.0,
                confidence_reward=0.0,
                compatibility=0.0,
            )
        ],
        final_answer="4",
        reward=SelectionRewardBreakdown(
            final_answer_correctness=1.0,
            confidence_reward=0.0,
            compatibility_reward=0.0,
            total_reward=1.0,
        ),
    )
    selection_rollout.selection.raw_payload["controller_prompt"] = "Assign workers."
    decomposition_rollout = DecompositionRollout(
        decomposition=fallback_decomposition,
        selections=[selection_rollout],
        base_decomposition_reward=1.0,
        decomposition_reward=1.0,
    )

    training_batch = trainer.orchestrator._build_training_batch(
        task=task,
        worker_pool=worker_pool,
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        schedule=TrainingScheduleConfig(
            mode=TrainingMode.JOINT,
        ),
        decompositions=[decomposition_rollout],
    )

    assert training_batch.worker_samples == []
    assert training_batch.worker_grpo_stats["num_decompositions_skipped_fallback"] == 1
    assert training_batch.worker_grpo_stats["num_worker_samples_skipped_fallback"] == 1


def test_worker_grpo_groups_do_not_cross_decomposition_boundaries() -> None:
    trainer = HierarchicalGRPOTrainer(
        train_worker_model=True,
        min_worker_grpo_group_size=1,
    )
    task = make_task("algebra", "Solve for x: 2x + 3 = 11.", "4", "5")
    worker_pool = make_worker_pool()
    worker_id = worker_pool.workers[0].worker_id

    def make_decomposition_rollout(
        decomposition_id: str,
        selection_id: str,
        raw_output_text: str,
        reward: float,
    ) -> DecompositionRollout:
        decomposition = DecompositionCandidate(
            decomposition_id=decomposition_id,
            summary="one-step solve",
            target_quantity="final answer",
            final_answer_format_hint="integer",
            nodes=[
                SubtaskNode(
                    node_id="1",
                    instruction="Solve the equation.",
                    output_key="final_answer",
                )
            ],
            final_node_id="1",
        )
        selection_rollout = SelectionRollout(
            selection=SelectionCandidate(selection_id=selection_id, assignments=[]),
            executions=[
                WorkerExecution(
                    node_id="1",
                    worker_id=worker_id,
                    output_text=raw_output_text,
                    raw_output_text=raw_output_text,
                    worker_prompt="Solve the equation.",
                    entropy=0.0,
                    confidence_reward=0.0,
                    compatibility=1.0,
                )
            ],
            final_answer=raw_output_text,
            reward=SelectionRewardBreakdown(
                final_answer_correctness=reward,
                confidence_reward=0.0,
                compatibility_reward=0.0,
                total_reward=reward,
            ),
        )
        decomposition.raw_payload["controller_prompt"] = "Decompose the task."
        selection_rollout.selection.raw_payload["controller_prompt"] = "Assign workers."
        return DecompositionRollout(
            decomposition=decomposition,
            selections=[selection_rollout],
            base_decomposition_reward=reward,
            decomposition_reward=reward,
        )

    training_batch = trainer.orchestrator._build_training_batch(
        task=task,
        worker_pool=worker_pool,
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
        decompositions=[
            make_decomposition_rollout("decomp-1", "sel-1", "4", 1.0),
            make_decomposition_rollout("decomp-2", "sel-2", "5", 0.0),
        ],
    )

    assert len(training_batch.worker_samples) == 2
    group_ids = {sample.group_id for sample in training_batch.worker_samples}
    assert len(group_ids) == 2
    assert any(
        group_id.startswith(f"task:{task.task_id}:decomposition:decomp-1:worker:{worker_id}:instr:")
        for group_id in group_ids
    )
    assert any(
        group_id.startswith(f"task:{task.task_id}:decomposition:decomp-2:worker:{worker_id}:instr:")
        for group_id in group_ids
    )
    assert {sample.metadata["decomposition_id"] for sample in training_batch.worker_samples} == {
        "decomp-1",
        "decomp-2",
    }
    assert all(sample.metadata["advantage_group_size"] == 1 for sample in training_batch.worker_samples)
    assert all(
        sample.metadata["advantage_group_kind"]
        == "worker_id_and_normalized_instruction_within_task_and_decomposition"
        for sample in training_batch.worker_samples
    )


def test_fallback_decomposition_receives_single_node_triviality_penalty() -> None:
    rollout_config = RolloutConfig(
        soft_hop_penalty=0.1,
        trivial_single_node_penalty=1.0,
    )

    fallback_decomposition = build_fallback_decomposition(
        task_id="task-1",
        task_prompt="Solve for x: 2x + 3 = 11.",
        raw_text="malformed output",
        error_message="format fallback",
        rollout_config=rollout_config,
    )

    assert len(fallback_decomposition.nodes) == 1
    assert fallback_decomposition.final_node_id == "1"
    assert fallback_decomposition.soft_penalty == 1.2


def test_shallow_two_node_plan_receives_triviality_penalty() -> None:
    rollout_config = RolloutConfig(
        soft_hop_penalty=0.1,
        trivial_shallow_two_node_penalty=0.5,
    )
    candidate = DecompositionCandidate(
        decomposition_id="decomp-1",
        summary="too shallow",
        target_quantity="best final answer requested by TASK",
        final_answer_format_hint="match the answer format requested by TASK",
        nodes=[
            SubtaskNode(
                node_id="1",
                instruction="Identify the relevant quantities.",
                output_key="quantities",
            ),
            SubtaskNode(
                node_id="2",
                instruction="Return the final answer.",
                dependencies=["1"],
                output_key="final_answer",
            ),
        ],
        final_node_id="2",
    )

    limited_candidate = apply_decomposition_limits(candidate, rollout_config)

    assert limited_candidate.soft_penalty == 0.6


def test_generic_simplify_wrapper_two_node_plan_receives_triviality_penalty() -> None:
    rollout_config = RolloutConfig(
        soft_hop_penalty=0.1,
        trivial_shallow_two_node_penalty=0.5,
    )
    candidate = DecompositionCandidate(
        decomposition_id="decomp-1",
        summary="wrapper plan",
        target_quantity="matrix M",
        final_answer_format_hint="matrix",
        nodes=[
            SubtaskNode(
                node_id="1",
                instruction="Solve for the entries of the matrix.",
                output_key="1_output",
            ),
            SubtaskNode(
                node_id="2",
                instruction="Simplify the expression.",
                dependencies=["1"],
                output_key="final_answer",
            ),
        ],
        final_node_id="2",
    )

    limited_candidate = apply_decomposition_limits(candidate, rollout_config)

    assert limited_candidate.soft_penalty == 0.6


def test_node_count_target_penalty_applies_outside_preferred_band() -> None:
    rollout_config = RolloutConfig(
        preferred_node_count_min=3,
        preferred_node_count_max=5,
        node_count_target_penalty_per_step=0.1,
        node_count_target_max_penalty=0.4,
    )
    candidate = DecompositionCandidate(
        decomposition_id="decomp-1",
        summary="too many nodes",
        target_quantity="answer",
        final_answer_format_hint="integer",
        nodes=[
            SubtaskNode(node_id="1", instruction="Compute a.", output_key="1_output"),
            SubtaskNode(node_id="2", instruction="Compute b.", dependencies=["1"], output_key="2_output"),
            SubtaskNode(node_id="3", instruction="Compute c.", dependencies=["2"], output_key="3_output"),
            SubtaskNode(node_id="4", instruction="Compute d.", dependencies=["3"], output_key="4_output"),
            SubtaskNode(node_id="5", instruction="Compute e.", dependencies=["4"], output_key="5_output"),
            SubtaskNode(node_id="6", instruction="Return the final answer.", dependencies=["5"], output_key="final_answer"),
        ],
        final_node_id="6",
    )

    limited_candidate = apply_decomposition_limits(candidate, rollout_config)

    assert limited_candidate.soft_penalty == 0.1


def test_selection_reward_adds_hierarchy_usage_bonuses() -> None:
    executions = [
        WorkerExecution(
            node_id="1",
            worker_id="symbolic_manipulation_worker",
            output_text="a = 2",
            raw_output_text="a = 2",
            entropy=0.1,
            confidence_reward=0.0,
            compatibility=1.0,
        ),
        WorkerExecution(
            node_id="2",
            worker_id="calculation_worker",
            output_text="a = 2, so b = 3",
            raw_output_text="a = 2, so b = 3",
            dependency_outputs={"1": "a = 2"},
            entropy=0.1,
            confidence_reward=0.0,
            compatibility=1.0,
        ),
        WorkerExecution(
            node_id="3",
            worker_id="logic_constraints_worker",
            output_text="Using a = 2 and a = 2, so b = 3.\nThe answer is 5",
            raw_output_text="Using a = 2 and a = 2, so b = 3.\nThe answer is 5",
            dependency_outputs={"1": "a = 2", "2": "a = 2, so b = 3"},
            entropy=0.1,
            confidence_reward=0.0,
            compatibility=1.0,
        ),
    ]

    reward = build_selection_reward(
        final_answer="Using a = 2 and a = 2, so b = 3.\nThe answer is 5",
        ground_truth="5",
        executions=executions,
        weights=RewardWeights(),
        final_node_id="3",
    )

    assert reward.positive_bonus_gate == 1.0
    assert reward.hierarchy_utilization_gate == 1.0
    assert reward.dependency_usage_rate == 1.0
    assert reward.final_dependency_usage_rate == 1.0
    assert reward.worker_unique_result_bonus > 0.0
    assert reward.worker_downstream_used_bonus > 0.0
    assert reward.final_stage_usage_bonus > 0.0
    assert reward.final_ignores_hierarchy_penalty == 0.0
    assert executions[0].downstream_used is True
    assert executions[1].dependency_used is True


def test_selection_reward_penalizes_final_answer_that_ignores_dependencies() -> None:
    executions = [
        WorkerExecution(
            node_id="1",
            worker_id="symbolic_manipulation_worker",
            output_text="a = 2",
            raw_output_text="a = 2",
            entropy=0.1,
            confidence_reward=0.0,
            compatibility=1.0,
        ),
        WorkerExecution(
            node_id="2",
            worker_id="calculation_worker",
            output_text="a = 2, so b = 3",
            raw_output_text="a = 2, so b = 3",
            dependency_outputs={"1": "a = 2"},
            entropy=0.1,
            confidence_reward=0.0,
            compatibility=1.0,
        ),
        WorkerExecution(
            node_id="3",
            worker_id="logic_constraints_worker",
            output_text="5",
            raw_output_text="5",
            dependency_outputs={"1": "a = 2", "2": "a = 2, so b = 3"},
            entropy=0.1,
            confidence_reward=0.0,
            compatibility=1.0,
        ),
    ]

    reward = build_selection_reward(
        final_answer="5",
        ground_truth="5",
        executions=executions,
        weights=RewardWeights(),
        final_node_id="3",
    )

    assert reward.hierarchy_utilization_gate == 0.0
    assert reward.final_dependency_usage_rate == 0.0
    assert reward.final_raw_score_usage_multiplier == 0.5
    assert reward.final_ignores_hierarchy_penalty > 0.0
    assert reward.worker_unique_result_bonus == 0.0
    assert reward.worker_downstream_used_bonus == 0.0
    assert reward.final_stage_usage_bonus == 0.0


def test_decomposer_reward_gets_dependency_usage_bonus_only_for_gated_correct_selections() -> None:
    trainer = HierarchicalGRPOTrainer(
        reward_weights=RewardWeights(decomposer_dependency_usage_bonus=0.05),
    )
    gated_selection = SelectionRollout(
        selection=SelectionCandidate(selection_id="sel-1", assignments=[]),
        executions=[],
        final_answer="5",
        reward=SelectionRewardBreakdown(
            final_answer_correctness=1.0,
            confidence_reward=0.0,
            compatibility_reward=0.0,
            total_reward=1.0,
            positive_bonus_gate=1.0,
            hierarchy_utilization_gate=1.0,
            dependency_usage_rate=1.0,
        ),
    )
    ungated_selection = SelectionRollout(
        selection=SelectionCandidate(selection_id="sel-2", assignments=[]),
        executions=[],
        final_answer="4",
        reward=SelectionRewardBreakdown(
            final_answer_correctness=0.0,
            confidence_reward=0.0,
            compatibility_reward=0.0,
            total_reward=0.5,
            positive_bonus_gate=0.0,
            hierarchy_utilization_gate=1.0,
            dependency_usage_rate=1.0,
        ),
    )

    reward = trainer.orchestrator._aggregate_decomposition_selection_reward(
        selection_rollouts=[gated_selection, ungated_selection],
        selection_training_rewards=[1.0, 0.5],
    )

    assert reward == pytest.approx(1.05)


def test_validate_decomposition_rejects_unused_nodes_outside_path_to_final() -> None:
    payload = {
        "decomposition_id": "decomp-1",
        "summary": "invalid disconnected plan",
        "target_quantity": "answer",
        "final_answer_format_hint": "integer",
        "final_node_id": "3",
        "nodes": [
            {
                "node_id": "1",
                "instruction": "Solve the task.",
                "dependencies": [],
            },
            {
                "node_id": "2",
                "instruction": "Unused detour.",
                "dependencies": [],
            },
            {
                "node_id": "3",
                "instruction": "Return the final answer.",
                "dependencies": ["1"],
            },
        ],
    }

    with pytest.raises(StructuredOutputError, match="unused nodes"):
        validate_decomposition_payload(
            payload=payload,
            rollout_config=RolloutConfig(),
            fallback_id="decomp-1",
        )


def test_validate_decomposition_rejects_final_node_without_dependencies_when_multinode() -> None:
    payload = {
        "decomposition_id": "decomp-1",
        "summary": "invalid wrapper plan",
        "target_quantity": "answer",
        "final_answer_format_hint": "integer",
        "final_node_id": "2",
        "nodes": [
            {
                "node_id": "1",
                "instruction": "Analyze the structure.",
                "dependencies": [],
            },
            {
                "node_id": "2",
                "instruction": "Return the final answer.",
                "dependencies": [],
            },
        ],
    }

    with pytest.raises(StructuredOutputError, match="must depend on earlier nodes"):
        validate_decomposition_payload(
            payload=payload,
            rollout_config=RolloutConfig(),
            fallback_id="decomp-1",
        )


def test_alternating_selector_phase_freezes_decomposer() -> None:
    trainer = HierarchicalGRPOTrainer(
        train_worker_model=True,
        min_worker_grpo_group_size=1,
    )
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

    assert len(rollout.decompositions) == 4
    assert len(rollout.decompositions[0].selections) == 3
    assert len(rollout.training_batch.decomposer_samples) == 0
    assert len(rollout.training_batch.selector_samples) == 12
    assert len(rollout.training_batch.worker_samples) > 0
    assert rollout.training_batch.frozen_roles == ["decomposer"]


def test_alternating_decomposer_phase_freezes_selector() -> None:
    trainer = HierarchicalGRPOTrainer(
        train_worker_model=True,
        min_worker_grpo_group_size=1,
    )
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
    assert all(len(decomposition.selections) == 2 for decomposition in rollout.decompositions)
    assert len(rollout.training_batch.decomposer_samples) == 4
    assert len(rollout.training_batch.selector_samples) == 0
    assert rollout.training_batch.worker_samples == []
    assert rollout.training_batch.frozen_roles == ["selector"]


def test_alternating_rollout_counts_use_wider_phase_specific_caps() -> None:
    trainer = HierarchicalGRPOTrainer(
        train_worker_model=True,
        min_worker_grpo_group_size=1,
    )
    task = make_task("algebra", "Solve for x: 2x + 3 = 11.", "4", "5")

    selector_rollout = trainer.run(
        task=task,
        worker_pool=make_worker_pool(),
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        rollout_config=RolloutConfig(num_decompositions=16, num_selections_per_decomposition=16),
        schedule=TrainingScheduleConfig(
            mode=TrainingMode.ALTERNATING,
            alternating_phase=AlternatingPhase.SELECTOR,
        ),
    )
    assert len(selector_rollout.decompositions) == 4
    assert all(len(decomposition.selections) == 4 for decomposition in selector_rollout.decompositions)

    decomposer_rollout = trainer.run(
        task=task,
        worker_pool=make_worker_pool(),
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        rollout_config=RolloutConfig(num_decompositions=16, num_selections_per_decomposition=16),
        schedule=TrainingScheduleConfig(
            mode=TrainingMode.ALTERNATING,
            alternating_phase=AlternatingPhase.DECOMPOSER,
        ),
    )
    assert len(decomposer_rollout.decompositions) == 8
    assert all(len(decomposition.selections) == 2 for decomposition in decomposer_rollout.decompositions)


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

    prompt_text = render_selector_prompt(
        task=second_task,
        decomposition=second_rollout.decompositions[0].decomposition,
        worker_pool=worker_pool,
        worker_performance=trainer.orchestrator.worker_memory.snapshot(worker_pool),
    )

    assert "WORKERS_BY_ID:" in prompt_text
    assert "calculation_worker" in prompt_text
    assert "skills=" in prompt_text
    assert "avg_reward=" in prompt_text
    assert '"workers_by_id"' not in prompt_text


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


def test_rollout_recorder_compact_mode_avoids_full_rollout_tree(tmp_path) -> None:
    trainer = HierarchicalGRPOTrainer(
        rollout_logging_config=RolloutLoggingConfig(
            output_dir=str(tmp_path),
            save_all_rollouts=False,
            save_best_rollouts=True,
            best_k=1,
            compact_mode=True,
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

    compact_record = json.loads((tmp_path / "best_selections.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "rollout" not in compact_record
    assert "assignments" in compact_record
    assert "executions" in compact_record
    assert "decomposition_raw_text" in compact_record
    assert "selection_raw_text" in compact_record
    assert compact_record["best_for_task"] is True
    assert compact_record["schedule"]["mode"] == "joint"
    assert not (tmp_path / "all_rollouts.jsonl").exists()
    assert (tmp_path / "topk_best_selections.jsonl").exists()


def test_best_rollout_logs_append_one_record_per_task(tmp_path) -> None:
    trainer = HierarchicalGRPOTrainer(
        rollout_logging_config=RolloutLoggingConfig(
            output_dir=str(tmp_path),
            save_all_rollouts=False,
            save_best_rollouts=True,
            best_k=1,
            compact_mode=True,
        )
    )
    tasks = [
        make_task("algebra", "Solve for x: 2x + 3 = 11.", "4", "5"),
        make_task("analysis", "Differentiate sin(x).", "cos(x)", "sin(x)"),
    ]
    for task in tasks:
        trainer.run(
            task=task,
            worker_pool=make_worker_pool(),
            policy_config=ControllerPolicyConfig(parameter_sharing=False),
            rollout_config=RolloutConfig(num_decompositions=2, num_selections_per_decomposition=2),
            schedule=TrainingScheduleConfig(mode=TrainingMode.ALTERNATING, alternating_phase=AlternatingPhase.SELECTOR),
        )

    best_decompositions = (tmp_path / "best_decompositions.jsonl").read_text(encoding="utf-8").splitlines()
    best_selections = (tmp_path / "best_selections.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(best_decompositions) == 2
    assert len(best_selections) == 2
    first_selection = json.loads(best_selections[0])
    assert first_selection["schedule"]["phase"] == "selector"
    assert "task_prompt" in first_selection


def test_extract_json_dict_accepts_tagged_controller_output() -> None:
    payload = extract_json_dict(
        "<selection_json>\n"
        "{\n"
        '  "selection_id": "sel-1",\n'
        '  "assignments": []\n'
        "}\n"
        "</selection_json>"
    )

    assert payload["selection_id"] == "sel-1"


def test_selector_prompt_uses_compact_decomposition_context() -> None:
    trainer = HierarchicalGRPOTrainer()
    task = make_task("analysis", "Differentiate sin(x).", "cos(x)", "sin(x)")
    rollout = trainer.run(
        task=task,
        worker_pool=make_worker_pool(),
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        rollout_config=RolloutConfig(num_decompositions=1, num_selections_per_decomposition=1),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
    )

    prompt = render_selector_prompt(
        task=task,
        decomposition=rollout.decompositions[0].decomposition,
        worker_pool=make_worker_pool(),
        worker_performance=trainer.orchestrator.worker_memory.snapshot(make_worker_pool()),
    )

    assert '"raw_payload"' not in prompt
    assert '"raw_text"' not in prompt
    assert "Allowed node IDs:" in prompt
    assert "- Use exactly one line per node in the form: `node_id: worker_id`." in prompt
    assert " | output=" not in prompt


def test_decomposer_prompt_declares_strict_output_contract() -> None:
    task = make_task("algebra", "Solve for x: 2x + 3 = 11.", "4", "5")

    prompt = render_decomposer_prompt(
        task=task,
        max_nodes_hint=4,
    )

    assert "- Return exactly one <decomposition_plan> block and nothing else." in prompt
    assert "Use only these keys: SUMMARY, TARGET_QUANTITY, FINAL_ANSWER_FORMAT_HINT, FINAL_NODE_ID, NODE_ID, INSTRUCTION, DEPENDENCIES, REQUIRED_SKILLS, REQUIRED_SKILLS_NOTE." in prompt
    assert "Allowed node IDs: 1, 2, 3, 4." in prompt
    assert "OUTPUT_KEY" not in prompt


def test_non_final_worker_output_does_not_blank_signed_values() -> None:
    task = make_task("algebra", "Solve x^2 + 3x = 0.", "3", "5")
    decomposition = DecompositionCandidate(
        decomposition_id="decomp-1",
        summary="solve roots",
        target_quantity="solution set",
        final_answer_format_hint="integer",
        nodes=[
            SubtaskNode(node_id="1", instruction="Solve the quadratic.", output_key="1_output"),
            SubtaskNode(
                node_id="2",
                instruction="Return the positive root only.",
                dependencies=["1"],
                output_key="final_answer",
            ),
        ],
        final_node_id="2",
    )

    normalized_output, invalid_reason, final_answer_leak, answer_containment, success = (
        _postprocess_worker_output(
            task=task,
            decomposition=decomposition,
            node=decomposition.nodes[0],
            raw_output_text="<worker_result>0; -3</worker_result>",
        )
    )

    assert normalized_output == "0; -3"
    assert invalid_reason == ""
    assert final_answer_leak is False
    assert answer_containment is False
    assert success is True


def test_malformed_worker_output_requires_closed_worker_result_block() -> None:
    task = make_task("algebra", "Find the integer solution.", "3", "4")
    decomposition = DecompositionCandidate(
        decomposition_id="decomp-1",
        summary="two-step plan",
        target_quantity="integer solution",
        final_answer_format_hint="integer",
        nodes=[
            SubtaskNode(node_id="1", instruction="Analyze the cases.", output_key="1_output"),
            SubtaskNode(
                node_id="2",
                instruction="Return the final answer.",
                dependencies=["1"],
                output_key="final_answer",
            ),
        ],
        final_node_id="2",
    )

    normalized_output, invalid_reason, final_answer_leak, answer_containment, success = (
        _postprocess_worker_output(
            task=task,
            decomposition=decomposition,
            node=decomposition.nodes[0],
            raw_output_text="<worker_scratchpad>\n<worker_result>\nconcise result",
        )
    )

    assert normalized_output == ""
    assert invalid_reason == "missing_worker_result"
    assert final_answer_leak is False
    assert answer_containment is False
    assert success is False


def test_worker_output_rejects_text_outside_worker_blocks() -> None:
    assert extract_worker_result_text("4\n<worker_result>\n4\n</worker_result>") == ""


def test_final_worker_output_recovers_worker_result_with_outer_text() -> None:
    task = make_task("algebra", "Find the integer solution.", "4", "5")
    decomposition = DecompositionCandidate(
        decomposition_id="decomp-1",
        summary="single-step plan",
        target_quantity="integer solution",
        final_answer_format_hint="integer",
        nodes=[
            SubtaskNode(
                node_id="1",
                instruction="Return the final answer.",
                output_key="final_answer",
            ),
        ],
        final_node_id="1",
    )

    normalized_output, invalid_reason, final_answer_leak, answer_containment, success = (
        _postprocess_worker_output(
            task=task,
            decomposition=decomposition,
            node=decomposition.nodes[0],
            raw_output_text="4\n<worker_result>\n4\n</worker_result>",
        )
    )

    assert normalized_output == "4"
    assert invalid_reason == "recovered_worker_result_with_outer_text"
    assert final_answer_leak is False
    assert answer_containment is False
    assert success is True


def test_final_worker_output_recovers_plain_text_answer() -> None:
    task = make_task("algebra", "Find the integer solution.", "1", "2")
    decomposition = DecompositionCandidate(
        decomposition_id="decomp-1",
        summary="single-step plan",
        target_quantity="integer solution",
        final_answer_format_hint="integer",
        nodes=[
            SubtaskNode(
                node_id="1",
                instruction="Return the final answer.",
                output_key="final_answer",
            ),
        ],
        final_node_id="1",
    )

    normalized_output, invalid_reason, final_answer_leak, answer_containment, success = (
        _postprocess_worker_output(
            task=task,
            decomposition=decomposition,
            node=decomposition.nodes[0],
            raw_output_text="1",
        )
    )

    assert normalized_output == "1"
    assert invalid_reason == "recovered_plain_text_final_answer"
    assert final_answer_leak is False
    assert answer_containment is False
    assert success is True


def test_non_final_worker_output_stays_strict_when_text_sits_outside_tags() -> None:
    task = make_task("algebra", "Find the integer solution.", "4", "5")
    decomposition = DecompositionCandidate(
        decomposition_id="decomp-1",
        summary="two-step plan",
        target_quantity="integer solution",
        final_answer_format_hint="integer",
        nodes=[
            SubtaskNode(node_id="1", instruction="Analyze the cases.", output_key="1_output"),
            SubtaskNode(
                node_id="2",
                instruction="Return the final answer.",
                dependencies=["1"],
                output_key="final_answer",
            ),
        ],
        final_node_id="2",
    )

    normalized_output, invalid_reason, final_answer_leak, answer_containment, success = (
        _postprocess_worker_output(
            task=task,
            decomposition=decomposition,
            node=decomposition.nodes[0],
            raw_output_text="4\n<worker_result>\n4\n</worker_result>",
        )
    )

    assert normalized_output == ""
    assert invalid_reason == "missing_worker_result"
    assert final_answer_leak is False
    assert answer_containment is False
    assert success is False


def test_non_final_worker_output_blanks_explicit_final_answer_clause() -> None:
    task = make_task(
        "algebra",
        "Find the integer solution.",
        "3",
        "4",
    )
    decomposition = DecompositionCandidate(
        decomposition_id="decomp-1",
        summary="two-step plan",
        target_quantity="integer solution",
        final_answer_format_hint="integer",
        nodes=[
            SubtaskNode(node_id="1", instruction="Analyze the cases.", output_key="1_output"),
            SubtaskNode(
                node_id="2",
                instruction="Return the final answer.",
                dependencies=["1"],
                output_key="final_answer",
            ),
        ],
        final_node_id="2",
    )

    normalized_output, invalid_reason, final_answer_leak, answer_containment, success = (
        _postprocess_worker_output(
            task=task,
            decomposition=decomposition,
            node=decomposition.nodes[0],
            raw_output_text="<worker_result>\nAfter simplifying, the answer is 3.\n</worker_result>",
        )
    )

    assert normalized_output == REDACTED_FINAL_ANSWER_LEAK_OUTPUT
    assert invalid_reason == "non_final_contains_ground_truth"
    assert final_answer_leak is True
    assert answer_containment is False
    assert success is False


def test_non_final_worker_output_blanks_boxed_final_answer() -> None:
    task = make_task(
        "algebra",
        "Find the integer solution.",
        "3",
        "4",
    )
    decomposition = DecompositionCandidate(
        decomposition_id="decomp-1",
        summary="two-step plan",
        target_quantity="integer solution",
        final_answer_format_hint="integer",
        nodes=[
            SubtaskNode(node_id="1", instruction="Analyze the cases.", output_key="1_output"),
            SubtaskNode(
                node_id="2",
                instruction="Return the final answer.",
                dependencies=["1"],
                output_key="final_answer",
            ),
        ],
        final_node_id="2",
    )

    normalized_output, invalid_reason, final_answer_leak, answer_containment, success = (
        _postprocess_worker_output(
            task=task,
            decomposition=decomposition,
            node=decomposition.nodes[0],
            raw_output_text="<worker_result>\n\\boxed{3}\n</worker_result>",
        )
    )

    assert normalized_output == REDACTED_FINAL_ANSWER_LEAK_OUTPUT
    assert invalid_reason == "non_final_contains_ground_truth"
    assert final_answer_leak is True
    assert answer_containment is False
    assert success is False


def test_non_final_worker_output_blanks_named_value_clause() -> None:
    task = make_task(
        "algebra",
        "Solve for y.",
        "\\frac{4}{13}",
        "1",
    )
    decomposition = DecompositionCandidate(
        decomposition_id="decomp-1",
        summary="two-step plan",
        target_quantity="value of y",
        final_answer_format_hint="fraction",
        nodes=[
            SubtaskNode(node_id="1", instruction="Solve the equation.", output_key="1_output"),
            SubtaskNode(
                node_id="2",
                instruction="Return the final answer.",
                dependencies=["1"],
                output_key="final_answer",
            ),
        ],
        final_node_id="2",
    )

    normalized_output, invalid_reason, final_answer_leak, answer_containment, success = (
        _postprocess_worker_output(
            task=task,
            decomposition=decomposition,
            node=decomposition.nodes[0],
            raw_output_text="<worker_result>\nThe value of y is $\\frac{4}{13}$.\n</worker_result>",
        )
    )

    assert normalized_output == REDACTED_FINAL_ANSWER_LEAK_OUTPUT
    assert invalid_reason == "non_final_contains_ground_truth"
    assert final_answer_leak is True
    assert answer_containment is False
    assert success is False


def test_non_final_worker_output_blanks_multiline_named_result_clause() -> None:
    task = make_task(
        "algebra",
        "Find the matrix.",
        "\\begin{pmatrix} 2 & -3 \\\\ 0 & 3 \\end{pmatrix}",
        "0",
    )
    decomposition = DecompositionCandidate(
        decomposition_id="decomp-1",
        summary="two-step plan",
        target_quantity="matrix",
        final_answer_format_hint="matrix",
        nodes=[
            SubtaskNode(node_id="1", instruction="Solve for the matrix.", output_key="1_output"),
            SubtaskNode(
                node_id="2",
                instruction="Return the final answer.",
                dependencies=["1"],
                output_key="final_answer",
            ),
        ],
        final_node_id="2",
    )

    normalized_output, invalid_reason, final_answer_leak, answer_containment, success = (
        _postprocess_worker_output(
            task=task,
            decomposition=decomposition,
            node=decomposition.nodes[0],
            raw_output_text=(
                "<worker_result>\n"
                "Thus, the matrix M is:\n"
                "\\[\n"
                "\\begin{pmatrix} 2 & -3 \\\\ 0 & 3 \\end{pmatrix}\n"
                "\\]\n"
                "</worker_result>"
            ),
        )
    )

    assert normalized_output == REDACTED_FINAL_ANSWER_LEAK_OUTPUT
    assert invalid_reason == "non_final_contains_ground_truth"
    assert final_answer_leak is True
    assert answer_containment is False
    assert success is False


def test_worker_prompt_explains_redacted_dependency_outputs() -> None:
    task = make_task("algebra", "Find the integer solution.", "3", "4")
    decomposition = DecompositionCandidate(
        decomposition_id="decomp-1",
        summary="two-step plan",
        target_quantity="integer solution",
        final_answer_format_hint="integer",
        nodes=[
            SubtaskNode(node_id="1", instruction="Analyze the cases.", output_key="1_output"),
            SubtaskNode(
                node_id="2",
                instruction="Return the final answer.",
                dependencies=["1"],
                output_key="final_answer",
            ),
        ],
        final_node_id="2",
    )
    worker_pool = make_worker_pool()
    prompt = render_worker_prompt(
        task=task,
        decomposition=decomposition,
        node=decomposition.nodes[1],
        worker=worker_pool.workers[0],
        dependency_outputs={"1": REDACTED_FINAL_ANSWER_LEAK_OUTPUT},
    )

    assert REDACTED_FINAL_ANSWER_LEAK_OUTPUT in prompt
    assert "Treat that dependency as unavailable evidence" in prompt


def test_worker_prompt_does_not_include_concise_result_placeholder() -> None:
    task = make_task("algebra", "Find the integer solution.", "3", "4")
    decomposition = DecompositionCandidate(
        decomposition_id="decomp-1",
        summary="two-step plan",
        target_quantity="integer solution",
        final_answer_format_hint="integer",
        nodes=[
            SubtaskNode(node_id="1", instruction="Analyze the cases.", output_key="1_output"),
            SubtaskNode(
                node_id="2",
                instruction="Return the final answer.",
                dependencies=["1"],
                output_key="final_answer",
            ),
        ],
        final_node_id="2",
    )
    worker_pool = make_worker_pool()
    prompt = render_worker_prompt(
        task=task,
        decomposition=decomposition,
        node=decomposition.nodes[1],
        worker=worker_pool.workers[0],
        dependency_outputs={"1": "x = 3"},
    )

    assert "concise result" not in prompt
    assert "From 2x = 8, divide both sides by 2." in prompt


def test_controller_sample_quality_filter_clean_only_keeps_only_clean_controller_samples() -> None:
    task = make_task("algebra", "Solve for x: 2x + 3 = 11.", "4", "5")
    rollout = TaskRollout(
        task=task,
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        rollout_config=RolloutConfig(),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
        decompositions=[],
        training_batch=HierarchicalTrainingBatch(
            decomposer_samples=[
                ControllerTrainingSample(
                    role="decomposer",
                    policy_id="decomposer_controller",
                    group_id="clean",
                    prompt_text="prompt",
                    completion_text="completion",
                    reward=1.0,
                    advantage=1.0,
                    metadata={
                        "format_validation": {
                            "fallback_used": False,
                            "attempt": 0,
                        }
                    },
                ),
                ControllerTrainingSample(
                    role="decomposer",
                    policy_id="decomposer_controller",
                    group_id="local",
                    prompt_text="prompt",
                    completion_text="completion",
                    reward=1.0,
                    advantage=1.0,
                    metadata={
                        "format_validation": {
                            "fallback_used": False,
                            "local_salvage_used": True,
                        }
                    },
                ),
                ControllerTrainingSample(
                    role="decomposer",
                    policy_id="decomposer_controller",
                    group_id="model",
                    prompt_text="prompt",
                    completion_text="completion",
                    reward=1.0,
                    advantage=1.0,
                    metadata={
                        "format_validation": {
                            "fallback_used": False,
                            "attempt": 1,
                            "errors_before_success": ["parse error"],
                        }
                    },
                ),
                ControllerTrainingSample(
                    role="decomposer",
                    policy_id="decomposer_controller",
                    group_id="fallback",
                    prompt_text="prompt",
                    completion_text="completion",
                    reward=1.0,
                    advantage=1.0,
                    metadata={
                        "format_validation": {
                            "fallback_used": True,
                        }
                    },
                ),
            ],
            selector_samples=[
                ControllerTrainingSample(
                    role="selector",
                    policy_id="selector_controller",
                    group_id="selector-clean",
                    prompt_text="prompt",
                    completion_text="completion",
                    reward=1.0,
                    advantage=1.0,
                    metadata={
                        "format_validation": {
                            "fallback_used": False,
                            "attempt": 0,
                        },
                        "decomposition_format_validation": {
                            "fallback_used": False,
                            "attempt": 0,
                        },
                    },
                ),
                ControllerTrainingSample(
                    role="selector",
                    policy_id="selector_controller",
                    group_id="selector-upstream-local",
                    prompt_text="prompt",
                    completion_text="completion",
                    reward=1.0,
                    advantage=1.0,
                    metadata={
                        "format_validation": {
                            "fallback_used": False,
                            "attempt": 0,
                        },
                        "decomposition_format_validation": {
                            "fallback_used": False,
                            "local_salvage_used": True,
                        },
                    },
                ),
            ],
        ),
    )

    samples = controller_samples_from_task_rollouts(
        task_rollouts=[rollout],
        roles=["decomposer", "selector"],
        controller_sample_quality_filter="clean_only",
    )

    assert [sample.group_id for sample in samples] == ["clean", "selector-clean"]


def test_controller_sample_quality_filter_allow_local_repair_excludes_model_repair_and_fallback() -> None:
    task = make_task("algebra", "Solve for x: 2x + 3 = 11.", "4", "5")
    rollout = TaskRollout(
        task=task,
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        rollout_config=RolloutConfig(),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
        decompositions=[],
        training_batch=HierarchicalTrainingBatch(
            decomposer_samples=[
                ControllerTrainingSample(
                    role="decomposer",
                    policy_id="decomposer_controller",
                    group_id="clean",
                    prompt_text="prompt",
                    completion_text="completion",
                    reward=1.0,
                    advantage=1.0,
                    metadata={"format_validation": {"fallback_used": False, "attempt": 0}},
                ),
                ControllerTrainingSample(
                    role="decomposer",
                    policy_id="decomposer_controller",
                    group_id="local",
                    prompt_text="prompt",
                    completion_text="completion",
                    reward=1.0,
                    advantage=1.0,
                    metadata={"format_validation": {"fallback_used": False, "local_salvage_used": True}},
                ),
                ControllerTrainingSample(
                    role="decomposer",
                    policy_id="decomposer_controller",
                    group_id="model",
                    prompt_text="prompt",
                    completion_text="completion",
                    reward=1.0,
                    advantage=1.0,
                    metadata={"format_validation": {"fallback_used": False, "attempt": 2}},
                ),
                ControllerTrainingSample(
                    role="decomposer",
                    policy_id="decomposer_controller",
                    group_id="fallback",
                    prompt_text="prompt",
                    completion_text="completion",
                    reward=1.0,
                    advantage=1.0,
                    metadata={"format_validation": {"fallback_used": True}},
                ),
            ],
            selector_samples=[
                ControllerTrainingSample(
                    role="selector",
                    policy_id="selector_controller",
                    group_id="selector-local",
                    prompt_text="prompt",
                    completion_text="completion",
                    reward=1.0,
                    advantage=1.0,
                    metadata={
                        "format_validation": {
                            "fallback_used": False,
                            "partial_completion_used": True,
                            "batch_repair_fallback": True,
                        },
                        "decomposition_format_validation": {
                            "fallback_used": False,
                            "attempt": 0,
                        },
                    },
                ),
                ControllerTrainingSample(
                    role="selector",
                    policy_id="selector_controller",
                    group_id="selector-upstream-model",
                    prompt_text="prompt",
                    completion_text="completion",
                    reward=1.0,
                    advantage=1.0,
                    metadata={
                        "format_validation": {
                            "fallback_used": False,
                            "attempt": 0,
                        },
                        "decomposition_format_validation": {
                            "fallback_used": False,
                            "attempt": 1,
                        },
                    },
                ),
            ],
        ),
    )

    samples = controller_samples_from_task_rollouts(
        task_rollouts=[rollout],
        roles=["decomposer", "selector"],
        controller_sample_quality_filter="allow_local_repair",
    )

    assert [sample.group_id for sample in samples] == ["clean", "local", "selector-local"]


def test_decomposition_output_keys_are_canonicalized() -> None:
    payload = {
        "decomposition_id": "decomp-1",
        "summary": "short plan",
        "target_quantity": "integer",
        "final_answer_format_hint": "integer",
        "final_node_id": "20",
        "nodes": [
            {
                "node_id": "10",
                "instruction": "Analyze the equation.",
                "dependencies": [],
                "required_skills": ["analysis"],
                "output_key": "1234567890abcdefghijklmnopQERTYUIOP1234567890abcdefgHJKLOMN",
            },
            {
                "node_id": "20",
                "instruction": "Return the final answer.",
                "dependencies": ["10"],
                "required_skills": ["algebra"],
                "output_key": "another_garbage_key",
            },
        ],
    }

    candidate = validate_decomposition_payload(
        payload=payload,
        rollout_config=RolloutConfig(),
        fallback_id="decomp-1",
    )

    assert [node.output_key for node in candidate.nodes] == ["1_output", "final_answer"]


def test_required_skills_normalize_new_role_like_labels() -> None:
    payload = {
        "decomposition_id": "decomp-1",
        "summary": "short plan",
        "target_quantity": "answer",
        "final_answer_format_hint": "integer",
        "final_node_id": "20",
        "nodes": [
            {
                "node_id": "10",
                "instruction": "Compute the intermediate quantity.",
                "dependencies": [],
                "required_skills": ["calculation", "casework"],
            },
            {
                "node_id": "20",
                "instruction": "Return the final answer.",
                "dependencies": ["10"],
                "required_skills": ["logic_constraints", "function_analysis"],
            },
        ],
    }

    candidate = validate_decomposition_payload(
        payload=payload,
        rollout_config=RolloutConfig(),
        fallback_id="decomp-1",
    )

    assert candidate.nodes[0].required_skills == ["arithmetic", "combinatorics"]
    assert candidate.nodes[1].required_skills == ["discrete_math", "analysis"]


def test_line_based_controller_plans_are_parseable() -> None:
    decomposition_payload = extract_decomposition_payload(
        "<decomposition_plan>\n"
        "DECOMPOSITION_ID: decomp-1\n"
        "SUMMARY: short plan\n"
        "FINAL_NODE_ID: 2\n"
        "NODE_ID: 1\n"
        "INSTRUCTION: analyze the structure\n"
        "DEPENDENCIES: none\n"
        "REQUIRED_SKILLS: analysis\n"
        "OUTPUT_KEY: structure\n"
        "NODE_ID: 2\n"
        "INSTRUCTION: produce final answer\n"
        "DEPENDENCIES: 1\n"
        "REQUIRED_SKILLS: algebra\n"
        "OUTPUT_KEY: final_answer\n"
        "</decomposition_plan>"
    )
    selection_payload = extract_selection_payload(
        "<selection_plan>\n"
        "SELECTION_ID: sel-1\n"
        "1: 4\n"
        "2: 2\n"
        "</selection_plan>"
    )

    assert decomposition_payload["final_node_id"] == "2"
    assert len(decomposition_payload["nodes"]) == 2
    assert selection_payload["selection_id"] == "sel-1"
    assert len(selection_payload["assignments"]) == 2


def test_selector_plan_accepts_minimal_assignment_lines() -> None:
    selection_payload = extract_selection_payload(
        "<selection_plan>\n"
        "SELECTION_ID: sel-compact\n"
        "1: 4\n"
        "2: 2\n"
        "</selection_plan>"
    )

    assert selection_payload["selection_id"] == "sel-compact"
    assert [assignment["node_id"] for assignment in selection_payload["assignments"]] == [
        "1",
        "2",
    ]
    assert [assignment["worker_index"] for assignment in selection_payload["assignments"]] == [
        4,
        2,
    ]


def test_controller_training_completions_are_clean_plan_blocks() -> None:
    trainer = HierarchicalGRPOTrainer()
    task = make_task("algebra", "Solve for x: 2x + 3 = 11.", "4", "5")
    rollout = trainer.run(
        task=task,
        worker_pool=make_worker_pool(),
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        rollout_config=RolloutConfig(num_decompositions=2, num_selections_per_decomposition=2),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
    )

    decomposer_completion = rollout.training_batch.decomposer_samples[0].completion_text
    selector_completion = rollout.training_batch.selector_samples[0].completion_text

    assert decomposer_completion.startswith("<decomposition_plan>")
    assert selector_completion.startswith("<selection_plan>")
    assert "controller_prompt" not in decomposer_completion
    assert "validation" not in decomposer_completion
    assert "controller_prompt" not in selector_completion
    assert "validation" not in selector_completion
    assert "compatibility=" not in selector_completion


def test_selector_compatibility_is_computed_even_for_minimal_output() -> None:
    trainer = HierarchicalGRPOTrainer()
    task = make_task("analysis", "Differentiate sin(x).", "cos(x)", "sin(x)")
    rollout = trainer.run(
        task=task,
        worker_pool=make_worker_pool(),
        policy_config=ControllerPolicyConfig(parameter_sharing=False),
        rollout_config=RolloutConfig(num_decompositions=1, num_selections_per_decomposition=1),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
    )

    selection = rollout.decompositions[0].selections[0].selection
    assert all(assignment.compatibility > 0.0 for assignment in selection.assignments)

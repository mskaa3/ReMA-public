import json

try:
    from verl.hierarchical_rema.backends import _postprocess_worker_output
    from verl.hierarchical_rema import (
        AlternatingPhase,
        ControllerPolicyConfig,
        DecompositionCandidate,
        DecompositionRollout,
        HierarchicalGRPOTrainer,
        RewardWeights,
        RolloutLoggingConfig,
        RolloutConfig,
        SelectionCandidate,
        SelectionRewardBreakdown,
        SelectionRollout,
        SubtaskNode,
        TaskExample,
        TrainingMode,
        TrainingScheduleConfig,
        REDACTED_FINAL_ANSWER_LEAK_OUTPUT,
        WorkerExecution,
        WorkerRewardMode,
        WorkerPoolConfig,
        WorkerSpec,
    )
    from verl.hierarchical_rema.demo import make_worker_pool as make_default_worker_pool
    from verl.hierarchical_rema.prompts import (
        render_decomposer_prompt,
        render_selector_prompt,
        render_worker_prompt,
    )
    from verl.hierarchical_rema.structured import (
        apply_decomposition_limits,
        build_fallback_decomposition,
        extract_decomposition_payload,
        extract_json_dict,
        extract_selection_payload,
        validate_decomposition_payload,
    )
except ModuleNotFoundError:
    from hierarchical_rema.backends import _postprocess_worker_output
    from hierarchical_rema import (
        AlternatingPhase,
        ControllerPolicyConfig,
        DecompositionCandidate,
        DecompositionRollout,
        HierarchicalGRPOTrainer,
        RewardWeights,
        RolloutLoggingConfig,
        RolloutConfig,
        SelectionCandidate,
        SelectionRewardBreakdown,
        SelectionRollout,
        SubtaskNode,
        TaskExample,
        TrainingMode,
        TrainingScheduleConfig,
        REDACTED_FINAL_ANSWER_LEAK_OUTPUT,
        WorkerExecution,
        WorkerRewardMode,
        WorkerPoolConfig,
        WorkerSpec,
    )
    from hierarchical_rema.demo import make_worker_pool as make_default_worker_pool
    from hierarchical_rema.prompts import (
        render_decomposer_prompt,
        render_selector_prompt,
        render_worker_prompt,
    )
    from hierarchical_rema.structured import (
        apply_decomposition_limits,
        build_fallback_decomposition,
        extract_decomposition_payload,
        extract_json_dict,
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
            mode=TrainingMode.ALTERNATING,
            alternating_phase=AlternatingPhase.DECOMPOSER,
        ),
        decompositions=[decomposition_rollout],
    )

    assert training_batch.worker_samples == []
    assert training_batch.worker_grpo_stats["num_decompositions_skipped_fallback"] == 1
    assert training_batch.worker_grpo_stats["num_worker_samples_skipped_fallback"] == 1


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
    assert fallback_decomposition.soft_penalty == 1.0


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

    assert limited_candidate.soft_penalty == 0.5


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

    prompt_text = render_decomposer_prompt(
        second_task,
        worker_pool,
        trainer.orchestrator.worker_memory.snapshot(worker_pool),
    )

    assert "AVAILABLE_WORKERS:" in prompt_text
    assert "arithmetic_prealgebra_worker" in prompt_text
    assert "complete=" in prompt_text
    assert "avg_reward=" in prompt_text
    assert '"available_workers"' not in prompt_text


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

"""Worker advantages compare one planned node and executor across repeated runs."""

import pytest

try:
    from verl import hierarchical_rema as api
    from verl.hierarchical_rema.rewarding import group_relative_advantages
except ModuleNotFoundError:
    import hierarchical_rema as api
    from hierarchical_rema.rewarding import group_relative_advantages


def make_batch(
    rewards_by_node=None,
    *,
    task_id="task-1",
    plan_id="plan-1",
    worker_ids=("executor",),
    model_path="checkpoint-1",
    adapter_path=None,
    instruction="Compute the result.",
    min_group_size=3,
    unscored_repeat=None,
):
    if rewards_by_node is None:
        rewards_by_node = {"A": [-0.9, 0.0, 0.9], "B": [0.6, 0.6, 0.6]}
    node_ids = list(rewards_by_node)
    repeats = len(rewards_by_node[node_ids[0]])
    task = api.TaskExample(task_id=task_id, prompt="Solve the problem.", ground_truth="42")
    pool = api.WorkerPoolConfig(
        base_model_path="pool-checkpoint",
        enable_role_lora=adapter_path is not None,
        workers=[api.WorkerSpec(
            worker_id=worker_id, description="Executor", skills=[], system_prompt="Execute.",
            base_model_path=model_path, lora_adapter_path=adapter_path,
        ) for worker_id in worker_ids],
    )
    plan = api.DecompositionCandidate(
        decomposition_id=plan_id, summary="Plan", target_quantity="Answer",
        final_answer_format_hint="Integer", final_node_id=node_ids[-1],
        nodes=[api.SubtaskNode(
            node_id=node_id, instruction=instruction,
            dependencies=node_ids[:index], output_key=f"result_{node_id}",
        ) for index, node_id in enumerate(node_ids)],
    )
    selections = []
    for worker_id in worker_ids:
        for repeat in range(repeats):
            selections.append(api.SelectionRollout(
                selection=api.SelectionCandidate(
                    selection_id=f"{worker_id}-repeat-{repeat}", assignments=[],
                ),
                executions=[api.WorkerExecution(
                    node_id=node_id, worker_id=worker_id, output_text=f"Result {repeat}",
                    raw_output_text=f"<worker_result>Result {repeat}</worker_result>",
                    worker_prompt=f"{instruction} Node {node_id}; upstream value: {repeat}",
                    dependency_outputs={parent: str(repeat) for parent in node_ids[:index]},
                    entropy=0.0, confidence_reward=0.0, compatibility=1.0,
                    reward_model_reward=rewards_by_node[node_id][repeat],
                ) for index, node_id in enumerate(node_ids)],
                final_answer="42",
                reward=api.SelectionRewardBreakdown(
                    final_answer_correctness=1.0, confidence_reward=0.0,
                    compatibility_reward=0.0,
                    total_reward=None if repeat == unscored_repeat else 0.99,
                    reward_model_source="gfam_v1",
                ),
            ))
    trainer = api.HierarchicalGRPOTrainer(
        train_worker_model=True, min_worker_grpo_group_size=min_group_size,
    )
    return trainer.orchestrator._build_training_batch(
        task=task, worker_pool=pool, policy_config=api.ControllerPolicyConfig(),
        schedule=api.TrainingScheduleConfig(
            mode=api.TrainingMode.ALTERNATING, alternating_phase=api.AlternatingPhase.EXECUTOR,
        ),
        decompositions=[api.DecompositionRollout(
            decomposition=plan, selections=selections,
            base_decomposition_reward=0.99, decomposition_reward=0.99,
        )],
    )


def group_ids(batch):
    return {sample.group_id for sample in batch.worker_samples}


def test_identical_instructions_on_distinct_nodes_keep_separate_local_advantages():
    batch = make_batch()
    assert len(batch.worker_samples) == 6
    assert len(group_ids(batch)) == 2
    by_node = {
        node: [sample for sample in batch.worker_samples if sample.metadata["node_id"] == node]
        for node in ("A", "B")
    }
    assert [sample.reward for sample in by_node["A"]] == [-0.9, 0.0, 0.9]
    assert [sample.reward for sample in by_node["B"]] == [0.6, 0.6, 0.6]
    assert [sample.advantage for sample in by_node["A"]] == pytest.approx(
        group_relative_advantages([-0.9, 0.0, 0.9])
    )
    assert [sample.advantage for sample in by_node["B"]] == [0.0, 0.0, 0.0]
    # Different upstream contexts remain in the same node comparison group.
    assert len({sample.prompt_text for sample in by_node["B"]}) == 3
    assert len({sample.group_id for sample in by_node["B"]}) == 1
    assert all(sample.metadata["advantage_group_size"] == 3 for sample in batch.worker_samples)
    assert {sample.policy_id for sample in batch.worker_samples} == {"shared_worker"}
    assert batch.decomposer_samples == []
    assert batch.selector_samples == []


def test_different_nodes_cannot_inflate_minimum_group_size():
    batch = make_batch({"A": [0.0, 1.0], "B": [0.0, 1.0]})
    assert batch.worker_samples == []
    assert batch.worker_grpo_stats["num_groups_total"] == 2
    assert batch.worker_grpo_stats["num_groups_skipped"] == 2
    assert batch.worker_grpo_stats["num_samples_skipped"] == 4


def test_unscored_execution_is_excluded_before_per_node_grouping():
    batch = make_batch({"A": [0.0, 0.1, 0.2, 0.3], "B": [0.0, 0.2, 0.4, 0.6]},
                       unscored_repeat=0)
    assert len(batch.worker_samples) == 6
    assert batch.worker_grpo_stats["num_worker_samples_skipped_unscored"] == 2
    assert all(sample.metadata["advantage_group_size"] == 3 for sample in batch.worker_samples)
    assert all(sample.metadata["selection_id"] != "executor-repeat-0" for sample in batch.worker_samples)


def test_different_executors_do_not_share_node_comparisons():
    batch = make_batch(worker_ids=("executor-a", "executor-b"))
    assert len(group_ids(batch)) == 4
    assert len(batch.worker_samples) == 12
    for group_id in group_ids(batch):
        samples = [sample for sample in batch.worker_samples if sample.group_id == group_id]
        assert len(samples) == 3
        assert len({sample.metadata["worker_id"] for sample in samples}) == 1
        assert len({sample.metadata["node_id"] for sample in samples}) == 1


@pytest.mark.parametrize("change", [
    {"task_id": "task-2"},
    {"plan_id": "plan-2"},
    {"worker_ids": ("another-executor",)},
    {"model_path": "checkpoint-2"},
    {"adapter_path": "adapter-2"},
])
def test_group_identity_separates_task_plan_and_executor_policy(change):
    assert group_ids(make_batch()).isdisjoint(group_ids(make_batch(**change)))


def test_group_identity_does_not_depend_on_instruction_text():
    assert group_ids(make_batch()) == group_ids(make_batch(instruction="Equivalent rewording."))


def test_group_metadata_records_policy_and_model_fallback():
    batch = make_batch(model_path=None, adapter_path="adapter-1")
    for sample in batch.worker_samples:
        assert sample.metadata["advantage_group_kind"] == (
            "node_id_and_executor_policy_within_task_and_decomposition"
        )
        assert sample.metadata["advantage_group_executor_policy"] == {
            "worker_id": "executor", "base_model_path": "pool-checkpoint",
            "lora_adapter_path": "adapter-1",
        }
        assert sample.metadata["normalized_node_instruction"] == "compute the result."

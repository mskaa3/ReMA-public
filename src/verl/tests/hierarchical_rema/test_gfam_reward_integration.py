import math
import pytest

try:
    from verl.hierarchical_rema import (
        ControllerPolicyConfig,
        DecompositionCandidate,
        HierarchicalGRPOTrainer,
        RolloutConfig,
        SelectionCandidate,
        SubtaskNode,
        TaskExample,
        TrainingMode,
        TrainingScheduleConfig,
        WorkerAssignment,
        WorkerExecution,
    )
    from verl.hierarchical_rema.demo import make_worker_pool as make_default_worker_pool
except ModuleNotFoundError:
    from hierarchical_rema import (
        ControllerPolicyConfig,
        DecompositionCandidate,
        HierarchicalGRPOTrainer,
        RolloutConfig,
        SelectionCandidate,
        SubtaskNode,
        TaskExample,
        TrainingMode,
        TrainingScheduleConfig,
        WorkerAssignment,
        WorkerExecution,
    )
    from hierarchical_rema.demo import make_worker_pool as make_default_worker_pool


class _FakeGFAMRewardScorer:
    def score_rollout(self, *, task, decomposition, selection, executions, final_answer):
        worker_rewards = {
            execution.node_id: {"reward": 0.11 if execution.node_id == "1" else 0.22}
            for execution in executions
        }
        selector_decision_rewards = {
            execution.node_id: {"reward": 0.61 if execution.node_id == "1" else 0.49}
            for execution in executions
        }
        return {
            "source": "gfam_v1",
            "compiled_rewards": {
                "decomposer": {"reward": 0.77},
                "selector": {"reward": 0.55},
                "selector_decisions": selector_decision_rewards,
                "workers": worker_rewards,
                "final": {"reward": 0.33},
            },
            "graph_summary": {"graph_final_correct_score": 0.5},
        }


def _make_task() -> TaskExample:
    return TaskExample(
        task_id="gfam-task",
        prompt="Solve for x: 2x + 3 = 11.",
        ground_truth="4",
        metadata={"skill_focus": "algebra", "distractor_answer": "5"},
    )


def _make_worker_pool():
    return make_default_worker_pool(base_model_path="mock-model")


def test_gfam_reward_overrides_handcrafted_selection_and_decomposer_rewards() -> None:
    trainer = HierarchicalGRPOTrainer(gfam_reward_scorer=_FakeGFAMRewardScorer())
    rollout = trainer.run(
        task=_make_task(),
        worker_pool=_make_worker_pool(),
        policy_config=ControllerPolicyConfig(parameter_sharing=True, shared_model_path="mock-shared"),
        rollout_config=RolloutConfig(
            num_decompositions=1,
            num_selections_per_decomposition=1,
            soft_hop_penalty=10.0,
        ),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
    )

    selection = rollout.decompositions[0].selections[0]
    decomposition = rollout.decompositions[0]

    assert selection.reward.reward_model_source == "gfam_v1"
    assert math.isclose(selection.reward.total_reward, 0.33)
    assert selection.reward.to_dict() == {
        "total_reward": 0.33,
        "reward_model_source": "gfam_v1",
    }
    assert selection.reward_model_outputs["source"] == "gfam_v1"
    assert math.isclose(selection.reward_model_outputs["selection_reward"], 0.33)
    assert math.isclose(selection.reward_model_outputs["selector_reward_mean"], 0.55)
    assignment_rewards = {
        assignment.node_id: assignment.reward_model_reward
        for assignment in selection.selection.assignments
    }
    assert math.isclose(assignment_rewards["1"], 0.61)
    assert math.isclose(assignment_rewards["2"], 0.49)
    assert math.isclose(decomposition.base_decomposition_reward, 0.77)
    assert math.isclose(decomposition.decomposition_reward, 0.77)


def test_gfam_worker_rewards_flow_into_worker_training_samples() -> None:
    trainer = HierarchicalGRPOTrainer(
        gfam_reward_scorer=_FakeGFAMRewardScorer(),
        train_worker_model=True,
        min_worker_grpo_group_size=1,
    )
    rollout = trainer.run(
        task=_make_task(),
        worker_pool=_make_worker_pool(),
        policy_config=ControllerPolicyConfig(parameter_sharing=True, shared_model_path="mock-shared"),
        rollout_config=RolloutConfig(num_decompositions=1, num_selections_per_decomposition=1),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
    )

    worker_samples = rollout.training_batch.worker_samples
    assert len(worker_samples) >= 2
    rewards_by_node = {sample.metadata["node_id"]: sample.reward for sample in worker_samples}
    assert math.isclose(rewards_by_node["1"], 0.11)
    assert math.isclose(rewards_by_node["2"], 0.22)


def test_gfam_worker_history_uses_reward_model_rewards() -> None:
    trainer = HierarchicalGRPOTrainer(
        gfam_reward_scorer=_FakeGFAMRewardScorer(),
        track_workers_history=True,
    )
    rollout = trainer.run(
        task=_make_task(),
        worker_pool=_make_worker_pool(),
        policy_config=ControllerPolicyConfig(parameter_sharing=True, shared_model_path="mock-shared"),
        rollout_config=RolloutConfig(num_decompositions=1, num_selections_per_decomposition=1),
        schedule=TrainingScheduleConfig(mode=TrainingMode.JOINT),
    )

    worker_ids = {
        execution.worker_id
        for execution in rollout.decompositions[0].selections[0].executions
    }
    assert worker_ids
    for worker_id in worker_ids:
        snapshot = trainer.worker_memory.snapshot_for(worker_id)
        assert snapshot.num_assignments > 0
        assert snapshot.average_reward != 0.0
        assert snapshot.recent_history
        for item in snapshot.recent_history:
            assert item["reward_source"] == "gfam_v1"
            assert "observed_reward" in item
            assert "confidence_reward" not in item


def test_gfam_inference_record_excludes_scratchpad_from_comparisons() -> None:
    pytest.importorskip("torch")
    try:
        from verl.hierarchical_rema.gfam_reward import _build_inference_record
    except ModuleNotFoundError:
        from hierarchical_rema.gfam_reward import _build_inference_record

    task = _make_task()
    decomposition = DecompositionCandidate(
        decomposition_id="decomp-1",
        summary="",
        target_quantity="value of x",
        final_answer_format_hint="integer",
        nodes=[
            SubtaskNode(
                node_id="1",
                instruction="Rewrite the equation into the form 2x = 8.",
                dependencies=[],
                required_skills=["algebra"],
                output_key="1_output",
            ),
            SubtaskNode(
                node_id="2",
                instruction="Use the equation from node 1 to isolate x and return the integer value of x.",
                dependencies=["1"],
                required_skills=["algebra"],
                output_key="final_answer",
            ),
        ],
        final_node_id="2",
        raw_text=(
            "<decomposer_scratchpad>\n"
            "Private note that should not reach GFAM.\n"
            "</decomposer_scratchpad>\n"
            "<decomposition_plan>\nTARGET_QUANTITY: value of x\nFINAL_ANSWER_FORMAT_HINT: integer\nFINAL_NODE_ID: 2\n"
            "NODE_ID: 1\nINSTRUCTION: Rewrite the equation into the form 2x = 8.\nDEPENDENCIES: none\nREQUIRED_SKILLS: algebra\n"
            "NODE_ID: 2\nINSTRUCTION: Use the equation from node 1 to isolate x and return the integer value of x.\nDEPENDENCIES: 1\nREQUIRED_SKILLS: algebra\n"
            "</decomposition_plan>"
        ),
    )
    selection = SelectionCandidate(
        selection_id="sel-1",
        assignments=[
            WorkerAssignment(
                node_id="1",
                worker_id="symbolic_manipulation_worker",
                rationale="Best fit.",
                compatibility=1.0,
            ),
            WorkerAssignment(
                node_id="2",
                worker_id="calculation_worker",
                rationale="Best fit.",
                compatibility=1.0,
            ),
        ],
        raw_text=(
            "<selector_scratchpad>\n"
            "Private routing note.\n"
            "</selector_scratchpad>\n"
            "<selection_plan>\n1: symbolic_manipulation_worker\n2: calculation_worker\n</selection_plan>"
        ),
    )
    executions = [
        WorkerExecution(
            node_id="1",
            worker_id="symbolic_manipulation_worker",
            output_text="2x = 8",
            raw_output_text=(
                "<worker_scratchpad>\n"
                "From the original equation, subtract 3.\n"
                "</worker_scratchpad>\n"
                "<worker_result>\n2x = 8\n</worker_result>"
            ),
            worker_prompt="prompt-1",
            dependency_outputs={},
            entropy=0.0,
            confidence_reward=0.0,
            compatibility=1.0,
        ),
        WorkerExecution(
            node_id="2",
            worker_id="calculation_worker",
            output_text="4",
            raw_output_text=(
                "<worker_scratchpad>\n"
                "Use 2x = 8 from node 1.\n"
                "</worker_scratchpad>\n"
                "<worker_result>\n4\n</worker_result>"
            ),
            worker_prompt="prompt-2",
            dependency_outputs={"1": "2x = 8"},
            entropy=0.0,
            confidence_reward=0.0,
            compatibility=1.0,
        ),
    ]

    record = _build_inference_record(
        task=task,
        decomposition=decomposition,
        selection=selection,
        executions=executions,
        final_answer="<worker_scratchpad>hidden</worker_scratchpad><worker_result>4</worker_result>",
    )

    assert "scratchpad" not in record["decomposition"]["raw_text"].lower()
    assert "scratchpad" not in record["selection"]["raw_text"].lower()
    assert "scratchpad" not in record["workers"][0]["raw_output_text"].lower()
    assert "scratchpad" not in record["workers"][1]["raw_output_text"].lower()
    assert "scratchpad" not in record["trajectory"]["final_answer"].lower()
    assert record["graph"]["used_dependency_edges"] == []

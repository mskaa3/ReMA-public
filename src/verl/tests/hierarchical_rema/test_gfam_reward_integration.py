import math

try:
    from verl.hierarchical_rema import (
        ControllerPolicyConfig,
        HierarchicalGRPOTrainer,
        RolloutConfig,
        TaskExample,
        TrainingMode,
        TrainingScheduleConfig,
    )
    from verl.hierarchical_rema.demo import make_worker_pool as make_default_worker_pool
except ModuleNotFoundError:
    from hierarchical_rema import (
        ControllerPolicyConfig,
        HierarchicalGRPOTrainer,
        RolloutConfig,
        TaskExample,
        TrainingMode,
        TrainingScheduleConfig,
    )
    from hierarchical_rema.demo import make_worker_pool as make_default_worker_pool


class _FakeGFAMRewardScorer:
    def score_rollout(self, *, task, decomposition, selection, executions, final_answer):
        worker_rewards = {
            execution.node_id: {"reward": 0.11 if execution.node_id == "1" else 0.22}
            for execution in executions
        }
        return {
            "source": "gfam_v1",
            "compiled_rewards": {
                "decomposer": {"reward": 0.77},
                "selector": {"reward": 0.55},
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
    assert math.isclose(selection.reward.total_reward, 0.55)
    assert selection.reward.to_dict() == {
        "total_reward": 0.55,
        "reward_model_source": "gfam_v1",
    }
    assert selection.reward_model_outputs["source"] == "gfam_v1"
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

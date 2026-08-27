from verl.rema_separated_trainer.ppo.multi_agent_rollout import MultiAgentRollout


def test_final_context_preserves_natural_worker_transcript():
    rollout = object.__new__(MultiAgentRollout)
    context = rollout._format_plan_and_worker_results_for_final(
        [("S1", "Compute an intermediate value."), ("S2", "Verify S1.")],
        [
            (
                "worker_stage_1",
                "general_math_worker",
                "S1",
                "REASONING:\nCompute carefully.\n\nLOCAL_RESULT: \\boxed{7}",
            ),
            (
                "worker_stage_2",
                "general_math_worker",
                "S2",
                "REASONING:\nThe previous value satisfies the constraint.\n\n"
                "LOCAL_RESULT: \\boxed{7}",
            ),
        ],
    )

    assert context.startswith("PLAN:\n")
    assert "WORK SO FAR:\nS1:\nREASONING:" in context
    assert "\n\nS2:\nREASONING:" in context
    assert "Compute carefully." in context
    assert "The previous value satisfies the constraint." in context
    assert "LOCAL RESULT SUMMARY" not in context
    assert "SUPPORTING WORKER REASONING" not in context

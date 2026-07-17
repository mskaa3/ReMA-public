import math

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.hierarchical_rema.gfam_reward import (
        DECOMPOSER_LABELS,
        FINAL_LABELS,
        GRAPH_LABELS,
        GFAMSmallModel,
        GraphExample,
        PRIMARY_FAILURE_STAGES,
        SELECTOR_LABELS,
        WORKER_LABELS,
        compile_rewards_from_predictions,
    )
except ModuleNotFoundError:
    from hierarchical_rema.gfam_reward import (
        DECOMPOSER_LABELS,
        FINAL_LABELS,
        GRAPH_LABELS,
        GFAMSmallModel,
        GraphExample,
        PRIMARY_FAILURE_STAGES,
        SELECTOR_LABELS,
        WORKER_LABELS,
        compile_rewards_from_predictions,
    )


def _make_example(*, final_answer: str = "42", final_worker_output: str = "42") -> GraphExample:
    return GraphExample(
        source_record={
            "trajectory": {
                "final_answer": final_answer,
                "final_node_id": "1",
            },
            "workers": [
                {
                    "node_id": "1",
                    "is_final_node": True,
                    "output_text": final_worker_output,
                }
            ],
        },
        text_embeddings=torch.zeros((1, 4), dtype=torch.float32),
        node_type_ids=torch.zeros(1, dtype=torch.long),
        role_ids=torch.zeros(1, dtype=torch.long),
        worker_bucket_ids=torch.zeros(1, dtype=torch.long),
        metadata_features=torch.zeros((1, 8), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_type_ids=torch.zeros(0, dtype=torch.long),
        decomposer_index=0,
        selector_index=0,
        final_index=0,
        worker_indices_by_node_id={"1": 0},
        edge_key_to_position={},
    )


def test_gfam_runtime_label_schema_matches_new_checkpoint() -> None:
    assert DECOMPOSER_LABELS == (
        "D_coverage",
        "D_dependency_correct",
        "D_subtasks_solvable",
        "D_role_drift",
        "D_under_decomposition",
    )
    assert WORKER_LABELS == (
        "W_subtask_solved",
        "W_used_dependencies",
        "W_format_correct",
        "W_reasoning_or_fact_error",
        "W_role_drift",
        "W_nonfinal_solved_final",
        "W_trivial_finalization",
        "W_contaminated_by_upstream",
        "W_contaminates_downstream",
    )
    assert FINAL_LABELS == (
        "F_aggregation_error",
        "F_verification_error",
        "F_answer_missing_or_invalid",
    )

    model = GFAMSmallModel(
        text_dim=4,
        metadata_dim=8,
        hidden_dim=16,
        message_passing_layers=1,
        dropout=0.0,
        worker_bucket_count=8,
    )
    assert model.decomposer_head.out_features == len(DECOMPOSER_LABELS) * 3 == 15
    assert model.worker_head.out_features == len(WORKER_LABELS) * 3 == 27
    assert model.final_head.out_features == len(FINAL_LABELS) * 3 == 9


def test_gfam_runtime_prediction_compiler_uses_centered_scores_and_new_outputs() -> None:
    example = _make_example(final_answer="42", final_worker_output="42")

    def neutral_logits(label_names: tuple[str, ...]) -> torch.Tensor:
        return torch.tensor([[0.0, 10.0, 0.0]] * len(label_names), dtype=torch.float32)

    outputs = {
        "graph_label_logits": neutral_logits(GRAPH_LABELS),
        "decomposer_logits": neutral_logits(DECOMPOSER_LABELS),
        "selector_logits": neutral_logits(SELECTOR_LABELS),
        "final_logits": neutral_logits(FINAL_LABELS),
        "worker_logits": {
            "1": neutral_logits(WORKER_LABELS),
        },
        "edge_logits": {},
        "primary_stage_logits": torch.zeros(len(PRIMARY_FAILURE_STAGES), dtype=torch.float32),
        "final_anchor_logit": torch.tensor(0.0, dtype=torch.float32),
        "ranking_score": torch.tensor(0.0, dtype=torch.float32),
    }

    compiled = compile_rewards_from_predictions(example, outputs)
    summary = compiled["graph_summary"]
    final_payload = compiled["node_rewards"]["final"]
    worker_payload = compiled["node_rewards"]["workers"]["1"]

    assert math.isclose(summary["graph_final_correct_score"], 0.0, abs_tol=1e-6)
    assert "hierarchy_bypass_score" in summary
    assert "final_failure_severity" in summary
    assert "final_answer_missing_or_invalid_score" in summary
    assert "reward_pre_cap" in final_payload
    assert "positive_cap" in final_payload
    assert "final_stage_penalty" in worker_payload
    assert "is_final_worker" in worker_payload

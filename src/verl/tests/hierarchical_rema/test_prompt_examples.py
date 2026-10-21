"""Few-shot diversity without exposing the task to downstream workers."""

import re

import pytest

try:
    from verl.hierarchical_rema.prompts import render_decomposer_prompt, render_worker_prompt
    from verl.hierarchical_rema.schema import (
        DecompositionCandidate, RolloutConfig, SubtaskNode, TaskExample, WorkerSpec,
    )
    from verl.hierarchical_rema.structured import (
        extract_decomposition_payload, validate_decomposition_payload,
    )
except ModuleNotFoundError:
    from hierarchical_rema.prompts import render_decomposer_prompt, render_worker_prompt
    from hierarchical_rema.schema import (
        DecompositionCandidate, RolloutConfig, SubtaskNode, TaskExample, WorkerSpec,
    )
    from hierarchical_rema.structured import (
        extract_decomposition_payload, validate_decomposition_payload,
    )


@pytest.mark.parametrize("cap,hard_cap,expected", [(1, None, [1]), (2, None, [2, 1]),
                                                  (3, None, [2, 3, 1]), (8, 2, [2, 1])])
def test_plan_examples_are_valid_and_respect_node_budget(cap, hard_cap, expected):
    task = TaskExample("test", "ACTUAL_PROBLEM_SENTINEL", "SECRET_REFERENCE")
    prompt = render_decomposer_prompt(task, max_nodes_hint=cap, hard_max_hops_hint=hard_cap)
    texts = re.findall(r"(?m)^<decomposition_plan>\n.*?^</decomposition_plan>", prompt, re.S)
    plans = [validate_decomposition_payload(extract_decomposition_payload(text),
             RolloutConfig(max_nodes_per_decomposition=cap, hard_max_hops=hard_cap), f"example-{i}")
             for i, text in enumerate(texts)]
    assert [len(plan.nodes) for plan in plans] == expected
    for text, plan in zip(texts, plans):
        assert text.splitlines()[-2] == f"FINAL_NODE_ID: {plan.final_node_id}"
        assert not plan.was_hard_truncated
        assert not plan.raw_payload.get("validation", {}).get("local_salvage_used")
    assert prompt.count("ACTUAL_PROBLEM_SENTINEL") == 1
    assert "SECRET_REFERENCE" not in prompt
    assert prompt.index("ACTUAL_PROBLEM_SENTINEL") > prompt.index("END EXAMPLES")
    assert "rewrite the equation into the form 2x = 8" not in prompt


def test_examples_teach_branching_and_carrying_givens_not_computed_answers():
    prompt = render_decomposer_prompt(TaskExample("test", "Actual task", "120"), max_nodes_hint=8)
    assert "DEPENDENCIES: 1,2" in prompt
    assert "given diagonal d = 24" in prompt
    assert "Only root workers receive TASK" in prompt
    assert "leave computation of unknown intermediate values and the answer to the workers" in prompt
    examples = prompt.split("ILLUSTRATIVE EXAMPLES", 1)[1].split("END EXAMPLES", 1)[0]
    for precomputed in ("s = 13", "scientist_pairs = 10", "artist_choices = 3", "0.375", "120"):
        assert precomputed not in examples


@pytest.mark.parametrize("node_index", [0, 1, 2])
def test_worker_examples_preserve_visibility_and_current_node_contract(node_index):
    task = TaskExample("task-sentinel", "ACTUAL_PROBLEM_SENTINEL", "SECRET_REFERENCE")
    plan = DecompositionCandidate(
        decomposition_id="plan", summary="", target_quantity="requested total",
        final_answer_format_hint="integer",
        nodes=[SubtaskNode(node_id="1", instruction="Compute first artifact."),
               SubtaskNode(node_id="2", instruction="Transform the supplied artifact.", dependencies=["1"]),
               SubtaskNode(node_id="3", instruction="Compute the total.", dependencies=["2"])],
        final_node_id="3",
    )
    worker = WorkerSpec("executor_worker", "Execute only the assigned node.", ["arithmetic"], "Execute.")
    node = plan.nodes[node_index]
    supplied = {dep: "DEPENDENCY_ARTIFACT_SENTINEL" for dep in node.dependencies}
    prompt = render_worker_prompt(task, plan, node, worker, supplied)
    actual = prompt.split("END EXAMPLES. Execute only the actual node below.", 1)[1]
    assert ("ACTUAL_PROBLEM_SENTINEL" in prompt) == (node_index == 0)
    assert "SECRET_REFERENCE" not in prompt
    assert f"NODE_INSTRUCTION: {node.instruction}" in actual
    assert f"FINAL_NODE: {'yes' if node_index == 2 else 'no'}" in actual
    assert ("DEPENDENCY_ARTIFACT_SENTINEL" in actual) == bool(supplied)
    examples = prompt.split("ILLUSTRATIVE EXAMPLES", 1)[1].split("END EXAMPLES", 1)[0]
    assert examples.count("<worker_result>") == 3
    assert "z^2 + 9" in examples
    assert "s = 13; d = 24" in examples
    assert "- NODE 1: scientist_pairs = 10" in examples
    assert "- NODE 2: artist_choices = 3" in examples
    assert "From 2x = 8" not in prompt

"""Footer output contract with backward-compatible header parsing."""

import re

import pytest

try:
    from verl.hierarchical_rema.backends import TransformersHierarchicalBackend
    from verl.hierarchical_rema.orchestrator import HierarchicalReMAOrchestrator
    from verl.hierarchical_rema.prompts import render_decomposer_prompt
    from verl.hierarchical_rema.schema import HFBackendConfig, RolloutConfig, TaskExample
    from verl.hierarchical_rema.structured import (
        StructuredOutputError,
        build_fallback_decomposition,
        extract_decomposition_payload,
        format_decomposition_plan,
        validate_decomposition_payload,
    )
except ModuleNotFoundError:
    from hierarchical_rema.backends import TransformersHierarchicalBackend
    from hierarchical_rema.orchestrator import HierarchicalReMAOrchestrator
    from hierarchical_rema.prompts import render_decomposer_prompt
    from hierarchical_rema.schema import HFBackendConfig, RolloutConfig, TaskExample
    from hierarchical_rema.structured import (
        StructuredOutputError,
        build_fallback_decomposition,
        extract_decomposition_payload,
        format_decomposition_plan,
        validate_decomposition_payload,
    )


def plan_text(position="footer", final="2"):
    fields = [
        "TARGET_QUANTITY: value of x",
        "FINAL_ANSWER_FORMAT_HINT: integer",
    ]
    nodes = [
        "NODE_ID: 1",
        "INSTRUCTION: Rewrite 2x + 3 = 11 as 2x = 8.",
        "DEPENDENCIES: none",
        "REQUIRED_SKILLS: algebra",
        "NODE_ID: 2",
        "INSTRUCTION: Use the equation from node 1 to isolate x.",
        "DEPENDENCIES: 1",
        "REQUIRED_SKILLS: algebra",
    ]
    final_line = [] if final is None else [f"FINAL_NODE_ID: {final}"]
    body = fields + final_line + nodes if position == "header" else fields + nodes + final_line
    return "\n".join(["<decomposition_plan>", *body, "</decomposition_plan>"])


def parse(text):
    return validate_decomposition_payload(extract_decomposition_payload(text), RolloutConfig(), "plan")


def assert_footer(text, final="2"):
    lines = text.splitlines()
    assert lines[-2:] == [f"FINAL_NODE_ID: {final}", "</decomposition_plan>"]
    assert sum(line.startswith("FINAL_NODE_ID:") for line in lines) == 1
    assert all(not line.startswith("FINAL_NODE_ID:") for line in lines[:-2])


def test_prompt_example_and_instructions_require_footer():
    task = TaskExample("test", "Solve 2x + 3 = 11.", "4")
    prompt = render_decomposer_prompt(task, max_nodes_hint=8)
    example = re.search(r"(?m)^<decomposition_plan>\n.*?^</decomposition_plan>", prompt, re.S)
    assert example is not None
    assert_footer(example.group())
    assert parse(example.group()).final_node_id == "2"
    assert "After the last node block, write exactly one FINAL_NODE_ID line" in prompt
    assert "Do not put FINAL_NODE_ID in the header" in prompt
    assert "not the allowed-ID range or hard node cap" in prompt


@pytest.mark.parametrize("position", ["header", "footer"])
def test_both_positions_have_identical_graph_and_canonical_footer(position):
    candidate = parse(plan_text(position))
    reference = parse(plan_text("header"))
    assert candidate.to_dict() == reference.to_dict()
    assert not candidate.raw_payload.get("validation")
    rendered = format_decomposition_plan(candidate)
    assert_footer(rendered)
    assert parse(rendered).to_dict() == candidate.to_dict()
    assert format_decomposition_plan(parse(rendered)) == rendered


@pytest.mark.parametrize("position", ["header", "footer"])
def test_training_completion_reformats_header_and_footer_candidates(position):
    candidate = parse(plan_text(position))
    candidate.raw_text = plan_text(position)
    before = candidate.to_dict()
    completion = HierarchicalReMAOrchestrator._canonical_decomposition_completion(candidate)
    assert_footer(completion)
    assert candidate.to_dict() == before


@pytest.mark.parametrize("position", ["header", "footer"])
def test_backend_keeps_original_output_for_audit_but_formats_footer(position):
    raw = plan_text(position)
    candidate = TransformersHierarchicalBackend._set_decomposition_artifacts(
        parse(raw), "prompt", {"raw_model_text": raw, "fallback_used": False},
    )
    assert_footer(candidate.raw_text)
    assert candidate.raw_payload["validation"]["raw_model_text"] == raw


@pytest.mark.parametrize("position", ["header", "footer"])
@pytest.mark.parametrize("final", ["8", "1", None])
def test_unique_sink_repair_remains_enabled_in_either_format(position, final):
    raw = plan_text(position, final)
    candidate = parse(raw)
    validation = candidate.raw_payload["validation"]
    assert validation["final_node_repair"]["resolved_final_node_id"] == "2"
    assert validation["local_salvage_used"] is True
    assert_footer(format_decomposition_plan(candidate))


@pytest.mark.parametrize("position", ["header", "footer"])
def test_unknown_dependencies_are_still_rejected(position):
    raw = plan_text(position).replace("DEPENDENCIES: 1", "DEPENDENCIES: 99")
    with pytest.raises(ValueError, match="unknown node"):
        parse(raw)


@pytest.mark.parametrize("position", ["header", "footer"])
def test_ambiguous_final_pointer_is_still_rejected(position):
    raw = plan_text(position, "8").replace("DEPENDENCIES: 1", "DEPENDENCIES: none")
    with pytest.raises(StructuredOutputError, match="ambiguous terminal nodes"):
        parse(raw)


def test_retry_prompt_requires_footer_too():
    prompt = TransformersHierarchicalBackend._decomposition_repair_prompt("TASK", "bad final")
    assert "After all node blocks, write exactly one FINAL_NODE_ID line" in prompt
    assert "immediately before </decomposition_plan>" in prompt
    assert "Do not put FINAL_NODE_ID in the header" in prompt


def test_single_node_fallback_also_uses_footer():
    candidate = build_fallback_decomposition("task", "Solve x = 2.", "bad", "invalid", RolloutConfig())
    assert_footer(candidate.raw_text, "1")
    assert candidate.raw_payload["validation"]["fallback_used"] is True


def test_footer_does_not_change_decoding_stop_tag():
    backend = TransformersHierarchicalBackend(HFBackendConfig(controller_constrained_decoding=True))
    overrides = backend._controller_sampling_overrides("decomposer")
    assert overrides["stop"] == ["</decomposition_plan>"]
    assert overrides["include_stop_str_in_output"] is True
    assert re.fullmatch(overrides["guided_regex"], plan_text())
    assert re.fullmatch(overrides["guided_regex"], plan_text("header"))

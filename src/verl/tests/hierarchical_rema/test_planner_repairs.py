"""Planner graph validation and round-batched repair without loading a model."""

from copy import deepcopy
import json

import pytest

try:
    from verl.hierarchical_rema.backends import (
        DecompositionRequest, TransformersHierarchicalBackend, RayVLLMHierarchicalBackend,
    )
    from verl.hierarchical_rema.schema import (
        ControllerPolicyConfig, HFBackendConfig, VLLMBackendConfig,
        RolloutConfig, TaskExample, WorkerPoolConfig,
    )
    from verl.hierarchical_rema.structured import (
        StructuredOutputError, extract_decomposition_payload,
        format_decomposition_plan, validate_decomposition_payload,
    )
    from verl.hierarchical_rema.controller_data import _controller_validation_bucket
except ModuleNotFoundError:
    from hierarchical_rema.backends import (
        DecompositionRequest, TransformersHierarchicalBackend, RayVLLMHierarchicalBackend,
    )
    from hierarchical_rema.schema import (
        ControllerPolicyConfig, HFBackendConfig, VLLMBackendConfig,
        RolloutConfig, TaskExample, WorkerPoolConfig,
    )
    from hierarchical_rema.structured import (
        StructuredOutputError, extract_decomposition_payload,
        format_decomposition_plan, validate_decomposition_payload,
    )
    from hierarchical_rema.controller_data import _controller_validation_bucket


def plan(final="20", label="value"):
    return {
        "target_quantity": label, "final_answer_format_hint": "integer", "final_node_id": final,
        # Declaration order is intentionally not dependency order.
        "nodes": [
            {"node_id": "20", "dependencies": ["10"], "instruction": f"Combine {label}; return the final answer."},
            {"node_id": "10", "dependencies": [], "instruction": f"Compute {label}. Preserve x < 3 and y > 2."},
        ],
    }


def validate(payload, **config):
    return validate_decomposition_payload(payload, RolloutConfig(**config), "test-plan")


@pytest.mark.parametrize("final,reason", [("99", "unknown_node"), ("", "missing"), ("10", "nonterminal_node")])
def test_repair_only_unique_terminal_node_with_audit(final, reason):
    payload = plan(final)
    original = deepcopy(payload)
    candidate = validate(payload)
    assert payload == original
    assert candidate.final_node_id == "2"
    assert [n.instruction for n in candidate.nodes] == [payload['nodes'][1]['instruction'], payload['nodes'][0]['instruction']]
    assert [n.dependencies for n in candidate.nodes] == [[], ['1']]
    audit = candidate.raw_payload['validation']['final_node_repair']
    assert audit['reason'] == reason
    assert audit['declared_final_node_id'] == final
    assert audit['resolved_final_node_id'] == '20'
    assert candidate.raw_payload['validation']['local_salvage_used'] is True
    # Canonicalization must not change the resolved graph on the next pass.
    again = validate(extract_decomposition_payload(format_decomposition_plan(candidate)))
    assert [n.to_dict() for n in again.nodes] == [n.to_dict() for n in candidate.nodes]
    assert 'final_node_repair' not in again.raw_payload.get('validation', {})


def test_valid_final_is_not_marked_repaired():
    candidate = validate(plan())
    assert candidate.final_node_id == '2'
    assert not candidate.raw_payload.get('validation')


def test_text_plan_without_final_header_is_resolved_and_audited():
    raw = """<decomposition_plan>
NODE_ID: 20
INSTRUCTION: Combine the intermediate result.
DEPENDENCIES: 10
NODE_ID: 10
INSTRUCTION: Compute the intermediate result.
DEPENDENCIES: none
</decomposition_plan>"""
    candidate, _ = TransformersHierarchicalBackend._parse_decomposition_candidate(
        raw, RolloutConfig(), 'text-plan',
    )
    assert candidate.final_node_id == '2'
    assert candidate.nodes[-1].instruction == 'Combine the intermediate result.'
    audit = candidate.raw_payload['validation']['final_node_repair']
    assert audit['reason'] == 'missing'
    assert audit['resolved_final_node_id'] == '20'


@pytest.mark.parametrize('kind', ['multiple_sinks', 'unknown_dependency', 'cycle', 'self_dependency', 'duplicate'])
def test_salvage_cannot_bypass_graph_validation(kind):
    payload = plan('99')
    if kind == 'multiple_sinks':
        payload['nodes'].append({'node_id': '30', 'instruction': 'Another answer.', 'dependencies': ['10']})
    elif kind == 'unknown_dependency':
        payload['nodes'][1]['dependencies'] = ['missing']
    elif kind == 'cycle':
        payload['nodes'][1]['dependencies'] = ['20']
    elif kind == 'self_dependency':
        payload['nodes'][1]['dependencies'] = ['10']
    else:
        payload['nodes'].append(deepcopy(payload['nodes'][0]))
    with pytest.raises(ValueError):
        TransformersHierarchicalBackend._parse_decomposition_candidate(
            json.dumps(payload), RolloutConfig(), 'invalid-plan',
        )


@pytest.mark.parametrize('final', ['99', '', '10'])
def test_multiple_sinks_rejected_even_with_final_answer_instruction(final):
    payload = plan(final)
    payload['nodes'].append({'node_id': '30', 'dependencies': ['10'], 'instruction': 'Return another final answer.'})
    with pytest.raises(StructuredOutputError, match='ambiguous terminal nodes'):
        validate(payload)


def test_disconnected_nodes_still_rejected_with_declared_final():
    payload = plan()
    payload['nodes'].append({'node_id': '30', 'dependencies': [], 'instruction': 'Unused work.'})
    with pytest.raises(StructuredOutputError, match='unused nodes'):
        validate(payload)


def test_unknown_dependency_not_repaired():
    payload = plan('99')
    payload['nodes'][1]['dependencies'] = ['missing']
    with pytest.raises(ValueError, match='depends on unknown node'):
        validate(payload)


def test_cycle_not_repaired_even_if_a_unique_sink_exists():
    payload = plan('99')
    payload['nodes'] += [
        {'node_id': '30', 'dependencies': ['40'], 'instruction': 'Cycle A'},
        {'node_id': '40', 'dependencies': ['30'], 'instruction': 'Cycle B'},
    ]
    with pytest.raises(ValueError, match='not a DAG'):
        validate(payload)


def test_graph_without_any_sink_rejected():
    payload = plan('99')
    payload['nodes'][1]['dependencies'] = ['20']
    with pytest.raises(StructuredOutputError, match='terminal sink'):
        validate(payload)


def test_self_dependency_is_not_silently_removed():
    payload = plan()
    payload['nodes'][1]['dependencies'] = ['10']
    with pytest.raises(StructuredOutputError, match='cannot depend on itself'):
        validate(payload)


def test_duplicate_ids_not_repaired():
    payload = plan('99')
    payload['nodes'].append(deepcopy(payload['nodes'][0]))
    with pytest.raises(StructuredOutputError, match='Duplicate node_id'):
        validate(payload)


def test_missing_instructions_and_empty_plans_rejected():
    payload = plan('99')
    payload['nodes'][0]['instruction'] = ''
    with pytest.raises(StructuredOutputError, match="missing 'instruction'"):
        validate(payload)
    payload['nodes'] = []
    with pytest.raises(StructuredOutputError, match='non-empty'):
        validate(payload)


def test_repaired_plan_keeps_execution_limits_and_audit():
    candidate = validate(plan('99'), max_nodes_per_decomposition=1)
    assert len(candidate.nodes) == 1
    assert candidate.was_hard_truncated
    assert candidate.raw_payload['validation']['final_node_repair']['declared_final_node_id'] == '99'


def request(name, model='mock-model'):
    return DecompositionRequest(
        task=TaskExample(f'task-{name}', f'TASK_{name}: Solve this problem.', '4'),
        worker_pool=WorkerPoolConfig(base_model_path='mock-worker', workers=[]),
        policy_config=ControllerPolicyConfig(decomposer_model_path=model),
        rollout_config=RolloutConfig(), decomposition_index=0, worker_performance={},
    )


@pytest.fixture(params=['hf', 'ray'])
def backend(request, monkeypatch):
    config = dict(max_format_retries=1, controller_batch_size=96, controller_temperature=.3,
                  decomposer_max_new_tokens=2048)
    value = (TransformersHierarchicalBackend(HFBackendConfig(**config)) if request.param == 'hf'
             else RayVLLMHierarchicalBackend(VLLMBackendConfig(**config)))
    def forbid_serial(**kwargs):
        pytest.fail('Batch repair must not call the serial generator')
    monkeypatch.setattr(value, '_generate_text', forbid_serial)
    return value


def test_pending_only_batched_rounds_preserve_alignment_and_fallback(backend, monkeypatch, capsys):
    requests = [request(name) for name in 'ABCD']
    batches = [
        [(json.dumps(plan(label='A')), .1), ('invalid B', .2), ('invalid C', .3), ('invalid D', .4)],
        [('still invalid B', .2), (json.dumps(plan(label='C')), .3), ('still invalid D', .4)],
        [(json.dumps(plan(label='B')), .2), ('permanently invalid D', .4)],
    ]
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return batches[len(calls)-1]
    monkeypatch.setattr(backend, '_generate_text_batch', generate)
    candidates = backend.sample_decompositions_batch(requests)
    assert [len(c['prompt_texts']) for c in calls] == [4, 3, 2]
    assert ['TASK_B' in p for p in calls[2]['prompt_texts']] == [True, False]
    assert 'TASK_D' in calls[2]['prompt_texts'][1]
    assert all(c['batch_size'] == 96 and c['max_new_tokens'] == 2048 and c['temperature'] == .3 for c in calls)
    assert calls[1]['log_label'] == 'decomposer_repair_round_1'
    assert all('Error:' in p for c in calls[1:] for p in c['prompt_texts'])
    assert [c.target_quantity for c in candidates[:3]] == ['A', 'B', 'C']
    assert [c.decomposition_id for c in candidates[:3]] == [f'task-{name}-decomp-0' for name in 'ABC']
    for c, name in zip(candidates, 'ABCD'):
        assert 'TASK_'+name in c.raw_payload['controller_prompt']
    b = candidates[1].raw_payload['validation']
    assert b['batch_repair_fallback'] is True and b['fallback_used'] is False
    assert b['attempt'] == 1 and b['repair_round'] == 2
    assert b['initial_raw_model_text'] == 'invalid B'
    assert len(b['errors_before_success']) == 1
    assert len(b['repair_history']) == 2 and b['repair_history'][-1]['accepted']
    assert _controller_validation_bucket(b) == 'model_repair'
    d = candidates[3].raw_payload['validation']
    assert d['fallback_used'] is True and len(d['errors']) == 2
    assert candidates[3].nodes[0].instruction == requests[3].task.prompt
    assert d['raw_text'] == 'permanently invalid D'
    assert _controller_validation_bucket(d) == 'fallback'
    log = capsys.readouterr().out
    assert 'event=round_start' in log and 'pending=3 total_candidates=4' in log
    assert 'recovered_after_repair=2 fallback_plans=1' in log
    assert 'request_index=4 total_requests=4' in log
    assert ' item=' not in log


def test_locally_repaired_candidates_do_not_need_model_retry(backend, monkeypatch):
    raw=json.dumps(plan('99'))
    calls=[]
    def generate(**kwargs):
        calls.append(kwargs)
        return [(raw,.1)]
    monkeypatch.setattr(backend,'_generate_text_batch',generate)
    candidate=backend.sample_decompositions_batch([request('A')])[0]
    assert len(calls)==1
    v=candidate.raw_payload['validation']
    assert v['raw_model_text']==raw
    assert v['final_node_repair']['resolved_final_node_id']=='20'
    assert _controller_validation_bucket(v)=='local_repair'
    assert not v.get('batch_repair_fallback')


def test_model_groups_keep_original_request_order(backend, monkeypatch):
    calls=[]
    def generate(**kwargs):
        calls.append(kwargs)
        if len(calls)==1:return [('bad',.0),(json.dumps(plan(label='C')),.0)]
        if len(calls)==2:return [(json.dumps(plan(label='A')),.0)]
        return [(json.dumps(plan(label='B')),.0)]
    monkeypatch.setattr(backend,'_generate_text_batch',generate)
    candidates=backend.sample_decompositions_batch([request('A','model-1'),request('B','model-2'),request('C','model-1')])
    assert [c.target_quantity for c in candidates]==['A','B','C']
    assert [c['base_model_path'] for c in calls]==['model-1','model-1','model-2']


@pytest.mark.parametrize('which_call',[0,1])
def test_result_count_mismatch_fails_instead_of_shifting_targets(backend,monkeypatch,which_call):
    calls=[]
    def generate(**kwargs):
        calls.append(kwargs)
        return [] if len(calls)-1==which_call else [('invalid',0.)]
    monkeypatch.setattr(backend,'_generate_text_batch',generate)
    with pytest.raises(RuntimeError,match='unexpected number'):
        backend.sample_decompositions_batch([request('A')])


def test_generation_runtime_errors_are_not_swallowed_as_format_errors(backend,monkeypatch):
    def fail(**kwargs):raise RuntimeError('GPU unavailable')
    monkeypatch.setattr(backend,'_generate_text_batch',fail)
    with pytest.raises(RuntimeError,match='GPU unavailable'):
        backend.sample_decompositions_batch([request('A')])


def test_empty_request_batch_never_generates(backend,monkeypatch):
    def fail(**kwargs):pytest.fail('No requests should mean no generation')
    monkeypatch.setattr(backend,'_generate_text_batch',fail)
    assert backend.sample_decompositions_batch([])==[]


def test_no_extra_round_after_all_candidates_recover(backend,monkeypatch):
    calls=[]
    def generate(**kwargs):
        calls.append(kwargs)
        if len(calls)==1:return [('invalid',0.)]
        assert len(calls)==2
        return [(json.dumps(plan()),0.)]
    monkeypatch.setattr(backend,'_generate_text_batch',generate)
    backend.sample_decompositions_batch([request('A')])
    assert len(calls)==2


def test_zero_format_retries_preserves_one_regeneration_after_initial_batch(backend,monkeypatch):
    backend.config.max_format_retries=0
    calls=[]
    def generate(**kwargs):
        calls.append(kwargs)
        return [('invalid',0.)]
    monkeypatch.setattr(backend,'_generate_text_batch',generate)
    candidate=backend.sample_decompositions_batch([request('A')])[0]
    assert len(calls)==2
    assert candidate.raw_payload['validation']['fallback_used']

"""Binary Graph+PRM inference for completed single-executor ReMA rollouts.

This module loads the frozen BGE/Qwen encoders and the small graph checkpoint.
It has no dependency on the offline training scripts, labels, or Python bytecode.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import torch

from . import graphprm_core as core
from .graphprm_model import FrozenQwenMarkerFeatureExtractor, build_fused_feature_tensor, instantiate_model_from_checkpoint
from .graphprm_compiler import compile_rewards_from_scores, validate_bad_class_penalty
from .graphprm_inputs import INPUT_VERSION, normalize_text_record, validate_marker_payload, apply_protocol_reward_guards

logger = logging.getLogger(__name__)


def normalize_record(record):
    record = normalize_text_record(record)
    # Reference answers and judge labels never enter model text or features.
    record['task'].pop('ground_truth', None)
    for name in list(record):
        if name.startswith('labels') or name.startswith('label_metadata'):
            record.pop(name)
    for name in ('declared_dependency_edges', 'used_dependency_edges'):
        for edge in record.get('graph', {}).get(name, []):
            edge['from_node_id'] = str(edge.get('from_node_id', edge.get('source')))
            edge['to_node_id'] = str(edge.get('to_node_id', edge.get('target')))
    subtasks = {str(x['node_id']): x for x in record['decomposition']['subtasks']}
    for worker in record.get('workers', []):
        worker['node_id'] = str(worker['node_id'])
        worker.setdefault('subtask', subtasks[worker['node_id']])
        worker.setdefault('declared_dependencies', worker.get('dependencies', []))
    record['trajectory']['final_answer'] = core.result_text(record['trajectory'].get('final_answer'))
    return record


def binary_predictions(logits, names):
    probabilities = torch.softmax(logits[:, [0, 2]], dim=-1).detach().cpu()
    if not torch.isfinite(probabilities).all():
        raise RuntimeError('Graph+PRM produced non-finite probabilities')
    return {name: {'prediction': int(p[1] >= p[0]),
                   'probabilities': {'0': float(p[0]), '1': float(p[1])},
                   'centered_score': float(p[1] - p[0])}
            for name, p in zip(names, probabilities)}


class GraphPRMRewardScorer:
    def __init__(self, checkpoint_path, device='cpu', encoder_backend=None, encoder_model=None,
                 prm_device='cpu', prm_torch_dtype='float32', prm_max_length=512,
                 cache_dir=None, encoder_device=None, bad_class_penalty=1.0):
        from .gfam_reward import resolve_device, build_sentence_encoder

        self.bad_class_penalty = validate_bad_class_penalty(bad_class_penalty)
        self.device = resolve_device(device)
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        spec = self.checkpoint.get('encoder_spec', {})
        if spec.get('backend') != 'graphprm_hybrid':
            raise ValueError('Expected a graphprm_hybrid checkpoint')
        for key, expected in [('node_types', core.NODE_TYPES), ('edge_types', core.EDGE_TYPES)]:
            if tuple(self.checkpoint.get(key, ())) != expected:
                raise ValueError(f'Incompatible Graph+PRM {key}')
        for scope, names in [('graph', core.GRAPH_LABELS), ('decomposer', core.DECOMPOSER_LABELS), ('worker', core.WORKER_LABELS), ('final', core.FINAL_LABELS)]:
            if tuple(self.checkpoint.get('label_schema', {}).get(scope, ())) != names:
                raise ValueError(f'Incompatible binary label schema for {scope}')
        state = self.checkpoint['model_state_dict']
        if state['worker_head.weight'].shape[0] != 2 * len(core.WORKER_LABELS):
            raise ValueError('Graph+PRM runtime requires binary checkpoint heads')
        self.model = instantiate_model_from_checkpoint(self.checkpoint, self.device)
        self.model.requires_grad_(False)
        self.layout = self.model.layout
        bge = spec['bge_encoder']
        self.encoder = build_sentence_encoder(encoder_backend or bge['backend'], encoder_model or bge['model_name'], resolve_device(encoder_device or device))
        self.prm_args = SimpleNamespace(
            prm_model=spec['prm_features']['model'], prm_device=prm_device,
            prm_device_map=None, prm_torch_dtype=prm_torch_dtype,
            prm_max_length=int(prm_max_length), prm_edge_mode='prefer_used',
            prm_max_problem_chars=3000, prm_max_step_chars=1400,
            prm_max_dependency_chars=450, include_final_prm_marker=True)
        if self.prm_args.prm_max_length <= 0:
            raise ValueError('PRM max length must be positive')
        self.extractor = None
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.lock = RLock()
        logger.info('Loaded binary Graph+PRM %s; PRM max_length=%d device=%s dtype=%s',
                    self.checkpoint_path, prm_max_length, prm_device, prm_torch_dtype)

    def _prm_features(self, record):
        if self.extractor is None:
            self.extractor = FrozenQwenMarkerFeatureExtractor(self.prm_args, include_hidden=True, include_score=True)
        render = self.extractor.render(record)
        alignment = self.extractor.alignment(render)
        identity = {'version': INPUT_VERSION, 'prompt': render['prompt'], **vars(self.prm_args)}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / key[:2] / (key + '.pt') if self.cache_dir else None
        if path and path.exists():
            payload = torch.load(path, map_location='cpu', weights_only=False)
        else:
            payload = self.extractor.extract_batch([render['prompt']])[0]
        validate_marker_payload(payload, alignment)
        hidden = payload.get('hidden_states')
        if hidden is not None and not torch.isfinite(hidden).all():
            raise RuntimeError('Non-finite Qwen marker states; use float32 extraction')
        scores = payload.get('scores', [])
        if scores and not torch.isfinite(torch.tensor(scores)).all():
            raise RuntimeError('Non-finite Qwen PRM scores')
        payload['targets'] = render['targets']
        missing = len(render['targets']) - int(payload['marker_count'])
        if missing:
            logger.warning('Graph+PRM: %d/%d markers truncated at %d tokens; using training-compatible zero features',
                           missing, len(render['targets']), self.prm_args.prm_max_length)
        if path and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f'.{os.getpid()}.tmp')
            torch.save(payload, temporary)
            os.replace(temporary, path)
        return payload, missing

    def score_record(self, record):
        # A scorer can be shared by executor-rollout threads; encoder access is serialized.
        with self.lock, torch.inference_mode():
            record = normalize_record(record)
            blueprint = core.prepare_graph_blueprint(record)
            bge = self.encoder.encode_texts(blueprint.node_texts).cpu()
            payload, missing = self._prm_features(record)
            text = build_fused_feature_tensor(blueprint=blueprint, bge_embeddings=bge, prm_payload=payload, layout=self.layout)
            if not torch.isfinite(text).all():
                raise RuntimeError('Non-finite Graph+PRM node features')
            edges = blueprint.edge_rows
            example = SimpleNamespace(source_record=record, text_embeddings=text,
                metadata_features=torch.tensor(blueprint.metadata_rows, dtype=torch.float32),
                node_type_ids=torch.tensor(blueprint.node_type_ids, dtype=torch.long),
                edge_index=torch.tensor([(s, d) for s, d, _ in edges], dtype=torch.long).T.contiguous(),
                edge_type_ids=torch.tensor([r for _, _, r in edges], dtype=torch.long),
                decomposer_index=blueprint.decomposer_index, final_index=blueprint.final_index,
                worker_indices_by_node_id=blueprint.worker_indices_by_node_id)
            outputs = self.model(example, self.device)
            predictions = {scope: binary_predictions(outputs[key], names) for scope, key, names in (
                ('graph', 'graph_label_logits', core.GRAPH_LABELS), ('decomposer', 'decomposer_logits', core.DECOMPOSER_LABELS),
                ('final', 'final_logits', core.FINAL_LABELS))}
            predictions['workers'] = {node: binary_predictions(logits, core.WORKER_LABELS) for node, logits in outputs['worker_logits'].items()}
            def centered(values):
                return {name: item['centered_score'] for name, item in values.items()}
            inputs = {'example': example, 'bad_class_penalty': self.bad_class_penalty,
                      'graph_scores': centered(predictions['graph']),
                      'decomposer_scores': centered(predictions['decomposer']), 'final_scores': centered(predictions['final']),
                      'worker_scores': {node: centered(values) for node, values in predictions['workers'].items()},
                      'final_anchor_score': float(2 * torch.sigmoid(outputs['final_anchor_logit']) - 1)}
            compiled = compile_rewards_from_scores(inputs)
            from .rewarding import WORKER_INVALID_RESULT_PENALTY
            invalid = apply_protocol_reward_guards(compiled['node_rewards'], record, WORKER_INVALID_RESULT_PENALTY)
            summary = compiled['graph_summary']
            summary.update(prm_marker_count=payload['marker_count'], missing_prm_markers=missing,
                           prm_max_length=self.prm_args.prm_max_length, reward_backend='graphprm_binary',
                           protocol_invalid_workers=invalid, input_version=INPUT_VERSION,
                           prm_total_tokens=payload['marker_alignment']['total_tokens'],
                           prm_retained_tokens=payload['marker_alignment']['retained_tokens'])
            return {'source': 'gfam_v1', 'backend': 'graphprm_binary', 'record': record,
                    'compiled_rewards': compiled['node_rewards'], 'graph_summary': summary,
                    'predictions': predictions, 'model_predictions': predictions,
                    'prm_step_scores': payload['scores']}

    def score_rollout(self, *, task, decomposition, selection, executions, final_answer):
        from .gfam_reward import _build_inference_record
        record = _build_inference_record(task, decomposition, selection, executions, final_answer)
        # Automatic assignments retain rollout IDs, but are not graph nodes/features.
        record['decomposition']['target_quantity'] = getattr(decomposition, 'target_quantity', '')
        record['decomposition']['final_answer_format_hint'] = getattr(decomposition, 'final_answer_format_hint', '')
        record['trajectory']['final_answer'] = final_answer
        executions_by_id = {str(execution.node_id): execution for execution in executions}
        for worker in record['workers']:
            execution = executions_by_id[str(worker['node_id'])]
            # Feed the shared cleaner the delivered fields, not the legacy cleaner's output.
            worker['output_text'] = execution.output_text
            worker['dependency_outputs'] = dict(execution.dependency_outputs)
            for context in worker.get('upstream_context', []):
                value = execution.dependency_outputs[context['node_id']]
                context['used_value'] = context['output_text'] = value
            worker['invalid_reason'] = execution.invalid_reason
            worker['raw_output_text'] = execution.raw_output_text
        return self.score_record(record)

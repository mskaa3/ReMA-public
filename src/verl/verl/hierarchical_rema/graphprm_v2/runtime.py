"""Deployment adapter using EXACTLY the same feature code as HPC training."""
from pathlib import Path
from threading import RLock
import tempfile
import os
import re
import torch

from . import FEATURE_SCHEMA
from . import graphprm_core as core
from .contract import canonical_record
from .features import DEFAULT_SPEC
from .live_features import LiveFeatureEngine
from .graphprm_model import instantiate_model_from_checkpoint
from .evaluation import predictions_from_outputs
from .rewards import compile_rewards, validate_failure_positive_scale, DEFAULT_VERIFIED_FAILURE_POSITIVE_SCALE
from .graphprm_sequence import resolve_device
from .live_logging import feature_progress


def live_record(task, decomposition, executions, final_answer):
    # Do not invoke the legacy adapter: it infers actual usage and downstream blame.
    return canonical_record({
        'task': {'prompt': task.prompt},
        'decomposition': {'subtasks': [node.to_dict() for node in decomposition.nodes],
            'final_node_id': str(decomposition.final_node_id),
            'target_quantity': decomposition.target_quantity,
            'final_answer_format_hint': decomposition.final_answer_format_hint},
        'trajectory': {'final_answer': final_answer, 'final_node_id': str(decomposition.final_node_id)},
        'workers': [{'node_id': str(ex.node_id), 'output_text': ex.output_text,
            'dependency_outputs': dict(ex.dependency_outputs), 'invalid_reason': ex.invalid_reason,
            'raw_output_text': ex.raw_output_text} for ex in executions],
    })


def validate_checkpoint(cp):
    if cp.get('feature_schema') != FEATURE_SCHEMA or cp.get('feature_spec', {}).get('feature_schema') != FEATURE_SCHEMA:
        raise ValueError('Wrong checkpoint feature schema; v1 and v2 are not interchangeable')
    if cp['feature_spec'].get('feature_code_sha256') != DEFAULT_SPEC['feature_code_sha256']:
        raise ValueError('Feature source code differs from training; install the matching shared package')
    for name, expected in [('node_types', core.NODE_TYPES), ('edge_types', core.EDGE_TYPES),
                           ('metadata_names', core.METADATA_NAMES)]:
        if tuple(cp.get(name, [])) != expected:
            raise ValueError(f'Incompatible checkpoint {name}')
    for scope, names in [('graph', core.GRAPH_LABELS), ('decomposer', core.DECOMPOSER_LABELS),
                         ('worker', core.WORKER_LABELS), ('final', core.FINAL_LABELS)]:
        if tuple(cp.get('label_schema', {}).get(scope, [])) != names:
            raise ValueError(f'Wrong binary labels: {scope}')


class GraphPRMV2RewardScorer:
    supports_verified_final_correctness = True

    def __init__(self, checkpoint_path, device='cpu', encoder_backend=None, encoder_model=None,
                 prm_device='cpu', prm_torch_dtype='auto', prm_max_length=0,
                 cache_dir=None, encoder_device=None, bad_class_penalty=1.,
                 verified_failure_positive_scale=None, **kwargs):
        self.verified_failure_positive_scale = validate_failure_positive_scale(
            os.environ.get('GFAM_VERIFIED_FAILURE_POSITIVE_SCALE', DEFAULT_VERIFIED_FAILURE_POSITIVE_SCALE)
            if verified_failure_positive_scale is None else verified_failure_positive_scale)
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        validate_checkpoint(self.checkpoint)
        spec = self.checkpoint['feature_spec']
        if prm_max_length not in (None, 0, spec['prm_max_length']):
            raise ValueError('PRM context override differs from training; use checkpoint settings (0/auto)')
        if prm_torch_dtype not in (None, 'auto', spec['prm_dtype']):
            raise ValueError('PRM dtype override differs from training; use auto')
        if encoder_model not in (None, '', spec['bge_model']):
            raise ValueError('BGE override differs from trained checkpoint')
        if encoder_backend not in (None, '', 'auto', 'sentence-transformers', 'sentence_transformers'):
            raise ValueError('v2 requires its trained SentenceTransformer encoder; no fallback backend')
        self.device = torch.device(resolve_device(device))
        self.model = instantiate_model_from_checkpoint(self.checkpoint, self.device)
        self.model.requires_grad_(False)
        self.rollout_batch_size = int(os.environ.get('GFAM_REWARD_BATCH_SIZE', '64'))
        if self.rollout_batch_size < 1:
            raise ValueError('GFAM_REWARD_BATCH_SIZE must be positive')
        local_cache = (Path(os.environ.get('TMPDIR_LOCAL') or tempfile.gettempdir()) /
                       f'graphprm_v2_{os.getuid()}_{os.environ.get("SLURM_JOB_ID", "local")}')
        self.engine = LiveFeatureEngine(spec, cache_dir or local_cache, device=prm_device,
            bge_batch_size=int(os.environ.get('GFAM_BGE_BATCH_SIZE', '32')),
            prm_batch_size=int(os.environ.get('GFAM_PRM_BATCH_SIZE', '16')),
            bge_cache_size=int(os.environ.get('GFAM_BGE_RAM_CACHE_SIZE', '8192')),
            graph_cache_size=int(os.environ.get('GFAM_GRAPH_RAM_CACHE_SIZE', '128')),
            encoder_device=encoder_device or device)
        self.bad_class_penalty = bad_class_penalty
        self.lock = RLock()

    def score_record(self, record, verified_final_correctness=None):
        return self.score_records([record], [verified_final_correctness])[0]

    def _overflow_result(self, row, exc):
        message = str(exc)
        prm_overflow = re.fullmatch(
            r'PRM context overflow: (\d+) > (\d+); no truncation/zero substitution is permitted\. Audit this rollout before training\.',
            message)
        if not prm_overflow and message != 'BGE context overflow; no silent truncation in v2':
            raise exc
        encoder = 'prm' if prm_overflow else 'bge'
        return {'source': 'gfam_v1', 'backend': 'graphprm_binary_v2',
                'status': 'unscored', 'record': row, 'compiled_rewards': {},
                'model_predictions': {}, 'prm_step_scores': [],
                'error': {'code': 'context_overflow', 'encoder': encoder,
                          'message': message,
                          'total_tokens': int(prm_overflow[1]) if prm_overflow else None,
                          'max_tokens': self.engine.spec[f'{encoder}_max_length']},
                'graph_summary': {'feature_schema': FEATURE_SCHEMA,
                                  'reward_status': 'unscored', 'reward_backend': 'graphprm_binary_v2'}}

    def score_records(self, records, verified_final_correctness=None):
        outcomes = [None] * len(records) if verified_final_correctness is None else verified_final_correctness
        if len(outcomes) != len(records):
            raise ValueError('One verified outcome is required per record')
        results = []
        with self.lock, torch.inference_mode(), feature_progress(
            visible=os.environ.get('GFAM_FEATURE_PROGRESS', '0').lower() in {'1', 'true', 'yes'}
        ):
            for start in range(0, len(records), self.rollout_batch_size):
                rows = [canonical_record(record) for record in records[start:start + self.rollout_batch_size]]
                errors = self.engine.prepare_safe(rows)
                for index, row in enumerate(rows):
                    if index in errors:
                        results.append(self._overflow_result(row, errors[index]))
                        continue
                    example = self.engine.example(row)
                    predictions = predictions_from_outputs(self.model(example, self.device))
                    compiled = compile_rewards(predictions, row, self.bad_class_penalty,
                                               verified_final_correctness=outcomes[start + index],
                                               verified_failure_positive_scale=self.verified_failure_positive_scale)
                    summary = compiled['graph_summary']
                    summary.update(feature_schema=FEATURE_SCHEMA, reward_backend='graphprm_binary_v2',
                                   missing_prm_markers=0, prm_total_tokens=example.prm_alignment['total_tokens'],
                                   prm_max_length=self.engine.spec['prm_max_length'])
                    results.append({'source': 'gfam_v1', 'backend': 'graphprm_binary_v2', 'status': 'scored',
                        'record': row, 'compiled_rewards': compiled['node_rewards'], 'graph_summary': summary,
                        'predictions': predictions, 'model_predictions': predictions, 'prm_step_scores': example.prm_scores})
        return results

    def score_rollouts(self, requests):
        rows = [live_record(item['task'], item['decomposition'], item['executions'], item['final_answer'])
                for item in requests]
        return self.score_records(rows, [item.get('verified_final_correctness') for item in requests])

    def score_rollout(self, *, task, decomposition, selection, executions, final_answer,
                      verified_final_correctness=None):
        return self.score_record(live_record(task, decomposition, executions, final_answer),
                                 verified_final_correctness=verified_final_correctness)

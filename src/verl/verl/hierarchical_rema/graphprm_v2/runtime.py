"""Deployment adapter using EXACTLY the same feature code as HPC training."""
from pathlib import Path
from threading import RLock
import os
import re
import torch

from . import FEATURE_SCHEMA
from . import graphprm_core as core
from .contract import canonical_record
from .features import FeatureEngine, DEFAULT_SPEC
from .graphprm_model import instantiate_model_from_checkpoint
from .evaluation import predictions_from_outputs
from .rewards import compile_rewards
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
                 cache_dir=None, encoder_device=None, bad_class_penalty=1., **kwargs):
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
        self.engine = FeatureEngine(spec, cache_dir or Path.home() / '.cache' / 'graphprm_v2',
                                    device=prm_device, bge_batch_size=16, prm_batch_size=1,
                                    encoder_device=encoder_device or device)
        self.engine.keep_models = True
        self.bad_class_penalty = bad_class_penalty
        self.lock = RLock()

    def score_record(self, record, verified_final_correctness=None):
        with self.lock, torch.inference_mode(), feature_progress(
            visible=os.environ.get('GFAM_FEATURE_PROGRESS', '0').lower() in {'1', 'true', 'yes'}
        ):
            row = canonical_record(record)
            try:
                self.engine.prepare([row])
            except ValueError as exc:
                # Only the pinned feature builder's explicit context errors are recoverable.
                # Alignment, nonfinite features, and incompatible checkpoints still fail loudly.
                message = str(exc)
                prm_overflow = re.fullmatch(
                    r'PRM context overflow: (\d+) > (\d+); no truncation/zero substitution is permitted\. Audit this rollout before training\.',
                    message)
                if not prm_overflow and message != 'BGE context overflow; no silent truncation in v2':
                    raise
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
            example = self.engine.example(row)
            predictions = predictions_from_outputs(self.model(example, self.device))
            compiled = compile_rewards(predictions, row, self.bad_class_penalty,
                                       verified_final_correctness=verified_final_correctness)
            summary = compiled['graph_summary']
            summary.update(feature_schema=FEATURE_SCHEMA, reward_backend='graphprm_binary_v2',
                           missing_prm_markers=0, prm_total_tokens=example.prm_alignment['total_tokens'],
                           prm_max_length=self.engine.spec['prm_max_length'])
            return {'source': 'gfam_v1', 'backend': 'graphprm_binary_v2', 'status': 'scored', 'record': row,
                    'compiled_rewards': compiled['node_rewards'], 'graph_summary': summary,
                    'predictions': predictions, 'model_predictions': predictions,
                    'prm_step_scores': example.prm_scores}

    def score_rollout(self, *, task, decomposition, selection, executions, final_answer,
                      verified_final_correctness=None):
        return self.score_record(live_record(task, decomposition, executions, final_answer),
                                 verified_final_correctness=verified_final_correctness)

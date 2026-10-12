"""Checkpoint-selected rendering, preserving the original v2 feature contract."""
from pathlib import Path

from .data import digest
from .features import FeatureEngine, DEFAULT_SPEC as LEGACY_SPEC
from .graphprm_inputs import marker_alignment
from .graphprm_model import render_graphprm_sequence
from .serialization import render_compact_sequence, SERIALIZATION_VERSION

DEFAULT_SPEC = dict(LEGACY_SPEC, serialization_version=SERIALIZATION_VERSION, implementation_version=2)
DEFAULT_SPEC['feature_code_sha256'] = digest({
    'legacy_feature_code_sha256': LEGACY_SPEC['feature_code_sha256'],
    **{name: (Path(__file__).parent / name).read_text()
       for name in ('serialization.py', 'versioned_features.py')},
})


def validate_feature_spec(spec):
    version = spec.get('serialization_version', 'legacy_v1')
    if version not in ('legacy_v1', SERIALIZATION_VERSION):
        raise ValueError(f'Unsupported PRM serialization version: {version}')
    expected = LEGACY_SPEC if version == 'legacy_v1' else DEFAULT_SPEC
    if spec.get('feature_code_sha256') != expected['feature_code_sha256']:
        raise ValueError('Feature source code differs from training; install the matching shared package')


class VersionedFeatureEngine(FeatureEngine):
    def __init__(self, spec, *args, **kwargs):
        validate_feature_spec(spec)
        super().__init__(spec, *args, **kwargs)

    def render_for_audit(self, row):
        tok = self._tokenizer()
        if self.spec.get('serialization_version', 'legacy_v1') == 'legacy_v1':
            render = render_graphprm_sequence(row, tokenizer=tok, edge_mode='provided',
                max_problem_chars=0, max_step_chars=0, max_dependency_chars=0, include_final_marker=True)
        else:
            render = render_compact_sequence(row, tok)
        alignment = marker_alignment(tok, render['prompt'], render['targets'], self.spec['prm_max_length'])
        return render, alignment

    def render(self, row):
        render, alignment = self.render_for_audit(row)
        if alignment['total_tokens'] > self.spec['prm_max_length']:
            raise ValueError(f"PRM context overflow: {alignment['total_tokens']} > {self.spec['prm_max_length']}; "
                             'no truncation/zero substitution is permitted. Audit this rollout before training.')
        return render, alignment

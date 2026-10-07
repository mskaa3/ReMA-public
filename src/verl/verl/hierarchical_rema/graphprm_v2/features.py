"""Frozen, revision-pinned encoders with atomic content-addressed CPU caches."""
import gc
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

from . import FEATURE_SCHEMA
from .data import digest, feature_key
from .graphprm_core import prepare_graph_blueprint
from .graphprm_inputs import marker_alignment, validate_marker_payload
from .graphprm_model import (GraphPRMFeatureLayout, render_graphprm_sequence,
                             marker_features_from_outputs, build_fused_feature_tensor)
from .graphprm_sequence import load_tokenizer, load_qwen_prm, resolve_device


DEFAULT_SPEC = {
    'feature_schema': FEATURE_SCHEMA, 'mode': 'bge_prm_hidden',
    'bge_model': 'BAAI/bge-m3', 'bge_revision': '5617a9f61b028005a4858fdac845db406aefb181',
    'bge_max_length': 8192, 'bge_dim': 1024,
    'prm_model': 'Qwen/Qwen2.5-Math-PRM-7B',
    'prm_revision': '0610740060112df12585d00a1c5f4624d2f59051',
    'prm_max_length': 4096, 'prm_dtype': 'bfloat16', 'prm_hidden_dim': 3584,
    'edge_mode': 'provided', 'overflow_policy': 'error', 'include_final_marker': True,
    'hidden_storage_dtype': 'float32', 'implementation_version': 1,
}
DEFAULT_SPEC['feature_code_sha256'] = digest({name: (Path(__file__).parent / name).read_text()
    for name in ('contract.py', 'graphprm_core.py', 'graphprm_model.py',
                 'graphprm_inputs.py', 'graphprm_sequence.py', 'features.py')})


def layout_from_spec(spec):
    return GraphPRMFeatureLayout('bge_prm_hidden', spec['bge_dim'], spec['prm_hidden_dim'],
                                 2, ('has_prm_marker', 'is_final_prm_marker'))


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_cache(path, identity):
    if not path.exists():
        return None
    # Cache files are trusted local artifacts, never untrusted downloads.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('identity') != identity:
        raise ValueError(f'Cache identity mismatch: {path}')
    return payload


class FeatureEngine:
    def __init__(self, spec, cache_dir, device='cuda', bge_batch_size=32, prm_batch_size=2, encoder_device=None):
        self.spec = dict(spec)
        if spec['feature_schema'] != FEATURE_SCHEMA:
            raise ValueError('Unsupported feature schema')
        if spec['prm_max_length'] > 4096 or spec['prm_max_length'] < 1:
            raise ValueError('This pinned Qwen checkpoint supports at most 4096 tokens')
        self.cache_dir, self.device = Path(cache_dir), resolve_device(device)
        self.encoder_device = resolve_device(encoder_device or device)
        self.bge_batch_size, self.prm_batch_size = bge_batch_size, prm_batch_size
        self.layout = layout_from_spec(spec)
        self.tokenizer = None
        self.keep_models = False
        self.encoder = self.prm_model = None

    def graph_path(self, row):
        return self.cache_dir / 'graphs' / (feature_key(row, self.spec) + '.pt')

    def example(self, row):
        identity = feature_key(row, self.spec)
        payload = load_cache(self.graph_path(row), identity)
        if payload is None:
            raise FileNotFoundError(f'Missing prepared graph: {identity}. Run prepare first.')
        fields = payload['example']
        if not torch.isfinite(fields['text_embeddings']).all():
            raise ValueError(f'Invalid cached graph features: {identity}')
        return SimpleNamespace(**fields, source_record=row)

    def _tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = load_tokenizer(self.spec['prm_model'], self.spec['prm_revision'])
        return self.tokenizer

    def render(self, row):
        tok = self._tokenizer()
        render = render_graphprm_sequence(row, tokenizer=tok, edge_mode='provided',
                                         max_problem_chars=0, max_step_chars=0, max_dependency_chars=0,
                                         include_final_marker=True)
        alignment = marker_alignment(tok, render['prompt'], render['targets'], self.spec['prm_max_length'])
        if alignment['total_tokens'] > self.spec['prm_max_length']:
            raise ValueError(f"PRM context overflow: {alignment['total_tokens']} > {self.spec['prm_max_length']}; "
                             'no truncation/zero substitution is permitted. Audit this rollout before training.')
        return render, alignment

    def prepare(self, rows):
        pending = {}
        for row in rows:
            identity = feature_key(row, self.spec)
            if self.graph_path(row).exists():
                self.example(row)
            else:
                pending[identity] = row
        print(f'[features] graphs={len(rows)} unique_uncached={len(pending)}', flush=True)
        if not pending:
            return
        blueprints = {key: prepare_graph_blueprint(row) for key, row in pending.items()}
        # Audit ALL sequences before spending time encoding; a cache hit cannot hide truncation.
        renders = {key: self.render(row) for key, row in tqdm(pending.items(), desc='audit PRM sequences')}
        bge_spec = {k: v for k, v in self.spec.items() if k.startswith('bge_')}
        text_keys = {text: digest({'spec': bge_spec, 'text': text})
                     for b in blueprints.values() for text in b.node_texts}
        embeddings, todo = {}, []
        for text, key in text_keys.items():
            payload = load_cache(self.cache_dir / 'bge' / (key + '.pt'), key)
            if payload is None:
                todo.append(text)
            else:
                embeddings[text] = payload['embedding']
        if todo:
            from sentence_transformers import SentenceTransformer
            encoder = self.encoder or SentenceTransformer(self.spec['bge_model'], revision=self.spec['bge_revision'], device=self.encoder_device)
            encoder.max_seq_length = self.spec['bge_max_length']
            encoder.eval().requires_grad_(False)
            for offset in tqdm(range(0, len(todo), self.bge_batch_size), desc='encode unique BGE texts'):
                texts = todo[offset:offset + self.bge_batch_size]
                # SentenceTransformer normally truncates silently too: reject that here.
                if any(len(encoder.tokenizer(t, truncation=False)['input_ids']) > encoder.max_seq_length for t in texts):
                    raise ValueError('BGE context overflow; no silent truncation in v2')
                with torch.inference_mode():
                    vectors = encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                                             batch_size=self.bge_batch_size, show_progress_bar=False).float().cpu()
                if vectors.shape[1] != self.spec['bge_dim'] or not torch.isfinite(vectors).all():
                    raise ValueError('Invalid BGE embeddings')
                for text, vector in zip(texts, vectors):
                    embeddings[text] = vector
                    key = text_keys[text]
                    atomic_torch(self.cache_dir / 'bge' / (key + '.pt'), {'identity': key, 'embedding': vector})
            if self.keep_models:
                self.encoder = encoder
            del encoder
            free_gpu()
        prm_payloads, prm_todo = {}, []
        for key, (render, alignment) in renders.items():
            identity = digest({'spec': self.spec, 'prompt': render['prompt'], 'targets': render['targets']})
            path = self.cache_dir / 'prm' / (identity + '.pt')
            saved = load_cache(path, identity)
            if saved:
                payload = saved['payload']
                self._validate(payload, alignment)
                prm_payloads[key] = payload
            else:
                prm_todo.append((key, identity, path))
        prm_todo.sort(key=lambda item: renders[item[0]][1]['total_tokens'])
        if prm_todo:
            args = SimpleNamespace(model=self.spec['prm_model'], revision=self.spec['prm_revision'],
                                   device=self.device, device_map=None, torch_dtype=self.spec['prm_dtype'])
            if self.prm_model is None:
                model, input_device = load_qwen_prm(args, self._tokenizer())
            else:
                model, input_device = self.prm_model, self.device
            model.eval().requires_grad_(False)
            sep_id = self.tokenizer.encode('<extra_0>', add_special_tokens=False)[0]
            for offset in tqdm(range(0, len(prm_todo), self.prm_batch_size), desc='extract Qwen markers'):
                batch = prm_todo[offset:offset + self.prm_batch_size]
                encoded = self.tokenizer([renders[k][0]['prompt'] for k, _, _ in batch],
                                         padding=True, truncation=False, return_tensors='pt').to(input_device)
                with torch.inference_mode():
                    outputs = model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
                    payloads = marker_features_from_outputs(outputs=outputs, encoded=encoded,
                                    step_sep_id=sep_id, include_hidden=True, include_score=True)
                del outputs, encoded
                for (key, identity, path), payload in zip(batch, payloads):
                    payload['targets'] = renders[key][0]['targets']
                    self._validate(payload, renders[key][1])
                    atomic_torch(path, {'identity': identity, 'payload': payload})
                    prm_payloads[key] = payload
            if self.keep_models:
                self.prm_model = model
            del model
            free_gpu()
        for key, row in tqdm(pending.items(), desc='assemble graph features'):
            bp = blueprints[key]
            text = build_fused_feature_tensor(blueprint=bp,
                bge_embeddings=torch.stack([embeddings[t] for t in bp.node_texts]),
                prm_payload=prm_payloads[key], layout=self.layout)
            if not torch.isfinite(text).all():
                raise ValueError('Non-finite fused features')
            edges = bp.edge_rows
            fields = dict(text_embeddings=text, metadata_features=torch.tensor(bp.metadata_rows, dtype=torch.float32),
                node_type_ids=torch.tensor(bp.node_type_ids),
                edge_index=torch.tensor([(s, d) for s, d, _ in edges], dtype=torch.long).T.contiguous(),
                edge_type_ids=torch.tensor([r for _, _, r in edges]), decomposer_index=bp.decomposer_index,
                final_index=bp.final_index, worker_indices_by_node_id=bp.worker_indices_by_node_id,
                prm_scores=prm_payloads[key]['scores'], prm_alignment=renders[key][1])
            atomic_torch(self.graph_path(row), {'identity': key, 'example': fields})

    def _validate(self, payload, alignment):
        validate_marker_payload(payload, alignment)
        hidden = payload['hidden_states']
        if hidden is None or hidden.shape[1] != self.spec['prm_hidden_dim'] or not torch.isfinite(hidden).all():
            raise ValueError('Qwen returned invalid marker states; nothing cached. Try float32 extraction.')
        scores = torch.tensor(payload['scores'])
        if not torch.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
            raise ValueError('Invalid PRM diagnostic scores')

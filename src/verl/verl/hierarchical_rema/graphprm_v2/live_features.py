"""Batched live execution of the pinned feature contract, without changing it.

The offline builder remains the reference implementation. This adapter changes
batching, transfers and caches only; rendering, validation and fusion are shared.
"""
from collections import OrderedDict
import gc
import time
from types import SimpleNamespace

import torch

from .features import FeatureEngine, atomic_torch, load_cache
from .data import digest, feature_key
from .graphprm_core import prepare_graph_blueprint
from .graphprm_model import build_fused_feature_tensor
from .graphprm_sequence import load_qwen_prm


class BoundedCache:
    def __init__(self, capacity):
        self.capacity = max(0, int(capacity))
        self.items = OrderedDict()

    def get(self, key):
        value = self.items.get(key)
        if value is not None:
            self.items.move_to_end(key)
        return value

    def put(self, key, value):
        if not self.capacity:
            return
        self.items[key] = value
        self.items.move_to_end(key)
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)


def marker_features_on_device(*, outputs, encoded, step_sep_id):
    """Transfer marker rows only; keep score arithmetic on CPU like the reference."""
    logits = outputs[0] if isinstance(outputs, (tuple, list)) else getattr(outputs, 'logits', None)
    states = getattr(outputs, 'hidden_states', None)
    if logits is None or logits.ndim != 3 or logits.shape[-1] < 2 or not states:
        raise RuntimeError('Qwen PRM must expose logits and final hidden states')
    ids = encoded['input_ids']
    attention = encoded.get('attention_mask', torch.ones_like(ids)).bool()
    mask = (ids == step_sep_id) & attention
    counts = mask.sum(dim=1).tolist()
    hidden = states[-1][mask].detach().cpu()
    probabilities = torch.softmax(logits[mask].detach().float().cpu(), dim=-1)
    # Token positions are in the unpadded sequence, exactly as in offline caches.
    positions = (attention.long().cumsum(dim=1) - 1)[mask].detach().cpu().tolist()
    rows, offset = [], 0
    for count in counts:
        end = offset + count
        rows.append(dict(scores=probabilities[offset:end, 1].tolist(),
                         hidden_states=hidden[offset:end].float().clone() if count else None,
                         marker_count=count, marker_positions=positions[offset:end]))
        offset = end
    return rows


def context_overflow(exc):
    text = str(exc)
    return (text.startswith('PRM context overflow: ') and
            text.endswith('no truncation/zero substitution is permitted. Audit this rollout before training.')) or (
                text == 'BGE context overflow; no silent truncation in v2')


class LiveFeatureEngine(FeatureEngine):
    def __init__(self, *args, bge_cache_size=8192, graph_cache_size=128, **kwargs):
        super().__init__(*args, **kwargs)
        if self.prm_batch_size < 1 or self.bge_batch_size < 1:
            raise ValueError('Live encoder batch sizes must be positive')
        self.keep_models = True
        self._texts = BoundedCache(bge_cache_size)
        self._graphs = BoundedCache(graph_cache_size)
        self.last_stats = {}

    def example(self, row):
        key = feature_key(row, self.spec)
        fields = self._graphs.get(key)
        if fields is None:
            example = super().example(row)
            fields = {k: v for k, v in vars(example).items() if k != 'source_record'}
            self._graphs.put(key, fields)
        return SimpleNamespace(**fields, source_record=row)

    def _encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.spec['bge_model'], revision=self.spec['bge_revision'],
                                               device=self.encoder_device)
            self.encoder.max_seq_length = self.spec['bge_max_length']
            self.encoder.eval().requires_grad_(False)
        return self.encoder

    def _batches(self, items, kind, operation):
        offset = 0
        attribute = f'{kind}_batch_size'
        while offset < len(items):
            batch = items[offset:offset + getattr(self, attribute)]
            retry = False
            try:
                values = operation(batch)
            except torch.cuda.OutOfMemoryError:
                if len(batch) <= 1:
                    raise
                reduced = max(1, len(batch) // 2)
                setattr(self, attribute, reduced)
                self.last_stats['oom_retries'] += 1
                print(f'[graphprm][batch] encoder={kind} CUDA_OOM batch={len(batch)} '
                      f'retry_batch={reduced}', flush=True)
                retry = True
            if retry:
                # The exception traceback is gone before reclaiming its tensors.
                gc.collect()
                torch.cuda.empty_cache()
                continue
            if len(values) != len(batch):
                raise ValueError(f'{kind} encoder returned the wrong number of rows')
            self.last_stats[f'{kind}_batches'] += 1
            self.last_stats[f'{kind}_max_batch'] = max(self.last_stats[f'{kind}_max_batch'], len(batch))
            yield batch, values
            offset += len(batch)

    def _encode_texts(self, texts):
        with torch.inference_mode():
            vectors = self._encoder().encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                                             batch_size=len(texts), show_progress_bar=False).float().cpu()
        if vectors.ndim != 2 or vectors.shape[1] != self.spec['bge_dim'] or not torch.isfinite(vectors).all():
            raise ValueError('Invalid BGE embeddings')
        # Independent storage keeps each disk/RAM entry from retaining a batch.
        return [vector.clone() for vector in vectors]

    def _extract(self, batch, renders):
        encoded = self.tokenizer([renders[key][0]['prompt'] for key, _, _ in batch],
                                 padding=True, truncation=False, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            outputs = self.prm_model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
            payloads = marker_features_on_device(outputs=outputs, encoded=encoded,
                step_sep_id=self.tokenizer.encode('<extra_0>', add_special_tokens=False)[0])
        self.last_stats['prm_padded_tokens'] += encoded['input_ids'].numel()
        self.last_stats['prm_input_tokens'] += int(encoded['attention_mask'].sum())
        return payloads

    def prepare_safe(self, rows):
        """Return overflow errors by original row index; never shift valid targets."""
        started = time.monotonic()
        self.last_stats = dict(rows=len(rows), cached_graphs=0, bge_ram_hits=0, oom_retries=0,
                               bge_batches=0, prm_batches=0, bge_max_batch=0, prm_max_batch=0,
                               prm_input_tokens=0, prm_padded_tokens=0)
        device = torch.device(self.device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        pending, blueprints, renders, errors = {}, {}, {}, {}
        text_lengths = {}
        for index, row in enumerate(rows):
            key = feature_key(row, self.spec)
            if self._graphs.get(key) is not None or self.graph_path(row).exists():
                self.example(row)
                self.last_stats['cached_graphs'] += 1
                continue
            if key in pending:
                continue
            try:
                render = self.render(row)
                bp = prepare_graph_blueprint(row)
                encoder = self._encoder()
                for text in bp.node_texts:
                    if text not in text_lengths:
                        text_lengths[text] = len(encoder.tokenizer(text, truncation=False)['input_ids'])
                    if text_lengths[text] > self.spec['bge_max_length']:
                        raise ValueError('BGE context overflow; no silent truncation in v2')
            except ValueError as exc:
                if not context_overflow(exc):
                    raise
                errors[index] = exc
                continue
            pending[key], blueprints[key], renders[key] = row, bp, render

        bge_spec = {k: v for k, v in self.spec.items() if k.startswith('bge_')}
        text_keys = {text: digest({'spec': bge_spec, 'text': text})
                     for bp in blueprints.values() for text in bp.node_texts}
        embeddings, todo = {}, []
        for text, key in text_keys.items():
            vector = self._texts.get(key)
            if vector is not None:
                self.last_stats['bge_ram_hits'] += 1
            else:
                saved = load_cache(self.cache_dir / 'bge' / (key + '.pt'), key)
                vector = saved['embedding'] if saved is not None else None
            if vector is None:
                todo.append(text)
            else:
                embeddings[text] = vector
                self._texts.put(key, vector)
        todo.sort(key=text_lengths.__getitem__)
        for texts, vectors in self._batches(todo, 'bge', self._encode_texts):
            for text, vector in zip(texts, vectors):
                key = text_keys[text]
                embeddings[text] = vector
                self._texts.put(key, vector)
                atomic_torch(self.cache_dir / 'bge' / (key + '.pt'), {'identity': key, 'embedding': vector})

        payloads, todo = {}, []
        for key, (render, alignment) in renders.items():
            identity = digest({'spec': self.spec, 'prompt': render['prompt'], 'targets': render['targets']})
            path = self.cache_dir / 'prm' / (identity + '.pt')
            saved = load_cache(path, identity)
            if saved is not None:
                self._validate(saved['payload'], alignment)
                payloads[key] = saved['payload']
            else:
                todo.append((key, identity, path))
        todo.sort(key=lambda item: renders[item[0]][1]['total_tokens'])
        if todo and self.prm_model is None:
            args = SimpleNamespace(model=self.spec['prm_model'], revision=self.spec['prm_revision'],
                                   device=self.device, device_map=None, torch_dtype=self.spec['prm_dtype'])
            self.prm_model, self.device = load_qwen_prm(args, self._tokenizer())
            self.prm_model.eval().requires_grad_(False)
        for batch, values in self._batches(todo, 'prm', lambda items: self._extract(items, renders)):
            for (key, identity, path), payload in zip(batch, values):
                payload['targets'] = renders[key][0]['targets']
                self._validate(payload, renders[key][1])
                atomic_torch(path, {'identity': identity, 'payload': payload})
                payloads[key] = payload
        for key, row in pending.items():
            bp = blueprints[key]
            fused = build_fused_feature_tensor(blueprint=bp,
                bge_embeddings=torch.stack([embeddings[t] for t in bp.node_texts]),
                prm_payload=payloads[key], layout=self.layout)
            if not torch.isfinite(fused).all():
                raise ValueError('Non-finite fused features')
            fields = dict(text_embeddings=fused, metadata_features=torch.tensor(bp.metadata_rows, dtype=torch.float32),
                node_type_ids=torch.tensor(bp.node_type_ids),
                edge_index=torch.tensor([(s, d) for s, d, _ in bp.edge_rows], dtype=torch.long).T.contiguous(),
                edge_type_ids=torch.tensor([r for _, _, r in bp.edge_rows]), decomposer_index=bp.decomposer_index,
                final_index=bp.final_index, worker_indices_by_node_id=bp.worker_indices_by_node_id,
                prm_scores=payloads[key]['scores'], prm_alignment=renders[key][1])
            atomic_torch(self.graph_path(row), {'identity': key, 'example': fields})
            self._graphs.put(key, fields)
        self.last_stats.update(overflows=len(errors), elapsed_s=round(time.monotonic() - started, 3),
                               prm_batch_limit=self.prm_batch_size, bge_batch_limit=self.bge_batch_size)
        if device.type == 'cuda':
            self.last_stats.update(cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                                   cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
        print(f'[graphprm][features-batch] {self.last_stats}', flush=True)
        return errors

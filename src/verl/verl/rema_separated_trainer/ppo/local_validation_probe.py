"""Job-local frozen inference, on a GPU outside the training placement groups."""

import math
import logging
from pathlib import Path
import re
import time

from .validation_probes import ProbeResponse, SYSTEM_PROMPT, probe_request_options


def local_probe_config(config):
    """Validate before creating any CUDA context or reserving resources."""
    result = dict(config)
    result['model'] = str(config.get('model') or '').strip()
    result['revision'] = str(config.get('revision') or '').strip() or None
    if not result['model']:
        raise ValueError('Local validation probe requires a model checkpoint')
    if config.get('base_url'):
        raise ValueError('Local validation probe does not use base_url; unset the probe URL')
    if not Path(result['model']).is_dir() and not re.fullmatch(r'[0-9a-fA-F]{40}', result['revision'] or ''):
        raise ValueError('Local validation probe requires a fixed 40-character Hugging Face revision, '
                         'or an immutable local model directory visible to the probe worker')
    for key, default in (('max_new_tokens', 1024), ('max_model_len', 16384),
                         ('batch_size', 8), ('replicas', 1)):
        value = config.get(key, default)
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError(f'validation_probe.{key} must be a positive integer')
        result[key] = int(value)
    for key, default in (('timeout_seconds', 180), ('startup_timeout_seconds', 900),
                         ('gpu_memory_utilization', 0.8)):
        value = float(config.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'validation_probe.{key} must be finite and positive')
        result[key] = value
    if result['gpu_memory_utilization'] > 1:
        raise ValueError('validation_probe.gpu_memory_utilization cannot exceed 1')
    if result['max_model_len'] <= result['max_new_tokens']:
        raise ValueError('Probe context must leave room for both input and output tokens')
    return result


def check_local_probe_dependencies():
    from importlib.metadata import PackageNotFoundError, version
    from packaging.version import Version

    for package, minimum in (('vllm', '0.8.5'), ('transformers', '4.51.0')):
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = None
        if installed is None or Version(installed) < Version(minimum):
            raise RuntimeError(f'Local frozen probe requires {package}>={minimum}. '
                               'Use a compatible container or the HTTP backend; '
                               'dependencies are never upgraded automatically.')


class LocalProbeWorker:
    """Inference-only actor: no optimizer, weight sync, or training-role methods."""

    def __init__(self, config):
        import ray
        check_local_probe_dependencies()
        if len(ray.get_gpu_ids()) != 1:
            raise RuntimeError('The local probe must own exactly one Ray-assigned GPU')
        # Do not let a nested vLLM engine create its own Ray GPU pool.
        from vllm import LLM

        self.config = local_probe_config(config)
        self.engine = LLM(
            model=self.config['model'], tokenizer=self.config['model'],
            revision=self.config['revision'], tokenizer_revision=self.config['revision'],
            tensor_parallel_size=1, distributed_executor_backend='uni',
            dtype='bfloat16', trust_remote_code=False, enforce_eager=True,
            max_model_len=self.config['max_model_len'],
            max_num_seqs=self.config['batch_size'],
            max_num_batched_tokens=self.config['max_model_len'],
            gpu_memory_utilization=self.config['gpu_memory_utilization'],
            generation_config='vllm', enable_prefix_caching=False, seed=0,
        )
        self.tokenizer = self.engine.get_tokenizer()

    def ready(self):
        from importlib.metadata import version
        return {'model': self.config['model'], 'revision': self.config['revision'],
                'vllm': version('vllm'), 'transformers': version('transformers')}

    def generate(self, messages, *, system_prompt=SYSTEM_PROMPT, max_new_tokens=None):
        from vllm import SamplingParams

        system_prompt, budget = probe_request_options(
            system_prompt, max_new_tokens, self.config['max_new_tokens'])
        if len(messages) > self.config['batch_size']:
            raise ValueError('Probe request exceeds its configured batch_size')
        prompts = []
        for message in messages:
            tokens = self.tokenizer.apply_chat_template(
                [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': message}],
                tokenize=True, add_generation_prompt=True,
            )
            if len(tokens) + budget > self.config['max_model_len']:
                raise ValueError('Frozen probe context exceeds max_model_len; '
                                 'increase the limit rather than truncate the measured information')
            prompts.append({'prompt_token_ids': tokens})
        if not prompts:
            return []
        sampling = SamplingParams(n=1, temperature=0, top_p=1, top_k=-1, max_tokens=budget, seed=0)
        outputs = self.engine.generate(prompts, sampling, use_tqdm=False)
        if len(outputs) != len(messages):
            raise RuntimeError('Incomplete local probe generation batch')
        responses = []
        for output in outputs:
            if len(output.outputs) != 1:
                raise RuntimeError('Frozen probe must produce one completion per input')
            completion = output.outputs[0]
            if not isinstance(completion.text, str) or completion.finish_reason not in ('stop', 'length'):
                raise RuntimeError('Malformed or aborted local probe generation')
            responses.append(ProbeResponse(completion.text, completion.finish_reason, len(completion.token_ids)))
        return responses


class LocalFrozenProbeClient:
    """Bounded parallel inference on identical, dedicated one-GPU replicas."""

    def __init__(self, config):
        self.config = local_probe_config(config)
        self.model = self.config['model']
        self.max_new_tokens = self.config['max_new_tokens']
        self.actors = []
        self.required_gpus = self.config['replicas']

    def check_capacity(self, training_gpus, *, probe_gpus=None):
        import ray
        check_local_probe_dependencies()
        probe_gpus = self.required_gpus if probe_gpus is None else probe_gpus
        available = ray.available_resources().get('GPU', 0)
        if available < training_gpus + probe_gpus:
            raise RuntimeError(f'Local frozen probes need {training_gpus} training GPUs plus {probe_gpus} '
                               f'exclusive inference GPUs; Ray currently has {available} available. '
                               'Reserve extra capacity; do not share a training GPU implicitly.')

    def start(self):
        import ray
        if self.actors:
            return
        if ray.available_resources().get('GPU', 0) < self.required_gpus:
            raise RuntimeError(f'Frozen probe requires {self.required_gpus} free GPUs outside training pools')
        actor_type = ray.remote(num_cpus=1, num_gpus=1, max_restarts=0,
                                max_task_retries=0)(LocalProbeWorker)
        # Explicitly avoid inheriting a training placement group.
        try:
            for _ in range(self.required_gpus):
                self.actors.append(actor_type.options(scheduling_strategy='DEFAULT').remote(self.config))
            info = ray.get([actor.ready.remote() for actor in self.actors],
                           timeout=self.config['startup_timeout_seconds'])
            print(f'[validation/probe] local frozen replicas={len(self.actors)} '
                  f'batch_size_per_replica={self.config["batch_size"]} ready: {info}', flush=True)
        except Exception:
            self.close()
            raise

    def generate(self, messages, *, system_prompt=SYSTEM_PROMPT, max_new_tokens=None):
        import ray
        system_prompt, budget = probe_request_options(system_prompt, max_new_tokens, self.max_new_tokens)
        if not messages:
            return []
        if not self.actors:
            raise RuntimeError('Local frozen probe has not been started')
        responses = [None] * len(messages)
        pending = {}
        next_start = 0

        def submit(actor):
            nonlocal next_start
            start = next_start
            chunk = messages[start:start + self.config['batch_size']]
            next_start += len(chunk)
            deadline = time.monotonic() + self.config['timeout_seconds']
            ref = actor.generate.remote(chunk, system_prompt=system_prompt, max_new_tokens=budget)
            pending[ref] = (actor, start, len(chunk), deadline)

        try:
            for actor in self.actors:
                if next_start < len(messages):
                    submit(actor)
            while pending:
                remaining = min(item[3] for item in pending.values()) - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Local frozen probe generation timed out')
                ready, _ = ray.wait(list(pending), num_returns=1, timeout=remaining)
                if not ready:
                    raise TimeoutError('Local frozen probe generation timed out')
                ref = ready[0]
                actor, start, size, deadline = pending.pop(ref)
                result = ray.get(ref, timeout=max(0, deadline - time.monotonic()))
                if len(result) != size:
                    raise RuntimeError('Incomplete local probe generation batch')
                responses[start:start + size] = result
                if next_start < len(messages):
                    submit(actor)
        except Exception:
            # A timed-out inference must not continue running after evaluation fails.
            self.close()
            raise
        return responses

    def close(self):
        if self.actors:
            import ray
            actors, self.actors = self.actors, []
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    # Preserve the original failure if the Ray cluster is already down.
                    logging.getLogger(__name__).warning('Could not stop local validation probe', exc_info=True)

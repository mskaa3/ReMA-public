"""Job-local frozen inference, on a GPU outside the training placement groups."""

import math
import logging
from pathlib import Path
import re

from .validation_probes import ProbeResponse, SYSTEM_PROMPT


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
    for key, default in (('max_new_tokens', 256), ('max_model_len', 16384), ('batch_size', 4)):
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
        from vllm import LLM, SamplingParams

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
        self.sampling = SamplingParams(n=1, temperature=0, top_p=1, top_k=-1,
                                       max_tokens=self.config['max_new_tokens'], seed=0)

    def ready(self):
        from importlib.metadata import version
        return {'model': self.config['model'], 'revision': self.config['revision'],
                'vllm': version('vllm'), 'transformers': version('transformers')}

    def generate(self, messages):
        if len(messages) > self.config['batch_size']:
            raise ValueError('Probe request exceeds its configured batch_size')
        prompts = []
        for message in messages:
            tokens = self.tokenizer.apply_chat_template(
                [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': message}],
                tokenize=True, add_generation_prompt=True,
            )
            if len(tokens) + self.config['max_new_tokens'] > self.config['max_model_len']:
                raise ValueError('Frozen probe context exceeds max_model_len; '
                                 'increase the limit rather than truncate the measured information')
            prompts.append({'prompt_token_ids': tokens})
        if not prompts:
            return []
        outputs = self.engine.generate(prompts, self.sampling, use_tqdm=False)
        if len(outputs) != len(messages):
            raise RuntimeError('Incomplete local probe generation batch')
        responses = []
        for output in outputs:
            if len(output.outputs) != 1:
                raise RuntimeError('Frozen probe must produce one completion per input')
            completion = output.outputs[0]
            if not isinstance(completion.text, str) or completion.finish_reason not in ('stop', 'length'):
                raise RuntimeError('Malformed or aborted local probe generation')
            responses.append(ProbeResponse(completion.text, completion.finish_reason))
        return responses


class LocalFrozenProbeClient:
    """Own a dedicated inference actor for the job, not an HTTP endpoint."""

    def __init__(self, config):
        self.config = local_probe_config(config)
        self.model = self.config['model']
        self.max_new_tokens = self.config['max_new_tokens']
        self.actor = None

    def check_capacity(self, training_gpus):
        import ray
        check_local_probe_dependencies()
        available = ray.available_resources().get('GPU', 0)
        if available < training_gpus + 1:
            raise RuntimeError(f'Local frozen probe needs {training_gpus} training GPUs plus 1 '
                               f'exclusive inference GPU; Ray currently has {available} available. '
                               'Reserve extra capacity; do not share a training GPU implicitly.')

    def start(self):
        import ray
        if self.actor is not None:
            return
        if ray.available_resources().get('GPU', 0) < 1:
            raise RuntimeError('No free GPU remains outside the training pools for the frozen probe')
        actor_type = ray.remote(num_cpus=1, num_gpus=1, max_restarts=0,
                                max_task_retries=0)(LocalProbeWorker)
        # Explicitly avoid inheriting a training placement group.
        self.actor = actor_type.options(scheduling_strategy='DEFAULT').remote(self.config)
        try:
            info = ray.get(self.actor.ready.remote(), timeout=self.config['startup_timeout_seconds'])
            print(f'[validation/probe] local frozen model ready: {info}', flush=True)
        except Exception:
            self.close()
            raise

    def generate(self, messages):
        import ray
        if not messages:
            return []
        if self.actor is None:
            raise RuntimeError('Local frozen probe has not been started')
        responses = []
        try:
            for start in range(0, len(messages), self.config['batch_size']):
                chunk = messages[start:start + self.config['batch_size']]
                result = ray.get(self.actor.generate.remote(chunk), timeout=self.config['timeout_seconds'])
                if len(result) != len(chunk):
                    raise RuntimeError('Incomplete local probe generation batch')
                responses.extend(result)
        except Exception:
            # A timed-out inference must not continue running after evaluation fails.
            self.close()
            raise
        return responses

    def close(self):
        if self.actor is not None:
            import ray
            actor, self.actor = self.actor, None
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                # Preserve the original failure if the Ray cluster is already down.
                logging.getLogger(__name__).warning('Could not stop local validation probe', exc_info=True)

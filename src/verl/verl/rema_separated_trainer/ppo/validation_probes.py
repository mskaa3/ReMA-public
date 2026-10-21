"""Evaluation-only, fixed-budget information probes. Never used by reward gates."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .local_results import extract_complete_boxed_answer


CONDITIONS = ('terminal_only', 'with_instructions', 'with_results')
METRICS = tuple(f'{name}_acc' for name in CONDITIONS) + ('information_gain', 'result_advantage')
SYSTEM_PROMPT = (
    'Solve the CURRENT TASK using the supplied information. '
    'Show concise reasoning and end with exactly one complete \\boxed{...}. '
    'If the requested answer cannot be inferred, end with \\boxed{UNKNOWN}.'
)


def make_probe_inputs(terminal_task, upstream_tasks, previous_results):
    """Use routed instructions/results, not the original question or any reasoning."""
    current = f'CURRENT TASK:\n{terminal_task}'
    return {
        'terminal_only': current,
        'with_instructions': (
            f'PREVIOUS SUBTASKS:\n{upstream_tasks}\n\n{current}' if upstream_tasks else current
        ),
        'with_results': f'{previous_results}\n\n{current}' if previous_results else current,
    }


@dataclass(frozen=True)
class ProbeExample:
    question_id: str
    dataset: str
    data_source: str
    ground_truth: object
    extra_info: object
    inputs: object
    system_score: float


@dataclass(frozen=True)
class ProbeResponse:
    text: str
    stop_reason: str


def select_probe_examples(examples, samples_per_dataset):
    """Stable hash-ranked questions; use the first trajectory if val.n > 1."""
    if samples_per_dataset < 1:
        raise ValueError('samples_per_dataset must be positive')
    groups = {}
    for example in examples:
        groups.setdefault(example.dataset, {}).setdefault(example.question_id, example)
    selected = []
    for dataset, group in sorted(groups.items()):
        ranked = sorted(group.values(), key=lambda item: hashlib.sha256(
            (dataset + '\0' + item.question_id).encode('utf-8')
        ).hexdigest())
        selected.extend(ranked[:samples_per_dataset])
    return selected


class FrozenProbeClient:
    """A separate, immutable model served through /v1/chat/completions.

    Service failures abort the evaluation instead of masquerading as model errors.
    The operator must keep the served checkpoint and model identifier unchanged.
    """

    def __init__(self, config):
        self.base_url = str(config.get('base_url') or '').rstrip('/')
        self.model = str(config.get('model') or '').strip()
        parsed = urlparse(self.base_url)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError('validation_probe.base_url must be an HTTP(S) API base URL (ending in /v1)')
        if not self.model:
            raise ValueError('validation_probe.model must identify the frozen served checkpoint')
        self.max_new_tokens = int(config.get('max_new_tokens', 256))
        self.timeout = float(config.get('timeout_seconds', 180))
        self.concurrency = int(config.get('concurrency', 4))
        if min(self.max_new_tokens, self.timeout, self.concurrency) <= 0:
            raise ValueError('Validation probe budgets, concurrency and timeout must be positive')

    def _generate_one(self, message):
        payload = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                         {'role': 'user', 'content': message}],
            'temperature': 0, 'top_p': 1, 'n': 1, 'max_tokens': self.max_new_tokens,
            'stream': False,
        }
        headers = {'Content-Type': 'application/json'}
        key = os.environ.get('VALIDATION_PROBE_API_KEY')
        if key:
            headers['Authorization'] = f'Bearer {key}'
        request = Request(self.base_url + '/chat/completions',
                          data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError('Frozen validation probe endpoint failed; no probe accuracy reported') from exc
        if result.get('model') != self.model:
            raise RuntimeError('Frozen probe returned a different model identifier')
        choices = result.get('choices', [])
        if len(choices) != 1:
            raise RuntimeError('Frozen probe must return exactly one completion per input')
        text = choices[0].get('message', {}).get('content')
        stop = choices[0].get('finish_reason')
        if not isinstance(text, str) or not isinstance(stop, str):
            raise RuntimeError('Malformed frozen probe response')
        return ProbeResponse(text, stop)

    def generate(self, messages):
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return list(executor.map(self._generate_one, messages))


def make_validation_probe_client(config):
    backend = config.get('backend', 'http')
    if backend == 'http':
        return FrozenProbeClient(config)
    if backend == 'local':
        from .local_validation_probe import LocalFrozenProbeClient
        return LocalFrozenProbeClient(config)
    raise ValueError(f'Unknown validation_probe.backend: {backend!r}; use http or local')


def evaluate_probes(examples, generate, parse_answers, score_responses):
    """Three paired measurements, with failures retained in each denominator."""
    unique_inputs, lookup, locations = [], {}, {}
    for index, example in enumerate(examples):
        inputs = example.inputs
        # Missing metadata is not a license to reconstruct a potentially leaky context.
        if not isinstance(inputs, dict) or any(
            not isinstance(inputs.get(condition), str) or not inputs[condition].strip()
            for condition in CONDITIONS
        ):
            continue
        for condition in CONDITIONS:
            message = inputs[condition]
            if message not in lookup:
                lookup[message] = len(unique_inputs)
                unique_inputs.append(message)
            locations[index, condition] = lookup[message]
    responses = generate(unique_inputs) if unique_inputs else []
    if len(responses) != len(unique_inputs):
        raise RuntimeError('Incomplete frozen probe generation batch')
    valid = [False] * len(responses)
    unknown = [False] * len(responses)
    parse_indices = []
    for index, response in enumerate(responses):
        answer = extract_complete_boxed_answer(response.text)
        if response.stop_reason != 'stop' or answer is None:
            continue
        unknown[index] = ''.join(answer.lower().split()) in {
            'unknown', r'\text{unknown}', r'\mathrm{unknown}',
        }
        if unknown[index]:
            valid[index] = True
        else:
            parse_indices.append(index)
    if parse_indices:
        parsed = parse_answers([responses[i].text for i in parse_indices])
        if len(parsed) != len(parse_indices):
            raise RuntimeError('Incomplete frozen probe parse batch')
        for index, value in zip(parse_indices, parsed):
            valid[index] = value is True

    records, scoring = [], []
    for index, example in enumerate(examples):
        record = {'question_id': example.question_id, 'dataset': example.dataset,
                  'system_score': example.system_score, 'conditions': {}}
        for condition in CONDITIONS:
            source = locations.get((index, condition))
            response = responses[source] if source is not None else None
            item = {
                'input': unique_inputs[source] if source is not None else '',
                'response': response.text if response else '',
                'valid': valid[source] if source is not None else False,
                'missing_context': source is None,
                'truncated': response.stop_reason == 'length' if response else False,
                'unknown': unknown[source] if source is not None else False,
                'score': 0.0,
            }
            record['conditions'][condition] = item
            if item['valid'] and not item['unknown']:
                scoring.append((example, item))
        records.append(record)
    if scoring:
        scores = score_responses(
            [example.data_source for example, _ in scoring],
            [item['response'] for _, item in scoring],
            [example.ground_truth for example, _ in scoring],
            [example.extra_info for example, _ in scoring],
            show_progress=False, timeout_score=float('nan'),
        )
        if len(scores) != len(scoring):
            raise RuntimeError('Incomplete frozen probe score batch')
        for (_, item), score in zip(scoring, scores):
            item['valid'] = math.isfinite(float(score))
            item['score'] = float(float(score) > 0) if item['valid'] else 0.0

    metrics = {}
    for dataset in sorted({record['dataset'] for record in records}):
        group = [record for record in records if record['dataset'] == dataset]
        count = len(group)
        prefix = f'val/probe/{dataset}'
        diagnostics = f'val_diagnostics/probe/{dataset}'
        metrics[f'{diagnostics}/sample_count'] = count
        metrics[f'{diagnostics}/system_acc_on_subset'] = sum(r['system_score'] for r in group) / count
        for condition in CONDITIONS:
            items = [record['conditions'][condition] for record in group]
            metrics[f'{prefix}/{condition}_acc'] = sum(item['score'] for item in items) / count
            for field in ('valid', 'missing_context', 'truncated', 'unknown'):
                metrics[f'{diagnostics}/{condition}/{field}_rate'] = sum(item[field] for item in items) / count
        metrics[f'{prefix}/information_gain'] = (
            metrics[f'{prefix}/with_results_acc'] - metrics[f'{prefix}/terminal_only_acc']
        )
        metrics[f'{prefix}/result_advantage'] = (
            metrics[f'{prefix}/with_results_acc'] - metrics[f'{prefix}/with_instructions_acc']
        )
    return metrics, records


def validation_metric_namespace(metrics, *, teacher_assisted=False):
    """Reserve val/ for the five probe metrics and ordinary dataset accuracy."""
    result = {}
    for key, value in metrics.items():
        if key.startswith('val_diagnostics/'):
            suffix = key[len('val_diagnostics/'):]
        elif key.startswith('val/'):
            suffix = key[len('val/'):]
            headline = suffix.startswith('acc/') or (
                suffix.startswith('probe/') and suffix.rsplit('/', 1)[-1] in METRICS
            )
            if headline and not teacher_assisted:
                result[key] = value
                continue
        else:
            result[key] = value
            continue
        prefix = 'val_diagnostics/teacher_assisted/' if teacher_assisted else 'val_diagnostics/'
        result[prefix + suffix] = value
    return result

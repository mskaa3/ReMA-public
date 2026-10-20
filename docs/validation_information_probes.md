# Frozen validation information probes

These are evaluation metrics, not rewards or eligibility gates. The existing
C3/prefix-probe training behavior is unchanged. They run at each validation,
including `trainer.val_only=True` evaluation/test jobs, when enabled.

## Measurements

For each sampled trajectory, let C be the actual terminal subtask instruction,
S the preceding routed subtask instructions, and R the preceding routed
LOCAL_RESULTs. A separate frozen small model receives three inputs:

| W&B metric under `val/probe/<dataset>/` | Meaning |
| --- | --- |
| `terminal_only_acc` | Correct-answer recovery from C |
| `with_instructions_acc` | Correct-answer recovery from C + S |
| `with_results_acc` | Correct-answer recovery from C + R |
| `information_gain` | `with_results_acc - terminal_only_acc` |
| `result_advantage` | `with_results_acc - with_instructions_acc` |

The last two are signed differences of paired success rates, not information
theoretic quantities or GRPO advantages. Negative values are retained.

Inputs are captured at the terminal worker's prompt construction boundary. No
original question, teacher solution, decomposer REASONING, worker reasoning, or
terminal answer is appended. Constants already present in a subtask instruction
are preserved: their recoverability is part of the measurement. R uses exactly
the accepted LOCAL_RESULT representation passed to the terminal, including its
extraction/length rules. Missing results are not reconstructed from reasoning.

All three conditions share one system prompt and generation budget. They use
greedy decoding and one attempt, with identical inputs deduplicated within an
evaluation. Ground truth is sent only to the existing mathematical verifier,
never to the model endpoint.

## Sampling and interpretation

The default is at most 32 unique questions per dataset, deterministically
selected by hash, independently of correctness or gate eligibility. If `val.n`
is greater than one, the first trajectory for a selected question is used.
Keep the dataset, validation settings, probe checkpoint, served model name,
prompt, and budget fixed across comparisons. The same questions are selected
even when validation batches are reordered; repeated-question trajectory order
must remain fixed. Single-subtask trajectories are included, not filtered out.

Invalid, unparseable, truncated, missing-context, or verifier-timeout responses
are unsuccessful and remain in the accuracy denominator. A complete
`\boxed{UNKNOWN}` is a valid unsuccessful response. Infrastructure failures
(unreachable endpoint, wrong served model, malformed API response) abort the
evaluation rather than generate misleading zero accuracies.

Diagnostics under `val_diagnostics/probe/<dataset>/` include sample count,
the actual system accuracy on the same subset, and validity, truncation,
missing-context, and UNKNOWN rates for each condition. Check these alongside
the headline metrics. Small subsets and a single attempt have considerable
uncertainty; the full-dataset `val/acc/<dataset>` is not directly paired with
the subset probe results.

High `terminal_only_acc` measures answer recoverability, not proof of copying.
Positive `result_advantage` means results help this frozen probe more than the
instructions do; it does not establish that the actual terminal causally used
them or that every worker performed the correct subtask.

## Frozen endpoint

The evaluator must be separate from the actors being trained. Serve one fixed
SLM checkpoint with an OpenAI-compatible `/v1/chat/completions` API. Do not
hot-swap its weights or reuse a moving actor behind the same model name. The
response's `model` identifier must match `VALIDATION_PROBE_MODEL`.

For example, in a separate allocated GPU session with vLLM available:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/immutable/probe-huggingface \
  --served-model-name eac3-probe-v1 \
  --host 0.0.0.0 --port 8000 \
  --generation-config vllm
```

Use a cluster-internal address accessible from the trainer's Ray TaskRunner,
not `localhost` on the login node. Follow cluster networking rules and restrict
access to the server. The training launcher does not allocate a GPU for this
service or download a probe checkpoint automatically.

Enable it for the usual Agent-1/Agent-2 or experiment launcher:

```bash
export VALIDATION_PROBE_URL="http://PROBE_NODE:8000/v1"
export VALIDATION_PROBE_MODEL="eac3-probe-v1"
export VALIDATION_PROBE_SAMPLES_PER_DATASET=32
export VALIDATION_PROBE_MAX_NEW_TOKENS=256
# Optional if the server requires authentication:
# export VALIDATION_PROBE_API_KEY="..."

# Then use your existing sbatch command and checkpoint/model settings.
```

Both URL and model must be specified. Without them the new probes remain off;
accuracy and the reorganized diagnostics still log normally. The same options
are available to direct Hydra invocations:

```text
trainer.validation_probe.enable=True
trainer.validation_probe.base_url=http://PROBE_NODE:8000/v1
trainer.validation_probe.model=eac3-probe-v1
trainer.validation_probe.samples_per_dataset=32
trainer.validation_probe.max_new_tokens=256
```

`timeout_seconds` (default 180) bounds each HTTP request; `concurrency` (default
4) bounds outstanding requests. Each evaluation uses at most three probe
generations per selected trajectory, before input deduplication. For seven
datasets with 32 questions each, that is at most 672 generations, independent
of the existing gate probes. Avoid silently truncating prompts at the server.

## Logging layout

Only dataset accuracy and the five probe metrics remain under `val/`:

```text
val/acc/gsm8k
val/probe/gsm8k/terminal_only_acc
val/probe/gsm8k/with_instructions_acc
val/probe/gsm8k/with_results_acc
val/probe/gsm8k/information_gain
val/probe/gsm8k/result_advantage
```

All other validation diagnostics move to `val_diagnostics/`, preserving their
suffixes, e.g. `val/leakage/gated_accuracy` becomes
`val_diagnostics/leakage/gated_accuracy`. Generation tables move there too.
Teacher-assisted validation is wholly under
`val_diagnostics/teacher_assisted/`, including its accuracy and probes, so it
cannot overwrite ordinary held-out results. Update saved W&B panels that use
the old paths; previously logged history is not renamed. Both namespaces use
the completed actor-update step as their W&B x-axis.

The console prints a `[probe/val]` summary per dataset (and a separate
`[probe/val_teacher_assisted]` summary if that evaluation is enabled).
Existing `[leakage/val]` console lines retain their labels.

When `trainer.validation_output_dir` is configured, each evaluation also saves
a sibling `*.probes.jsonl` with selected question IDs, all three inputs,
responses, validity flags, and scores. Its checksum is recorded in the normal
evaluation metadata; the resolved config records the endpoint/model/budget.

## Local checks

```bash
python3 src/verl/tests/rema_separated_trainer/test_validation_probes.py -v
```

The stdlib suite exercises paired scoring, failure denominators, endpoint
contracts, stable selection, and namespace isolation without GPUs. The existing
`test_agent2_launch.py` additionally checks captured probe inputs at the actual
rollout boundary when the trainer dependencies are installed.

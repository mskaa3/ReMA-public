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

## Frozen model and backends

The default evaluator is **`Qwen/Qwen3-4B-Instruct-2507`**, using its original
instruction-tuned weights, independently of the worker and decomposer models.
This choice applies only to these five evaluation metrics, not the existing
training leakage probes or reward gates.

The evaluator must be separate from the actors being trained. Two backends use
the same input construction, system prompt, greedy decoding, verifier, sampling,
and metrics: a job-managed local vLLM actor, or an external HTTP endpoint.
Neither backend receives gradients or training-role weight updates. Do not
switch backends or inference versions midway through a comparison: the
measurement protocol is shared, but bitwise identical generations are not
guaranteed across inference environments.

## Local inference inside the training job

This mode needs no HTTP server, endpoint address, or separate interactive
session. A Ray actor loads the fixed model once on one exclusive GPU, outside
both training placement groups. It stays resident across validations and is
killed when the trainer finishes or fails. Inference is batched; no training
optimizer is created for the probe and its weights are never synchronized
with the workers or decomposer.

For experiment manifests, add the immutable HF revision when generating runs:

```bash
: "${PROBE_REVISION:?Set the fixed 40-character HF commit hash first}"
python3 scripts/eac3_experiments.py plan \
  --config config/eac3-round2-gated.json \
  --phase adapt-workers \
  --local-probe-revision "$PROBE_REVISION" \
  --output-dir experiments/runs/round2-local-probe-gated
```

This generates commands only. For a repeat of an existing experiment, use a
fresh study name and explicitly retain its original adapted planner first;
see [the round-2 rerun recipe](eac3_round2.md#rerun-workers-with-local-frozen-probes).
Changing only the output directory does not change the S3 study identity.

The manifest records `local_validation_probe.revision` and requests one extra
node by default, without changing `nodes` for the training pools. For the
current 4-node x 4-GPU configurations, Slurm allocates **5 nodes x 4 GPUs**:
16 GPUs remain assigned to the two training pools, one GPU runs the probe,
and three GPUs are unused. This is a homogeneous Slurm request, not a promise
that an entire particular host is used only for the probe. Actual GPU placement
is managed by Ray. Both arms together allocate 40 GPUs rather than 32.

Optional manifest settings (defaults shown) are:

```json
"local_validation_probe": {
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "revision": "REPLACE_WITH_IMMUTABLE_HF_COMMIT_HASH",
  "extra_nodes": 1,
  "samples_per_dataset": 32,
  "max_new_tokens": 256,
  "max_model_len": 16384,
  "batch_size": 4
}
```

The experiment wrapper clears inherited HTTP probe settings for local runs.
All nodes still join the Ray cluster and receive the code/container staging;
`TRAINING_NNODES` controls only the two training pools. Teacher generation, if
used outside these question-only experiments, remains a separate earlier phase.

For ordinary launchers (without an experiment manifest), reserve capacity
explicitly. For example, a five-node allocation with four training nodes:

```bash
export VALIDATION_PROBE_BACKEND=local
export VALIDATION_PROBE_MODEL=Qwen/Qwen3-4B-Instruct-2507
export VALIDATION_PROBE_REVISION="$PROBE_REVISION"
export TRAINING_NNODES=4
unset VALIDATION_PROBE_URL
# Submit the usual launcher with --nodes=5 and its existing 4 GPUs per node.
```

Missing capacity is an error, not a request to share a training GPU. The shell
checks requested capacity, the trainer checks Ray capacity before initializing
training models, and the client checks for a free GPU after training placement
groups are created. Actor startup has a 900-second limit; each inference batch
uses `timeout_seconds` (180 by default). Failed startup/inference kills the
probe actor and fails the run rather than manufacturing zero probe accuracy.
Long prompts are rejected, never silently truncated.

**Container prerequisite:** local mode requires vLLM >= 0.8.5 and Transformers
>= 4.51.0 in the job's container. The dependency check fails early rather than
upgrading packages automatically. Unlike HTTP mode, a separate compatible
server environment cannot compensate for an incompatible training container.
Validate that a newer container still supports the existing trainer before
launching long runs. Local inference startup prints model revision and library
versions; the experiment manifest and resolved config retain the revision.

## HTTP endpoint (optional)

Serve one fixed SLM checkpoint with an OpenAI-compatible
`/v1/chat/completions` API. Do not hot-swap its weights or reuse a moving actor
behind the same model name. The response's `model` identifier must match
`VALIDATION_PROBE_MODEL`.

Use a separate allocated GPU session with a Qwen3-compatible environment. The
[official model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
specifies vLLM >= 0.8.5 and Transformers >= 4.51.0. This is the non-thinking
variant; no thinking-mode flag is needed. The training container does not need
these upgrades because it only sends HTTP requests to the evaluator.

Choose one immutable Hugging Face commit revision for the experiment and set
`PROBE_REVISION` to its full hash. Reuse it on every server restart. The client
checks the served name, not the underlying weights, so retain the revision and
server environment with the experiment records.

```bash
: "${PROBE_REVISION:?Set the fixed Hugging Face commit hash for the probe}"
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --revision "$PROBE_REVISION" \
  --tokenizer-revision "$PROBE_REVISION" \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 16384 \
  --generation-config vllm
```

The server context limit above includes input and output tokens; it is not the
output budget. Increase it if the captured inputs require it rather than
silently truncating them. Before training, check `/v1/models` at the chosen
server address and confirm the served model name.

Use a cluster-internal address accessible from the trainer's Ray TaskRunner,
not `localhost` on the login node. Follow cluster networking rules and restrict
access to the server. The training launcher does not allocate a GPU for this
service or download a probe checkpoint automatically.

Enable it for the usual Agent-1/Agent-2 or experiment launcher:

```bash
export VALIDATION_PROBE_URL="http://PROBE_NODE:8000/v1"
export VALIDATION_PROBE_BACKEND=http
# Optional: already the default when VALIDATION_PROBE_URL is supplied.
export VALIDATION_PROBE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export VALIDATION_PROBE_SAMPLES_PER_DATASET=32
export VALIDATION_PROBE_MAX_NEW_TOKENS=256
# Optional if the server requires authentication:
# export VALIDATION_PROBE_API_KEY="..."

# Then use your existing sbatch command and checkpoint/model settings.
```

Setting the URL with no explicit backend selects HTTP and enables the metrics.
An explicit `VALIDATION_PROBE_MODEL` can override the served name. With neither
variable or backend set, probes remain off; specifying only a model is a configuration
error. Accuracy and the reorganized diagnostics still log normally when probes
are off. The same options are available to direct Hydra invocations:

```text
trainer.validation_probe.enable=True
trainer.validation_probe.backend=http
trainer.validation_probe.base_url=http://PROBE_NODE:8000/v1
trainer.validation_probe.model=Qwen/Qwen3-4B-Instruct-2507
trainer.validation_probe.samples_per_dataset=32
trainer.validation_probe.max_new_tokens=256
```

`timeout_seconds` (default 180) bounds each HTTP request; `concurrency` (default
4) bounds outstanding requests. Each evaluation uses at most three probe
generations per selected trajectory, before input deduplication. For seven
datasets with 32 questions each, that is at most 672 generations, independent
of the existing gate probes. Avoid silently truncating prompts at the server.

The output budget remains 256 tokens. Calibrate truncation and recovery rates
on a small sample before adopting this budget for the study; changing models
does not establish that 256 tokens is sufficient. Freeze the chosen budget
and rerun both experimental conditions if it changes.

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
contracts, local inference/resource contracts with mocked Ray/vLLM, stable
selection, and namespace isolation without GPUs. It is not a GPU integration
test. The existing
`test_agent2_launch.py` additionally checks captured probe inputs at the actual
rollout boundary when the trainer dependencies are installed.

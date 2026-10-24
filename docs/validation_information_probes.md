# Frozen validation information probes

These are evaluation metrics, not rewards or eligibility gates. Their frozen
inference model can optionally also serve the existing C3/prefix-probe gates,
using separate request prompts and budgets. The metrics run at each validation,
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

The default evaluator is **`Qwen/Qwen2.5-7B-Instruct`**, using its original
instruction-tuned weights. This is the decomposer's base model, but loaded as
a separate frozen instance, not the live decomposer. A trained decomposer HF
export can also be selected as the metrics model; see below.
Changing `DECOMPOSER_MODEL_PATH` does not change the probe. Pin a revision of
`Qwen/Qwen2.5-7B-Instruct`; do not reuse a revision from the previous Qwen3 probe.
This remains the default evaluator for these five metrics. Training gates use
this model when `prefix_probe.solver_backend=frozen_probe` is selected without
a separate `trainer.frozen_gate_probe` checkpoint.

The evaluator must be separate from the actors being trained. Two backends use
the same input construction, system prompt, greedy decoding, verifier, sampling,
and metrics: a job-managed local vLLM actor, or an external HTTP endpoint.
Neither backend receives gradients or training-role weight updates. Do not
switch backends or inference versions midway through a comparison: the
measurement protocol is shared, but bitwise identical generations are not
guaranteed across inference environments.

## Throughput without changing the measurement

Local inference defaults to 8 requests per replica and one replica. Increase
replicas explicitly; no GPU is silently taken from either training pool. For
the four-node allocation with two six-GPU training pools, use:

```bash
export VALIDATION_PROBE_BATCH_SIZE=8
export VALIDATION_PROBE_REPLICAS=4
export KEEP_FROZEN_MODELS_RESIDENT=true
```

With `PREFIX_PROBE_SOLVER_BACKEND=frozen_probe` and no separate gate checkpoint,
the same four replicas serve metrics and gates. The allocation is 6 decomposer
GPUs + 6 worker GPUs + 4 probe GPUs. Replicas hold identical frozen weights;
each input is evaluated once, not voted on four times. Prompt/output budgets,
sampling, verifier, eligibility rules, and C3's 8 x 4 grouping do not change.

For generated experiment manifests, set the fields in the study configuration
before calling `write_runs` (exporting shell variables does not override a
manifest):

```python
cfg['local_validation_probe'].update(batch_size=8, replicas=4)
cfg['keep_frozen_models_resident'] = True
```

A separate gate model uses `FROZEN_GATE_BATCH_SIZE` and `FROZEN_GATE_REPLICAS`,
or `frozen_gate_probe.batch_size` / `frozen_gate_probe.replicas` in the study.
Counts add: four metrics replicas plus one separate gate replica require five
extra GPUs. The experiment planner rounds up to enough extra nodes, while the
ordinary launcher rejects insufficient allocations before starting models.
The local probe still requires one exclusive GPU per replica and a checkpoint
visible on every potential host.

At most one batch is in flight on each replica. Completed replicas receive new
work immediately, but responses are reassembled in original input order.
Timeouts, incomplete outputs, or actor failures terminate all replicas and
raise an error; partial results never become zeros or selected training data.

Shared worker roles enter/exit their physical generation pool only once per
rollout batch. With `trainer.keep_frozen_models_resident=True`, a vLLM pool
excluded from **all** configured training roles stays resident between
generation calls. Such a pool synchronizes weights on first use, not on every
batch. Any pool that can train later in the curriculum retains the ordinary
sleep/sync lifecycle, and updating a resident-frozen pool is explicitly
rejected. Checkpoint save/load releases cached inference weights; the next
generation resynchronizes them. Generation/actor RNG scopes remain separate.
Use `KEEP_FROZEN_MODELS_RESIDENT=false` (or the corresponding trainer/study key)
to disable residency without changing the deduplication fix.

Regenerate manifests under new run names after these changes. Keep batch size,
replica count, checkpoint and inference version identical across comparison
arms: batching can change floating-point behavior even with greedy decoding.
Benchmark completed updates/hour and phase timings, including backward-pass
memory peaks; no particular speedup is guaranteed. Replica scheduling alone
does not pipeline rollout groups. The separate opt-in below overlaps frozen
inference, without policy-stale asynchronous training.

## Optional rollout/probe pipeline

For local frozen training gates, enable bounded overlap within each existing
generation batch. In the experiment configuration, before generating fresh
manifests for both arms:

```python
cfg['rollout_probe_pipeline'] = {'enable': True, 'groups_per_chunk': 12}
```

For ordinary launchers, the equivalent settings are:

```bash
export ROLLOUT_PROBE_PIPELINE=true
export ROLLOUT_PIPELINE_GROUPS_PER_CHUNK=12
```

The default remains disabled, so old manifests and non-hierarchical training
keep their original behavior. This option does not change the selected gate
model or backend. It activates only for scoped C3 with prefix probes and a
local frozen gate client (separate or shared with evaluation). Role-based gates,
HTTP gates, and validation remain synchronous; the console prints the reason.
Having only frozen **validation metrics** does not make role-based training
gates eligible for overlap. No extra GPUs are requested.

For 48 question groups and 32 rollouts per question, generation is split into
four chunks of 12 complete groups. Each group retains all 8 focal actions and
4 continuations per action. While chunk B generates, the independent frozen
probe evaluates inputs from chunk A. There is at most one probe chunk in
flight, and only one caller of each generation pool. A slow probe blocks the
next submission instead of growing an unbounded queue. Generation pools enter
and exit their lifecycle once per logical batch, not once per chunk.

Only frozen-model inference runs in a background thread. Answer verification,
symbolic equivalence, gates, C3 scoring, console examples and group filtering
still run on the reassembled batch in the original order on the trainer thread.
This avoids moving the verifier's process/signal machinery into another thread.
Identical probe inputs with identical scoring contexts are deduplicated across
all chunks, just as in the synchronous path. Raw probe responses are cached only
within that logical batch, never across updates; missing responses are fatal,
not zero scores or an invitation to rerun with another evaluator.

All pending inference is drained before scoring/filtering and before any model
update, role switch, validation or checkpoint. Probe/generation failures abort
the batch and cancel outstanding frozen inference. No first-completed selection,
additional dataloader reads, cross-update look-ahead, or partial C3 groups are
introduced. The logical batch and generation-attempt counters are unchanged.
The five evaluation metrics and their sampling remain unchanged.

Monitor `rollout/pipeline/enabled`, `chunks`, `unique_probe_requests`,
`generation_seconds`, `probe_seconds`, `probe_wait_seconds`, `overlap_seconds`,
and `wall_seconds`. The intervals measure host-observed generation/inference
calls for that logical batch, not CUDA kernel occupancy or a sum over earlier
group-filter retries. `timing_s/gen` now includes the pipeline and
its final probe wait when enabled; `timing_s/pipeline_generation` isolates the
generation calls and `timing_s/pipeline_probe_wait` reports exposed waiting.
The ordinary reward/prefix-probe timer covers the remaining synchronous scoring,
so do not interpret its reduction alone as the end-to-end speedup.

Both arms should use identical chunk and probe batching settings. Smaller
chunks can lower GPU inference efficiency and batching can change generated
outputs; the same learning objective is not a promise of bitwise equivalence.
Benchmark updates/hour and peak host/GPU memory before long runs. Splitting
decomposer generation from worker execution is not implemented by this option.

## Fixed versus adaptive training gates

The choice is independent of `prefix_probe.apply_to_training`, which still
decides whether measured eligibility affects training:

| `prefix_probe.solver_backend` | Gate solver | Five evaluation metrics |
| --- | --- | --- |
| `role` (default) | Current `solver_role`, normally the decomposer | Configured frozen metrics model |
| `frozen_probe` | The same frozen inference instance used for evaluation | Configured frozen metrics model |
| `frozen_probe` + `trainer.frozen_gate_probe.enable=True` | Separate immutable checkpoint, e.g. a trained decomposer snapshot | Configured frozen metrics model, unchanged |

For ordinary Agent-1/Agent-2 launchers, set `PREFIX_PROBE_SOLVER_BACKEND=role`
or `PREFIX_PROBE_SOLVER_BACKEND=frozen_probe`. Enable the frozen evaluator in
both arms using `VALIDATION_PROBE_BACKEND=local` with a pinned revision and
reserved GPU, or `http` with the endpoint settings below. Frozen gates require
either the evaluator or a separate frozen gate checkpoint to be enabled, and
never fall back to another model on failure.

The equivalent Hydra key is
`algorithm.hierarchy.scoped_c3_grpo.prefix_probe.solver_backend`.
Experiment manifests use `gate_solver_backend`, with the same choices. The
experiment planner also accepts `--gate-solver-backend`; frozen-gate manifests
require a pinned `local_validation_probe` or `frozen_gate_probe`. Regenerate existing manifests after
changing the code, model, or backend; do not reuse old run/output identities.

Only inference routing changes. Both gate solvers receive the same gate inputs,
`prefix_probe.system_prompt`, greedy decoding, and `prefix_probe.max_new_tokens`.
L_T, diagnostic L_D, mathematical worker/final equivalence, invalid-response
handling, and C3 credit rules are unchanged. Role probes reject inputs longer
than `rollout.prompt_length` instead of silently truncating; local frozen probes
likewise reject inputs exceeding their context window. Keep inputs within both
limits and preserve equivalent prompt/output budgets in a comparison.

The five metrics retain their own system prompt and output budget: a gate
request cannot change subsequent evaluation settings. The local frozen model
is loaded once per replica, on dedicated probe GPUs, not once per purpose.
Gate inference runs every applicable training batch, so frozen gating adds
work to that inference pool and may be slower than the decomposer's replicas.

`[prefix_probe]` and `[leakage/...]` console records identify `solver_backend`.
Resolved W&B configuration, experiment manifests, evaluation metadata, and saved
training trajectories record the choice. `val_diagnostics/leakage/*` describes
the selected gate solver, not an independent fixed audit in `role` mode. Compare
the five `val/probe/<dataset>/*` metrics under the fixed evaluator across arms.
If that evaluator also supplies training gates, its metrics are not an
independent guarantee against probe-specific exploitation.

See [the controlled gate comparison](eac3_round2.md#compare-fixed-and-adaptive-gates)
for commands that preserve identical starting partners rather than mixing the
two existing alternating histories.

## Trained decomposer for metrics

The five metrics are not restricted to the base Qwen checkpoint. You can select
an immutable, trained decomposer HF export and keep that copy frozen even while
the live decomposer continues learning. Input construction, prompt, budgets,
verifier, metric names, and dataset accuracy logging do not change.

For ordinary launchers:

```bash
: "${DECOMPOSER_MODEL_REMOTE:?Set the immutable decomposer HF export on S3}"
export VALIDATION_PROBE_BACKEND=local
export VALIDATION_PROBE_MODEL_REMOTE="$DECOMPOSER_MODEL_REMOTE"
unset VALIDATION_PROBE_URL VALIDATION_PROBE_MODEL VALIDATION_PROBE_REVISION

# Optional: use this same frozen instance for both metrics and gates.
export PREFIX_PROBE_SOLVER_BACKEND=frozen_probe
unset FROZEN_GATE_MODEL_REMOTE FROZEN_GATE_MODEL_PATH FROZEN_GATE_REVISION
```

The export is staged on every node into `$JOB_TMP/validation_probe_model` and
validated before inference starts. It is independent of the training actor and
never receives its weight updates. A failed download/invalid export aborts;
there is no fallback to base Qwen. The original remote source is retained in
`trainer.validation_probe.checkpoint_source`, resolved logging configuration,
and evaluation metadata, not only in a transient scratch path.

Set `PREFIX_PROBE_SOLVER_BACKEND=role` instead to keep adaptive decomposer gates
while still measuring metrics with this frozen snapshot. Alternatively, retain
a distinct `FROZEN_GATE_MODEL_REMOTE` to use two different frozen checkpoints.
With the default one replica per model, only the last option needs two probe
GPUs. Sharing the metrics model for gates uses one inference pool with separate
per-request prompts/budgets. Explicitly setting both checkpoint configurations
creates two replica pools, even for equal paths;
omit the separate gate configuration when you intend to share.

In experiment JSON, set `local_validation_probe.model` to the immutable export
and omit `revision` (HF model IDs still require an immutable HF revision):

```json
"local_validation_probe": {
  "model": "s3v2:YOUR_BUCKET/path/to/global_step_110/decomposer/huggingface",
  "extra_nodes": 1,
  "samples_per_dataset": 32,
  "max_new_tokens": 1024
}
```

The CLI equivalent is `--local-probe-checkpoint "$DECOMPOSER_MODEL_REMOTE"`.
Use it with the usual `plan --config ... --phase ... --output-dir ...` command
and a fresh study identity. `--gate-solver-backend frozen_probe` shares this
metrics instance when the configuration has no `frozen_gate_probe` entry.
An absolute local HF directory visible inside the container on all nodes is
also accepted (`VALIDATION_PROBE_MODEL` for ordinary launchers).

**Comparison rule:** keep the same metrics checkpoint, budget and protocol
across all arms and training steps. Changing the probe from base Qwen to a
trained decomposer changes the measurement instrument; do not merge those
scores into an unlabelled curve or attribute that difference to worker quality.
Sharing metrics and gates also means this is not an independent auditor of
the gate's own blind spots. Raw `val/acc/<dataset>` remains unchanged.

## Frozen trained-decomposer snapshot

To train a decomposer while using its initial checkpoint as a fixed gate solver,
initialize two independent copies from the same immutable HF export:

- Trainable decomposer: starts at D_0 and receives actor updates.
- Gate solver: stays at D_0 for the entire run, with no optimizer or weight sync.
- Metrics evaluator: stays at its separately selected frozen checkpoint.

The snapshot still receives the existing solver/probe prompt, not the planning
prompt. Gate equations and C3 rules do not change. Sharing initial weights does
not mean sharing a Ray actor, tokenizer, KV cache, or optimizer. A resumed run
loads the same gate export, not the resumed actor weights. Refreshing gates
between alternating phases must be an explicit experiment decision.

For ordinary launchers, retain the usual model/data/budget settings and add:

```bash
: "${DECOMPOSER_MODEL_REMOTE:?Set the immutable decomposer HF export on S3}"
export PREFIX_PROBE_SOLVER_BACKEND=frozen_probe
export FROZEN_GATE_MODEL_REMOTE="$DECOMPOSER_MODEL_REMOTE"
unset FROZEN_GATE_MODEL_PATH FROZEN_GATE_REVISION
# Keep VALIDATION_PROBE_* unchanged for comparable evaluation metrics.
```

The common launcher stages the gate export into a separate
`$JOB_TMP/frozen_gate_model` on every node, validates the HF files, and loads
one inference-only GPU actor per configured replica. It never uploads over the source checkpoint.
Use a step-specific S3 export that will not be overwritten. Alternatively,
`FROZEN_GATE_MODEL_PATH` accepts an immutable local HF directory visible inside
the container on every node, or a HF model ID with a fixed 40-character
`FROZEN_GATE_REVISION`. Raw FSDP shards must be merged to HF format first.

Experiment JSON accepts a separate configuration alongside the existing
`local_validation_probe`:

```json
"gate_solver_backend": "frozen_probe",
"frozen_gate_probe": {
  "model": "s3v2:YOUR_BUCKET/path/to/global_step_110/decomposer/huggingface",
  "extra_nodes": 1,
  "max_model_len": 16384,
  "batch_size": 8,
  "replicas": 1
}
```

Replace the example path with the actual immutable export. The planner CLI also
accepts `--frozen-gate-checkpoint "$DECOMPOSER_MODEL_REMOTE"` (and
`--frozen-gate-revision` for a HF model ID); this selects `frozen_probe` and does
not replace `local_validation_probe`. Regenerate under a fresh study/run name.

**Capacity with one replica per model:** a separate local gate plus local
evaluation consumes two reserved GPUs, not one. `nodes=3` with four GPUs/node
and `extra_nodes=1` still fits:
6 decomposer + 6 worker + 1 metric probe + 1 gate probe = 14 of 16 GPUs.
The two probes share the extra allocation, not a GPU or model. Capacity checks
include every replica in both pools. Gate-only runs can omit evaluation probes, but should not
be used for comparisons requiring the five fixed-probe metrics.

This controls gate drift while keeping metrics fixed. It does not eliminate
blind spots inherited from D_0, nor guarantee independence from the policy's
errors. Keep that distinction explicit when interpreting the experiment.

## Local inference inside the training job

This mode needs no HTTP server, endpoint address, or separate interactive
session. Each Ray replica loads the fixed model once on one exclusive GPU, outside
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
node by default, without automatically changing `nodes` for the training pools.
The CLI example above preserves the original pool sizes and allocates five
nodes. The round-2 rerun recipe explicitly reduces `nodes` to three and requests
four probe replicas instead:

| Configuration | Decomposer GPUs | Worker GPUs | Probe GPUs | Unused GPUs |
| --- | --- | --- | --- | --- |
| `nodes=3`, `extra_nodes=1`: 4 nodes total | 6 | 6 | 1 | 3 |
| `nodes=3`, `extra_nodes=1`, `replicas=4`: rerun recipe | 6 | 6 | 4 | 0 |
| `nodes=4`, `extra_nodes=1`: 5 nodes total | 8 | 8 | 1 | 3 |

These are homogeneous 4-GPU-per-node requests, not a promise that an entire
particular host is used only for the probe. Actual placement is managed by Ray.
The four-node recipe allocates 32 GPUs for both arms rather than 40. Smaller
training pools can increase per-GPU memory use and change throughput; cluster
testing is still required. Generate new manifests for the chosen allocation
rather than overriding only the node count on an old submission command.

Optional manifest settings (defaults shown) are:

```json
"local_validation_probe": {
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "revision": "REPLACE_WITH_IMMUTABLE_HF_COMMIT_HASH",
  "extra_nodes": 1,
  "samples_per_dataset": 32,
  "max_new_tokens": 1024,
  "max_model_len": 16384,
  "batch_size": 8,
  "replicas": 1
}
```

The experiment wrapper clears inherited HTTP probe settings for local runs.
All nodes still join the Ray cluster and receive the code/container staging;
`TRAINING_NNODES` controls only the two training pools. Teacher generation, if
used outside these question-only experiments, remains a separate earlier phase.

For ordinary launchers (without an experiment manifest), reserve capacity
explicitly. For example, a four-node allocation with three training-node equivalents:

```bash
export VALIDATION_PROBE_BACKEND=local
export VALIDATION_PROBE_MODEL=Qwen/Qwen2.5-7B-Instruct
export VALIDATION_PROBE_REVISION="$PROBE_REVISION"
export TRAINING_NNODES=3
unset VALIDATION_PROBE_URL
# Submit the usual launcher with --nodes=4 and its existing 4 GPUs per node.
```

Missing capacity is an error, not a request to share a training GPU. The shell
checks requested capacity, the trainer checks Ray capacity before initializing
training models, and the client checks for a free GPU after training placement
groups are created. Actor startup has a 900-second limit; each inference batch
uses `timeout_seconds` (180 by default). Failed startup/inference kills the
probe actor and fails the run rather than manufacturing zero probe accuracy.
Long prompts are rejected, never silently truncated.

**Container prerequisite:** local mode requires vLLM >= 0.8.5 and Transformers
>= 4.51.0 in the job's container. These are the current backend's conservative
dependency checks, retained from the previous probe, not minimum requirements
claimed for Qwen2.5. The dependency check fails early rather than
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

Use a separate allocated GPU session with an environment that supports
`Qwen/Qwen2.5-7B-Instruct`. Use its own tokenizer and chat template, not the
worker's tokenizer. The HTTP client does not import the server's inference
libraries, so their versions need not match the training container.

Choose one immutable Hugging Face commit revision for the experiment and set
`PROBE_REVISION` to its full hash. Reuse it on every server restart. The client
checks the served name, not the underlying weights, so retain the revision and
server environment with the experiment records.

```bash
: "${PROBE_REVISION:?Set the fixed Hugging Face commit hash for the probe}"
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --revision "$PROBE_REVISION" \
  --tokenizer-revision "$PROBE_REVISION" \
  --served-model-name Qwen/Qwen2.5-7B-Instruct \
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
export VALIDATION_PROBE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export VALIDATION_PROBE_SAMPLES_PER_DATASET=32
export VALIDATION_PROBE_MAX_NEW_TOKENS=1024
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
trainer.validation_probe.model=Qwen/Qwen2.5-7B-Instruct
trainer.validation_probe.samples_per_dataset=32
trainer.validation_probe.max_new_tokens=1024
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

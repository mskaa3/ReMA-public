# Training throughput controls

These changes do not alter label weights, correctness gates, node identities,
GRPO grouping, or the number of requested plans/executions. They do not require
retraining Graph+PRM. Defaults live in `hierarchical-rema-trainer.sh` and can be
overridden in the job environment. The `hierarchical-rema-train.sh` preset now
enables persistent engines; the low-level wrapper still defaults to disabled.

| Setting | Default | Purpose |
| --- | --- | --- |
| `GFAM_OVERLAP_REWARDS` | `true` | Score completed trajectories while later worker chunks generate. |
| `GFAM_REWARD_QUEUE_BATCHES` | `2` | Maximum running plus queued reward batches. One GPU consumer. |
| `REMA_LENGTH_BUCKETING` | `true` | Group similar-length replay rows without shortening them. |
| `REMA_VERIFIER_WORKERS` | `1` | Independent isolated verifier processes, allowed range 1-16. |
| `REMA_PERSISTENT_ROLLOUT` | `true` in training preset; `false` in wrapper | Experimental persistent vLLM engines with explicit learner GPU handoff. |

## Reward/generation overlap

Worker generation is submitted in `WORKER_BATCH_SIZE` chunks. A trajectory is
eligible for scoring only when all its nodes have finished. Results are keyed by
`(task index, plan index, execution-repeat index)` and restored in original order.
All results are drained before advantages are calculated or a learner updates a
policy. Duplicate/missing identities and scorer alignment errors remain fatal.
Existing per-trajectory context-overflow and unverified exclusions remain intact.

Reward batches use the scorer's existing rollout batch limit; partial batches
flush at generation-chunk boundaries. This avoids waiting for longer graphs to
fill a batch. A full queue applies backpressure rather than accumulating unlimited
GPU requests. Speedup depends on trajectory lengths and relative generation/scoring
times; it is not a claim that every generation step can overlap.

Log `[reward-pipeline] consumer_wait_s` is time the producer spends waiting on
reward futures, including the final drain, not total reward compute time. Compare
whole-segment duration, GPU utilization and completed trajectories per second.

## Learner batching and verification

Length buckets are reshuffled each replay epoch. Each global batch is striped
across learner ranks, giving equal step counts and equal row counts at each step
while spreading long sequences across ranks. No rows are dropped; at most
`world_size - 1` padding repeats are added, as with the previous distributed
sampler. The same bucketing is used for old-policy log-probability extraction and
replay validation. Token limits and masks are unchanged. Batch order changes, so
training is not expected to be bitwise identical.

Start with one verifier and inspect `[verification-batch] elapsed_s`. Set
`REMA_VERIFIER_WORKERS=2` or `4` if verification takes material time. Calls are
dispatched concurrently within a reward batch to separate persistent subprocesses.
Stable request hashing retains cache affinity; identical requests share one cache.
The existing hard timeout, child termination and `unverified` result are preserved.
A timeout is never converted into an incorrect answer. This pool does not make
the startup reference audit or every other serial verifier caller parallel.

## Persistent engines: enabled in training preset, GPU smoke test required

`REMA_PERSISTENT_ROLLOUT=true` replaces the rollout-only FSDP/vLLM worker with a
vLLM-only worker. It reuses one actor group across segments and planner/executor
switches, rather than retaining multiple full-GPU bundles. Unchanged weights are
not synchronized on each generation call. Updated checkpoints are loaded on CPU
and copied into the existing vLLM parameters; prefix caches are invalidated.

Requirements and limitations:

- The current adapter targets the repository's vLLM V0/SPMD API and TP=1.
  Unsupported APIs fail explicitly. CUDA/vLLM behavior has not been tested locally.
- Planner and executor must have identical architecture, tokenizer and chat
  template. Engine context limits must remain fixed, including during validation.
- Use distributed Ray offline learning. The existing rollout placement groups keep
  their GPU reservations. Every rollout worker must acknowledge sleep before the
  learner borrows its exact device. Learner actors are pinned to those nodes, use
  the acknowledged CUDA device, and must release memory before rollout can resume.
- Learner actors request zero *additional* Ray GPUs/CPUs because the sleeping
  rollout pool already reserves that capacity. This is an explicit exclusive
  handoff, not concurrent learning and rollout on the same GPU. Do not launch
  unrelated GPU work inside these reservations.
- A local checkpoint's file identity is pinned for each rollout segment. A change
  mid-segment, incomplete weight load or stale worker acknowledgement aborts
  rather than generating from silently mixed weights. Initial Hub model IDs
  should refer to a fixed cached snapshot; mutable remote revisions are not polled.
- Sleeping vLLM engines retain host-side weight copies and some CUDA context
  overhead. Leave CPU RAM and GPU headroom for the learner. The adapter does not
  promise lower peak memory on every vLLM version.

Before a long job, run a separate short HPC job with at least two update segments
and both planner/executor phases. Keep its outputs separate from research runs.
Verify one initial `bundle_build_start`, subsequent `weight_refresh` events,
`event=sleep` before each learner, advancing checkpoint versions, valid generated
plans, and memory recovery after each update. Compare generation on one checkpoint
against a fresh engine with deterministic sampling, and confirm reward/node
alignment. Local tests cover orchestration, exact CPU cache parity and refresh
failure handling; they cannot substitute for this GPU/container check.

Commit and push the launcher and implementation files before pulling on HPC;
pulling cannot transfer uncommitted local changes. Submit a new job after pulling.
To roll back, use `REMA_PERSISTENT_ROLLOUT=false sbatch hierarchical-rema-train.sh`.
It does not change checkpoint formats. `GFAM_OVERLAP_REWARDS=false` and `REMA_LENGTH_BUCKETING=false`
independently disable the other optimizations. No launcher settings are changed
in an already-running job by pulling this commit.

## The two kinds of validation

`EVAL_EVERY_STEPS` controls teacher-forced replay validation inside a policy update:
the learner sees saved prompts and completions and measures its loss on those
completions. It does not solve fresh tasks or measure mathematical accuracy. A
wrong saved answer can still have low likelihood loss. Increasing the interval
from 20 to 100 reduces repeated evaluation; 0 disables interim evaluation while
retaining the final evaluation. Integrated training still selects the final update
checkpoint. This setting is unchanged by the throughput implementation.

External validation generates new plans and executions on held-out tasks and
verifies final answers. `VALIDATION_MODE=auto` now uses routine validation when
there is one plan and one execution, and periodic full reward diagnostics.

| Setting | Default | Meaning |
| --- | --- | --- |
| `VALIDATION_MODE` | `auto` | Routine/diagnostic scheduling; `routine` or `diagnostic` forces one mode. |
| `VAL_DIAGNOSTICS_EVERY` | `4` | Every fourth external validation event is diagnostic; 0 means final only. |
| `VAL_DIAGNOSTIC_EXAMPLES` | `30` | Maximum complete rollout examples saved on diagnostic rounds. |

The accuracy schedule is unchanged. The current training preset has
`EXTERNAL_VALIDATION_EVERY_SUBSET_ROUNDS=6`, so accuracy appears at rounds
6, 12, 18, 24, ... and the final round. Diagnostics run at 24, 48, ... and the
final round in auto mode. These are outer subset rounds, not optimizer steps or
full passes over the training dataset. With the current `VAL_TASKS_PER_SUBSET=200`
and `VAL_TASKS_PER_EPOCH=0`, each validation uses the same first 200 available
tasks per subset (or all tasks if the subset is smaller), not the full benchmark.

Routine validation retains final-answer verification and all plan/protocol
checks. It reports accuracy, verification coverage, parse errors, retries, local
repairs, fallback/invalid-output rates, output lengths in characters, and wall
time per task. It does not construct the Graph+PRM scorer. Uncomputed reward
metrics are null in summaries and are not sent as zeros to W&B. Unverified
answers remain excluded from verified accuracy and are exposed through coverage.

Diagnostic validation uses Graph+PRM on those same generated rollouts, with no
second generation pass. Metrics include predicted-outcome disagreement at the
0.5 threshold, incorrect-answer positive final rewards, and mean positive local
credit before/after the final-correctness gate. A node is counted once, using its
actual final/worker reward route. All evaluated tasks contribute to metrics;
`epoch_NNNN/validation_diagnostics.jsonl` contains up to 30 detailed examples in
input order for inspection, not a representative statistical sample. They retain
the task/reference, planner output, node instructions, dependency inputs, worker
outputs and available model predictions/reward breakdowns.

Both modes keep `val/acc/<subset>`, verification coverage, and the existing
`validation_accuracy.csv`. Detailed measures are under `val/diagnostics/...`
and `validation_summary.json`. Reward-specific curves will have points only on
diagnostic rounds. Best-checkpoint selection still uses verified accuracy.

With more than one plan or execution, auto mode retains full reward scoring on
every validation; explicit routine mode is rejected. This avoids silently
disabling scoring for candidate selection. The existing multi-candidate accuracy
definition is not changed. These validation settings are independent of
`REMA_PERSISTENT_ROLLOUT` and `EVAL_EVERY_STEPS`.

# Answer Verification and Live Progress

## Skip Unusable References Before Rollouts

A missing or unsupported reference answer is **unverified**, not incorrect.
The initial audit now filters these examples out of both training and external
validation before epoch sampling, model loading, or rollout generation. No
planner/executor calls or policy updates are performed for them. The original
dataset files are unchanged. Audit issues are marked `skipped_before_rollout`,
and the audit and training summary contain input, kept, and skipped row counts.
If no usable training examples remain, the run exits without loading models,
with status `skipped_no_usable_training_tasks` in `training_summary.json`.

Accuracy is measured on the retained dataset, not the original unfiltered set.
Read it alongside the filtering counts when comparing runs.

## Runtime Safety Net

### Bounded Mathematical Checks

Final correctness, reference parseability, and intermediate-answer matching are
still checked mathematically. Symbolic operations now run in a persistent,
CPU-only child process, rather than in the training process. Each request has
a **10-second hard deadline**, including parsing, solution-set conversion and
equivalence checking. On timeout the parent kills the child, records
`verification_timeout`, and starts a fresh child for the next uncached request.
The child does not import the training entry point or inherit CUDA/Ray state.
Successful and unverified results are kept in a bounded 4,096-entry RAM cache
so repeated executions do not repeatedly stall on the same expression.

Optional environment settings (defaults work without changing the launcher):

```bash
export REMA_VERIFIER_TIMEOUT_SECONDS=10
export REMA_VERIFIER_STARTUP_TIMEOUT_SECONDS=30
```

The second limit allows library imports before the first request; it is not
added to each check. These are positive finite limits, not switches to disable
verification. Missing dependencies, child crashes and unexpected bugs remain
explicit errors. No numerical guessing or exact-match-only fallback is added.

A timeout is **unknown**, not incorrect, and not a clean intermediate result.
For a final-answer timeout, correctness and reward are both `null`. If final
correctness was already verified but an intermediate check fails, that numeric
correctness is retained while the rollout reward becomes `null`. The rollout
is saved for inspection but excluded from policy updates; its plan is not
trained on a selectively successful subset of repeated executions.
Thus `unscored_rollouts` can increase without increasing `unverified_rollouts`,
which counts missing *final correctness*, not incomplete auxiliary checks.

Per-node `verification_checks` record `verified`, `not_applicable` or
`unverified`; failed match fields are `null`, never a fabricated `false`.
`verification_failures.jsonl` also includes the failed check, node ID, delivered
outputs and any already-known final correctness. Matching the reference at an
intermediate node is only an observation, not proof of cheating or mathematical
correctness of the assigned operation. Delivered outputs and reward-model
features are never changed using the reference.

If a verification data failure still arises at runtime, training and external
validation continue, but the affected rollout has no numeric reward (`null`
in JSON). Correctness is also `null` when the final check itself failed, rather
than a later auxiliary check. It is excluded from worker
policy updates. If any execution of a plan is unscored, that plan's planner
sample is excluded too; we do not average only its successfully scored repeats.
There is no fallback to predicted success or a fabricated zero reward.

Ordinary wrong predictions against supported references still receive zero
correctness. Missing verifier libraries, incompatible checkpoints, and unexpected
programming errors remain fatal rather than silently poisoning training.

Earlier versions caught verifier/import failures and used exact string matching.
That avoided crashes but missed mathematically equivalent answers. The stricter
verifier exposed unsupported source references; the new data-error boundary
handles them explicitly without restoring that silent fallback.

The verifier now supports interval set-builder forms such as `{x|-2\lex<1}`
and compact TeX fractions such as `\frac \pi4`. It does not guess the contents
of empty answers, missing fraction operands, or unsupported expressions.
Reference answers remain outside the reward-model inputs and delivered artifacts.

Compound inequalities are rebuilt with native SymPy boolean operators before
conversion to real solution sets. This avoids latex2sympy's `_unsorted_args`
AttributeError when a compound predicate simplifies to a single relation.
Recoverable symbolic parsing/conversion/comparison exceptions are recorded as
unverified, not scored as incorrect; their error codes identify the failed stage.
For example, an unsupported periodic solution set is not guessed or approximated.

## Audit and Coverage

Before model loading, training audits the selected train and validation inputs
and writes `answer_reference_audit.json` under the run output directory.
Each dataset report includes task IDs, row indices, prompts, and problematic
references. The report also records verifier package versions.

To audit files without generating rollouts, from the repository root and in
the training environment:

```bash
PYTHONPATH=src/verl/verl python -m hierarchical_rema.audit_answers \
  --task-source data/MATH/train_lv3to5_8k.parquet \
  --task-source data/overall_math/test.parquet \
  --output outputs/reference_audit.json
```

Parseability is not proof of reference correctness or complete coverage of all
future generated notation. Unsupported references need review, not automatic
rewriting. The standalone audit does not modify or filter dataset files.

Pre-filtered examples appear in `answer_reference_audit.json`, not the runtime
failure file, because they never generate rollouts. Runtime failures are saved
as `verification_failures.jsonl` inside the relevant
epoch or validation output directory, including the generated final answer.
Rollout summaries and W&B report verification coverage and unverified counts.
Accuracy excludes unverified tasks; a partially verified best-of-N task is also
excluded so the denominator does not silently mix different numbers of trials.
Always read accuracy together with coverage. An all-unverified batch produces
no policy update and undefined accuracy, not 0% accuracy.

## Progress and Deployment

Live logs show epoch, phase, task batch, and reward-scoring progress. A heartbeat
appears every 30 seconds while scoring is running, including when one call is
slow. Counts also update every 10 scored trajectories and at completion.
Line-oriented output is flushed so it is readable through `tail -f` in Slurm
and Apptainer, without needing an interactive terminal.

The repetitive single-graph BGE/Qwen progress bars are hidden during live
scoring. Set `GFAM_FEATURE_PROGRESS=1` to restore them for debugging. Offline
feature extraction keeps its usual bars. This is a logging-only change; the
checkpoint-hashed feature builder and checkpoint contract are unchanged.

Sync the updated repository source to HPC and submit a new job. The existing
launcher stages the Python source into its container, so this patch does not
require a new SIF, reward-model checkpoint, or feature extraction cache.
The container must still contain the verifier dependencies already required by
this branch. A running Python process does not pick up source edits automatically.

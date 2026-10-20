# Full Rollout Audit and Temporary Training Filter

The launcher defaults are `ROLLOUT_LOG_MODE=all` and `ROLLOUT_LOG_DETAIL=full`.
Existing environment overrides still take precedence. Each task record in
`all_rollouts.jsonl` contains every sampled plan and every executor repeat:

```text
rollout.task                         task ID, prompt, reference (audit only)
rollout.policy_config                policy/checkpoint configuration
rollout.decompositions[]
  decomposition.decomposition_id    planner candidate ID
  decomposition.raw_payload.raw_model_text  original generated planner text, when available
  decomposition.raw_text            canonical plan text
  decomposition.nodes[]             parsed node instructions and dependencies
  selections[]                     repeated executions of this plan (legacy field name)
    selection.selection_id         executor-repeat ID
    executions[]                   node IDs, workers, delivered upstream inputs,
                                   delivered outputs and raw output text
    reward                         verified final correctness and applied rollout reward
    reward_model_outputs           label probabilities, pre/post-gate rewards, diagnostics
    training_eligible              whether this execution may enter policy learning
```

All-mode logging preserves this full tree even if compact best-selection views
are requested. Worker system prompts are omitted from the execution records as
before; instructions and actual dependency artifacts are retained. Original
planner text is not inferred if the backend did not supply it.

## Suspicious Verified Failures

`GFAM_SKIP_SATURATED_FAILURES=true` excludes an execution if its reference-based
final correctness is zero and **all executed non-final nodes** have
`reward_before_outcome_gate >= GFAM_SATURATED_FAILURE_THRESHOLD` (default `0.95`).
At least one intermediate node and finite pre-gate scores for every such node
are required; an intermediate that already has a negative applied penalty does
not qualify as saturated positive credit. Unknown verification is handled
separately as unscored. The filter
does not change labels, predictions, verified correctness, or numerical rewards.

All completions in that execution, including the final worker, are excluded
before worker GRPO groups/advantages and worker-history updates. A plan with any
excluded repeat is omitted from the planner update; its complete repeated-run
mean stays in the audit log. Other executor repeats can still train when their
groups meet the minimum size. Exclusion counts appear in rollout/epoch summaries.
This is a temporary heuristic, not proof that the intermediate work was wrong:
correct intermediate steps followed by a bad final synthesis can also trigger it.
Review `training_exclusions.jsonl` when updating the reward model.

The filter inspects pre-gate scores because the verified-failure positive-credit
scale defaults to `0.25`. With valid binary probabilities and symmetric utility,
the applied worker/final reward on a verified failure cannot exceed `0.25`.
Scores near `1` can still appear in raw/pre-gate predictions. With lambda=1,
per-label utility is `2*P(good)-1`; e.g. P(good)=0.997 gives utility 0.994.

## S3 Location

The existing epoch uploader copies these files together with `best_selections.jsonl`:

```text
${S3_OUTPUT_PATH}/epochs/epoch_NNNN/rollouts/segment_NNNN/all_rollouts.jsonl
${S3_OUTPUT_PATH}/epochs/epoch_NNNN/rollouts/segment_NNNN/training_exclusions.jsonl
```

`S3_EPOCHS_PATH` can override the epochs prefix. For the training launcher, the
default run prefix is
`s3v2:s3min-tomasznaskret-1712063354/user/jmoska/hierarchical_rema/train/${JOB_ID}`.
Uploads follow the existing completed-epoch sync/cleanup schedule, not each JSONL
append. Full logs are larger. Existing running jobs and old best-only logs are
not retroactively updated; pull these source changes and start a new job.

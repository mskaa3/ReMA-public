# Fixed-prefix worker rollouts

This is an optional sampling mode, independent of `REWARD_SETTING`. The default
`repeated_execution` mode is unchanged. Select it in `hierarchical-rema-train.sh`:

```bash
export WORKER_ROLLOUT_MODE=fixed_prefix
export WORKER_ACTION_SAMPLES=8
export WORKER_CONTINUATIONS_PER_ACTION=4
export WORKER_BRANCH_POINTS_PER_PLAN=1
export WORKER_UPDATE_RULE=both_signs
export WORKER_BRANCH_SEED=42
```

## Sampling

For each sampled plan, select up to `WORKER_BRANCH_POINTS_PER_PLAN` distinct nodes
uniformly before generating worker outputs. Each selected node gets a fresh,
independent prefix. Sample the prefix once using the current executor; do not
select a best/correct prefix or reuse previous-epoch executions. The selection
seed includes the configured seed, update-segment context, call index, task,
and plan. It controls node selection, not the backend's token-generation RNG.

For a group targeting W3 in a chain:

```text
Q -> D -> W1 -> W2                 generated once and frozen
                  -> W3 action 1  -> four independent remaining executions
                  -> ...
                  -> W3 action 8  -> four independent remaining executions
```

The actual rendered worker prompt, system prompt, dependency snapshots, and
executor policy are fixed across actions. Each action is generated once. For a
DAG, complete every not-yet-executed node in topological order, deriving each
node's dependencies from that continuation. All completed graphs contain the
original prefix and the focal action. No downstream result is reused across
continuations. Terminal actions have one completed graph each, not four copies.

The policy checkpoint stays fixed until the existing update-segment boundary.
Only focal actions become worker training samples. Prefix and suffix outputs
are recorded but are not additionally optimized in that comparison group.
Different groups may target different nodes; this is not recursive branching.

In alternating executor phases these counts replace ordinary executor-repeat
counts. Planner phases retain their existing repeated-execution sampling and
rewards. Joint mode generates independent ordinary repeats for planner credit
and separate worker branches; correlated branches do not enter the planner mean.
External validation remains unbranched, so validation accuracy is still measured
with the configured validation sampling budget.

## Scores and updates

The selected reward provider scores each complete graph with existing validation,
protocol checks, and correctness gates. For action j, average that focal node's
compiled reward across its M continuations:

`q[j] = mean(r_focal[j, m])`.

Do not average the rewards of different nodes and do not normalize K*M graphs
directly. For the K action scores:

```text
b[j] = sum(q[l] for l != j) / (K - 1)
delta[j] = q[j] - b[j]
A[j] = delta[j] / (sample_std(delta, ddof=1) + 1e-8)
```

Equal scores provide no policy signal and are not submitted to the learner.
Exactly one sample per eligible sampled action carries `q[j]` and `A[j]`.
The existing offline learner applies the scalar advantage to the action's
response tokens; it does not recompute the group normalization.

- `both_signs`: retain positive and negative advantages.
- `positive_only_nonterminal`: skip nonpositive advantages at intermediate
  nodes, but retain both signs at the final node. Mask after computing baselines.

Scored actions excluded by existing training-safety or leak gates still
contribute their scores to the other actions' baselines, but receive no updates.
This differs deliberately from filtering before grouping in the legacy sampler.
If any required continuation is unscored, unverified, missing, or nonfinite,
skip the whole comparison. Do not substitute zero or average only valid suffixes.
An invalid delivered artifact remains subject to existing protocol penalties;
it is never silently replaced with a raw rejected artifact.

## Cost and auditing

One nonterminal branch point uses K focal generations and K*M suffixes, in
addition to its single prefix. Defaults therefore produce 32 completed graphs
for eight directly trained actions. Branching can be much more expensive than
ordinary repeats. Use smaller task batches for initial HPC memory/throughput
checks; the launcher reports a conservative generation upper bound.

When rollout logging is enabled, fixed-prefix runs always retain
`all_rollouts.jsonl`, even under `ROLLOUT_LOG_MODE=best`. It uses the existing S3
sync path. Each selection contains `branch_metadata`; each decomposition has
`worker_comparisons` with prompt/policy identities, dependency snapshots, action
scores, continuation rewards, LOO baselines, advantages, and exclusion reasons.
Worker prompts are retained in these full branch logs. Fallback plans are logged
as skipped comparisons. `best_*` files remain convenience views, not training
selection mechanisms.

GraphPRM/coach predictions for other nodes remain available for analysis. Report
validation accuracy, generated-token cost, comparison counts, and optimizer
updates when comparing rollout modes; their training best-of-N metrics have
different sampling budgets and are not directly comparable.

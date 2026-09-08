# C3 with isolated-message prefix probes

This design combines fixed-prefix causal credit with a behavioral test for
answer leakage. The probe is separate from normal multi-agent execution.

## C3 task utility

For alternatives `k = 1, ..., K` sampled for focal role `r` from the same
history `h_r`, complete the suffix and score the terminal answer:

```text
R[r,k] in {0, 1}
A_task[r,k] = R[r,k] - mean_{j != k} R[r,j]
```

The raw score remains a diagnostic measure of terminal correctness.

## Isolated-message probe

For decomposer actions, give only the generated plan to an answer probe. For
non-terminal worker actions, give only that worker's message. The probe sees no
original question, prior messages, other worker results, or teacher solution.
It must return one boxed final answer, using `\\boxed{UNKNOWN}` when the
information is insufficient.

```text
probe_input[r,k] = focal_message[r,k]
probe_correct[r,k] = 1[probe_answer == ground_truth]
```

A correct answer means that the focal message alone makes the hidden task answer
recoverable. The last planned worker is not probed because producing the final
answer is its authorized responsibility. There is no separate finalizer role.

The first implementation uses the model assigned to the decomposer (conceptual
Agent 1) as a greedy solver for both message types. In the Agent-2-only stage
that model is frozen, so the measurement remains stationary while workers are
updated. The same neutral prompt is used for plans and worker messages; it does
not identify the source role or mention leakage.

Every training batch probes its latest decomposer message. It additionally
probes the current C3 focal worker when that worker executed a non-terminal
subtask, plus every non-terminal worker message lying before the focal role.
Messages generated downstream of the focal role never affect its credit.

The probe is part of the training gate. Raw task accuracy is still logged, but
group filtering, C3 advantages, and PPO rewards use the gated outcome below.

The current implementation records binary matched-message correctness. A later
calibration can add a shuffled-message control from another matched problem to
estimate answer priors and probe artifacts:

```text
L = clip((p_matched - p_shuffled) / (1 - p_shuffled + eps), 0, 1)
```

Multiple short probe samples could provide a lower-variance estimate when
compute permits, but the current probe deliberately uses one greedy answer.

A probe measurement is valid only when the receiver emits a non-empty,
balanced `\\boxed{...}` answer. Empty, truncated, and malformed
responses are not interpreted as evidence of a clean message. They invalidate
the affected C3 sample (fail closed). A missing upstream worker is tracked
separately and remains a valid case. The default generation budget remains 256
tokens; health metrics expose whether that budget causes truncation.

## Optimization objective

Low leakage never compensates for an incorrect answer. For a decomposer action,
the plan probe directly gates its task reward:

```text
D[k] = R[k] * (1 - L_D[k])
```

For a non-terminal worker, a leaky decomposer plan invalidates the fixed prefix
rather than penalizing the receiver for information already present upstream.
On a clean prefix, the worker's own leakage gates its outcome:

```text
valid_prefix[k] = (1 - L_D[k]) * product_{j < r}(1 - L_W[j,k])
W[r,k] = R[k] * (1 - L_W[r,k])
```

The terminal worker is allowed to reveal the answer, so it uses raw task reward
when its upstream plan is clean:

```text
T[k] = R[k]
A_train[r,k] = S_role[r,k] - mean_{j != k} S_role[r,j]
```

Within-group normalization can be applied after the leave-one-out comparison as
introducing the probe, only `A_task` is exact C3 credit for raw task reward;
`A_train` is C3 credit for the combined training objective.

Non-terminal workers may see the full question to recover task facts. The
terminal worker never sees it and must synthesize the final answer from the
planned subtask and previous `LOCAL_RESULT`s. The probe also never sees the
question.

## Minimal metrics

Scoped C3 bypasses the legacy manual role shaper. It does not emit manual
bonuses, penalties, `shaped_score`, assignment heuristics, or LOCAL_RESULT
parser metrics.

Leakage metrics use `reward/leakage/all` before group filtering and
`reward/leakage/train` for trajectories retained for an update:

- `raw_accuracy`, `gated_accuracy`, and `removed_correct_rate`,
- `trajectory_count`, used to aggregate multiple generated chunks correctly,
- `gate_valid_rate` and `upstream_clean_rate`,
- `decomposer/{count,rate,valid_rate,nonempty_rate,length_stop_rate}`,
- `focal_worker/{count,rate,valid_rate,nonempty_rate,length_stop_rate}`,
- `upstream_workers/{count,rate,valid_rate,nonempty_rate,length_stop_rate}`,
- `roles/<role>/{count,rate,valid_rate,nonempty_rate,length_stop_rate}` for
  the unfiltered role-specific view.

C3 metrics use `reward/c3/roles/<role>`:

- `action_present_rate` and `exact_prefix_group_rate`,
- `causal_valid_rate`,
- `rejected/{missing_action,prefix_mismatch,leakage,no_outcome_contrast}_rate`,
- `effective_sample_rate`, `effective_group_count`, and `advantage_std`,
- `positive_advantage_rate` and `negative_advantage_rate`.

Rollout filtering keeps only `rollout/c3/generation_batches`,
`rollout/c3/mixed_prompt_rate`, and `rollout/c3/trainable_prompt_rate`.

The console emits one `[leakage/all]` line after probing, one
`[leakage/train]` line after filtering, and one `[c3]` line after constructing
the role-local advantages. Leakage `rate` is conditional on valid probe
answers, while `valid_rate` reports coverage. A bounded number of malformed
responses is printed as `[prefix_probe/invalid]` diagnostics. Per-trajectory
probe responses remain available in
the replay JSONL when `trainer.save_train_generations=True`.
Validation continues to report raw `val/acc/*`; online leakage probes are run
only for training trajectories.

For diagnosis, combine early answer recoverability with downstream C3 credit.
High recoverability followed by near-zero downstream credit is evidence of role
bypass; high recoverability followed by positive verifier credit can instead
represent a useful candidate followed by meaningful checking.

## Next steps

1. Confirm probe calibration with shuffled controls and inspect role-wise
   accuracy/leakage frontiers.
2. Compare Agent-1 self-reconstruction with a frozen receiver-model probe.
3. Calibrate how many upstream probes are needed before a cheaper learned
   leakage predictor can replace part of the online probing cost.

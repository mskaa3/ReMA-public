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
It must either return one boxed final answer or declare the information
insufficient.

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

Current metrics are grouped under `reward/prefix_probe`:

- `request_count`, `unique_generation_count`, and `deduplication_rate`,
- `decomposer/leakage_rate` and `decomposer/boxed_rate`,
- `nonterminal_worker/leakage_rate` and `nonterminal_worker/boxed_rate`,
- `upstream_worker/leakage_rate` for contamination before the focal role,
- `roles/<role>/...` for role-specific views,
- `task_correct_and_leaky_rate` and `task_correct_and_nonleaky_rate` for each
  measured source kind,
- `raw_outcome_mean`, `gated_outcome_mean`, and `removed_positive_count`,
- `gate_valid_rate` and `upstream_clean_rate`.

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

# Scoped C3-GRPO

This training mode assigns role-local credit directly from exact fixed-prefix
rollouts and terminal correctness, without manual role rewards.

## Rollout construction

For every problem UID and currently trained role:

1. All `n` alternatives share the same hierarchy through the configured
   `branch_turn`.
2. On that turn, generation remains shared until the focal role.
3. The focal action and every later action are sampled independently.
4. Only groups with both outcome values and at least one trainable focal action
   enter the optimizer.

The action stored for PPO is always the focal action at `branch_turn`, even if
the trajectory continues for later rounds. Its final outcome therefore measures
the long-horizon effect of that action under subsequent repair and backtracking.
With `branch_turn: 0`, the comparison starts in the first round and the complete
remaining trajectory is sampled independently.

## Causal credit

For outcome `R_i` in a fixed-prefix group of size `K`, the role-local
leave-one-out effect is

```text
C_i = R_i - (1 / (K - 1)) * sum_{j != i} R_j.
```

When enabled, `C_i` is divided by its within-group standard deviation.
Invalid actions do not receive gradients, but an exact-prefix action can still
serve as a baseline donor for another action.

The C3 advantage is used directly for every trained role. Role specialization
is intentionally not inferred from teacher-forced scope contrasts. A separate
prefix-probe evaluator can be added without changing the fixed-prefix causal
comparison.

## Key metrics

- `exact_prefix_group_rate`: verifies that C3 groups really share one state.
- `selected_action_turn`: one-based round whose focal action is optimized.
- `rejected_missing_action_count`: focal role did not emit an action.
- `rejected_prefix_mismatch_count`: rollout construction failed exact pairing.
- `rejected_no_outcome_contrast_count`: all exact alternatives had one outcome.
- `effective_sample_count`: actions that actually reach PPO.
- `positive_advantage_count` and `negative_advantage_count`: the sign of the
  fixed-prefix LOO signal among effective actions.
- `advantage_mean` and `advantage_std`: the role-local C3 signal distribution.
- `rollout/c3_trainable_prompt_rate`: generated prompt groups with a nonzero
  C3 training signal.

The intended healthy run has an `exact_prefix_group_rate` near one. A low
`c3_trainable_prompt_rate` with a high exact-prefix rate indicates sparse
outcome exploration rather than a rollout-construction bug.

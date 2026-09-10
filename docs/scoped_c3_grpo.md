# Scoped C3-GRPO

This training mode assigns role-local credit from exact fixed-prefix rollouts
and terminal correctness. The Agent-2-only launcher enables the gates described
in [C3 with terminal-instruction and answer gates](c3_prefix_probe.md).

## Rollouts and baseline

For each (question, teacher attempt) group and current focal role:

1. Generate one shared history before the focal action.
2. Sample n different focal actions, then independently generate each suffix.
3. Score the terminal answer of each complete trajectory.
4. Compare factual alternatives from identical focal prefixes.

For K valid alternatives:

    C_j = Y_j - sum_{l != j}(Y_l) / (K - 1)

When enabled, divide C by its within-group sample standard deviation.
Groups with no outcome contrast provide no update.

Action presence and exact-prefix agreement determine baseline validity.
Leakage measurements determine actor-update eligibility separately. Even an
unparseable or equivalent worker result can remain a baseline donor if its
focal action and raw trajectory outcome are available.

## Masks

The combined leakage gate requires a multi-subtask plan and valid L_T=0:
the terminal instruction alone must not recover the final answer when previous
LOCAL_RESULTs are withheld. Whole-plan L_D is logged but does not gate training.
Non-terminal actions also require a parsed LOCAL_RESULT not equivalent to
the terminal answer. Terminal actions are exempt only from this comparison.

C3 uses raw correctness, not gated correctness. Apply the gate after computing
advantages by masking the focal action's training tokens. Both positive and
negative updates are skipped for rejected actions. A group must contain at
least one eligible nonzero advantage to contribute to the optimizer.

The gates require one round, branch_turn=0; initialization rejects multi-round
configurations with these gates enabled. Generic C3 without the gates still
supports later branching. Extending the leakage experiment to multiple rounds
requires aligning its measurements to the selected action.

## Metrics

Under reward/c3/roles/<role>:

- action_present_rate and exact_prefix_group_rate;
- causal_valid_rate and update_eligible_rate;
- rejected/{missing_action,prefix_mismatch,leakage_gate,probe_invalid,
  no_outcome_contrast}_rate;
- effective_sample_rate, effective_group_count, and advantage_std;
- positive_advantage_rate and negative_advantage_rate;
- positive/negative_before_gate_count, after_gate_count, and removed_count.

Batch filtering uses rollout/c3/mixed_prompt_rate and trainable_prompt_rate.
A high mixed-group rate but low update eligibility points to gate rejection;
a low mixed-group rate points to sparse raw-outcome exploration.

These are noisy relative-action comparisons under independently sampled
suffixes, not worker-removal tests or a guarantee of role specialization.

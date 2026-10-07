# Scoped C3-GRPO

This training mode assigns role-local credit from exact fixed-prefix rollouts
and terminal correctness. The Agent-2-only launcher enables the gates described
in [C3 with terminal-instruction and answer gates](c3_prefix_probe.md).

## Rollouts and baseline

For each (question, teacher attempt) group and current focal role:

1. Generate one shared history before the focal action.
2. Sample K focal actions, then independently generate M suffixes per action.
3. Score the terminal answer of each complete trajectory.
4. Compare factual alternatives from identical focal prefixes.

The generic default is M=1, preserving the original estimator. The Agent-2-only
launcher now defaults to rollout.n=32 and continuations_per_action=4: K=8
independently sampled focal actions, each followed by M=4 continuations. Samples
can coincide in content; they are not forced to be textually unique.

For K valid actions:

    q_j = mean_m(Y_jm)
    C_j = q_j - sum_{l != j}(q_l) / (K - 1)

When enabled, divide C by its within-group sample standard deviation.
Groups with no contrast between action means provide no update, even if their
individual binary outcomes are mixed. All four outcomes enter an action's mean:
do not select only successful or gate-eligible suffixes.

The sampler shares all roles before the focal role across the group, shares
the exact focal output only within its four continuation slots, and generates
all later roles independently. Nested sampling currently requires one round.
When the focal role is terminal, it is generated once per action and there is
no suffix to sample: the four bookkeeping copies carry the same outcome.
They are not four independent observations of terminal correctness.

Only the suffix-zero representative is a baseline entry and an actor training
row. The other three copies have no actor gradient, entropy, or KL contribution.
Raw trajectory scores remain unchanged for accuracy and diagnostics. The mean
is used only for role-local C3 credit. Validation remains ordinary generation,
not a vote or a best-of-32 evaluation.

Action presence and exact-prefix agreement determine baseline validity.
Leakage measurements determine actor-update eligibility separately. Even an
unparseable or equivalent worker result can remain a baseline donor if its
focal action and raw trajectory outcome are available.

For M>1, an action requires all M causal records with identical focal token IDs
and a consistent terminal/non-terminal status. Incomplete records cannot be
averaged over a convenient surviving subset. The actor gate conservatively
requires every continuation's existing gate to pass; one rejection or unknown
vetoes the shared action's update. Its complete raw outcome mean still remains
a baseline donor. This all-suffix rule can reduce update coverage; it is a
deliberate conservative gate, not a consequence of Monte Carlo estimation.

## Configuration and cost

The Agent-2-only shell defaults are ROLLOUT_N=32 and
C3_CONTINUATIONS_PER_ACTION=4. The common launcher passes them to rollout.n and
algorithm.hierarchy.scoped_c3_grpo.continuations_per_action. The ratio must be an
integer of at least two actions. Set ROLLOUT_N=16 and
C3_CONTINUATIONS_PER_ACTION=1 to reproduce the previous sampling layout.

With the current batch, 3 questions x 16 teacher attempts x 32 slots produces
1536 trajectory rows rather than 768. Relative to 16x1, this halves focal action
samples but doubles post-focal suffixes when a suffix exists. Prefix sharing
reduces generation, but tensor/history storage and scoring still use all slots.
Runtime is not necessarily doubled. Memory pressure may increase; token limits,
GPU allocation, and sparse-rank safeguards are unchanged. Eight focal actions
are not 32 trainable actions, so sparse updates can still be skipped.

## Statistical interpretation and limitations

For a fixed prefix and focal action, q_j is a Monte Carlo estimate of downstream
success under the current sampling policy. Under independent suffix sampling,
Var(q_j | prefix, action) = p_j * (1 - p_j) / M. M=4 halves conditional standard
error relative to M=1, not necessarily the variance of the entire policy update.
The K/M allocation trades action exploration for return-estimation precision.

Repeated suffixes reduce lucky downstream compensation. They do not verify the
subtask: systematic downstream repair, ignored messages, answer leakage missed
by the gates, and scoring mistakes can still get high estimated success. Four
samples remain noisy. A 1/4-success action can receive positive relative credit
if the alternatives are worse. Conversely equal action means give no signal.
If true action values are identical, group standardization can expand small
sampling differences back into order-one advantages; positive-only selection
can then reinforce random apparent winners. Averaging is not a confidence test.
Within-group normalization, positive-only selection, and output-dependent gates
retain the original method's biases; only the unfiltered, unnormalized return
and leave-one-out estimates have the straightforward unbiased interpretation.

## Masks

The combined leakage gate requires a multi-subtask plan and valid L_T=0:
the terminal instruction alone must not recover the final answer when previous
LOCAL_RESULTs are withheld. Whole-plan L_D is logged but does not gate training.
Non-terminal actions also require a parsed LOCAL_RESULT not equivalent to
the terminal answer. Terminal actions are exempt only from this comparison.

C3 uses raw correctness, not gated correctness. Apply the gate after computing
advantages by masking the focal action's training tokens. Both positive and
negative updates are skipped for rejected actions.

The default config also requires an eligible success in each optimizer group
(require_eligible_success). Mixed raw outcomes alone are insufficient if every
success is gated out. With positive_only_nonterminal_workers, negative
non-terminal actions are masked after normalization, while terminal actions
retain signed advantages. The actual terminal role is read per trajectory,
not inferred from a fixed final slot. Decomposer/selector actions remain signed.
All factual alternatives remain baseline donors; advantages are not re-centered
after masking. Disabling both flags restores the previous signed C3 policy.

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
- positive/negative_before_gate_count, after_gate_count, and removed_count;
- positive/negative_after_policy_count and policy_removed_count;
- no_eligible_success_group_count and rejected/no_eligible_success_rate.

Repeated-continuation runs additionally log action_count, continuation_row_count,
continuations_per_action, valid_action_count, all_suffix_gate_eligible_action_count,
action_success_mean, action_success_std, within_action_outcome_variance, and
suffix_gate_disagreement_rate for the latest generated batch. Actor-batch
metrics optimizer_action_count, effective_action_rate, and masked_replica_row_count
count actual action representatives. Existing effective_sample_rate still uses
all trajectory rows as its denominator and is therefore at most 1/M. Console
[c3/action_means] distinguishes actions from continuations; [c3] retains the
positive/negative action counts. Replay JSONL records action/suffix indices,
action mean scores, representative masks, and action update eligibility. The
latter is eligibility, not proof that a distributed optimizer step occurred.

The gate counters refer only to leakage eligibility; after_policy_count records
the actions actually retained by the conservative update rule. Attach-time
rejection counters describe the latest generated batch, while advantage-time
counters describe the collected optimizer batch.

Batch filtering uses rollout/c3/mixed_prompt_rate and trainable_prompt_rate.
A high mixed-group rate but low update eligibility points to gate rejection;
a low mixed-group rate points to sparse raw-outcome exploration.

These are noisy relative-action comparisons under independently sampled
suffixes, not worker-removal tests or a guarantee of role specialization.

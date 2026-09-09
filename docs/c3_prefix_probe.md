# C3 with plan probes and answer-equivalence gates

The Agent-2-only experiment learns a shared worker model under a frozen
decomposer. The teacher supplies solution attempts to the decomposer.
Non-terminal workers see the question, their assigned subtask, and earlier
LOCAL_RESULTs. The last planned worker is terminal: it sees its assigned
subtask and earlier LOCAL_RESULTs, but no separately supplied full question.
There is no separate finalizer.

## Raw outcome and C3

For alternatives j sampled at focal role r from exactly the same prefix:

    Y_j = 1[terminal_answer_j is correct]
    C_j = Y_j - mean_{l != j}(Y_l)
    A_j = C_j / (std(C) + epsilon)

Only mixed groups with at least two factual alternatives and at least one
eligible focal action are used for optimization. Masked actions remain
baseline donors. Their outcomes are not replaced by zero before computing
C3, normalizing advantages, or checking outcome contrast.

## Plan gate

Generate a greedy answer probe from the decomposer output alone. The probe
does not receive the question or teacher attempt as separate inputs.

    L_D = 1[probe_answer is correct]
    G(P) = 1[number_of_routed_subtasks >= 2] * valid(L_D) * (1 - L_D)

This gate applies to every role, including the terminal worker. A one-subtask
plan, a recoverable answer, or an invalid plan probe prevents updates from
that plan. In Agent-2-only, the plan is shared by the focal worker's entire
C3 group, so the whole group is excluded from optimization.

The router records the actual subtask count in decomposer history metadata.
The gate uses this count, including routing limits/fallbacks, rather than
guessing the count from the number of available worker slots.

The probe uses one greedy response per unique plan/scoring context in a
generated batch. A complete boxed UNKNOWN is a valid negative recovery result;
an empty, malformed, or unparseable mathematical answer is an invalid
measurement. Surfaced verifier timeouts are also invalid, rather than evidence
that the plan is clean. The generation limit remains configurable and defaults to 256
tokens.

Plan gating currently happens after trajectory generation, before group
selection and actor optimization. It saves optimizer work on rejected groups,
not suffix generation. Moving it before branching is a separate performance
optimization.

## Non-terminal worker gate

Extract the boxed answer z_k from the worker's LOCAL_RESULT, and compare it
against the boxed terminal answer y_hat from the same trajectory:

    E_k = MathVerify(z_k, y_hat)
    M_k = G(P) * valid(E_k) * (1 - E_k)

This comparison uses the generated terminal answer, not dataset ground truth.
It does not read other boxes in the worker's reasoning. Missing LOCAL_RESULT,
empty/incomplete boxes, parser failures, and comparison errors/timeouts are
unknown measurements and exclude the affected action from updates.

Math-Verify 0.7's symbolic comparison is used with strict variable matching
and without extraction fallback strings. Its public verify function converts
comparison exceptions into False; the gate instead preserves those failures
as unknown.

For the terminal worker, no equality comparison is required:

    M_terminal = G(P)

For decomposer/selector updates, the same plan gate applies. The decomposer
is frozen during Agent-2-only training.

Worker before/after LLM probes and their information-gain metrics have been
removed. Worker comparisons require no additional model generation.

## Actor updates

    A_train_j = M_j * A_j

The actor's token mask is zeroed for excluded focal actions, for both positive
and negative advantages. Other alternatives remain in the C3 baseline.
No manual penalties, PRD, CPCR, or TSS enter this training path.

This is selective policy optimization. Because M depends on generated
outputs, it is not an unbiased policy gradient of raw accuracy alone.

## Validation and logging

Raw val/acc/* continues to cover the full validation set, including one-task
plans. Leakage diagnostics run on the configured validation sample budget,
including examples printed live; those examples are not probed again later.

Training evaluates E only for the focal non-terminal role. Validation evaluates
every executed non-terminal worker and requires all comparisons to be valid
and unequal for a trajectory to pass. Thus validation gated accuracy is a
trajectory-wide diagnostic, while training gated accuracy is role-local.

Metrics use reward/leakage/all, reward/leakage/train, and val/leakage:

- trajectory_count, raw_accuracy, gated_accuracy, removed_correct_rate;
- plan_eligible_rate, gate_valid_rate, update_eligible_rate, subtask_count_mean;
- rejected/{single_subtask,plan_probe_invalid,plan_recoverable,
  comparison_invalid,equivalent_answer}_rate (exclusive rejection reasons);
- decomposer/rate and count: recovery rate over valid plan measurements;
- decomposer_valid/rate and count: validity over requested plan probes;
- decomposer_nonempty and decomposer_length_stopped: probe health;
- worker_match/rate and count: equivalence over valid worker comparisons;
- worker_match_valid/rate and count: validity over requested comparisons;
- roles/<role>/worker_match and worker_match_valid: the same per role.

C3 metrics under reward/c3/roles/<role> additionally report positive and negative
before_gate_count, after_gate_count, and removed_count. These counts are for
the batch reaching advantage construction; leakage/all includes generated
groups discarded before optimization.

The console prints [leakage/all], [leakage/train], [leakage/val], and live
[leakage/example] lines with raw/gated scores, actual terminal role, subtask
count, Ld, E per measured worker, plan/update eligibility, and rejection reason.
Replay JSONL stores plan-probe responses, comparisons, counts, masks, and raw
and diagnostic gated scores. Old B/A fields are no longer emitted.

## Interpretation

Equivalent non-terminal and terminal outputs are excluded by design, including
accidental equality. Different answers do not prove subtask obedience.
The plan probe measures answer recoverability by a particular solver, not
literal answer disclosure. Partial answer leakage can escape it, and a
sufficiently informative legitimate plan may also be rejected.

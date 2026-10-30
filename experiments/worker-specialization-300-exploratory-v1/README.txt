EXPLORATORY MODEL-REVIEWED WORKER COMPARISON
==========================================

Status (2026-10-30)
-------------------
The assistant authored and self-reviewed the reserved selection, at the user's
explicit request for an exploratory comparison before later human review.
No reserved worker outputs were inspected in preparing this package. Historical
logs/development examples were previously discussed; this is not unseen data.
No inference or training was run as part of preparing this package.

All 300 selected questions are accounted for:
  276 provisionally included with plans, local answers, and mathematical checks.
   24 explicitly deferred for missing/ambiguous data, reference issues, or an
      unresolved proof. They were not replaced by easier sampled questions.
  520 local tasks; 764 prompts per model; 2292 responses across three models.
   40 single-task questions remain terminal-only, outside the primary endpoint.
  236 questions / 244 tasks contribute to each nonterminal-compute endpoint.

This is MODEL SELF-REVIEW, not independent mathematical certification.
The untouched source selection and pending human-review batches remain in
experiments/worker-specialization-300-v1/. Running this new package exposes
outcomes on that selection: later reviewing the same questions does not turn
them back into a previously uninspected confirmatory test set.

Files
-----
review-cards.txt: readable mathematical review of every sampled question.
review.json: all 300 decisions, tasks, references, review notes, provenance.
review-summary.json: coverage, deferrals, and exact-check inventory.
reference-verification.json: executed CPU verifier/arithmetic checks and hashes.
frozen/dataset.json: fixed evaluation inputs and hidden scoring references.
frozen/eval-config.json: populated checkpoint paths; no placeholders.
build_review.py: reproducible authoring source, not a model-generation script.
check_math.py: exact arithmetic and enumeration checks for part of the review.
verify_references.py: additional production-verifier and symbolic checks.

Included / sampled by subset
---------------------------
MATH500          45 / 46
AIME24            2 / 3
AIME25            1 / 3
AMC23             4 / 4
Gaokao           31 / 35
GSM8K           121 / 122
Minerva Math     23 / 25
OlympiadBench    49 / 62

Example of a genuine reference issue: 80 indivisible stones weighing 75 lb each
cannot fit in three 2000 lb truckloads. Each fits at most 26 stones, so four
loads are necessary, despite the source answer of three. This case is deferred,
not silently given a new source answer. Difficult proof questions without a
justified derivation are also deferred, not certified by copying their answers.
These omissions change the difficulty mix; report the coverage table.

Eleven final-reference notation normalizations are explicit and retain the
original labels. They include scientific e-notation, np.arcsin, binomial
coefficient notation, one subscript, and a flattened list of two five-tuples.
They change the mathematical representation, not the intended answer. Some
word-problem conventions and supplied physical constants are noted per card.

What was checked
----------------
287 exact arithmetic/enumeration checks cover 150 questions, including the
grade-school arithmetic, coin-change DP, integer enumeration, and divisibility.
Four additional symbolic checks support selected algebraic references.
All 336 distinct scoring-reference strings passed the production Math-Verify
0.7.0 self-check; all 276 terminal references matched their reviewed final
references. Four known-incorrect comparison controls were rejected.
These are partial checks, NOT automated proofs of all 276 plans.

The production prompt-capture harness checked all 764 inputs: assigned task
only, reference question only where permitted, correct fixed predecessor
results, and no hidden reference/review metadata in model messages. The worker
study and fixed-prefix test suites passed 29 tests when this package was made.
The cluster runner repeats its own runtime preparation before loading models.

Comparison
----------
W0: shared workers, job 6055460, step 50 (already trained).
Ungated: job 6106013, step 63.
Gated: job 6106014, step 65.

Every model gets the same assigned task and correct fixed predecessor outputs.
Nonterminal tasks run both with full Q and without Q. Terminal tasks run once,
without Q. No gating filters evaluation examples. This isolates local worker
behavior from differences in planner output and upstream arithmetic errors.
It does not evaluate the complete multi-agent system or prove that gating
caused any checkpoint difference: histories and update budgets differ.

Primary comparison remains gated minus ungated on production-delivered correct
local results for nonterminal compute tasks with full Q, weighting questions
equally. Keep all secondary/context/W0 results, including unfavorable ones.
The stated-answer diagnostic is separate from production delivery correctness.
Correct final values do not certify sound reasoning or role specialization;
inspect paired, blinded outputs after the quantitative comparison.

Precision caveat: the existing subset-stratified bootstrap deliberately refuses
to supply a pooled CI when an endpoint contains a stratum with only one question.
Here AIME25 has one included question. Expect N/A CIs for affected aggregate
endpoints, with point estimates and counts still available. Do not interpret
N/A as zero uncertainty or choose exclusions after observing outputs to obtain
significance. Resolving the deferred references before inference, or defining
a separate prospective sampling/analysis protocol, is necessary for that
stronger statistical claim. This package is for checking the direction now.

Launch from the repository root on the cluster
---------------------------------------------
First transfer/version this folder and the accompanying script changes. The
following configuration is already frozen; no manual JSON editing is needed.

sbatch --partition=lem-gpu-normal --time=08:00:00 \
  scripts/eac3_fixed_prefix_eval.sh \
  experiments/worker-specialization-300-exploratory-v1/frozen/eval-config.json

One node, one Hopper GPU. The three models run sequentially. Eight hours is a
wall-time request, not a measured runtime guarantee. Existing launcher options
PREFIX_EVAL_SIF (local image) and PREFIX_EVAL_REMOTE (S3 result backup) still work.
No S3 weights were downloaded or cluster job submitted during preparation.

Results: results/fixed-prefix-<job_id>/comparison/report.json, report.md and
review.jsonl. Keep raw responses, manifests and prepared.json as well.

Before inference, fix the output audit at 60 retained questions, seed 20261101,
including every model, task and context for those questions. After inference:

python3 -B scripts/eac3_worker_study.py audit \
  --review results/fixed-prefix-<job_id>/comparison/review.jsonl \
  --output results/fixed-prefix-<job_id>/blind-audit \
  --questions 60 --seed 20261101

Keep PRIVATE-key.json away from annotators. Assistant labels can be provisional;
subsequent independent human review must remain distinguishable.

Optional CPU recheck inside the training container (no GPU needed):

python3 -B experiments/worker-specialization-300-exploratory-v1/verify_references.py \
  --output /tmp/exploratory-reference-recheck.json

The output path must not already exist. The builder also refuses to overwrite
frozen artifacts. Version a new package for changes rather than silently
revising inputs after observing checkpoint outputs.

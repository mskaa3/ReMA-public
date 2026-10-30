# Reserved 300-question worker comparison

## Current status

Prepared from the existing selection
`c5baddf888a22c799519b83aae2cbafd2cbf7f7d86ccb1e172af7cbb19fa853e`.
No resampling after development results. No reserved worker outputs inspected.
The 12 review batches contain all 300 questions, but their tasks and independent
reviews are **not yet completed**. This is not a launchable evaluation dataset.

| Subset | Reserved questions |
|---|---:|
| GSM8K | 122 |
| OlympiadBench | 62 |
| MATH500 | 46 |
| Gaokao | 35 |
| Minerva Math | 25 |
| AMC23 | 4 |
| AIME24 | 3 |
| AIME25 | 3 |
| Total | 300 |

Keep all decisions, including exclusions and naturally single-task questions.
Do not replace difficult cases or add questions until a desired result appears.
The final number of eligible nonterminal compute questions can be below 300.
Small subsets are coverage, not well-powered subset-specific experiments.

## Fixed comparison

- W0: shared, already-trained workers, job 6055460 step 50.
- Ungated: job 6106013 step 63.
- Gated: job 6106014 step 65.
- One identical reviewed plan and correct prefix for each paired comparison.
- Nonterminal tasks: full question versus subtask-only context.
- Terminal tasks: no full question, once per model.
- Every task of the reviewed plan is evaluated. No gate filters evaluation.
- Same prompts/tokenization, greedy decoding, 1024 response tokens, 8192 context,
  batch size 8; three checkpoints evaluated separately.

This is a comparison of historical checkpoints on historical validation data.
It is not an estimate of training-seed variability or a causal isolation of
gating. The checkpoints have different training histories. Do not call 300
questions 300 independent training replications.

## Outcomes and decision rules, before reserved inference

**Primary:** question-weighted gated-minus-ungated difference in correct,
production-routable results for nonterminal compute tasks with full Q.
Keep this endpoint unchanged from development. A paired, subset-stratified
question bootstrap supplies a 95% interval conditional on these checkpoints.

**Secondary:** the same comparison without Q; the context interaction;
terminal delivery; W0 comparisons; parser/missing/unknown/truncation rates;
and the separately reported stated-answer diagnostic. Report every endpoint,
not only the favorable one. Subset breakdowns and secondary findings are
exploratory, not many independent significance claims.

The evaluation-only `standalone-local-result-v1` diagnostic ignores inline
quotations of upstream `LOCAL_RESULT:` labels when there is exactly one explicit,
standalone `LOCAL_RESULT:` section. It refuses multiple standalone sections,
multiple boxes in that section, malformed boxes, or oversized answers. Without
a standalone section it requires one box in the entire output. It never chooses
an answer by looking at the reference. Production routing/rewards are unchanged.
This diagnostic is **stated-answer correctness**, not proof of sound reasoning.
Ambiguous/unverified responses remain visible and need audit; they are not
silently relabelled wrong or dropped from denominators.

For the primary endpoint, a positive interval excluding zero supports a
checkpoint-level advantage on this sampled population. Treat 5 percentage points
as a practical-effect reference, not a guarantee or a post hoc stopping rule.
An interval crossing zero is inconclusive about direction; it does not establish
equivalence. Claiming an advantage of at least 5 points needs a lower confidence
bound above 5 points, not just a point estimate above 5.

A fixed sample of 300 improves precision, but neither guarantees significance
nor excludes small effects. Development has very few discordant cases and is
not a reliable promise of power. Complete the fixed study regardless of the
direction of results. Do not pool development into the reserved headline result.

**Role specialization needs an independent outcome audit.** Before inference,
fix a random audit of 60 retained questions (seed 20261101), retaining all their
models, tasks, contexts, and correct/incorrect/undelivered responses. Use the
existing five-field rubric for task adherence, local math, reasoning soundness,
whole-question solving, and visible previous-result use. Have two independent
human reviewers label at least a random 20-question overlap and adjudicate blind
to model identities. Model-only labels are provisional, not human evidence.
Do not classify legitimate forwarding, repeated intermediate values, or terminal
whole-question solving as violations automatically. Positive task adherence
without reasoning correctness is not sufficient evidence of better reasoning.

## 1. Complete the review batches

Files `review/review-01.json` through `review/review-12.json` each contain 25
questions. A model may author candidates, but these must not be produced or
revised after examining the evaluated worker responses. A distinct human must
check local references, task sufficiency, no supplied compute-task answers,
coherence, and terminal consistency. The reference answer in a review card is
author/checker metadata, not an instruction to expose it to the workers.

See `docs/eac3_worker_specialization_study.md` for the task schema. Record honest
author/checker identity and types. Do not approve blank checks automatically.
The approved development self-review exception does NOT apply here.

After all 12 batches have been authored and independently checked, from the
repository root on the cluster:

```bash
python3 -B scripts/eac3_worker_study_pack.py assemble \
  --package experiments/worker-specialization-300-v1/review \
  --output experiments/worker-specialization-300-v1/completed-review.json

python3 -B scripts/eac3_worker_study.py freeze \
  --selection experiments/worker-specialization-300-v1/review/selection.json \
  --review experiments/worker-specialization-300-v1/completed-review.json \
  --output experiments/worker-specialization-300-v1/frozen
```

Pending, missing, duplicated, substituted or self-certified reserved questions
are rejected. `freeze` additionally validates the full task schema and creates
the only launchable configuration. The package/frozen directories and review
decisions should be versioned before observing reserved outputs. A local hash
records integrity; it is not an externally timestamped preregistration.

## 2. Run inference, not training

Only after successful freeze:

```bash
sbatch --partition=lem-gpu-normal --time=08:00:00 \
  scripts/eac3_fixed_prefix_eval.sh \
  experiments/worker-specialization-300-v1/frozen/eval-config.json
```

This requests one node and one Hopper GPU, not a training allocation. Adjust the
wall-time request to the site's permitted limit and observed development
throughput; eight hours is a request, not a measured runtime guarantee. The
default two-hour short job was for the pilot. Up to 2100 prompts per model and
6300 responses overall are possible with four-task plans. Additional diagnostic
scoring uses CPU Math-Verify checks, not extra LLM generations. Review and scoring
are real costs as well as inference.

Optional existing-SIF and S3 settings are `PREFIX_EVAL_SIF` and
`PREFIX_EVAL_REMOTE`. Model paths are inherited from the original selection,
not placeholders. The runner downloads them using the existing launcher.

## 3. Inspect the report and audit, regardless of the outcome

Results are under `results/fixed-prefix-<job_id>/`, including manifests,
`prepared.json`, raw responses, scores, and `comparison/report.json` plus
`comparison/review.jsonl`. Keep all of them, not just accuracy aggregates.

```bash
read -r -p 'Completed evaluation job ID: ' EVAL_JOB_ID
python3 -B scripts/eac3_worker_study.py audit \
  --review "results/fixed-prefix-${EVAL_JOB_ID}/comparison/review.jsonl" \
  --questions 60 --seed 20261101 \
  --output "results/fixed-prefix-${EVAL_JOB_ID}/blind-audit"
```

Annotators receive `examples.jsonl` and `labels.jsonl`, not `PRIVATE-key.json`.
Outcome audit is separate from checking the reference tasks before inference.
Keep uncertainty, parser disagreement and audit disagreements in the report.

## What this can settle

It can provide a substantially better estimate of **how these three checkpoints
differ on the same local tasks**, whether full-Q context changes that gap, and
whether an apparent gap is mathematical, behavioral, or just delivery-related.

It cannot on its own prove that EAC3 caused the difference, that cooperation is
causally necessary, or that end-to-end accuracy improved. Those need matched
training replications, dependency interventions, or end-to-end evaluations,
respectively. A negative or mixed result is still a completed comparison.

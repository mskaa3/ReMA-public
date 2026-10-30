# Worker specialization: from pilot to credible evidence

The existing 300-question sample is packaged in
`experiments/worker-specialization-300-v1/review/`. It is not a second sample.
See `experiments/worker-specialization-300-v1/README.md` for its current pending
review status, fixed analysis rules, and launch commands. Evaluation now also
reports a secondary stated-answer diagnostic; production routing and the primary
accepted-local-result endpoint remain unchanged.

## The claim we test

**Given the same assigned mathematical task and correct previous results, does
the gated worker checkpoint deliver more correct local results, and follow its
assigned role more faithfully, than the ungated checkpoint? Does access to the
full question change that difference?**

This is an evaluation of workers, not a new reward or training algorithm. No
decomposer generates plans during evaluation. No eligibility gate filters the
evaluated responses. All checkpoints get exactly the same reviewed instructions
and prefixes. The existing evaluator captures the actual production prompts.

| Model | Source | Meaning |
|---|---|---|
| w0 | 6055460, step 50 | Common initialization, already trained |
| ungated | 6106013, step 63 | First ungated worker-training round |
| gated | 6106014, step 65 | First gated worker-training round |

The exact S3 exports are in `config/eac3-fixed-prefix-pilot.json`; the study freezes
them into its selection manifest. These are NOT the second-round workers in the
earlier A/B/C/D audit. Historical checkpoints differ in training history/budget.
Their comparison cannot isolate the causal effect of EAC3 training.

## Design fixed before looking at outputs

1. Sample 30 development questions and reserve 300 more, proportionally stratified
   by `subset` (GSM8K, MATH500, etc.), using a recorded seed. Use development questions to settle
   the annotation protocol and check the runner. Do not tune on reserved results.
2. For every selected question, author one coherent plan of 1-4 tasks. Review the
   local references, each prefix and the sufficiency of each individual task.
   Evaluate every task, not a hand-picked successful stage.
3. Evaluate every nonterminal task under both `full_question` and
   `subtask_context`. Only the Q block changes. Evaluate the terminal once, without
   Q, because the current production protocol never separately supplies it with Q.
4. Run all three checkpoints greedily with the same limits. Compare paired
   outcomes and inspect a model-blinded random sample of responses.

The 30/300 sizes are starting design choices, NOT a completed power calculation.
Before inspecting reserved model outputs, choose a minimum useful difference
(for example 5 percentage points) and assess whether the planned precision is
adequate using development question-level variability. Do not increase the sample
repeatedly until a desired result becomes significant. A small or inconclusive
difference is a valid result.

The supplied pool, `data/overall_math/test.parquet`, is historical validation
data. "Reserved" means reserved for this new protocol, NOT unseen by earlier
evaluation or checkpoint selection. The metadata explicitly says
`historical_validation_exploratory`. For genuinely confirmatory generalization
evidence, select an untouched pool and preregister a separate protocol. The
current script deliberately does not relabel this pool as confirmatory.

Sampling removes exact whitespace-normalized overlap with the configured known
training files, duplicate questions, and the diagnostic pilot questions.
The default `missing_reference_policy: "exclude"` records and excludes rows with
missing, empty or nonfinite reference answers BEFORE sampling; it never invents
an answer from a solution trace. These rows remain visible in
`input-exclusions.json` and the selection manifest. Use `"error"` for strict
failure instead. Unsupported structured answers and malformed question/source
fields still fail with the file path and zero-based row index.
The default `stratify_by: "subset"` uses actual benchmark names because
`data_source` is the shared reward-routing label `ReMA-math` in this dataset.
Other datasets may explicitly select `stratify_by: "data_source"`.
Conflicting duplicate references/strata fail instead of selecting arbitrarily.
This is NOT semantic deduplication or proof of no pretraining contamination.
Check that `known_training_files` lists every relevant training dataset before
sampling; the default is the repository's MATH training file, not a complete
reconstruction of all historical runs.

## 1. Sample questions, without GPUs

From the repository root, with pandas and a parquet engine installed:

```bash
python3 -B scripts/eac3_worker_study.py sample \
  --config config/eac3-worker-specialization-study.json \
  --output experiments/worker-specialization-v1/selection
```

If the login machine lacks pyarrow, use the existing training SIF inside an
interactive allocation. Set `PREFIX_EVAL_SIF` to the actual existing SIF path;
this step needs no GPU and does not download models:

```bash
test -s "${PREFIX_EVAL_SIF:?Set PREFIX_EVAL_SIF to an existing verl-rema-v3.sif}"
apptainer exec --writable-tmpfs --bind "$PWD:/repo" --pwd /repo \
  "$PREFIX_EVAL_SIF" python3 -B scripts/eac3_worker_study.py sample \
  --config config/eac3-worker-specialization-study.json \
  --output experiments/worker-specialization-v1/selection
```

Outputs: `selection.json`, `input-exclusions.json`, `development-review.json`,
`PRIVATE-reserved-review.json`. All output directories must be new. File hashes,
sampled IDs, strata, exclusions and checkpoint sources are recorded. No model
outputs or correctness scores are used to select questions.

Keep the reserved file with the benchmark author until development decisions
are settled. A filename is not access control. Hashes help detect accidental
changes; they are not a signed preregistration.

## 2. Author and independently review the tasks

The review templates are intentionally incomplete. Do not mark checks true
without doing them. A model can draft tasks; record its actual identity and
`author_type: "model"`. A distinct checker must inspect them. The reserved split
requires `checker_type: "human"`, including exclusions. Different IDs alone do
not ensure independent judgement; use a genuinely independent reviewer.

For development only, an explicit opt-in permits a model author to self-review:
set top-level `review_mode: "model_self_review"`, record the SAME actual
author/checker ID with both types `model`, and pass
`--allow-development-self-review` to `freeze`. Do not invent a second reviewer.
Every inclusion decision/check must still be substantively completed. The frozen
dataset and reports prominently label this as development-only and NOT
independently reviewed. Without the flag the normal independent-review rule
applies. Reserved questions reject both this mode and this flag; their human
review requirement is unchanged.

For each question choose `decision: "include"` or `"exclude"`. Every exclusion
needs a reason. Do not drop rows, silently replace questions, or exclude based on
checkpoint outputs. Publish exclusion counts/reasons and scope the claim to the
retained population. A high exclusion rate is itself a limitation.

Fill `tasks` with objects like this (JSON requires doubled backslashes):

```json
{
  "instruction": "Compute 6 times 9. Return the number of blue pens.",
  "kind": "compute",
  "reference_answers": ["54"],
  "checked_result": "LOCAL_RESULT: \\boxed{54}",
  "reference_check": "Six bags, nine pens per bag: 6*9=54."
}
```

Use `compute` for genuine calculation/derivation, `forward` for extracting or
relaying supplied data, and `synthesize` for combining established results into
the requested final form. Classify before observing responses. Do not force
every problem into many subtasks: one-task plans are permitted, but contribute
only to the terminal endpoint, not evidence about nonterminal specialization.

The review checklist requires:

- The instruction does not already state the answer to the assigned calculation.
- The task can be executed from its own instruction and the routed previous
  results WITHOUT Q. It must not rely on hidden earlier subtask instructions.
- Units, domains, indexing conventions, modulus, precision and output meaning
  are explicit where needed. For a digit lookup, for example, explicitly say
  "1-based indexing" in the lookup task, not only in the preceding task.
- Previous results are correct and parseable; local references are mathematically
  correct; the terminal reference answers the original question.
- The plan is coherent. Later tasks may use earlier local answers, but a prefix
  should not pre-solve the later assigned calculation.

Keep `checked_result` minimal, normally one labelled box. It is actual model
input. References and `reference_check` are evaluator-only metadata, never inputs
to the target worker. Automated checks enforce parser/format consistency, NOT
mathematical truth, answer-free instructions or sufficiency. Those require review.

## 3. Freeze and run development first

After filling the development review:

```bash
python3 -B scripts/eac3_worker_study.py freeze \
  --selection experiments/worker-specialization-v1/selection/selection.json \
  --review experiments/worker-specialization-v1/selection/development-review.json \
  --output experiments/worker-specialization-v1/development

sbatch scripts/eac3_fixed_prefix_eval.sh \
  experiments/worker-specialization-v1/development/eval-config.json
```

This reuses the one-node, one-GPU inference-only launcher. Three models load
sequentially. Up to seven prompts per four-task question are generated per model:
two contexts for each of three nonterminals plus one terminal. Thus 300 four-task
questions mean up to 6,300 responses overall, not 300 responses. The launcher's
two-hour request is not a measured runtime guarantee for the larger study; use
development throughput to set the reserved job wall time appropriately.

Preparation checks reference notation using Math-Verify before loading models.
Reports reject different input tokenizations/settings, missing model runs,
missing context pairs or unexpected checkpoint sources. Current code reconstructs
prompts; it does not reproduce historical training inputs byte-for-byte.

Once development is complete, finalize the rubric/sample size BEFORE reserved
responses. Independently review the reserved questions, then freeze with
`--review .../PRIVATE-reserved-review.json --output .../reserved` and submit its
`eval-config.json` in the same way. If changing the selection design, create a
new version with explicit provenance; never overwrite the existing sample.

## 4. Read the right endpoints

The primary comparison is **gated minus ungated accepted local correctness on
compute/nonterminal/full-question tasks**. W0 comparisons establish whether
either later checkpoint improved over the already-trained initialization.
Compute/no-Q, forwarding, synthesis and terminal results are separate endpoints.

The main binary endpoint is confirmed correct delivery: a routable result which
matches a local reference. Unknown verification and missing results are counted
as unconfirmed delivery, not labelled mathematically wrong. Report their counts
and manually inspect them. An answer match with the final Q answer is only a
diagnostic: a legitimate intermediate result can have the same value.

`comparison/report.json` includes equal-question means (average targets within
each question first), paired differences and 95% question-bootstrap intervals
stratified by source. Questions, not responses, are the sampling unit. CIs are
unavailable when an endpoint has a stratum with fewer than two questions.
The target population is the included historical question pool, not all math
problems. Report source-specific coverage/exclusions. Secondary comparisons
are exploratory; do not treat many unadjusted tests as independent confirmations.

The context interaction is:

```text
(gated_fullQ - ungated_fullQ) - (gated_taskOnly - ungated_taskOnly)
```

A positive interaction alone does not show that gated is better: both full-Q
scores might be worse. Read it alongside the primary effect and all four means.
Intervals condition on these checkpoints and reviewed plans; they do not account
for training seeds, plan-author choices, or historical checkpoint selection.

## 5. Independent role/reasoning audit

Use development responses to calibrate the rubric. For the reserved comparison,
select random questions and retain every checkpoint/context, irrespective of
correctness, gates or parser status:

```bash
python3 -B scripts/eac3_worker_study.py audit \
  --review results/fixed-prefix-JOB_ID/comparison/review.jsonl \
  --questions 50 --seed 20261101 \
  --output experiments/worker-specialization-v1/audit
```

Use the actual job ID. If fewer than 50 questions exist (e.g. development), choose
an explicit smaller number. Outputs are `examples.jsonl`, blank `labels.jsonl`
and `PRIVATE-key.json`. Annotators receive only the first two. Model identity,
automatic correctness and gates are withheld; context cannot be hidden because
the actual input is needed to judge compliance. These are local-target labels,
not the older whole-trajectory labeler's schema.

Assign `yes`, `no`, `unknown`, or `not_applicable` to the blank fields:

| Field | Question |
|---|---|
| assigned_task_followed | Does the response perform the assigned task, rather than substitute another one? |
| local_math_correct | Is the intended local result correct, including when formatting failed? |
| reasoning_sound | Are the substantive mathematical steps valid? |
| whole_question_solving | Does the worker go beyond its scope to solve Q? |
| previous_results_used | Does the visible response actually use relevant prior results? |

`reasoning_sound` can be unknown for an answer-only response; `previous_results_used`
is not applicable when there is no dependency. Whole-question solving is not an
overreach violation for the terminal. Forwarding is valid if assigned, not evidence
of new calculation. Correct results with flawed reasoning stay distinguishable.
Visible citation of prior results is not proof of causal dependence.

Use separate labels from two independent reviewers on at least a random overlap,
report agreement and disagreements, and adjudicate without model identities.
Automated/model annotations alone remain provisional. The helper exports the
audit but does not validate or aggregate filled labels automatically.

## What would constitute stronger evidence?

- Higher local correctness on paired answer-free compute tasks supports improved
  local execution, rather than gains caused solely by better plans or leakage.
- Better blinded role-compliance judgements support specialization beyond the
  same gate used for training. Report unknown judgements and confidence intervals.
- Neither finding alone proves cooperation. A later, separately reviewed
  dependency intervention must test reliance and appropriate repair. With full Q,
  correcting an erroneous prefix may be good reasoning, not noncooperation.
- To attribute improvements to EAC3, retrain a gated/ungated pair with the same W0,
  frozen D, corrected code, task order and declared budget. Repeat with independent
  training seeds; report generated tokens, updates and wall time. Gate filtering
  changes update frequency, so equal-update and equal-compute conclusions differ.
- Only after that local comparison, use end-to-end fixed-plan tests and alternating
  training to measure downstream system impact. Do not conflate these claims.

This workflow prepares the historical checkpoint comparison. It does not launch
training, manufacture reference correctness, guarantee a positive result, or turn
the existing pilot into a statistical proof.

## CPU checks

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s src/verl/tests/rema_separated_trainer -p 'test_worker_study.py' -v
```

These use synthetic JSONL fixtures and mocked inference, plus actual production
prompt construction. GPU inference and the real Math-Verify installation still
need the cluster environment. Synthetic fixture review metadata is test data,
not certified research annotation.

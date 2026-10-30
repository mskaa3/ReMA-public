# Fixed-prefix worker evaluation

This inference-only pilot asks: **given the same assigned task and correct prior
results, do the trained workers produce better local answers than their common
initialization?** It neither trains nor changes the gates, rewards or prompts.

## What runs

The default configuration, `config/eac3-fixed-prefix-pilot.json`, compares:

| ID | Worker checkpoint | Interpretation |
|---|---|---|
| w0 | 6055460 / step 50 | Shared initialization; already trained, not an untrained base model |
| ungated | 6106013 / step 63 | First subsequent ungated worker training |
| gated | 6106014 / step 65 | First subsequent gated worker training |

These paths come from the existing round-two configuration. Their availability
on S3 must still be checked on the cluster. This is **not** the second worker
round from the earlier A/B/C/D audit. Checkpoints also have different training
budgets: a difference here is not by itself an equal-compute causal estimate of
the effect of gating.

For each example, the evaluator:

1. Supplies a curated fixed plan to the production sequential router.
2. Supplies the checked prefix results from the dataset instead of generating
   earlier workers. There is no teacher or decomposer model invocation.
3. Captures the exact current production messages at the target worker and stops
   the router before generation. Only the target response is sampled, greedily.
4. Repeats that same input for each checkpoint in a separate process.
5. Extracts `LOCAL_RESULT` using the production communication boundary and checks
   it against the **subtask reference**, with process-isolated Math-Verify.

Nonterminal workers see Q, their task, and earlier accepted LOCAL_RESULTs.
Terminal workers see their task and earlier LOCAL_RESULTs, but no separate Q
block or full plan. Reference answers/check notes are never appended to a prompt.
The prepared prompts are current-code reconstructions, not a claim to reproduce
historical prompts byte-for-byte.

Unlike `replay-workers`, this does **not** regenerate the earlier workers. A bad
upstream calculation therefore cannot confound the target worker comparison.
It also does not measure robustness to bad upstream results or end-to-end ACC.

## Launch

On the cluster, after these files are available in your repository:

```bash
cd ~/projects/ReMA-public
python3 -B scripts/eac3_fixed_prefix_eval.py check \
  --dataset experiments/fixed-prefix-pilot.json
sbatch scripts/eac3_fixed_prefix_eval.sh
```

One node and **one Hopper GPU**, not four or six nodes. The three 1.5B models load
sequentially. The launcher uses the existing SIF remote, stages model exports on
current-job scratch, and stores small results on shared storage under
`results/fixed-prefix-<job_id>/`. It deliberately ignores inherited `JOB_TMP`.
The two-hour request is a limit, not a measured runtime estimate.

Optional S3 backup (after each complete model and again after the report):

```bash
export PREFIX_EVAL_REMOTE="s3v2:s3min-tomasznaskret-1712063354/user/ajanz/eac3-experiments/fixed-prefix-pilot"
sbatch scripts/eac3_fixed_prefix_eval.sh
```

Exports go to `$PREFIX_EVAL_REMOTE/job-<job_id>/`. Optional `PREFIX_EVAL_SIF`
points to an existing container file; `PREFIX_EVAL_OUTPUT` overrides the shared
output directory, which must not already exist. An alternative config can be
passed as the first script argument. Submit from the repository root.

Inference is greedy, n=1, seed=0, up to 1024 new tokens and 8192 total tokens.
There is no silent prompt truncation. The report rejects mismatched prompt token
hashes, sampling settings, code/verifier versions, duplicate IDs or missing rows.
GPU numerical nondeterminism is still possible despite greedy decoding.
Preparation also checks that the verifier can parse and compare each reference
with itself before loading models. This is a notation/dependency check, not an
independent proof that the supplied mathematical reference is right.

## Outputs and interpretation

`prepared.json` stores every fixed input, references, provenance, messages and
source fingerprints. Each model directory contains a manifest, raw responses
and scored records. Interrupted generation leaves `.partial`, which cannot be
mistaken for a complete run.

`comparison/report.md` and `report.json` show accepted local correctness,
decidable correctness, accepted-result rate, length stops, token usage and paired
gains/losses. There are breakdowns for compute/forward/synthesize tasks and for
terminal/nonterminal stages. No eligibility gate is applied.

Statuses mean:

- `correct`: accepted local result is symbolically equivalent to a reference.
- `wrong`: accepted, symbolically decidable result is not equivalent.
- `unknown`: result accepted by routing, but comparison failed to parse or timed out.
- `no_accepted_result`: the communication boundary rejected the response.

The fraction `correct / all examples` measures **confirmed successful delivery**,
not intrinsic reasoning accuracy. Unknown and missing results are not evidence
of incorrect mathematics. The conditional accuracy over decidable cases must be
read alongside coverage, never alone. Equivalent-answer checking inherits the
limits of Math-Verify; inspect disputed results manually.

`comparison/review.jsonl` places all model outputs beside each fixed input.
Manual role-compliance and reasoning-correctness fields are intentionally blank.
Use them to distinguish calculation, valid forwarding, solving all of Q,
incorrect reasoning with a correct result, and ignoring dependencies.
The optional full-question-answer-match diagnostic is **not a leakage label**:
`digit-s1` deliberately has the same correct local and final numeric answer.

## Pilot limits and next dataset

The 18 rows cover eight small questions/families from known failure patterns,
with rewritten, answer-free instructions and checked prefixes. They are an
assistant-curated smoke/diagnostic test, not a random benchmark or validated human
annotation set. No confidence intervals or publication-level frequency claims
are justified from this hand-picked, correlated sample.

For the actual experiment, select questions and target stages randomly before
viewing model outputs, freeze that selection, independently check the local
references and prefixes, and include difficult tasks and all outcome classes.
Preserve the schema in `experiments/fixed-prefix-pilot.json`; each row needs a
question, plan of 1-4 tasks, 1-based target, exactly target-1 checked prior results,
local reference alternatives, final-answer diagnostic and a reference rationale.
Do not import gold final answers into prior results unless the preceding task
legitimately asks for them. Document exclusions and review provenance explicitly.

Correct-prefix local competence is only one axis. After this pilot, keep the
existing fixed-plan end-to-end evaluation and manual audit for communication,
error repair and cooperation. Strong local scores alone do not prove causal use
of earlier workers, reduced leakage, or gains on harder tasks.

## CPU checks

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s src/verl/tests/rema_separated_trainer -p 'test_fixed_prefix_eval.py' -v
```

The test harness executes production rollout definitions with canned outputs,
without Ray, model loading or GPU access. Actual inference requires the cluster
container's vLLM, Transformers, Math-Verify and Pebble installations.

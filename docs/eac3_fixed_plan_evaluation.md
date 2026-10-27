# Round-2 workers on identical frozen plans

This is evaluation only. It compares worker changes within each training chain
and transfer to the other chain's planner. It does not retrain any model.

## Models and inputs

Configuration: `config/eac3-round2-fixed-plans.json`.

| Label | Worker export | Role in comparison |
| --- | --- | --- |
| `u_before` | 6106013 / global_step_63 | Ungated chain before this round |
| `u_after` | 6154441 / global_step_45 | Ungated chain after 30 updates |
| `g_before` | 6106014 / global_step_65 | Gated chain before this round |
| `g_after` | 6154442 / global_step_45 | Gated chain after 30 updates |

`du` and `dg` are the two round-2 frozen decomposers. **Both decomposers were
trained with eligibility**; the labels refer to their worker partners' histories.
Check the six S3 exports in the config before submitting. Paths are assumptions
about your completed exports, not a remote-existence guarantee. Do not replace
these exports in place. If a path or experiment changes, use a new study name.

Every condition uses the full held-out `data/overall_math/test.parquet`.
No teacher solutions are given to the decomposer. Nonterminal workers see Q,
their assigned subtask and earlier extracted results. The terminal worker sees
its subtask and earlier results, without separately appended Q. Workers always
generate their own results; only the decomposer output is replayed.

There are two cache-producing runs followed by eight replay runs:

| Saved plans | u_before | u_after | g_before | g_after |
| --- | --- | --- | --- | --- |
| du | eval | eval | eval | eval |
| dg | eval | eval | eval | eval |

All runs use `val_only=true`, greedy `val_kwargs.n=1`, one round, and disabled
resume. The launcher's unused training settings still contain 32 / 4 to satisfy
existing C3 configuration validation. **No 8x4 expansion or optimizer update
occurs in validation.** Four nodes with four GPUs each are configured, using
the existing two-model launcher. Eight jobs may be queued together or submitted
one by one from `commands.txt`; they do not need to run concurrently.

## 1. Generate and save plans once

Run from the updated repository on the cluster. Manifest creation needs ordinary
Python; cache building and reporting additionally need pandas and pyarrow
(available in the training container). Do not change code/prompts between phases.

```bash
cd ~/projects/ReMA-public
python3 scripts/eac3_experiments.py plan \
  --config config/eac3-round2-fixed-plans.json \
  --phase cache-plans \
  --output-dir experiments/runs/fixed-plans-cache
bash experiments/runs/fixed-plans-cache/commands.txt
```

These two runs reuse standard full validation, so `u_before` also executes the
plans. That extra reference rollout is a compatibility cost, not plan selection:
its answer, success and probes are **not** used to retain/reject any plan.
Each decomposer generates only one plan per question, independently of workers.

After both jobs finish:

```bash
bash experiments/runs/fixed-plans-cache/fetch-results.sh

python3 scripts/eac3_fixed_plan_evaluation.py cache \
  --config config/eac3-round2-fixed-plans.json \
  --source du experiments/runs/fixed-plans-cache/results/eac3-round2-fixed-plans-v1--cache-plans-du-s42/val-step*-update0.jsonl \
  --source dg experiments/runs/fixed-plans-cache/results/eac3-round2-fixed-plans-v1--cache-plans-dg-s42/val-step*-update0.jsonl \
  --output-dir experiments/runs/fixed-plans-bank

bash experiments/runs/fixed-plans-bank/upload.sh
```

Each source must provide exactly one complete JSONL, its metadata, and `run.json`.
Missing questions, partial files, wrong checkpoints and differing code fingerprints
are rejected. Empty, malformed, single-subtask and answer-supplying plans remain.
An empty plan is replayed verbatim and follows the normal router's fallback; it
is never silently regenerated or removed. No intermediate/final correctness is
used to construct the cache. `answer_free_reviewed=false` is explicit.

The builder produces `du.parquet`, `dg.parquet`, provenance JSONs, `upload.sh`, and
`replay-config.json`. The latter pins the bank checksums. Caches are uploaded to:

```text
<remote_root>/eac3-round2-fixed-plans-v1/plan-cache/{du,dg}.parquet
```

## 2. Replay four worker checkpoints on both banks

```bash
python3 scripts/eac3_experiments.py plan \
  --config experiments/runs/fixed-plans-bank/replay-config.json \
  --phase replay-workers \
  --output-dir experiments/runs/fixed-plans-replay
bash experiments/runs/fixed-plans-replay/commands.txt
```

The planner model loaded by every replay job is frozen `du`, solely for diagnostic
probes. **It does not generate the plan**, including on `dg`-plan runs. The full
cached text is routed through the existing prompt/parser implementation.

Gates are diagnostic, not applied to training or example selection. Standard
`val/acc/*`, console examples and `val_diagnostics/leakage/*` are retained. Leakage uses the
same `du` probe across all eight runs, on up to 128 items selected by the existing
console-example plus stratified sampling procedure, not the whole benchmark. Keep unknown
probe outcomes separate; do not interpret gated accuracy as verified cooperation.
No external auditor is added.

Full trajectories, raw scores, question/plan IDs, worker tokens, input fingerprints,
resolved config and aggregate diagnostics are stored at:

```text
<remote_root>/<generated run name>/evaluations/
```

Each job validates the downloaded bank checksum before initializing training
workers. Changed caches fail rather than silently comparing different inputs.

## 3. Download and summarize

After all eight jobs complete:

```bash
bash experiments/runs/fixed-plans-replay/fetch-results.sh
python3 scripts/eac3_fixed_plan_evaluation.py report \
  --config experiments/runs/fixed-plans-bank/replay-config.json \
  --results-root experiments/runs/fixed-plans-replay/results \
  --output-dir experiments/runs/fixed-plans-report
```

`summary.csv` contains full-benchmark and per-dataset rows for each plan source:
before/after accuracy, paired deltas, question-bootstrap intervals, worker token
counts, executed workers, result parse rates and length-stop rates. `report.json`
also retains paired comparison details, diagnostic leakage metrics and checksums.
The report reads current `val_diagnostics/leakage/*` metrics and legacy
`val/leakage/*` metrics, preserving their recorded keys and values. Teacher-assisted
diagnostics are not included in this question-only comparison.
All eight results must cover identical question/plan pairs within each source.

Interpret `u_after - u_before` and `g_after - g_before` under each fixed plan bank.
Better performance on both banks supports worker improvement and partner transfer.
Do not call the difference between chains a clean causal gate effect: their
starting workers and preceding training histories already differ. Bootstrap
intervals quantify evaluation-question uncertainty, not training-seed variability.

## 4. Optional independent plan and trajectory review

Use the same cached-source JSONLs to prepare a plan-only review, blinded to worker
outcomes and planner source:

```bash
python3 scripts/analyze_eac3_experiments.py plan-review \
  --source du experiments/runs/fixed-plans-cache/results/eac3-round2-fixed-plans-v1--cache-plans-du-s42/val-step*-update0.jsonl \
  --source dg experiments/runs/fixed-plans-cache/results/eac3-round2-fixed-plans-v1--cache-plans-dg-s42/val-step*-update0.jsonl \
  --output experiments/runs/plan-review.csv
```

Review every plan with `yes/no/unknown`, including rejections, and a reviewer ID.
Only `executable=yes` and `supplies_computed_answers=no` enter the selected subset.
After annotation, repeat `report` with `--labels experiments/runs/plan-review.csv`
and a **new output directory**. This adds reviewed-subset rows while retaining full
results. It requires no new model inference and never rewrites saved trajectories.

Raw accuracy and parse success do not establish correct intermediate reasoning or
assignment compliance. For those claims use the existing `audit-export` /
`audit-summary` commands on paired before/after JSONLs, sampling questions without
selecting successes. The audit includes assigned subtasks, worker reasoning and
the final answer. Report unknowns and sample size; do not treat format validity as
subtask correctness.

For example, a blinded before/after worker audit under `du` plans:

```bash
python3 scripts/analyze_eac3_experiments.py audit-export \
  --left experiments/runs/fixed-plans-replay/results/eac3-round2-fixed-plans-v1--replay-workers-u_before-under-du-s42/val-step*-update0.jsonl \
  --right experiments/runs/fixed-plans-replay/results/eac3-round2-fixed-plans-v1--replay-workers-u_after-under-du-s42/val-step*-update0.jsonl \
  --questions 100 --seed 42 \
  --output-dir experiments/runs/fixed-plans-worker-audit
```

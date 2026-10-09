# Agent 1 training after Agent 2

Submit `agent1-only-rl-trainer-multinode.sh` after the Agent-2-only run has
successfully exported its merged Hugging Face worker model. This wrapper reuses
`agent12-curriculum-trainer-multinode.sh`; it does not introduce another trainer.

## Launch

From the repository root on the cluster, replace `<AGENT2_JOB_ID>` with the job
that produced the worker model:

```bash
WORKER_MODEL_REMOTE="s3v2:s3min-tomasznaskret-1712063354/user/ajanz/agent12/agent2-only-<AGENT2_JOB_ID>/workers/huggingface" \
  sbatch agent1-only-rl-trainer-multinode.sh
```

`WORKER_MODEL_REMOTE` is required. Use the merged `workers/huggingface` directory,
not a raw per-node FSDP checkpoint. The launcher downloads it to
`$JOB_TMP/worker_model` on every allocated node and uses that path in both the
host and container. It checks the model config, tokenizer files, and all weight
files referenced by a sharded index before starting Ray. A missing, empty, or
incomplete export stops the job; there is no fallback to the base Qwen model.
Preflight verifies file presence and index completeness, not numerical weight
integrity or whether the model fits GPU memory. Existing staging directories
are rejected; resubmit with a fresh job-local `JOB_TMP` rather than mixing exports.

The default decomposer is `Qwen/Qwen2.5-7B-Instruct`. Set `DECOMPOSER_MODEL_PATH`
to override it. Resource allocation remains 6 nodes with 4 GPUs per node, split
between the decomposer and worker pools as in Agent-2-only.

## Inputs and optimization

| Component | Input | Output | Optimized? |
| --- | --- | --- | --- |
| Agent 0 | No online call; cached attempts are reused | Teacher reasoning attempts | No |
| Agent 1 | Question and, probabilistically, one cached teacher attempt | Reasoning and a sequential plan | Yes |
| Non-terminal Agent 2 roles | Question, assigned task, prior LOCAL_RESULTs | Reasoning and boxed LOCAL_RESULT | No |
| Terminal Agent 2 role | Assigned task and prior LOCAL_RESULTs; no separate question | Boxed system answer | No |

`train_agent_roles=[decomposer]` enforces the optimization boundary. All worker
roles still execute and their shared model is saved, but no worker optimizer
step is selected, including after the teacher fade ends. The implementation
still constructs the existing worker actor/rollout objects; this is an
optimization freeze, not an inference-only memory optimization.

The wrapper preserves four available worker stages, up to four planned tasks,
deterministic routing, one round, and the last planned worker as the answer.
C3 samples 8 decomposer outputs from each shared input, each followed by four
independent worker continuations (32 trajectory slots). Credit uses mean final
correctness with the existing LOO normalization and gates. Decomposer credit
remains signed; the positive-only rule applies to non-terminal workers, not D.

No rewards or probe rules are changed. The terminal-instruction probe gates
decomposer updates; whole-plan L_D remains diagnostic. Worker/final answer
comparisons are still reported during validation, but training comparisons are
role-local and do not gate D based on every worker. The probe solver remains
Agent 1, whose weights now change during training; probe rates therefore reflect
both plan changes and changes in that solver, not a fixed external evaluator.

## Curriculum and data

- `WORKER_BOOTSTRAP_STEPS=0`: do not retrain workers.
- `DECOMPOSER_TRANSFER_STEPS=200`: teacher-attempt visibility decreases from 1
  toward 0 over completed optimizer updates, not skipped rollout attempts.
- `TOTAL_STEPS=800`: the upper limit on rollout attempts in the new run.
- After 200 completed updates, planning is question-only and only D continues
  learning. The shared scheduler labels this phase `joint`, but the trainable
  role list still contains only D. Workers never become trainable.
- Non-terminal worker question visibility remains 1 throughout and in validation.
- Validation uses question-only planning from the start, with the existing
  `val/acc/*` and leakage metrics. It does not receive teacher solutions.

Set `DECOMPOSER_TRANSFER_STEPS` and `TOTAL_STEPS` before `sbatch` to override
their defaults. If many batches are skipped, a run can reach its rollout limit
before completing the fade. Check `curriculum/teacher_attempt_probability` and
`train/actor_update_step`; the launcher does not claim that 800 attempts imply
800 successful updates.

The wrapper disables teacher generation and reuses the full cached dataset at
the common launcher's `TEACHER_TRAIN_REMOTE`. All 16 attempts per question,
correct or incorrect, are retained. To use another complete cache or one shard,
also set `TEACHER_TRAIN_REMOTE`. Missing cached data fails explicitly rather than
silently invoking Agent 0. Teacher visibility is shared within each C3 group.

## Logging and exports

The default W&B/run name is `agent1-only-<SLURM_JOB_ID>`. Existing accuracy,
leakage, C3 action-mean, and curriculum metrics remain enabled. Check that
`train/roles/decomposer/actor_update_count` is the only increasing role counter.

The default export destination is:

```text
s3v2:s3min-tomasznaskret-1712063354/user/ajanz/agent12/agent1-only-<SLURM_JOB_ID>/decomposer/huggingface/
```

The frozen worker model is also exported under `workers/huggingface` in the same
run, and `metadata.txt` records its source remote. Local saves occur every 50
completed optimizer updates and at session boundaries. S3 model export remains
end-of-run only; this launcher does not add periodic or crash-time backup.
An HF export initializes a later stage's weights; it is not a full optimizer
resume. No Slurm training or S3 access is performed by the local CPU tests.

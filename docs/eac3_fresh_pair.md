# Fresh, paired worker-to-decomposer training

This repeats the teacher-assisted curriculum, not the question-only adaptation
experiments. Both arms start from pretrained Instruct weights, not random weights
and not previous project RL checkpoints. No Agent 0 training or new teacher
generation is performed. Both arms reuse the cached Phi-4-mini-reasoning attempts,
including incorrect attempts.

| Stage | Gated branch | Ungated branch | Shared controls |
| --- | --- | --- | --- |
| Workers | Train fresh Qwen2.5-1.5B with eligibility | Train fresh Qwen2.5-1.5B without eligibility | Fresh frozen Qwen2.5-7B planner, cached teacher attempts, function-balanced C3 |
| Decomposer | Train fresh Qwen2.5-7B with eligibility; freeze gated workers | Train fresh Qwen2.5-7B without eligibility; freeze ungated workers | Same transfer schedule, original base planner, same update budget |

Both arms compute leakage diagnostics. The treatment changes
`prefix_probe.apply_to_training` and `require_eligible_success`, not the presence
of the probes. C3 uses 8 actions x 4 continuations in both branches. Positive-only
nonterminal updates remain enabled in both. Function balancing alternates bounded
nonterminal/terminal optimizer attempts, not guaranteed successful updates.

Nonterminal workers see the original question. The terminal worker is still
question-blocked by the existing protocol; the paired launcher does not change
prompts or routed inputs. The selector remains deterministic.

## 1. Prepare and submit workers

Run on the cluster from the repository root, after updating the code. Edit
`config/eac3-fresh-pair.json` **before preparing either branch** if necessary.
Defaults: 4 nodes x 4 GPUs per job, 100 completed updates, 1200 rollout-attempt cap,
evaluation before training and every 10 completed updates, local checkpoint every
10 completed updates. The configured partition/time are submission requests, not
a guarantee of availability or permitted wall time on your cluster.

```bash
python3 -B scripts/eac3_fresh_pair.py prepare \
  --stage workers \
  --output-dir experiments/fresh-pair-v1/workers

bash experiments/fresh-pair-v1/workers/commands.sh
```

The second command submits two independent jobs, gated first and ungated second;
it prints their Slurm IDs. They can run concurrently when resources permit.
Together they request 8 nodes / 32 GPUs. Each job splits its allocation into two
equal model pools. There are 3 questions x 16 teacher attempts x 32 trajectory
slots = 1536 slots per generated batch, before filtering/resampling.

Preparing does not submit anything. Each manifest includes code/prompt hashes;
editing training code after preparation requires regenerating both manifests in
a new directory. Every submitted job gets a distinct W&B/S3 name containing its
Slurm ID. Previous model, scratch, resume, probe and experiment settings in the
shell or `env.sh` do not override the generated experiment settings. Credentials
and a site-specific SIF path can still come from `env.sh`.

## 2. Check completion, then prepare decomposers

Do not start this stage just because a checkpoint exists. First verify that both
worker jobs reached the requested `actor_update_step=100`. A job reaching the
1200-attempt cap first is a shorter training run, not a matched-update result.
The wrapper reports a nonzero exit on a completed run whose update budget was
not reached, even if the common launcher already exported its model.

After successful model export, the wrapper also uploads `evaluations/run.json`,
`training_progress.json`, `teacher-shards.sha256`, `eval-data.sha256`, validation
trajectories and training metrics. Check the update counts and compare the shard
and evaluation-file hashes across the branches. Cached teacher coverage may be
partial; the existing loader uses all available complete shards, not necessarily
all 8523 original questions. Do not alter that cache during this experiment.

```bash
export ROOT_REMOTE=s3v2:s3min-tomasznaskret-1712063354/user/ajanz/eac3-experiments

# Enter the IDs printed when submitting the two worker jobs.
read -r -p 'Gated worker job ID: ' GATED_JOB
read -r -p 'Ungated worker job ID: ' UNGATED_JOB

export GATED_WORKERS="$ROOT_REMOTE/eac3-fresh-pair-v1-workers-gated-s11-${GATED_JOB}/workers/huggingface"
export UNGATED_WORKERS="$ROOT_REMOTE/eac3-fresh-pair-v1-workers-ungated-s11-${UNGATED_JOB}/workers/huggingface"

rclone cat "${GATED_WORKERS%/workers/huggingface}/evaluations/training_progress.json"
rclone cat "${UNGATED_WORKERS%/workers/huggingface}/evaluations/training_progress.json"

python3 -B scripts/eac3_fresh_pair.py prepare \
  --stage decomposers \
  --gated-workers "$GATED_WORKERS" \
  --ungated-workers "$UNGATED_WORKERS" \
  --output-dir experiments/fresh-pair-v1/decomposers

bash experiments/fresh-pair-v1/decomposers/commands.sh
```

Adapt the path prefixes if you change `study`, `remote_root`, or `seed` in the
config. Missing/invalid worker exports fail model staging rather than falling
back to base weights. The two exports must be distinct; the launcher cannot infer
from arbitrary paths whether you accidentally swapped their experimental labels.

The decomposer starts from the original pretrained 7B model: its copy stayed
frozen throughout worker training. Each branch retains its gate setting. Teacher
attempt availability falls from 1 to 0 over the first 50 completed decomposer
updates by default. It is then trained from question-only input for the remaining
50 updates. Workers remain frozen throughout. Validation is question-only for the
decomposer from the beginning, including during teacher-assisted worker training.

## 3. Saved outputs and interpretation

Local checkpoints on the allocated nodes:

```text
/mnt/lscratch/slurm/<job_id>/agent12/checkpoints/agent12/global_step_<rollout_step>/
```

At successful completion the common launcher gathers and merges the distributed
weights and exports both `workers/huggingface` and `decomposer/huggingface` under:

```text
<remote_root>/<study>-<workers|decomposers>-<gated|ungated>-s<seed>-<job_id>/
```

**Saving every 10 updates is local, not an automatic S3 backup every 10 updates.**
If a job dies, use the existing checkpoint recovery/export tools. Do not assume
the final HF export or evaluation upload completed. Use the progress JSON, not
the `global_step_*` directory number, to determine the successful update budget.

Compare at equal completed optimizer updates and also report trajectories,
tokens, wall time, skipped updates, and terminal/nonterminal exposure. Eligibility
can require more resampling. Identical seeds do not guarantee identical plans
after the branches consume different amounts of randomness. This is a controlled
training comparison, not fixed-plan paired evaluation.

During worker training the loaded decomposer is a common frozen probe. During
decomposer training the probe is the branch's changing decomposer. Its leakage
metrics are useful diagnostics but are **not a shared independent judge** across
branches; use a common frozen audit/fixed-prefix evaluation for final claims.
Stage two measures the full alternating pipelines, not an isolated effect of
worker training, since both workers and decomposer objectives differ by then.

One seed is a pilot/comparison, not sufficient evidence of statistical robustness.
Change `seed` and use new output directories for later replications. Keep the same
base model/cache artifacts and code revision across both arms of each pair.

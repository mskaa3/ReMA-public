# Second alternating round: adapt planners, then workers

This extends the existing experiment launcher, not the teacher curriculum.
Both chains start from the same EAC3 Qwen-7B planner export at checkpoint 110.
They use different Qwen-1.5B worker exports, matched at 40 completed updates:

| Chain | Source job | Worker checkpoint directory | Configuration |
| --- | --- | --- | --- |
| Ungated | 6106013 | global_step_63 | config/eac3-round2-ungated.json |
| Gated | 6106014 | global_step_65 | config/eac3-round2-gated.json |

The logs contain save messages, not proof that all archived shards are complete.
The configuration paths assume the recovery procedure below has succeeded.
They do not mean these exports have already been uploaded.

## 1. Recover and upload starting workers

If the two exports already exist, skip recovery and set `initial_worker` in
each JSON configuration to its actual merged HF directory. Do not use a raw
actor directory or the archived `worker_model` input cache.

Otherwise, inside a new interactive allocation with access to the archives:

```bash
cd ~/projects/ReMA-public
bash scripts/recover_archived_model.sh --list 6106013 6106014
bash scripts/recover_archived_model.sh 6106013 6106014
```

Select step **63**, role **worker_stage_4** for job 6106013, then step **65**,
role **worker_stage_4** for job 6106014. The recovery script gathers all ranks,
merges on CPU, validates the HF files, and uploads to:

```text
s3v2:s3min-tomasznaskret-1712063354/user/ajanz/agent12/recovered-6106013/snapshots/global_step_63/workers/huggingface
s3v2:s3min-tomasznaskret-1712063354/user/ajanz/agent12/recovered-6106014/snapshots/global_step_65/workers/huggingface
```

No new decomposer export is needed at this point; both configs reuse the
existing `agent1-eac3-qwen7-20261013-144231/.../global_step_110` HF export.

## 2. Train the two decomposers

The new code must be present in the cluster checkout first. Generate manifests
after updating the code; they fingerprint source and data. From the repository
root on the login node:

```bash
(
  set -euo pipefail
  cd ~/projects/ReMA-public
  RUN_DIR="experiments/runs/eac3-round2-planners-$(date +%Y%m%d-%H%M%S)"

  # Lightweight remote-path check. The launcher validates full HF files later.
  python3 - <<'PY' | while IFS= read -r remote; do
import json
paths = set()
for chain in ('ungated', 'gated'):
    with open(f'config/eac3-round2-{chain}.json') as stream:
        cfg = json.load(stream)
    paths.add(cfg['initial_worker'])
    paths.add(cfg['planners']['eac3'])
print('\n'.join(sorted(paths)))
PY
    rclone cat "$remote/config.json" >/dev/null
  done

  for chain in ungated gated; do
    python3 scripts/eac3_experiments.py plan \
      --config "config/eac3-round2-${chain}.json" \
      --phase adapt-planners --output-dir "$RUN_DIR/$chain"
  done
  bash "$RUN_DIR/ungated/commands.txt"
  bash "$RUN_DIR/gated/commands.txt"
)
```

Each run trains **only the decomposer**, with its corresponding workers frozen.
Both decomposers use EAC3 training gates, regardless of the chain name. Planning
is question-only: curriculum disabled, no teacher traces or teacher generation.
Existing non-terminal full-question context, terminal task/results context,
one-round sequential protocol, and C3 8 actions x 4 continuations stay unchanged.

Both configurations use 40 completed actor updates per phase, with a safety cap
of 1200 rollout steps. Evaluation is at initialization, updates 1, 2, 5, then
every 10 updates; local checkpoints save every 10 updates before evaluation.
Each job uses 4 nodes x 4 GPUs on `lem-gpu-short`; both together request 32 GPUs.
These are weight-initialized new stages, not optimizer/dataloader resumes.

Successful end-of-job exports go under:

```text
s3v2:s3min-tomasznaskret-1712063354/user/ajanz/eac3-experiments/
  eac3-round2-ungated-v1--adapt-planners-eac3-s11/decomposer/huggingface
  eac3-round2-gated-v1--adapt-planners-eac3-s11/decomposer/huggingface
```

Periodic saves remain node-local, not periodic S3 uploads. If a job fails,
recover a complete intermediate checkpoint before continuing. Do not assume a
failed job produced its expected end-of-job export or reached 40 updates.

## 3. Continue the two worker models

Only after **both adapted planner exports exist** and their completed-update
counts are verified, generate and submit worker continuations:

```bash
(
  set -euo pipefail
  cd ~/projects/ReMA-public
  RUN_DIR="experiments/runs/eac3-round2-workers-$(date +%Y%m%d-%H%M%S)"
  ROOT="s3v2:s3min-tomasznaskret-1712063354/user/ajanz/eac3-experiments"
  for chain in ungated gated; do
    rclone cat "$ROOT/eac3-round2-${chain}-v1--adapt-planners-eac3-s11/decomposer/huggingface/config.json" >/dev/null
  done
  for chain in ungated gated; do
    python3 scripts/eac3_experiments.py plan \
      --config "config/eac3-round2-${chain}.json" \
      --phase adapt-workers --output-dir "$RUN_DIR/$chain"
  done
  bash "$RUN_DIR/ungated/commands.txt"
  bash "$RUN_DIR/gated/commands.txt"
)
```

Each run starts its workers from the same recovered checkpoint used in phase 2,
not the base model. Its newly adapted planner is frozen. Ungated workers retain
diagnostic probes without applying their gates; gated workers apply them.
Both keep the existing positive-only non-terminal rule and C3 validity checks.
Their final HF exports are under the same root, with names:

```text
eac3-round2-ungated-v1--adapt-workers-eac3-s11/workers/huggingface
eac3-round2-gated-v1--adapt-workers-eac3-s11/workers/huggingface
```

The `adapt-workers` phase derives the adapted planner path from the study name
and seed. Do not change those between phases. Do not resubmit a completed study
into the same output prefix; choose a new study identity for a separate attempt.
The older `planners` phase remains evaluation-only; `workers` still uses the
original planner in the config. Use the explicit `adapt-*` phases for round 2.

## Interpretation

Evaluate before and after each phase. This compares two alternating histories,
not only the worker gate in isolation once the planners have diverged. Use fixed
held-out plans or cross-partner evaluation to test whether worker gains transfer.
Track actual sampling cost as well as completed updates. The training probe is
the decomposer, so it evolves during planner training; its gate rates alone are
not a fixed independent measure of improved collaboration.

The legacy `transfer` and `cross` phases resolve the original `workers` exports,
not these `adapt-workers` exports. Do not use them unchanged for round-2 transfer.

## Rerun workers with local frozen probes

Use this after the adapted planner exports already exist. It preserves the
round-2 input models, creates fresh output identities, and evaluates both arms
with the same pinned Qwen3-4B probe inside each job. No probe server is needed.
This recipe requests **4 nodes x 4 GPUs per run**: six GPUs per model pool,
one dedicated probe GPU, and three unused GPUs. It sets the training-pool
size to three node equivalents and adds one node of probe capacity. Actual
placement is managed by Ray; a particular host is not reserved exclusively
for the probe. See [resource and dependency details](validation_information_probes.md#local-inference-inside-the-training-job).

Set `PROBE_REVISION` to the same immutable HF commit for both runs. To resolve
the current revision once (internet access required):

```bash
export PROBE_REVISION="$(curl --fail --silent --show-error \
  https://huggingface.co/api/models/Qwen/Qwen3-4B-Instruct-2507 \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')"
echo "$PROBE_REVISION"
```

Retain that hash for future repeats rather than resolving a new revision.
Then, from the updated repository on the login node:

```bash
(
  set -euo pipefail
  cd ~/projects/ReMA-public
  : "${PROBE_REVISION:?Set the fixed probe revision first}"
  export RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
  export RUN_DIR="experiments/runs/round2-workers-local-probe-${RUN_STAMP}"

  python3 - <<'PY'
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import eac3_experiments as suite

for chain in ('ungated', 'gated'):
    cfg = suite.read_json(f'config/eac3-round2-{chain}.json')
    assert len(cfg['seeds']) == 1
    original = suite.make_runs(cfg, 'adapt-workers')[0]
    for remote in (original['planner'], original['worker']):
        print(f'Checking {remote}', flush=True)
        subprocess.run(['rclone', 'cat', remote.rstrip('/') + '/config.json'],
                       stdout=subprocess.DEVNULL, check=True)
    cfg['planners'] = {'eac3': original['planner']}
    cfg['study'] += '-local-probe-' + os.environ['RUN_STAMP']
    cfg['nodes'] = 3  # Two 6-GPU model pools; one extra node makes four total.
    cfg['local_validation_probe'] = {
        'revision': os.environ['PROBE_REVISION'],
        'extra_nodes': 1,
    }
    suite.write_runs(cfg, 'workers', Path(os.environ['RUN_DIR']) / chain)
PY

  bash "$RUN_DIR/ungated/commands.txt"
  bash "$RUN_DIR/gated/commands.txt"
)
```

This deliberately uses `workers` with the explicitly resolved adapted planner:
renaming a study and using `adapt-workers` directly would look for a nonexistent
new planner export. Starting weights, seeds, global batch size, C3 and gate
settings are unchanged. Each model pool shrinks from eight GPUs to six;
per-GPU memory usage and throughput can change, so smoke-test this allocation
before a long run. Both arms together request 32 GPUs. Set `cfg['nodes'] = 4`
instead to retain the earlier pool sizes and request five nodes per run.
Regenerate manifests when changing capacity; do not override only `sbatch --nodes`.
The original round-2 JSON configurations are left unchanged.
Standard checkpoint paths are assumed; replace them with actual
recovered HF paths if necessary. These remain two alternating histories, not a
gate-only ablation with identical starting partners. New probe scores are
evaluation diagnostics, not additions to either training reward.

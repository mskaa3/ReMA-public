# Hierarchical ReMA MVP

This module adds a small, self-contained MVP for the controller stack:

1. `Decomposer` samples `M` DAG decompositions.
2. `Selector` samples `N` worker assignments for each decomposition.
3. Worker agents execute the DAG in topological order.
4. Rewards are computed bottom up:
   - worker confidence proxy from output entropy
   - selector reward from final correctness + confidence + compatibility
   - decomposer reward as the mean selector reward over a decomposition branch, minus any soft hop penalty
5. Advantages are computed within selector groups first, then across decomposition groups.
6. Controller prompts contain a frozen snapshot of worker specs plus performance history:
   - EMA outcome prior
   - assignment count
   - completion rate
   - success rate
   - average reward
   - recent execution history

Two training schedules are supported:

- `joint`: `M` decompositions and `N` selections per decomposition, both controllers receive training samples.
- `alternating`: one phase trains only the selector with a frozen decomposer, the next phase trains only the decomposer with a frozen selector.

Backend modes:

- `MockHierarchicalBackend`: deterministic path for tests and workflow debugging.
- `TransformersHierarchicalBackend`: real controller and worker model calls through HuggingFace `transformers`, with structured-output retries and fallback JSON-safe rollouts.
- `RayVLLMHierarchicalBackend`: repo-native rollout generation through `RayWorkerGroup` + vLLM, while keeping the same hierarchical decomposition/selection/DAG control flow.

Current scope:

- Controller outputs are strict JSON strings stored in `raw_text`.
- Decomposer and selector outputs are parsed and validated against the expected schema.
- If a controller returns invalid JSON, the real backend retries with a repair prompt and falls back to a safe structured output if needed.
- Decompositions can be constrained by:
  - `soft_max_hops`: above this, the decomposer gets a penalty that grows with exceedance
  - `hard_max_hops`: above this, the DAG is truncated to a safe executable form
  - `max_nodes_per_decomposition`: existing hard cap on node count
- Worker roles are prompt-defined and non-trainable by default. Future worker-role LoRA training is left as an explicit placeholder, not an active training path.
- Worker history is frozen per task rollout group, then updated after the rollout finishes, which matches the intended GRPO grouping much better than updating inside the `M x N` tree.
- Best rollouts and full rollout traces can be saved locally as JSONL logs.

Run the demo:

```bash
PYTHONPATH=src/verl/verl python -m hierarchical_rema.demo \
  --backend mock \
  --mode joint \
  --num-decompositions 3 \
  --num-selections 2 \
  --soft-max-hops 3 \
  --hard-max-hops 5
```

Run with real models:

```bash
PYTHONPATH=src/verl/verl python -m hierarchical_rema.demo \
  --backend vllm \
  --decomposer-model-path /path/to/controller-model \
  --selector-model-path /path/to/controller-model \
  --worker-base-model-path /path/to/worker-model \
  --ray-n-gpus-per-node 4 \
  --rollout-prompt-length 2048
```

The demo now defaults to a compact rollout summary in stdout and can write JSONL rollout logs under `outputs/hierarchical_rema`.

Run integrated controller training:

```bash
PYTHONPATH=src/verl/verl python -m hierarchical_rema.train \
  --task-source data/overall_math/all_test_data.jsonl \
  --backend vllm \
  --mode joint \
  --num-epochs 2 \
  --model-path /path/to/controller-model \
  --output-dir outputs/hierarchical_rema_train/run_002
```

This path does the full outer loop inside one job:

1. load task examples
2. sample decompositions and selections
3. execute workers and compute rewards
4. derive controller samples and advantages
5. run GRPO-style controller updates
6. save checkpoints, summaries, and optional replay copies

Useful flags:

- `--rollout-task-batch-size`: how many tasks are rolled out together in one batched hierarchical pass
- `--controller-batch-size`: batched decomposer/selector generation size for HF or vLLM rollouts
- `--worker-batch-size`: batched worker generation size for HF or vLLM rollouts
- `--rollout-prompt-length`: prompt truncation length for the Ray/vLLM rollout engine
- `--ray-n-gpus-per-node`: how many GPUs the Ray rollout worker group should use per node
- `--vllm-tensor-parallel-size`: tensor parallelism for vLLM rollout workers
- `--tasks-per-epoch`: limit how many tasks are rolled out in one outer epoch
- `--epochs`: number of GRPO update epochs per outer epoch
- `--save-replay-copy`: save train/val/all controller samples per policy for inspection
- `--enable-wandb`: enable W&B logging
- `--disable-rollout-logging`: skip JSONL rollout logging if you only want training artifacts

The integrated trainer now supports batched hierarchical rollout collection with either HF generation or repo-native Ray/vLLM generation. The hierarchical controller logic stays in this separate folder, while the rollout engine underneath can now use the same `RayWorkerGroup` stack as the rest of the repo.

If you still want the old replay-buffer workflow, it is available separately:

```bash
PYTHONPATH=src/verl/verl python -m hierarchical_rema.replay_train \
  --input outputs/hierarchical_rema/5158272 \
  --model-path /path/to/controller-model \
  --output-dir outputs/hierarchical_rema_train/run_002
```

Training artifacts are written per policy id, so shared-controller runs will produce one policy folder and separate decomposer/selector runs will produce two. Worker-role LoRA optimization is not implemented yet; see `worker_training.py` for the placeholder.

## Optional Result-Only Failure Judge

Set `REMA_MAST_ENABLED=true` in `hierarchical-rema-train.sh` to audit the same
rollouts at every external validation event (routine and diagnostic). Default is
off, with no judge model loading. Validation accuracy cadence does not change.
The default frozen judge is `google/gemma-4-31B-it`; its Hugging Face revision is
resolved to an immutable commit and recorded with package versions and prompt hash.

This is an eight-category, artifact-level adaptation of
[MAST](https://arxiv.org/abs/2503.13657), not a reproduction of the full taxonomy:
task requirements (1.1), role violations (1.2), unnecessary repetition (1.3),
context loss (1.4), derailment (2.3), missing handoff (2.4), disregarded input (2.5),
and incomplete execution (3.1). The generic prompt and response schema live in
`failure_judge_contract.py` and are also saved with every evaluation.

The judge sees the task, role instructions, structured executed plan, assembled
dependency snapshots, delivered outputs and final answer. It does not receive
reference answers, correctness verdicts, rewards, private scratchpads, or rejected
raw outputs. Non-root workers are not assumed to see the full task. Context loss
means information was established but lost from an expected downstream input;
missing handoff means it was never delivered; disregarded input means it arrived
but was visibly ignored. Snapshots do not prove post-tokenization visibility.
Equal output strings alone do not establish redundant work.

Labels are 1 (observed failure), 0 (no observed failure with assessable artifacts),
or null (not applicable / insufficient evidence). Positive labels need exact
source excerpts and JSON Pointer paths; information-flow failures need two-sided
evidence. These are observational judgments, not ground-truth error labels.
They never influence rewards, advantages, or best-checkpoint selection.

W&B adds only `val/failures/<category>` rates. Rates average assessed executions
within each task, then weight tasks equally. Nulls and failed requests are excluded;
coverage and status counts are saved per category, overall and per dataset. No
valid judgments means an absent metric, not a zero failure rate.

Each validation directory contains `validation_failures.jsonl` (safe input plus
judgments/evidence/errors), `failure_judge_metrics.jsonl`,
`failure_judge_definition.json`, and `failure_judge.log`. Run-level
`failure_judge_history.csv` and `.jsonl` track numerical changes over evaluations.
These use the existing epoch/root S3 uploader. Successful content-identical audits
are cached against input, prompt, resolved model revision and inference identity;
failed requests are not cached. Re-evaluating an epoch replaces its history rows.

The judge runs after generation and IDS, in an evaluation-scoped subprocess on
the isolated reward GPU. GraphPRM's cached models are released first and lazily
reload on the next reward call. A supervised local coach uses explicit sleep/wake;
Gemma exits before it is restored. An unsupported sleep API skips the judge;
failure to restore a coach stops training rather than silently changing rewards.
CPU offload during coach sleep requires sufficient host RAM for its weights.

Requests are length-sorted and batched (`REMA_MAST_BATCH_SIZE=4` by default).
The default context is 16,384 tokens, with 3,072 reserved for JSON output. Inputs
that do not fit are recorded as unavailable, not truncated. Batch errors are
isolated by splitting; only failed annotations are retried once. A 7,200-second
evaluation timeout retains completed judgments and releases the subprocess.
Batch size, context, output budget, GPU utilization, retries and timeout are
configurable through the corresponding `REMA_MAST_*` variables in the launcher.

Gemma 4 requires compatible vLLM/Transformers and authorized HF access. The policy
container's pinned old vLLM is not assumed compatible. Nothing is installed or
upgraded automatically: set `REMA_MAST_PYTHON` to a Gemma-compatible interpreter
visible inside the container (or use an updated compatible container). A startup
failure is recorded as unavailable, while accuracy/IDS and training continue.
See the [official serving guide](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html).
Real GPU memory and coach handoff behavior must be smoke-tested in the HPC image.

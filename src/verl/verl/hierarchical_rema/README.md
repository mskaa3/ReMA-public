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

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

Current scope:

- Controller outputs are strict JSON strings stored in `raw_text`.
- Decomposer and selector outputs are parsed and validated against the expected schema.
- If a controller returns invalid JSON, the real backend retries with a repair prompt and falls back to a safe structured output if needed.
- Decompositions can be constrained by:
  - `soft_max_hops`: above this, the decomposer gets a penalty that grows with exceedance
  - `hard_max_hops`: above this, the DAG is truncated to a safe executable form
  - `max_nodes_per_decomposition`: existing hard cap on node count
- Worker roles are prompt-defined and non-trainable by default, but `WorkerSpec` already carries `lora_adapter_path` and `trainable` flags for later role-specific LoRA work.
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
  --backend hf \
  --decomposer-model-path /path/to/controller-model \
  --selector-model-path /path/to/controller-model \
  --worker-base-model-path /path/to/worker-model
```

The demo prints the full structured rollout tree plus the decomposer/selector training batches that would be consumed by a future GRPO integration, and it can write JSONL rollout logs under `outputs/hierarchical_rema`.

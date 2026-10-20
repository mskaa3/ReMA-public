# Live Graph+PRM Training Performance

The optimizations preserve per-node rewards, verified-correctness gates, and
executor comparison groups (`task + plan + node_id + executor policy`). The
checkpoint-pinned graph/text feature contract is unchanged; no reward-model
retraining is required for these runtime changes.

## Batch Controls

Defaults in `hierarchical-rema-train.sh`:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `CONTROLLER_BATCH_SIZE` | 96 | Policy-generation batch limit |
| `WORKER_BATCH_SIZE` | 96 | Executor-generation batch limit |
| `GFAM_REWARD_BATCH_SIZE` | 64 | Completed trajectories collected per reward batch |
| `GFAM_PRM_BATCH_SIZE` | 16 | Maximum Qwen sequences per encoder forward |
| `GFAM_BGE_BATCH_SIZE` | 32 | Maximum unique node texts per BGE forward |
| `GFAM_BGE_RAM_CACHE_SIZE` | 8192 | LRU text-embedding entries; 0 disables RAM caching |
| `GFAM_GRAPH_RAM_CACHE_SIZE` | 128 | LRU completed-feature entries; 0 disables RAM caching |

Qwen sequences and BGE texts are sorted by token length within each reward batch.
Duplicate texts are encoded once. Results are restored to trajectory/node order
before applying the existing reward compiler. The small GNN still runs per graph;
batching targets the expensive frozen encoders.

To try larger Qwen batches without editing the preset:

```bash
GFAM_PRM_BATCH_SIZE=32 sbatch hierarchical-rema-train.sh
```

On CUDA OOM, the affected encoder halves its batch size and remembers the lower
limit for the lifetime of that scorer. A single-example OOM still raises an error.
An explicit PRM or BGE context overflow marks only that trajectory as unscored;
it receives no fabricated reward or policy-training sample. Alignment errors,
non-finite features, and checkpoint incompatibility remain fatal.

## Memory, Caches, and Measurement

Only PRM marker rows are selected on the GPU and copied to CPU, rather than the
full last-layer sequence tensor. BGE and complete graph features have bounded
RAM caches. Encoders remain resident; garbage collection and CUDA cache clearing
run only after a recoverable OOM, not between ordinary scoring batches.

`GFAM_FEATURE_CACHE_DIR` defaults to a job-specific directory under
`TMPDIR_LOCAL`, or `/tmp/$USER` if local scratch is unavailable. The launcher
binds it into the driver container. This cache is disposable; it is separate from
shared policy checkpoint storage and does not need to survive a job restart.
An explicit cache directory override is still supported.

Each `[graphprm][features-batch]` log reports elapsed time, actual batch sizes,
cache hits, padding/input token counts, OOM retries, and CUDA peak allocated and
reserved memory in GiB. CUDA peaks include resident models on that device, not
just temporary activations. Compare throughput on equivalent rollout sets;
warm caches and different sequence lengths otherwise confound the comparison.
Real Qwen GPU throughput and peak memory must be measured on the HPC: deterministic
CPU tests cannot establish a safe maximum batch size or a speedup factor.

## Distributed Learner

Each rank computes old-policy log probabilities for a deterministic padded shard,
not the entire replay dataset. Equal shard lengths keep DDP forward collectives
aligned. Compact CPU completion log probabilities are gathered to every rank so
later shuffled replay epochs can use the same complete cache. At most
`world_size - 1` sampler-padding examples are recomputed across the job.
The complete log-probability cache remains replicated in CPU RAM; this change
reduces forward computation, not total per-rank cache memory.

Learner ranks are hard-pinned to exactly `OFFLINE_GRPO_NNODES` eligible Ray nodes,
with `OFFLINE_GRPO_GPUS_PER_NODE` ranks each, preferring the driver for rank 0.
For one node and three GPUs all three ranks stay on one node. For two nodes and
three GPUs they use six GPUs. Insufficient declared Ray GPU capacity fails early;
resources occupied by other actors can still delay actor startup.

Pull the updated repository before starting a new job. The launcher stages the
local `src/verl` code into containers, so these Python-only changes do not require
rebuilding the SIF or changing the reward checkpoint. They do not hot-update an
already running Python process.

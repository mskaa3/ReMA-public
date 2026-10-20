# Plug Reward and Play

One reward setting is fixed for each independent training run. This branch does
not switch rewards during learning or train all reward settings simultaneously.

## Launch independent comparisons

Use the existing `hierarchical-rema-train.sh` (there is no `hierarchical_rema.sh`):

```bash
# Run A: existing binary Graph+PRM checkpoint, staged from the existing S3 path.
REWARD_SETTING=gnn_prm sbatch hierarchical-rema-train.sh

# Run B: deterministic label-aligned heuristic baseline; no reward model download.
REWARD_SETTING=handcrafted sbatch hierarchical-rema-train.sh

# Run C: frozen Qwen3-32B, automatically served locally on the reward GPU.
REWARD_SETTING=llm_coach sbatch hierarchical-rema-train.sh
```

Alternatively edit `export REWARD_SETTING=${REWARD_SETTING:-gnn_prm}` in the
training preset. Accepted future settings `frozen_prm`, `finetuned_prm` and
`gnn_only` raise `NotImplementedError` before staging models or generating data.
There is no automatic fallback to a different reward setting.

Use the SAME initial policy checkpoint, tasks/split, seed, sampling, plan/repeat
counts, update cadence, effective learner batch size and validation schedule.
Do not initialize the second experiment from the policy trained in the first.
Default W&B names include the reward setting and Slurm job ID. Job output/S3
directories remain job-specific. Python also saves `reward_setting.json` and
rejects an output directory carrying a different reward configuration.

The GPU reservation defaults deliberately stay unchanged between settings:
Graph+PRM or the local coach uses the reserved driver-node reward GPU;
handcrafted and remote-coach runs leave it idle. The second node's reserved GPU
remains unused, just as for Graph+PRM. This avoids changing the policy GPU/rank count and effective learner
batch size just by changing the reward source. Resource-efficiency comparisons
may explicitly change the reservation, but record that as a separate setting.

## What stays shared

- Planner generation, validation/repair, DAG execution and delivered artifacts.
- Node grouping identity: task + plan + node + executor policy. Upstream inputs
  can still differ between executions of that node.
- Repeated plan execution and configured planner aggregation (default: mean).
- Mathematical answer verification, including timeout/unverified behavior.
- Reward compiler: existing label weights, binary utility, anti-collapse and
  correctness gates, protocol-invalid-output guards. Negative penalties are not
  softened when a verified failure reduces positive execution credit.
- Existing controller format/provenance filters and saturated-failure exclusion.
- GRPO objective, policy training, recording, routine and diagnostic validation.

The existing `GFAM_*` environment names for compiler/gating/batching controls are
retained for compatibility and apply to the new label-aligned sources too.
No correctness check has been removed. Unverified or unscored trajectories
never receive invented rewards or silently fall back to the old reward function.
Partial repeat groups do not become optimistically biased planner averages.

## Implemented sources

### `gnn_prm`

Delegates to the existing scorer/checkpoint and preserves its numerical rewards
and feature construction. Requires binary verified-outcome support. Does not
retrain the reward model. Existing `GFAM_REWARD_MODEL_PKL_PATH`, encoder, dtype,
context and cache settings remain valid.

### `llm_coach`

The default backend is `local_vllm`, serving **Qwen/Qwen3-32B** from Hugging Face.
It is frozen inference only: the coach never joins policy training, receives
policy weight refreshes, or shares Ray's policy GPU pool. One server remains alive
for the entire run, including policy-update and validation segments.

The launcher partitions GPUs first, then a supervisor starts vLLM inside the
same SIF on the driver node. It binds to `127.0.0.1` with an automatically chosen
port and per-run authentication token. No external paid API or manually started
server is required. Model downloads use the existing job-local `HF_HOME`; provide
`HF_TOKEN` through the environment if needed, not through source control.

Defaults:

| Setting | Default |
| --- | --- |
| `REMA_COACH_BACKEND` | `local_vllm` |
| `REMA_COACH_MODEL` | `Qwen/Qwen3-32B` |
| `REMA_COACH_REVISION` | `main`, resolved and pinned to a Hub commit before loading |
| `REMA_COACH_MAX_MODEL_LEN` | 8192 total input + output tokens |
| `REMA_COACH_MAX_TOKENS` | 4096 output tokens; reserved from that total |
| `REMA_COACH_MAX_NUM_SEQS` | 1 active sequence |
| `REMA_COACH_WORKERS` | 1 concurrent scoring request |
| `REMA_COACH_GPU_MEMORY_UTILIZATION` | 0.90 |
| `REMA_COACH_TIMEOUT` | 300 seconds per scoring request |
| `REMA_COACH_STARTUP_TIMEOUT` | 1800 seconds for downloading/loading/readiness |
| `REMA_COACH_SEED` | 42 |

BF16, tensor parallelism 1, eager execution (no CUDA graph capture), V0 engine,
non-thinking mode (`enable_thinking=False`), greedy decoding, and structured JSON
output are explicit. Non-thinking is a throughput choice, not a claim of equal
judge quality to thinking mode. Pin `REMA_COACH_REVISION` to the recorded commit
for subsequent reproducible runs. Actual model identity and library versions are
recorded in `coach_server.json`, `reward_setting.json`, and cache keys; ephemeral
ports do not invalidate reusable annotation caches.

The SIF must contain vLLM >=0.8.5 and Transformers >=4.51.0. Repository pins meet
this minimum; the supervisor checks the actual container and does NOT upgrade it.
These minimums follow the [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-32B#quickstart).
Startup must pass readiness and a JSON generation smoke check before the ReMA
driver starts. If the server dies during training, the supervisor stops the
driver with an error rather than changing rewards. Both process groups are
cleaned up on normal exit or cancellation. See `coach_server.log` for loading,
memory, and serving diagnostics; those files follow the normal output/S3 sync.

A 32.8B BF16 model needs about 65.6 GB for weights alone, plus KV cache and runtime
memory. A reserved 80 GB GPU with these conservative settings is a candidate, not
a guaranteed fit. No automatic quantization, CPU offload, or extra policy GPU
borrowing is performed. Test memory/latency and judge quality before a long run.
Context overflow is unscored without truncating the rollout. Syntax constraints
do not prove that the coach's judgments are correct.

Inspired by the per-action, context-aware coaching in
[Scaling Multiagent Systems with Process Rewards](https://arxiv.org/abs/2601.23228).
This is a **ReMA adaptation, not a MAPPA reproduction**: it keeps ReMA GRPO,
uses one full-rollout request with ID-keyed per-node criterion scores, and retains
the common reward compiler instead of changing the learning algorithm.

The coach sees the question, plan, each instruction, actual dependency snapshots,
delivered outputs and final answer. It never receives reference answers, old
rewards or rejected raw output. Verified correctness is passed separately to the
compiler, not into the coach prompt. The rubric distinguishes local mathematics,
behavior, counterfactual contribution, plan quality and graph collapse.

Each criterion gets an integer 0..10 and a short explanation. Scores are
normalized to 0..1 and compiled with the same weights/gates as Graph+PRM.
These are **rubric scores, not class probabilities**; artifacts store them under
`reward_evidence`, not `model_predictions`. Uncertainty is expressed in reasons,
not additional confidence labels.

An externally managed endpoint remains optional:

```bash
export REMA_COACH_BACKEND=api
export REMA_COACH_BASE_URL=https://YOUR-SERVICE/v1
export REMA_COACH_MODEL=YOUR-COACH-MODEL
# Supply REMA_COACH_API_KEY through your secret environment if needed.
REWARD_SETTING=llm_coach sbatch hierarchical-rema-train.sh
```

API-backend defaults are 4 workers and a 120-second request timeout. Both backends
support `REMA_COACH_RETRIES=2` and `REMA_COACH_MAX_TOKENS=4096`.
For external endpoints, `REMA_COACH_API_KEY_ENV` selects the credential variable
(default `REMA_COACH_API_KEY`). No server is launched for `api`.

Responses require exact scopes, labels, node IDs, finite integer scores and
reasons. API/format failures have bounded retries and are recorded as unscored.
Concurrent requests restore input order. Successful responses are cached under
`OUTPUT_DIR/coach_cache` by model identity (endpoint for external APIs), prompt/version and serialized input;
failed responses are not cached. Verifier gating is recomputed on cache hits.
Use immutable model identifiers: an API alias changing its underlying model
cannot be detected by this local cache.

Tests use a lightweight fake local server, not GPU models or paid services.
Real GPU compatibility, memory, latency and judge quality still require HPC testing.

### `handcrafted`

Inspired by the older branch's protocol, duplication and hierarchy checks, but
does not copy its unsafe answer extraction or assume textual overlap proves use.
All label dimensions have explicit evidence entries:

| Dimension | Deterministic evidence |
| --- | --- |
| Global outcome | Existing verified correctness; neutral if unavailable |
| Plan fulfillment | Structural defects; semantic sufficiency remains unknown |
| Plan discipline | DAG validity, final-path reachability, repeated instructions, premature final-answer instructions |
| Anti-collapse | Those plan defects and repeated intermediate prose; no mandatory depth for atomic tasks |
| Worker fulfillment | Empty/invalid output, bounded exact arithmetic/equalities; other mathematical completion is unknown |
| Worker discipline | Protocol validity, extreme repetition, visible reward/instruction manipulation |
| Worker causal utility | Empty/off-path artifacts, redundancy; provision alone does not prove use |
| Final fulfillment | Existing verified final correctness |

Unknown semantic evidence maps to 0.5 (neutral at bad-class penalty 1), not
unearned full credit. Each distinct plan/collapse flag subtracts 0.35 from its
structural score, down to zero. Each visible behavioral flag subtracts 0.5;
invalid delivery is zero. Redundancy gives utility 0.25, otherwise unproven
causality stays 0.5. Valid arithmetic equalities alone score 0.75; a fully
specified atomic calculation can score 0 or 1. These provisional constants are
versioned in `handcrafted_v1`, not learned or empirically validated.

This is not semantically equivalent to a trained model: general proofs, nuanced
fake verification and true counterfactual contribution cannot be established by
rules alone. The baseline uses verifier correctness directly for final
fulfillment; model/coach process scores remain predictions with shared outcome
gates. Report this information-use difference when interpreting experiments.

## Extension points

`reward_providers/base.py`: reference-free input contract, common score compiler
and result validation. A provider implements `score_rollout` and ordered
`score_rollouts`, returning compiled planner/final/every-worker rewards or an
explicit unscored result. The orchestrator does not load models or call APIs.

`reward_providers/factory.py`: registration/configuration and per-run provenance.
New sources should not alter execution, grouping, protocol checks or GRPO.

`validation.py` and `metrics.py` remain independent from the provider. Routine
validation uses verified correctness without creating a provider; diagnostic
validation uses the selected provider. Add future evaluation metrics here,
without turning them into training rewards implicitly.

Legacy embedded `gfam_reward_scorer=` callers remain supported. Standalone demo
rollouts accept the same `--reward-setting` and checkpoint flags; without an
explicit setting the old embedded demo remains compatible. Replay-only training
consumes saved rewards and does not re-score them with a newly selected provider.

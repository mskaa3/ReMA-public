# Binary Graph+PRM Reward Model

## New Declared/Provided v2 Checkpoints

The new `GraphPRM_training_HPC` bundle produces checkpoints with
`feature_schema=graphprm_declared_provided_v2`. `GFAMRewardScorer` dispatches these
to `hierarchical_rema.graphprm_v2.runtime`; older unversioned checkpoints still
use the original implementation described below. **The rest of this document
describes v1 unless explicitly marked v2.** Do not use v2 features with v1 weights.

For a retrained v2 checkpoint, set `GFAM_PRM_MAX_LENGTH=0` and
`GFAM_PRM_TORCH_DTYPE=auto` (now the launcher defaults). These resolve to the
checkpoint's pinned settings, normally 4096 / bfloat16. Explicit mismatches fail
rather than silently changing features. Legacy checkpoints resolve these defaults
to 512 / float32, preserving their behavior. The HPC bundle stores versioned
backups and can publish its winner to this branch's existing S3
`hierarchical_rema/best_overall_model.pkl` alias. Local staging filenames are
job-specific, so a new job refreshes the alias rather than reusing an older
job's `.stage_complete`. Already running jobs keep their loaded checkpoint.

v2 uses the SAME vendored package as HPC training: graph builder, full-text PRM
renderer, pinned encoders, feature schema, binary heads and reward compiler.
`dependency_outputs` from live executions is authoritative, including `{}`.
Declared edges encode the plan; `provided_output` edges encode supplied artifacts.
There is no guessed actual-use edge or downstream blame from reachability.
Recipient counts include context-coverage metadata, so zero does not silently mean
confirmed non-use. Historical reconstructed context is marked as an assumption.

Worker/final reward is weighted binary label utility with a soft collapse gate
and a verified-final-failure gate on positive credit.
Decomposer reward includes plan quality, a plan-quality-gated success bonus and
soft consistency/collapse penalties. These are heuristics, not causal attribution.
PRM scalar scores remain diagnostics, not features or direct reward terms.
`GFAM_BAD_CLASS_PENALTY=1` retains symmetric per-label utility; higher values
increase the cost of predicted bad behavior. Protocol-invalid outputs remain
non-positive and receive a separate penalty.

v2 full-text extraction rejects context overflow and marker misalignment. No
silent truncation/zero substitution. BGE and Qwen stay frozen. Only the small
graph model and label/auxiliary heads need retraining. Old feature caches are not
reused under the new schema. All generated rollouts need actual encoder inference;
the encoders are reused within the scorer process, not reloaded for each rollout.

The bundle's audited source split had 78 shared normalized task prompts; 102
training rows are quarantined before training, leaving 4060 train / 734 test.
Original source files and labels are preserved. Ten retained rollouts have empty
delivered output with nonempty raw audit text and are listed for separate review.
This is not an automatic relabeling operation. See the bundle README for SIF,
Slurm, caching, search, S3 publishing and remaining validation requirements.

This branch supports the binary `graphprm_hybrid` checkpoint produced by
`train_graphprm_no_selector_simplified.py`. Upload the **best model `.pkl`**;
no offline training scripts, prepared graphs, or feature caches are required.
Only load trusted checkpoints: the PyTorch checkpoint loader permits pickle.

## Live v2 Pilot Defaults

The main training launcher now reserves one GPU **per allocated node** outside
Ray. On each 4-GPU node the first three allocated devices form the policy pool;
the driver uses the last device for frozen Qwen, BGE and the small GNN. Reserved
GPUs on other nodes are idle. This symmetric reservation keeps generation and
distributed learner resource pools consistent. Masks are derived from Slurm's
`CUDA_VISIBLE_DEVICES`, including noncontiguous IDs or UUIDs, not guessed host IDs.
The reward driver sees only its reserved GPU, renamed `cuda:0` locally.

```bash
export GFAM_REWARD_GPU_RESERVATION=1
export GFAM_REWARD_MODEL_DEVICE=cuda:0
export GFAM_PRM_DEVICE=cuda:0
export GFAM_ENCODER_DEVICE=cuda:0
export GFAM_PRM_TORCH_DTYPE=auto
export GFAM_PRM_MAX_LENGTH=0
export OFFLINE_GRPO_DISTRIBUTED=true
export ALTERNATING_DECOMPOSER_NUM_EXECUTOR_ROLLOUTS=4
export GFAM_DECOMPOSER_MAX_EXECUTOR_ROLLOUTS=4
export DECOMPOSER_REWARD_AGGREGATION=mean
```

These are defaults; remove old CPU and full-GPU-count overrides from your job
environment. The launcher rejects policy GPU counts that overlap the reservation.
It requires Slurm, at least two allocated GPUs per node, the vLLM/Ray backend and
Ray-distributed policy updates. Tensor parallel size must divide the policy GPU
count (default 1 works with three GPUs). For an explicit CPU-only reward setup,
set the reservation to 0 and all three reward devices to `cpu`.
Encoders stay resident; scoring remains sequential, not server-side batching.
No checkpoint retraining or feature-cache invalidation is needed for this policy.

### Unscored Context Overflow

The checkpoint still defines the exact feature contract: 4096 PRM tokens for the
new model, including the question, instructions, dependency context and results.
Only explicit PRM/BGE context overflow is recoverable. It produces `status=unscored`,
an error code, token-limit diagnostics, empty predictions/rewards, and a JSON
`null` rollout reward. It does **not** truncate, substitute zero vectors, assign
a zero/negative reward, or fall back to the handcrafted reward formula.

Unscored executions are excluded from worker learning and worker memory. If any
repeat of a plan is unscored, that entire plan is excluded from the decomposer
update and its reward/advantage normalization; averaging only surviving repeats
would introduce length-dependent selection bias. Other successfully scored worker
repeats can still train, subject to the usual minimum GRPO group size.

With rollout logging enabled (the default), failures are always appended to
`reward_scoring_failures.jsonl` in the rollout log directory, even in `best` mode.
They include the plan, delivered and raw outputs, and scoring diagnostics.
Unscored entries never enter best-rollout rankings. Summaries report counts and
use only scored denominators for reward means. An entirely unavailable diagnostic
is printed as `nan` and saved as JSON `null`, never used as a training target.
Marker misalignment, nonfinite features, GPU OOM and incompatible checkpoints
remain fatal: they must not be disguised as ordinary context overflow.

### Decomposer Reward and Repeated Execution

Repeated executor rollouts already run the **same plan** using the same executor
checkpoint and sampling configuration. Planner-phase defaults previously capped
this at two; they now allow four (a smaller explicit total-rollout setting still
applies). No policy update occurs between these repeats. Rewards are averaged,
not selected by best outcome. Averaging reduces executor randomness but does not
make the signal a causal measurement of plan quality.

For v2, the existing final-answer verifier supplies `Y=0/1` to the reward compiler
**after** feature extraction and prediction. Ground truth never enters the model
record or downstream artifacts. Without a reference, the compiler explicitly
records that it is using predicted success instead.
For verifiers with partial/format credit, only a full score of 1 counts as success.

Let `u(label)=P(good)-lambda*P(bad)` (default lambda=1),
`Q=.5*u(task_fulfillment)+.5*u(execution_discipline)`,
`G=min(P_good(task_fulfillment), P_good(execution_discipline))`, and
`A=P_good(anti_collapse)`. Each repeat gives:

```text
planner_reward = min(Q, 0) + max(Q, 0)*A
               + 0.15*G*A*Y
               - 0.15*(1-G)*(1-Y)
               - 0.20*(1-A)
```

Only positive plan quality is attenuated by the collapse gate. The success bonus
requires both plan-quality and anti-collapse support; a correct bypass therefore
earns less bonus if the model detects it. Failure is penalized more when the plan
also looks weak, not equally for every executor mistake. Collapse has a separate
penalty. These are soft gates: a missed bypass can still receive positive reward.
All components and verified/predicted outcome provenance are logged. Model
predictions are unchanged by the verified outcome. Worker/final positive rewards
now also use the verified-failure gate below; the planner formula is unchanged.

### Verified Failure Gate for Every Execution

Set `GFAM_VERIFIED_FAILURE_POSITIVE_SCALE=0.25` (the default) to retain 25% of
positive local credit when the final answer is verified incorrect. For the
collapse-gated local reward `r`, compilation applies:

```text
failed final answer: min(r, 0) + 0.25 * max(r, 0)
successful answer:   r
```

This applies to every worker node, including the final execution node, and to
the separate final head used for the rollout-level reward. Negative local
penalties are unchanged; protocol penalties are applied afterwards at full
strength. A reward of +0.94 becomes +0.235; -0.40 stays -0.40. Worker GRPO uses
these gated per-node rewards, not the trajectory reward substituted for each node.
The 0.25 factor is provisional and configurable in [0,1]; 0 removes all positive
credit on verified failure, while 1 restores the previous worker behavior.

The compiler records pre-gate reward, effective scale and verified outcome in
each rollout. No reference means no verified-failure gate; runtime verification
errors instead exclude the rollout from training, with no fabricated reward.
This changes the inference-time policy only, not checkpoint features or learned
probabilities. No reward-model retraining is required. Existing reward exports
and replay generated under the old policy are not automatically recomputed.

GRPO still normalizes within task + plan + node + executor-policy groups. If
every reward in a group is positive and scaled by the same factor, standardized
advantages can remain effectively unchanged; this soft gate is not an absolute
ban on reinforcing the best member of an all-failed group.

### Provisional Weights and Future Tuning

The current `equal_labels_v1` policy uses equal label weights: decomposer
`task_fulfillment=.5, execution_discipline=.5`; worker and final each use
`task_fulfillment=1/3, execution_discipline=1/3, causal_utility=1/3`.
For worker/final local quality `q`, first compute
`r=min(q,0) + max(q,0)*(.5+.5*A)`. Positive local credit retains 50-100%
depending on anti-collapse probability, then 25% of that on verified failure
by default; negative credit is not softened.
There is no separate ground-truth success bonus for worker/final rewards.
Invalid or empty delivered outputs get `min(reward,0)-.10` after compilation.
PRM scalars do not directly enter these rewards. Planner repeats are averaged;
the compiler does not guarantee that final rewards lie in [-1,1].

Equal weights avoid an unsupported preference between labels; they are not a
claim that the labels are equally reliable or equally important. The remaining
coefficients (.15 success bonus, .15 failure penalty, .20 collapse penalty,
.50 collapse-gate floor, .25 verified-failure positive scale, .10 protocol penalty, lambda=1) are also
provisional heuristics. This change affects compiled rewards, not probabilities,
classes, loss weights or frozen features; old exported rewards must be recomputed.

Verify these choices empirically on task-disjoint development rollouts, with
per-source correctness and violation rates and a final untouched test set.
An outer-loop search (e.g. Optuna) can compare short policy-training runs using
independent verified outcomes and audited violations, NOT the reward being tuned.
Classifier-loss weighting and reward-component weighting are separate problems.

One future option is adaptive reward weighting: increase a criterion's weight
when the policy violates it above an agreed target. Estimate role-specific
violation rates on fresh audit rollouts, smooth them, change weights slowly
between update blocks, bound them and keep their total scale controlled. Freeze
weights during each rollout/GRPO update block, log their versions, and do not
mix stale reward versions in replay without recompiling. A more principled variant
uses adaptive Lagrange penalties for constraint violations; see
[Stooke et al., PID Lagrangian Methods](https://proceedings.mlr.press/v119/stooke20a.html).
This is a proposal, not implemented behavior. If a label's CLASSIFIER is failing,
increasing its reward weight amplifies noise: improve/audit that predictor instead.
The frozen Graph+PRM signal can be a proxy for violation rates, but should be
checked against verified outcomes and independently judged samples.

## Legacy v1 Launch Configuration

Keep the existing job launch command and set these variables before submitting
`hierarchical-rema-train.sh`:

```bash
export GFAM_REWARD_MODEL_PKL_PATH="s3v2:YOUR_BUCKET/YOUR_PREFIX/best_overall_model.pkl"
export GFAM_REWARD_MODEL_DEVICE=cpu
export GFAM_ENCODER_DEVICE=cpu
export GFAM_PRM_DEVICE=cpu
export GFAM_PRM_TORCH_DTYPE=float32
export GFAM_PRM_MAX_LENGTH=512
export GFAM_BAD_CLASS_PENALTY=1.0
export GFAM_FEATURE_CACHE_DIR="/persistent/cache/graphprm"
```

Replace the S3 path with the rclone remote/path used on your cluster and the cache
path with a writable persistent directory visible inside the training container.
The existing launcher downloads the `.pkl` with rclone and passes its local path
to Python. Use a versioned S3 object and fresh job scratch when replacing a model;
the launcher's existing staging-complete marker can reuse a previous local file.

These explicit CPU settings are conservative, **not fast**, and require
`GFAM_REWARD_GPU_RESERVATION=0` with the current launcher. The v2 pilot settings
above provide an isolated CUDA device instead. Float32 Qwen weights alone require
roughly 28 GB, plus activations. BGE and the graph network have separate device
settings. Do not place reward encoders on an already-full policy GPU.

BGE-M3 and Qwen Math PRM are separate frozen pretrained dependencies, loaded from
Hugging Face using the model IDs saved in the checkpoint. They are not contained
in the `.pkl`. Pre-cache them on offline compute nodes; an optional `HF_TOKEN`
helps with download limits. Qwen uses its repository's custom model code.

The current best model was trained with **512-token PRM truncation**. Preserve
that value for matched inference. Later markers may be absent and receive the
same zero-feature fallback used during training; the runtime logs the count.
Increasing this limit changes the features and is not a transparent speed/quality
switch. Float16 extraction previously produced invalid values on MPS; float32 is
the safe starting setting. Non-finite features now raise an error, not a reward.

`GFAM_BAD_CLASS_PENALTY` controls the binary node-label utility used by the reward
compiler: `P(good) - lambda * P(bad)`. The default `1.0` preserves the original
`2*P(good)-1` behavior. Set it to `2.0` for positive label utility only above
`P(good)=2/3`. The direct Python flag is `--gfam-bad-class-penalty`.
Values must be finite and non-negative. The utility range is `[-lambda, 1]`;
it is not rescaled or clipped to `[-1, 1]`.

This changes decomposer/worker/final quality terms, not predicted classes,
probabilities, or the exported `centered_score`. Existing probability-based
failure/collapse/downstream penalties and caps remain unchanged. The threshold
therefore describes an individual label utility, not a guarantee of positive
final compiled reward. `graph_summary` records the penalty and threshold.
Changing it requires neither retraining nor recomputing cached Qwen features.

For direct Python launches, the corresponding options are
`--gfam-reward-model-pkl`, `--gfam-reward-model-device`, `--gfam-encoder-device`,
`--gfam-prm-device`, `--gfam-prm-torch-dtype`, `--gfam-prm-max-length`, and
`--gfam-feature-cache-dir`.

## Inference And Rewards

1. Build one no-selector graph for each completed executor trajectory. Question,
   decomposer, subtask, execution, and final nodes remain distinct. Multiple
   executions by `executor_worker` are indexed by **subtask node ID**, not worker
   identity.
2. Encode cleaned plan/result text with BGE-M3. Reference-answer and judge-label
   fields are excluded from the scorer. Scratchpad text is removed. The live
   backend validates the output protocol without comparing it to ground truth.
3. Render the DAG-aware Qwen sequence and attach marker hidden states to worker
   and final nodes. All nodes retain BGE features. The current checkpoint does
   **not** use the PRM scalar as a feature or directly mix it into rewards.
4. Run the checkpoint's graph architecture and binary label heads. Labels are
   `0=bad/risky`, `1=good`; each output contains the predicted class, probabilities
   for both classes, and `centered_score = p_good - p_bad`.
5. Apply the inference-time reward compiler in `graphprm_compiler.py`. It preserves
   the offline simplified compiler; no retraining is needed to edit this policy,
   but a changed compiler should be evaluated separately.

| Scope | Labels |
| --- | --- |
| Graph | global_outcome, anti_collapse |
| Decomposer | task_fulfillment, execution_discipline |
| Worker / final | task_fulfillment, execution_discipline, causal_utility |

Worker/final quality weights are 0.42, 0.33, 0.25 in the label order above.
Decomposer quality weights are 0.58 and 0.42. Additional compiler terms penalize
predicted failure, collapse, and downstream consequences and cap positive rewards.
Failure gating uses the auxiliary final-correctness prediction; graph
`global_outcome` is also exported for inspection. These are model estimates, not
oracle correctness checks or proven causal attribution.

The decomposer reward includes plan quality, a collapse penalty, a final-failure
penalty, and a plan-fault-weighted downstream penalty. Thus good-looking plans
are not rewarded solely on their text. Rewards are computed separately for each
executor trajectory; the existing planner aggregation configuration combines
trajectory rewards for the same plan. The existing planner/executor/joint phase
selection determines which policy samples are trained. No selector samples or
selector predictions are generated.

## Caching And Inspection

Frozen encoders are retained across rollout segments within a process. Optional
disk caching reuses Qwen marker extraction for identical rendered inputs and
extraction settings. New generated trajectories generally require new extraction;
the runtime currently scores one trajectory at a time, not a server batch.

Rollout `reward_model_outputs` includes `backend=graphprm_binary`, binary
`model_predictions`, `compiled_rewards`, `graph_summary`, and diagnostic
`prm_step_scores`. The legacy internal source identifier `gfam_v1` is retained so
existing orchestration selects the learned rewards rather than heuristic rewards.

## Verification

The portable graph builder and sequence renderer matched all 734 current test
rollouts. The best checkpoint produced identical logits to the offline reference
on matching synthetic features; both relational and GAT layers were checked.
Unit tests use lightweight encoder stubs to check binary outputs, caching,
reference/scratchpad exclusion, missing markers, invalid features, execution IDs,
and planner/executor/joint reward routing without downloading models.

A real GPU/Hugging Face extraction and cluster policy-training run are still
required as a deployment smoke test.

## Shared Input Contract And Safety Checks

`graphprm_inputs.py` is shared with the offline trainer as a byte-identical,
stdlib-only module. Its canonical offline location is
`models/gnn_prm/scripts/graphprm_inputs.py`; copy changes to both locations.

- Only recognized scratchpad/result/answer protocol tags are removed. Expressions
  such as `x < 3 and y > 2` and unknown non-protocol markup are preserved. An
  unclosed scratchpad is discarded, not exposed as result text.
- The first present accepted-result field is authoritative, even if it is empty.
  `raw_output_text` is audit data only and cannot supply missing model input.
  The live adapter preserves delivered fields until this shared cleaning step.
- Protocol validity is handled separately from learned predictions. Empty
  delivered outputs or an existing `invalid_reason` cap compiled worker reward
  at zero, then subtract the existing 0.10 invalid-result penalty. The same guard
  applies to an empty final answer or invalid final execution. Predictions stay
  unchanged; reward payloads record the pre-guard reward and reason. This is a
  compiled-reward guard, not a guarantee about later group-normalized advantages.
- Literal reserved tokens in task text, instructions, node IDs and results are
  escaped before rendering. Only the renderer inserts real `<extra_0>` markers.
  Full tokenized marker count must match node targets; right truncation must
  preserve a token prefix; extracted unpadded marker positions and cached feature
  counts must match the verified targets. Misalignment raises an error rather
  than silently assigning another worker's hidden state.

Offline prepared-graph caches have a new version and are rebuilt. Unchanged PRM
sequence caches can still be reused after alignment validation; changed prompts
get new feature keys. Runtime caches include the input-contract version. No
dataset or checkpoint is overwritten by these changes. Re-run evaluation before
deployment because formerly corrupted inputs now differ from earlier results.

The post-fix audit compared **independently prepared** offline and runtime inputs
for all 734 test rollouts: node text, metadata, edges and rendered PRM sequences
match. The checkpoint logits and both graph-layer implementations also match on
identical test features. Regression tests cover math preservation, authoritative
empty results, audit retention, protocol penalties, reserved tokens and alignment.

## Remaining Input Limitations

The 512-token limit covers the whole chat sequence, not each worker: system text,
problem, all step instructions/results, repeated dependency results, and a final
answer marker. Removing scratchpads does not make this sequence short. A local
tokenizer-only audit after the cleanup fixes found 195 of 734 rollouts missing
their final marker and four retaining no markers. Missing markers retain the
training-compatible zero-feature fallback. Each runtime summary reports total
tokens, retained tokens, marker count and missing markers. Marker-preserving
windows or a longer context require a separate training/evaluation change.

Additional tokenizer-only coverage audit of the current rendered sequences:

| Token limit | Train rows losing final marker (4,162) | Test rows losing final marker (734) |
| --- | ---: | ---: |
| 512 | 1,091 | 195 |
| 1,024 | 116 | 21 |
| 2,048 | 3 | 0 |
| 4,096 | 0 | 0 |

The longest rendered sequence is 2,235 train / 1,679 test tokens. These counts
are after the renderer's existing character limits; one training worker result
exceeds its 1,400-character cap. Full marker coverage alone does not guarantee
full text coverage. For the next model, use token-budgeted full-text rendering,
length-bucketed dynamic padding, explicit overflow checks, and matching extraction
settings in training and deployment. Re-extract changed PRM features and retrain
the graph model; do not silently switch the existing checkpoint's feature policy.

The dependency-schema mismatch remains: training data often equates declared
dependencies with used dependencies; live inference guesses use and treats an
any-dependency-used flag as all-used. It also populates downstream-use metadata
absent from the current test rows. Agree on provided-versus-demonstrably-used
semantics, rebuild features and retrain before claiming equivalent live features.

## Reference-Based Evaluation Is Not Execution

The live backend no longer redacts non-final results or marks them invalid merely
because they match ground truth. Protocol validation and existing reference-free
recovery still apply. Well-formed premature answers remain visible to downstream
workers and the learned execution-discipline/anti-collapse heads.

After a rollout completes, `build_selection_reward` computes the new audit field
`WorkerExecution.reference_answer_match` for non-final results. It is exported in
execution/training metadata, not supplied to Graph+PRM, and never triggers the
legacy leak flag, sample exclusion, or protocol-invalid penalty. A matching
intermediate quantity is not by itself evidence of a reward hack. Existing final
correctness and handcrafted reward calculations may still consult ground truth;
Graph+PRM replaces those handcrafted reward values when configured.

Regression tests hold generated completions fixed and vary the reference (correct,
different, or absent). Serial and batched execution preserve delivered text,
downstream prompts, PRM cache keys, model predictions and compiled learned rewards;
only reference-based evaluation diagnostics change. No policy/encoder models are
loaded for this check. Real model-generated trajectories still require a smoke test.

Historical redacted rollouts cannot be repaired just by changing this code:
their downstream outputs were generated from redacted context. Keep them marked
as legacy data or re-execute the affected trajectories; do not fill scorer inputs
from old raw-output fields while retaining mismatched downstream context.

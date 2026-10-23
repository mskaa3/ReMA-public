# ReMA Project
This is an open-source framework for Multi-Agent Multi-Turn Reinforcement Learning, i.e. **Re**inforced **M**ulti-**A**gents (ReMA), for the work of **Re**inforced **M**eta-thinking **A**gents (ReMA)

## Overview
![ReMA](./assets/demo.png)
We implement the paper *Learning to Meta-think for LLMs with Multi-Agent Reinforcement Learning* as an initial example.
In this work, we decouples the reasoning process into two hierarchical agents: a high-level meta-thinking agent responsible for generating strategic oversight and plans, and a low-level reasoning agent for detailed executions.

We support two types of training:

![Training](./assets/train.png)

### ReMA with Separate Parameters
In this mode, the high-level agent (meta-thinking) and the low-level agent (reasoning) are parameterized separately. The high-level agent is frozen while the low-level agent is trainable, or vice versa. This allows for independent optimization of meta-thinking and reasoning abilities.

### ReMA with Shared Parameters
In this mode, both the high-level and low-level agents share the same set of parameters. The training is performed end-to-end, and feedback is used to update the shared parameters jointly. This enables unified learning of both meta-thinking and reasoning within a single model.
You can choose the training mode according to your research needs. For more details, please refer to the documentation and the code examples in the src directory.


### Turn-level Ratio 

![Turn-level Ratio](./assets/turn_ratio.png)

We also proposed a turn-level ratio for clip and loss averaging, which is more efficient and effective than token-level ratio.


## Installation
we recommend using cuda12.4 + torch2.6 + python3.10.
```bash
conda create -n rema python=3.10
conda activate rema
# install flash-attn (faster)
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation

# install flash-attn
pip install flash-attn==2.7.4.post1 --no-build-isolation

# for sft
cd src/360-LLaMA-Factory
pip install -e .

# for RL
cd src/verl
pip install -e .

# update libs
pip install -r requirements.txt
```

## Training


### How to run scripts
To run the training, you can see `scripts/rl/rema/example_7b.sh` as an example.
```bash
bash scripts/rl/rema/example_7b.sh
```

key arguments:
- Rollout
    - `data.max_prompt_length`: max length of prompt, **for each turn's input**
    - `data.max_response_length`: max length of response **per turn**
    - `actor_rollout_ref.rollout.stop_when_truncated=True`: whether to stop when truncated, i.e. last agent's response is stopped by length

- Training:
    - `actor_rollout_ref.actor.clip_mode`: ratio clip mode, `"turn"` or `"batch"`
    - `actor_rollout_ref.actor.agg_mode`: ratio aggregation mode, `"trajectory"` or `"batch"` or `"turn"`

- For Separated Trainer:
    - `algorithm.switch_agent.enable=True`: whether to use separated trainer
    - `algorithm.switch_agent.freq=10`: switch agent frequency
    - `algorithm.switch_agent.model_paths`: model paths of two agents, **Previous `actor_rollout_ref.model.path` will be ignored**

see `src/verl/verl/rema_trainer/config/ppo_trainer.yaml` and `src/verl/verl/rema_separated_trainer/config/ppo_trainer.yaml` for more details.



## GraphPRM Process-Cost Reward (Hierarchical Trainer)

An optional objective, separate from the existing `gnn_prm` reward. Keep your
current rollout mode and alternating/joint schedule. In `hierarchical-rema-train.sh`,
replace the reward-setting line with:

```bash
export REWARD_SETTING=graph_prm_process_costs
# Same binary v2 checkpoint and reserved reward GPU as gnn_prm:
# export GFAM_REWARD_MODEL_PKL_PATH=s3v2:.../chosen_model.pkl

# Provisional experiment settings, not empirically optimized targets:
export REMA_PROCESS_COST_TARGET_COLLAPSE=0.10
export REMA_PROCESS_COST_TARGET_DISCIPLINE=0.10
export REMA_PROCESS_COST_TARGET_TASK_FULFILLMENT=0.10
export REMA_PROCESS_COST_TARGET_PLANNER_DISCIPLINE=0.10
export REMA_PROCESS_COST_TARGET_PLANNER_TASK_FULFILLMENT=0.10
export REMA_PROCESS_COST_INITIAL_COLLAPSE=0.10
export REMA_PROCESS_COST_INITIAL_DISCIPLINE=0.10
export REMA_PROCESS_COST_INITIAL_TASK_FULFILLMENT=0.10
export REMA_PROCESS_COST_INITIAL_PLANNER_DISCIPLINE=0.10
export REMA_PROCESS_COST_INITIAL_PLANNER_TASK_FULFILLMENT=0.10
export REMA_PROCESS_COST_LEARNING_RATE=0.01
export REMA_PROCESS_COST_MIN_COVERAGE=0.95
```

The frozen GraphPRM still provides predictions, but its old reward compiler is
bypassed. Let `Y` be verified binary final correctness, `C_c = 1 - P(anti_collapse=1)`,
`C_d,i = 1 - P(execution_discipline_i=1)`, and
`C_tf,i = 1 - P(task_fulfillment_i=1)`. Planner-specific `C_D,d` and `C_D,tf`
are the failure probabilities from the decomposer's own discipline and
task-fulfillment prediction heads:

- Worker and final execution: `R_i = Y - lambda_d * C_d,i - lambda_tf * C_tf,i`.
- Planner per completed execution: `R_D = Y - lambda_c * C_c - lambda_D,d * C_D,d - lambda_D,tf * C_D,tf - lambda_d * mean_i(C_d,i) - lambda_tf * mean_i(C_tf,i)`.
- The final execution uses the separate final prediction head, and is counted
  exactly once in the mean over executed nodes. Invalid protocol results or empty
  delivered artifacts explicitly receive discipline cost 1, not an additional
  reward cap. Task fulfillment remains model-predicted, without a protocol override.
  Raw rejected output is never substituted for the delivered artifact.

Task fulfillment penalizes incorrect/incomplete local operations even if another
worker recovers the final answer. Its multiplier and target are independent of
discipline and collapse. The planner receives both its own predicted costs and
mean execution costs. Its own costs have separate multipliers and targets; mean
execution costs reuse the worker multipliers, without extra fixed mixing weights.
The direct planner costs are evaluated once per graph, not once per worker.
This rewards plans associated with reliable execution, but does not establish
that a planner caused every worker failure. Causal utility remains available for
auditing but does not contribute to this objective. Predicted global outcome
never substitutes for missing verified
correctness. Context overflow and unknown required verification remain unscored.
The old positive-failure scaling, label weights, planner bonuses, collapse gates,
controller format penalties and saturated-positive-failure filter are not applied.
Plan/protocol validation and existing training-sample eligibility checks remain;
synthetic fallback planner completions are not learned from.

Planner returns always use the mean over independent executions of the same plan,
even if the legacy aggregation option says `best`. In repeated-execution mode,
worker responses keep their own rewards. In fixed-prefix mode, each sampled focal
action's own rewards are averaged across all its continuations before advantages
are calculated; the final node has one continuation. Joint mode keeps ordinary
planner executions separate from worker branches. No extra generation is needed.

After each complete rollout/policy-update segment:

```text
lambda_k_next = max(0, lambda_k + learning_rate * (mean_cost_k - target_k))
```

Weights are fixed throughout all rollout batches and policy updates in that
segment. A segment can still update weights if policy samples have zero signal;
statistics are collected before eligibility/sign/best-selection filters. Statistics
average nodes within graphs, repeats within plans, plans within tasks, and then
tasks equally. Joint-mode statistics use ordinary executions, not the additional
correlated branches. Executor-only fixed-prefix statistics use complete action /
continuation groups, equally averaged across branch points. A plan with missing
required costs/outcomes is excluded as a whole. If fewer than 95% of sampled plans
are complete, multiplier adaptation pauses. Missing results are never counted as
zero cost; coverage and observed means are logged. Even above this threshold,
missingness may bias the estimate, so inspect verification/scoring failures.
The same update rule applies separately to all five costs: collapse, worker
discipline, worker task fulfillment, planner discipline, and planner task fulfillment.
The provisional initial multipliers and cost targets are all 0.10.

Validation reads the current weights but never adapts them. W&B records
`process_costs/mean/*`, `process_costs/target/*`, `process_costs/lambda/*`,
`process_costs/verified_correctness`, coverage and dual-update counters.
`<output-dir>/process_costs/state.json` contains weights, checkpoint/config identity
and the segment update ledger. Every training segment also saves
`process_cost_update.json` and a portable `process_cost_state.json` beside its policy
artifacts. Epoch S3 uploads include these files and sync the root `process_costs`
directory independently of validation cadence. Recreating a scorer never resets
weights; replaying the same recorded segment cannot apply the dual update twice.

To carry weights into a new job, download the matching state or segment sidecar to
a shared path visible inside the container and set
`REMA_PROCESS_COST_INITIAL_STATE=/shared/path/process_cost_state.json`. The model
checksum and process-cost configuration must match. This restores penalty weights,
not policy checkpoints or the full training schedule; restore those separately.
An existing output-directory state takes precedence over this initialization file.
This objective is versioned as `role_aware_process_costs_v3`. State files from
earlier two- or three-cost versions are intentionally rejected; start a new output directory
without the old `REMA_PROCESS_COST_INITIAL_STATE`. The frozen GraphPRM checkpoint
and compatible feature caches can be reused; no reward-model retraining is needed.

These are soft predicted costs, not measured violation rates or guaranteed safety
constraints. Large multipliers can rank an incorrect low-cost execution above a
correct high-cost execution. GRPO may reinforce a relatively better action even
when its absolute reward is negative. This role-aware Lagrangian-style experiment
does not guarantee constrained optimality, and targets/update rate need validation.

A planned ablation removes only the two mean-worker-cost terms from the planner
reward, retaining verified correctness, collapse, and the planner's own costs.
Worker rewards stay unchanged. Comparing this against the full objective can
test whether execution costs add useful planner feedback or duplicate penalties
and transfer blame for independent worker mistakes. This is a proposed experiment,
not a new reward-mode switch. Setting shared worker multipliers to zero would not
isolate this ablation, because it would also change worker rewards.

## Evaluation
A very easy way is to set
```
+trainer.val_before_train=True \
+trainer.val_only=True \
```
in the training scripts.


## Basic Results
![Basic Results](./assets/results.png)
- Multi-agent Multi-turn RL training is still highly sensitive and often unstable(collapse)
- Turn-level ratio is more efficient and effective than token-level ratio.



## Citation
If you find this work useful, please consider citing:
```
@misc{wan2025remalearningmetathinkllms,
      title={ReMA: Learning to Meta-think for LLMs with Multi-Agent Reinforcement Learning}, 
      author={Ziyu Wan and Yunxiang Li and Xiaoyu Wen and Yan Song and Hanjing Wang and Linyi Yang and Mark Schmidt and Jun Wang and Weinan Zhang and Shuyue Hu and Ying Wen},
      year={2025},
      eprint={2503.09501},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2503.09501}, 
}
```

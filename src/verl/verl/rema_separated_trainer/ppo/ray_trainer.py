# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import copy
import json
import os
from pathlib import Path
import uuid
import jsonlines
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, Tuple
from copy import deepcopy
from collections import defaultdict

import ray
import numpy as np
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.rema_trainer.ppo import core_algos
from verl.rema_trainer.ppo.metric_utils import compute_data_metrics, compute_reward_diagnostic_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rema_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils import torch_functional as verl_F
from verl.rema_separated_trainer.ppo.multi_agent_rollout import MultiAgentRollout
from verl.rema_separated_trainer.ppo.scoped_c3_grpo import (
    estimate_scoped_c3_grpo,
)
from verl.rema_separated_trainer.ppo.prefix_probe import (
    apply_prefix_probe_gate,
    answer_round_records,
    compare_worker_final_answers,
    count_planned_subtasks,
    collect_prefix_probe_requests,
    extract_complete_boxed_answer,
    parse_boxed_math_answer,
    select_console_probe_indices,
    select_stratified_probe_indices,
)


WorkerType = Type[Worker]


@dataclass(frozen=True)
class Agent12CurriculumState:
    phase: str
    phase_id: int
    phase_step: int
    teacher_attempt_probability: float
    worker_question_probability: float


def expand_agent12_teacher_attempt_batch(
    batch_dict: Dict,
    *,
    attempts_key: str,
    attempt_key: str,
    expected_attempts: int,
) -> Tuple[Dict, int]:
    """Expand each question into one prompt group per Agent 0 attempt."""
    if attempts_key not in batch_dict:
        raise ValueError(
            f"Agent 1/2 curriculum requires dataset column {attempts_key!r}"
        )
    if expected_attempts <= 0:
        raise ValueError("expected_attempts must be positive")

    attempts_batch = batch_dict[attempts_key]
    base_batch_size = len(batch_dict['question'])
    source_indices = []
    selected_attempts = []
    selected_attempt_indices = []

    for source_index in range(base_batch_size):
        attempts = attempts_batch[source_index]
        if isinstance(attempts, str):
            attempts = [attempts]
        elif isinstance(attempts, np.ndarray):
            attempts = attempts.tolist()
        elif isinstance(attempts, (list, tuple)):
            attempts = list(attempts)
        else:
            attempts = []
        if len(attempts) != expected_attempts:
            raise ValueError(
                f"Question at batch index {source_index} has {len(attempts)} "
                f"teacher attempts; expected {expected_attempts}. Regenerate "
                "the teacher parquet with matching TEACHER_ROLLOUT_N."
            )
        for attempt_index, attempt in enumerate(attempts):
            source_indices.append(source_index)
            selected_attempts.append(
                attempt.strip() if isinstance(attempt, str) else ""
            )
            selected_attempt_indices.append(attempt_index)

    source_indices_np = np.asarray(source_indices, dtype=np.int64)
    expanded_batch = {}
    for key, value in batch_dict.items():
        if key == attempts_key:
            continue
        if isinstance(value, torch.Tensor):
            index_tensor = torch.as_tensor(
                source_indices_np,
                dtype=torch.long,
                device=value.device,
            )
            expanded_batch[key] = value.index_select(0, index_tensor)
        elif isinstance(value, np.ndarray):
            expanded_batch[key] = value[source_indices_np]
        else:
            raise TypeError(
                f"Cannot expand batch field {key!r} of type "
                f"{type(value).__name__}"
            )

    expanded_batch[attempt_key] = np.asarray(
        selected_attempts,
        dtype=object,
    )
    expanded_batch['teacher_attempt_index'] = np.asarray(
        selected_attempt_indices,
        dtype=np.int64,
    )
    expanded_batch['teacher_attempt_count'] = np.full(
        len(selected_attempts),
        expected_attempts,
        dtype=np.int64,
    )
    return expanded_batch, base_batch_size


def compute_agent12_curriculum_state(
    global_step: int,
    *,
    worker_bootstrap_steps: int,
    decomposer_transfer_steps: int,
    worker_question_fade_steps: int,
    worker_question_final_probability: float,
    worker_question_bootstrap_probability: float = 1.0,
) -> Agent12CurriculumState:
    """Return curriculum phase and context probabilities for one PPO step."""
    completed_steps = max(int(global_step) - 1, 0)
    worker_bootstrap_steps = max(int(worker_bootstrap_steps), 0)
    decomposer_transfer_steps = max(int(decomposer_transfer_steps), 0)
    worker_question_fade_steps = max(int(worker_question_fade_steps), 0)
    final_probability = min(
        max(float(worker_question_final_probability), 0.0),
        1.0,
    )
    bootstrap_probability = min(
        max(float(worker_question_bootstrap_probability), 0.0),
        1.0,
    )

    if completed_steps < worker_bootstrap_steps:
        return Agent12CurriculumState(
            phase="worker_bootstrap",
            phase_id=0,
            phase_step=completed_steps,
            teacher_attempt_probability=1.0,
            worker_question_probability=bootstrap_probability,
        )

    transfer_step = completed_steps - worker_bootstrap_steps
    if transfer_step < decomposer_transfer_steps:
        teacher_probability = 1.0 - (
            transfer_step / max(decomposer_transfer_steps, 1)
        )
        return Agent12CurriculumState(
            phase="decomposer_transfer",
            phase_id=1,
            phase_step=transfer_step,
            teacher_attempt_probability=teacher_probability,
            worker_question_probability=1.0,
        )

    joint_step = transfer_step - decomposer_transfer_steps
    if worker_question_fade_steps <= 0:
        worker_question_probability = final_probability
    else:
        fade_fraction = min(joint_step / worker_question_fade_steps, 1.0)
        worker_question_probability = 1.0 - (
            (1.0 - final_probability) * fade_fraction
        )
    return Agent12CurriculumState(
        phase="joint",
        phase_id=2,
        phase_step=joint_step,
        teacher_attempt_probability=0.0,
        worker_question_probability=worker_question_probability,
    )


def select_agent12_training_role(
    curriculum_state: Agent12CurriculumState,
    *,
    decomposer_role: str,
    worker_roles,
    switch_freq: int,
    train_decomposer: bool,
) -> str:
    """Choose the role updated by one Agent 1/2 curriculum step."""
    if not worker_roles:
        raise ValueError("Agent 1/2 curriculum requires worker roles")
    switch_freq = max(int(switch_freq), 1)

    if curriculum_state.phase == 'worker_bootstrap' or not train_decomposer:
        role_index = curriculum_state.phase_step // switch_freq
        return worker_roles[role_index % len(worker_roles)]
    if curriculum_state.phase == 'decomposer_transfer':
        return decomposer_role

    block_index = curriculum_state.phase_step // switch_freq
    if block_index % 2 == 0:
        return decomposer_role
    worker_index = (block_index // 2) % len(worker_roles)
    return worker_roles[worker_index]


def compute_usable_filtered_prompt_count(
    available_prompt_count,
    target_prompt_count,
    prompt_minibatch_size,
    allow_sub_minibatch=False,
):
    """Return a full target batch or the largest usable partial batch."""
    available_prompt_count = int(available_prompt_count)
    target_prompt_count = int(target_prompt_count)
    prompt_minibatch_size = int(prompt_minibatch_size)
    if available_prompt_count < 0:
        raise ValueError("available_prompt_count must be non-negative")
    if target_prompt_count <= 0 or prompt_minibatch_size <= 0:
        raise ValueError(
            "target_prompt_count and prompt_minibatch_size must be positive"
        )
    if available_prompt_count >= target_prompt_count:
        return target_prompt_count
    complete_minibatch_count = (
        available_prompt_count // prompt_minibatch_size
    ) * prompt_minibatch_size
    if complete_minibatch_count > 0:
        return complete_minibatch_count
    if allow_sub_minibatch:
        return available_prompt_count
    return 0


def build_trainable_rank_partitions(
    sequence_lengths,
    world_size,
    trainable_mask,
):
    """Balance tokens while placing a trainable sample on every DP rank."""

    sequence_lengths = [int(length) for length in sequence_lengths]
    trainable_mask = [bool(value) for value in trainable_mask]
    world_size = int(world_size)
    sample_count = len(sequence_lengths)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if len(trainable_mask) != sample_count:
        raise ValueError("trainable_mask must match sequence_lengths")
    if sample_count == 0 or sample_count % world_size != 0:
        return None

    rank_capacity = sample_count // world_size
    trainable_indices = [
        idx for idx, is_trainable in enumerate(trainable_mask)
        if is_trainable
    ]
    if len(trainable_indices) < world_size:
        return None

    trainable_indices.sort(
        key=lambda idx: (-sequence_lengths[idx], idx)
    )
    seed_indices = trainable_indices[:world_size]
    seed_set = set(seed_indices)
    partitions = [[idx] for idx in seed_indices]
    partition_lengths = [sequence_lengths[idx] for idx in seed_indices]

    remaining_indices = [
        idx for idx in range(sample_count)
        if idx not in seed_set
    ]
    remaining_indices.sort(
        key=lambda idx: (-sequence_lengths[idx], idx)
    )
    for idx in remaining_indices:
        available_ranks = [
            rank_idx for rank_idx, partition in enumerate(partitions)
            if len(partition) < rank_capacity
        ]
        rank_idx = min(
            available_ranks,
            key=lambda candidate: (
                partition_lengths[candidate],
                len(partitions[candidate]),
                candidate,
            ),
        )
        partitions[rank_idx].append(idx)
        partition_lengths[rank_idx] += sequence_lengths[idx]

    return partitions


def extract_round_score_role_outputs(
    histories,
    num_turns,
    agent_roles,
    score_role,
    max_num_turns,
    score_roles=None,
):
    """Extract the scoring-role candidate produced in every executed round."""

    if score_role not in agent_roles:
        raise ValueError(f"score_role={score_role!r} is not present in agent_roles")
    if len(histories) != len(num_turns):
        raise ValueError("histories and num_turns must have equal lengths")
    if score_roles is not None and len(score_roles) != len(histories):
        raise ValueError("score_roles and histories must have equal lengths")

    batch_size = len(histories)
    outputs = [["" for _ in range(batch_size)] for _ in range(max_num_turns)]
    executed = torch.zeros((batch_size, max_num_turns), dtype=torch.bool)
    roles_per_round = len(agent_roles)

    for sample_idx, (sample_history, sample_num_turns) in enumerate(
        zip(histories, num_turns)
    ):
        sample_score_role = (
            score_roles[sample_idx]
            if score_roles is not None
            and score_roles[sample_idx] in agent_roles
            else score_role
        )
        sample_num_turns = min(int(sample_num_turns), max_num_turns)
        for turn_idx in range(sample_num_turns):
            start = turn_idx * roles_per_round
            turn_history = sample_history[start:start + roles_per_round]
            final_message = next(
                (
                    message
                    for message in turn_history
                    if isinstance(message, dict)
                    and message.get("role") == sample_score_role
                ),
                None,
            )
            if final_message is None:
                raise ValueError(
                    f"Missing {sample_score_role} history slot for sample={sample_idx}, "
                    f"turn={turn_idx + 1}"
                )
            content = final_message.get("content", "")
            outputs[turn_idx][sample_idx] = content if isinstance(content, str) else ""
            executed[sample_idx, turn_idx] = bool(
                final_message.get("executed", True)
            )

    return outputs, executed


def select_score_role_rewards(
    reward_tensor,
    default_score_role,
    agent_roles,
    dynamic_score_roles=None,
):
    """Select each sample's terminal-role reward tensor."""

    default_key = f'{default_score_role}_turn_level_reward'
    if dynamic_score_roles is None:
        return reward_tensor[default_key]
    if len(dynamic_score_roles) != reward_tensor[default_key].shape[0]:
        raise ValueError(
            "dynamic_score_roles and reward batch must have equal lengths"
        )
    return torch.stack([
        reward_tensor[
            f'{dynamic_role}_turn_level_reward'
            if dynamic_role in agent_roles
            else default_key
        ][sample_idx]
        for sample_idx, dynamic_role in enumerate(dynamic_score_roles)
    ])


def carry_forward_round_scores(candidate_scores, executed):
    """Build answer-state scores after each round, preserving stopped samples."""

    if candidate_scores.shape != executed.shape:
        raise ValueError("candidate_scores and executed must have equal shapes")
    if candidate_scores.ndim != 2:
        raise ValueError("candidate_scores must have shape [batch, turns]")

    states = torch.zeros_like(candidate_scores)
    current = torch.zeros(
        candidate_scores.shape[0],
        dtype=candidate_scores.dtype,
        device=candidate_scores.device,
    )
    for turn_idx in range(candidate_scores.shape[1]):
        current = torch.where(
            executed[:, turn_idx].to(device=current.device),
            candidate_scores[:, turn_idx],
            current,
        )
        states[:, turn_idx] = current
    return states


def compute_round_transition_metrics(round_state_scores, round_executed):
    """Measure answer repairs and regressions between actually executed rounds."""

    if round_state_scores.shape != round_executed.shape:
        raise ValueError("round_state_scores and round_executed must have equal shapes")
    if round_state_scores.ndim != 2:
        raise ValueError("round_state_scores must have shape [batch, turns]")

    metrics = {}
    total_correct_to_wrong = 0
    total_wrong_to_correct = 0
    total_previous_correct = 0
    total_previous_wrong = 0

    for turn_idx in range(1, round_state_scores.shape[1]):
        transition_name = f"round_{turn_idx}_to_{turn_idx + 1}"
        eligible = round_executed[:, turn_idx].bool()
        previous_correct = round_state_scores[:, turn_idx - 1] > 0.0
        current_correct = round_state_scores[:, turn_idx] > 0.0
        correct_to_wrong_count = int(
            (eligible & previous_correct & ~current_correct).sum().item()
        )
        wrong_to_correct_count = int(
            (eligible & ~previous_correct & current_correct).sum().item()
        )
        previous_correct_count = int((eligible & previous_correct).sum().item())
        previous_wrong_count = int((eligible & ~previous_correct).sum().item())
        prefix = f"val/transitions/{transition_name}"
        metrics[f"{prefix}/correct_to_wrong_count"] = float(correct_to_wrong_count)
        metrics[f"{prefix}/wrong_to_correct_count"] = float(wrong_to_correct_count)
        metrics[f"{prefix}/previous_correct_count"] = float(previous_correct_count)
        metrics[f"{prefix}/previous_wrong_count"] = float(previous_wrong_count)
        metrics[f"{prefix}/correct_to_wrong_rate"] = (
            correct_to_wrong_count / previous_correct_count
            if previous_correct_count else 0.0
        )
        metrics[f"{prefix}/wrong_to_correct_rate"] = (
            wrong_to_correct_count / previous_wrong_count
            if previous_wrong_count else 0.0
        )

        total_correct_to_wrong += correct_to_wrong_count
        total_wrong_to_correct += wrong_to_correct_count
        total_previous_correct += previous_correct_count
        total_previous_wrong += previous_wrong_count

    prefix = "val/transitions"
    metrics[f"{prefix}/correct_to_wrong_count"] = float(total_correct_to_wrong)
    metrics[f"{prefix}/wrong_to_correct_count"] = float(total_wrong_to_correct)
    metrics[f"{prefix}/previous_correct_count"] = float(total_previous_correct)
    metrics[f"{prefix}/previous_wrong_count"] = float(total_previous_wrong)
    metrics[f"{prefix}/correct_to_wrong_rate"] = (
        total_correct_to_wrong / total_previous_correct
        if total_previous_correct else 0.0
    )
    metrics[f"{prefix}/wrong_to_correct_rate"] = (
        total_wrong_to_correct / total_previous_wrong
        if total_previous_wrong else 0.0
    )
    return metrics


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Agent0_Actor = 0
    Agent0_Rollout = 1
    Agent0_ActorRollout = 2
    Agent0_Critic = 3
    Agent0_RefPolicy = 4
    Agent0_RewardModel = 5
    Agent0_ActorRolloutRef = 6
    Agent1_Actor = 7
    Agent1_Rollout = 8
    Agent1_ActorRollout = 9
    Agent1_Critic = 10
    Agent1_RefPolicy = 11
    Agent1_RewardModel = 12
    Agent1_ActorRolloutRef = 13


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        import time
        import logging
        
        timeout = 300  # 300 seconds = 5 minutes
        retry_interval = 10  # seconds
        start_time = time.time()
        
        while True:
            node_available_resources = ray.state.available_resources_per_node()
            node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

            # check total required gpus can be satisfied
            total_available_gpus = sum(node_available_gpus.values())
            total_required_gpus = sum(
                [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
            
            # Check for resource pool satisfaction
            pools_satisfied = True
            error_msgs = []
            
            if total_available_gpus < total_required_gpus:
                pools_satisfied = False
                error_msgs.append(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")
            else:
                # check each resource pool can be satisfied, O(#resource_pools * #nodes)
                for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
                    num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
                    for node, available_gpus in node_available_gpus.items():
                        if available_gpus >= num_gpus:
                            node_available_gpus[node] -= num_gpus
                            num_nodes -= 1
                            if num_nodes == 0:
                                break
                    if num_nodes > 0:
                        pools_satisfied = False
                        error_msgs.append(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster")
            
            # If all resources are available, return
            if pools_satisfied:
                return
            
            # Check if we've exceeded the timeout
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                # If we've timed out, raise the error with all collected error messages
                raise ValueError(f"Resource allocation timed out after {timeout} seconds. Errors: {'; '.join(error_msgs)}")
            
            # Log waiting message and sleep before retry
            remaining = timeout - elapsed_time
            logging.info(f"Waiting for resources to be available. Retrying in {retry_interval} seconds. Timeout in {remaining:.1f} seconds.")
            logging.info(f"Resource issues: {'; '.join(error_msgs)}")
            time.sleep(retry_interval)


from verl.utils.torch_functional import masked_mean
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        raise NotImplementedError('GAE is not implemented yet')
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_sparse_rewards = torch.zeros_like(data.batch['token_level_rewards'])
        grpo_sparse_rewards[:, -1] = data.batch['turn_level_reward'].sum(-1)
        index = data.non_tensor_batch['uid']
        # responses = data.batch['responses']
        # response_length = responses.size(-1)
        # attention_mask = data.batch['attention_mask']
        # response_mask = attention_mask[:, -response_length:]
        step_mask = data.batch['step_ids'] != -100
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=grpo_sparse_rewards,
                                                                        eos_mask=step_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch['token_level_rewards']
        # responses = data.batch['responses']
        # response_length = responses.size(-1)
        # attention_mask = data.batch['attention_mask']
        # response_mask = attention_mask[:, -response_length:]
        step_mask = data.batch['step_ids'] != -100
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=step_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        raise NotImplementedError('REMAX is not implemented yet')
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        raise NotImplementedError('RLOO is not implemented yet')
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data

def get_last_index_of_turn(step_ids: torch.Tensor, i_turn: int) -> torch.Tensor:
    mask = step_ids == i_turn
    seq_tensor = torch.arange(step_ids.size(1), device=step_ids.device).expand_as(step_ids)
    last_indices = torch.where(mask, seq_tensor, torch.tensor(-1, device=step_ids.device))
    last_indices, _ = torch.max(last_indices, dim=1)  # shape: [bsz]
    
    return last_indices

def compute_token_level_scores(data: DataProto, dtype=torch.float32)->torch.Tensor:
    max_num_turns = data.meta_info['max_num_turns']
    bsz, seq_len = data.batch['input_ids'].shape
    token_level_scores = torch.zeros((bsz, seq_len), dtype=torch.float32)
    step_ids = data.batch['step_ids']
    turn_level_return = data.batch['turn_level_return']
    for i_turn in range(max_num_turns):
        last_indices = get_last_index_of_turn(step_ids, i_turn)
        valid_mask = last_indices != -1
        # Hierarchical rollouts optimize the latest factual prompt/action pair.
        # Earlier turn ids can therefore be intentionally absent.
        if (~valid_mask).all():
            continue
        batch_indices = torch.arange(bsz)
        token_level_scores[batch_indices[valid_mask], last_indices[valid_mask]] = \
            turn_level_return[:, i_turn][valid_mask]
    
    return token_level_scores


def split_batch_for_agents(data: DataProto) -> Dict[str, DataProto]:
    agent_roles = data.meta_info['agent_roles']
    new_tensor_batches = {role: {} for role in agent_roles}
    for key in data.batch.keys():
        role_name = ''
        for role in agent_roles:
            if role in key:
                role_name = role
                break
        if role_name in agent_roles:
            v_name = key.replace(f'{role_name}_', '')
            new_tensor_batches[role_name][v_name] = data.batch[key]
        else:
            for role in agent_roles:
                new_tensor_batches[role][key] = data.batch[key].clone()

    for role in agent_roles:
        new_tensor_batches[role]['num_turns'] = torch.tensor(
            data.non_tensor_batch['num_turns'].tolist()
        )
        if 'labels' in new_tensor_batches[role]:
            role_idx = agent_roles.index(role)
            new_tensor_batches[role]['agent_role_ids'] = torch.full_like(
                new_tensor_batches[role]['labels'], fill_value=role_idx, dtype=torch.long
            )
    
    # build non_tensor_batch
    new_non_tensor_batches = {role: {} for role in agent_roles}
    uid_list = data.non_tensor_batch['uid'].tolist()
    for role in agent_roles:
        new_non_tensor_batches[role]['uid'] = np.array(uid_list, dtype=object)
    
    all_agent_batches = {}
    for role in agent_roles:
        all_agent_batches[role] = DataProto.from_dict(new_tensor_batches[role], 
                                                      non_tensors=new_non_tensor_batches[role], 
                                                      meta_info=data.meta_info)

    return all_agent_batches

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayReMASeparatedTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self._current_train_agent = None
        self._current_train_agent_idx = None

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.Agent0_ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'
            assert Role.Agent1_ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.Agent0_RefPolicy in role_worker_mapping
        self.use_rm = Role.Agent0_RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        
        self._create_dataloader()
        self._init_scoped_c3_grpo()
        self._init_prefix_probe()

    def _tokenizer_for_role(self, role):
        return getattr(self, 'role_tokenizers', {}).get(role, self.tokenizer)

    def _init_scoped_c3_grpo(self):
        hierarchy_config = self.config.algorithm.get('hierarchy', {})
        scoped_config = (
            hierarchy_config.get('scoped_c3_grpo', {})
            if hierarchy_config else {}
        )
        self.scoped_c3_grpo_config = (
            OmegaConf.to_container(scoped_config, resolve=True)
            if scoped_config else {}
        )
        self.scoped_c3_grpo_enabled = bool(
            self.scoped_c3_grpo_config.get('enable', False)
        )
        if not self.scoped_c3_grpo_enabled:
            return

        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise ValueError(
                "Scoped C3 GRPO requires algorithm.adv_estimator='grpo'"
            )
        if not bool(hierarchy_config.get('enable', False)):
            raise ValueError(
                "Scoped C3 GRPO requires algorithm.hierarchy.enable=True"
            )
        rollout_config = self.config.actor_rollout_ref.rollout
        if int(rollout_config.get('n', 1)) < 2:
            raise ValueError("Scoped C3 GRPO requires rollout.n >= 2")
        if not bool(rollout_config.get('do_sample', True)):
            raise ValueError("Scoped C3 GRPO requires stochastic rollouts")
        if float(rollout_config.get('temperature', 1.0)) <= 0.0:
            raise ValueError("Scoped C3 GRPO requires rollout.temperature > 0")

        max_num_turns = int(rollout_config.get('max_num_turns', 1))
        configured_branch_turn = self.scoped_c3_grpo_config.get(
            'branch_turn',
            'latest',
        )
        if configured_branch_turn != 'latest':
            try:
                branch_turn = int(configured_branch_turn)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Scoped C3 GRPO branch_turn must be 'latest' or an integer"
                ) from exc
            if not 0 <= branch_turn < max_num_turns:
                raise ValueError(
                    f"Scoped C3 GRPO branch_turn={branch_turn} must be in "
                    f"[0, {max_num_turns})"
                )

        stage_roles = list(hierarchy_config.get('stage_roles', []))
        if not stage_roles:
            stage_roles = [
                f"worker_stage_{stage_idx}"
                for stage_idx in range(
                    1,
                    int(hierarchy_config.get('num_worker_stages', 0)) + 1,
                )
            ]
        configured_agent_roles = {
            hierarchy_config.get('decomposer_role', 'decomposer'),
            hierarchy_config.get('selector_role', 'selector'),
            *stage_roles,
        }
        train_agent_roles = set(
            hierarchy_config.get(
                'train_agent_roles',
                configured_agent_roles,
            )
        )
        unknown_train_roles = train_agent_roles - configured_agent_roles
        if unknown_train_roles:
            raise ValueError(
                "Scoped C3 GRPO train_agent_roles contains roles absent from "
                f"the hierarchy: {sorted(unknown_train_roles)}"
            )

    def _init_prefix_probe(self):
        hierarchy_config = self._get_hierarchy_config()
        probe_config = self.scoped_c3_grpo_config.get('prefix_probe', {})
        self.prefix_probe_config = probe_config or {}
        self.prefix_probe_enabled = bool(
            self.prefix_probe_config.get('enable', False)
        )
        if not self.prefix_probe_enabled:
            return
        if not self.scoped_c3_grpo_enabled:
            raise ValueError("Prefix probe requires scoped C3 GRPO")
        if int(self.config.actor_rollout_ref.rollout.get('max_num_turns', 1)) != 1:
            raise ValueError(
                "Plan/equivalence gates require max_num_turns=1; multi-round "
                "C3 needs probes aligned to the selected branch action."
            )
        try:
            from math_verify import parse, verify
        except ImportError as exc:
            raise ImportError("Leakage gates require math-verify on the trainer node") from exc

        solver_role = self.prefix_probe_config.get(
            'solver_role',
            hierarchy_config.get('decomposer_role', 'decomposer'),
        )
        if solver_role not in hierarchy_config.get('agent_roles', []):
            raise ValueError(
                f"Prefix-probe solver_role={solver_role!r} is absent from "
                "algorithm.hierarchy.agent_roles"
            )
        if int(self.prefix_probe_config.get('max_new_tokens', 256)) <= 0:
            raise ValueError("Prefix-probe max_new_tokens must be positive")
        if int(self.prefix_probe_config.get('diagnostic_samples', 2)) < 0:
            raise ValueError(
                "Prefix-probe diagnostic_samples must be non-negative"
            )
        if int(
            self.prefix_probe_config.get('validation_max_samples', 128)
        ) < 0:
            raise ValueError(
                "Prefix-probe validation_max_samples must be non-negative"
            )

    @staticmethod
    def _prefix_probe_context_key(message, data_source, ground_truth, extra_info):
        try:
            extra_key = json.dumps(extra_info, sort_keys=True, default=str)
        except (TypeError, ValueError):
            extra_key = repr(extra_info)
        return (
            str(message),
            str(data_source),
            repr(ground_truth),
            extra_key,
        )

    def _generate_prefix_probe_responses(self, messages):
        if not messages:
            return [], [], []

        hierarchy_config = self._get_hierarchy_config()
        solver_role = self.prefix_probe_config.get(
            'solver_role',
            hierarchy_config.get('decomposer_role', 'decomposer'),
        )
        system_prompt = str(self.prefix_probe_config.get(
            'system_prompt',
            "Infer and solve the mathematical task recoverable from the "
            "provided information. Do not ask for more context. Show concise "
            "reasoning and always end with exactly one non-empty, complete "
            "\\boxed{...}. If the requested answer cannot be inferred, end "
            "with \\boxed{UNKNOWN}.",
        ))
        chats = [[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ] for message in messages]

        tokenizer = self._tokenizer_for_role(solver_role)
        probe_meta_info = {
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
            'recompute_log_prob': False,
            'do_sample': False,
            'validate': True,
            'is_multi_turn': False,
        }
        rollout = self.actor_rollout_wg[solver_role]
        dummy_batch = DataProto.from_dict({
            'dummy_tensor': torch.arange(max(len(messages), 1)),
        })
        padded_dummy_batch, _ = pad_dataproto_to_divisor(
            dummy_batch,
            rollout.world_size,
        )
        rollout.enter_generate_context(padded_dummy_batch)
        try:
            (
                outputs,
                num_gen_tokens,
                stop_reasons,
                _,
                output_token_ids,
            ) = (
                self.multi_agent_rollout._generate_from_chat_list(
                    solver_role,
                    chats,
                    self.role_tokenizers,
                    probe_meta_info,
                    response_length=(
                        self.config.actor_rollout_ref.rollout.response_length
                    ),
                    max_new_tokens=int(
                        self.prefix_probe_config.get('max_new_tokens', 256)
                    ),
                )
            )
        finally:
            rollout.exit_generate_context(padded_dummy_batch)

        # Validation-style vLLM calls may return token ids while leaving
        # ``RequestOutput.text`` empty when detokenization is disabled.
        decoded_outputs = tokenizer.batch_decode(
            output_token_ids,
            skip_special_tokens=True,
        )
        outputs = [
            output
            if isinstance(output, str) and output.strip()
            else decoded_output
            for output, decoded_output in zip(outputs, decoded_outputs)
        ]
        return outputs, num_gen_tokens, stop_reasons

    @staticmethod
    def _record_leakage_metrics(data_batch, metrics, prefix):
        """Accumulate gates and comparison coverage over generation chunks."""

        raw = data_batch.batch['prefix_probe_raw_outcome_score'].float().cpu().numpy()
        gated = data_batch.batch['prefix_probe_gated_outcome_score'].float().cpu().numpy()
        reasons = data_batch.non_tensor_batch['prefix_probe_rejection_reason']

        def accumulate(name, values):
            count_key = f'{prefix}/{name}/count'
            previous = float(metrics.get(count_key, 0.0))
            total = previous + len(values)
            if not total:
                return
            key = f'{prefix}/{name}/rate'
            metrics[key] = (
                float(metrics.get(key, 0.0)) * previous + float(sum(values))
            ) / total
            metrics[count_key] = total

        count_key = f'{prefix}/trajectory_count'
        previous_count = float(metrics.get(count_key, 0.0))
        total_count = previous_count + len(data_batch)
        if not total_count:
            return
        rates = {
            'raw_accuracy': float(raw.sum()),
            'gated_accuracy': float(gated.sum()),
            'removed_correct_rate': float(((raw > 0) & (gated <= 0)).sum()),
            'plan_eligible_rate': float(data_batch.batch['prefix_probe_plan_eligible'].sum()),
            'gate_valid_rate': float(data_batch.batch['prefix_probe_gate_valid'].sum()),
            'update_eligible_rate': float(data_batch.batch['prefix_probe_collaboration_eligible'].sum()),
            'subtask_count_mean': float(data_batch.batch['prefix_probe_subtask_count'].sum()),
        }
        for reason in (
            'single_subtask', 'plan_probe_invalid', 'plan_recoverable',
            'comparison_invalid', 'equivalent_answer',
        ):
            rates[f'rejected/{reason}_rate'] = float(sum(value == reason for value in reasons))
        for key, summed_value in rates.items():
            key = f'{prefix}/{key}'
            metrics[key] = (
                float(metrics.get(key, 0.0)) * previous_count + summed_value
            ) / total_count
        metrics[count_key] = total_count

        requested = data_batch.batch['prefix_probe_decomposer_requested'].bool().cpu().numpy()
        valid = data_batch.batch['prefix_probe_decomposer_valid'].bool().cpu().numpy()
        ld = np.asarray(data_batch.non_tensor_batch['prefix_probe_decomposer_score'], dtype=float)
        accumulate('decomposer', [float(value > 0) for value in ld[valid]])
        accumulate('decomposer_valid', valid[requested].tolist())
        for field in ('nonempty', 'length_stopped'):
            values = data_batch.batch[f'prefix_probe_decomposer_{field}'].bool().cpu().numpy()
            accumulate(f'decomposer_{field}', values[requested].tolist())

        by_role = defaultdict(list)
        for comparisons in data_batch.non_tensor_batch['prefix_probe_worker_comparisons']:
            for comparison in comparisons:
                by_role[comparison['role']].append(comparison['match'])
        all_matches = [match for values in by_role.values() for match in values]
        for name, matches in [('worker_match', all_matches)] + [
            (f'roles/{role}/worker_match', values) for role, values in by_role.items()
        ]:
            accumulate(f'{name}_valid', [value is not None for value in matches])
            accumulate(name, [float(value) for value in matches if value is not None])

    @staticmethod
    def _print_validation_leakage_examples(data_batch, scope='val'):
        """Print L_D, each measured E_k, and the exact gate rejection reason."""

        def format_score(value):
            return 'n/a' if not np.isfinite(float(value)) else f'{float(value):.1f}'

        for index in range(len(data_batch)):
            comparisons = data_batch.non_tensor_batch['prefix_probe_worker_comparisons'][index]
            matches = ','.join(
                f"{item['role']}:" + ('n/a' if item['match'] is None else str(int(item['match'])))
                for item in comparisons
            ) or 'exempt'
            questions = data_batch.non_tensor_batch.get('question', [''] * len(data_batch))
            question = ' '.join(str(questions[index]).split())
            print(
                f'[leakage/example] scope={scope} '
                f'raw_score={float(data_batch.batch["prefix_probe_raw_outcome_score"][index]):.1f} '
                f'gated_score={float(data_batch.batch["prefix_probe_gated_outcome_score"][index]):.1f} '
                f'terminal_role={data_batch.non_tensor_batch["terminal_stage_role"][index]} '
                f'subtasks={int(data_batch.batch["prefix_probe_subtask_count"][index])} '
                f'Ld={format_score(data_batch.non_tensor_batch["prefix_probe_decomposer_score"][index])} '
                f'E=[{matches}] '
                f'plan_eligible={int(data_batch.batch["prefix_probe_plan_eligible"][index])} '
                f'update_eligible={int(data_batch.batch["prefix_probe_collaboration_eligible"][index])} '
                f'reason={data_batch.non_tensor_batch["prefix_probe_rejection_reason"][index]} '
                f'question={question!r}',
                flush=True,
            )

    def _print_leakage_summary(self, metrics, scope, metric_prefix=None):
        prefix = metric_prefix or f'reward/leakage/{scope}'

        def value(name):
            result = metrics.get(f'{prefix}/{name}')
            return 'n/a' if result is None else f'{float(result):.3f}'

        print(
            f'[leakage/{scope}] step={self.global_steps} role={self._current_train_agent} '
            f'raw_acc={value("raw_accuracy")} gated_acc={value("gated_accuracy")} '
            f'plan_eligible={value("plan_eligible_rate")} '
            f'update_eligible={value("update_eligible_rate")} '
            f'Ld={value("decomposer/rate")} '
            f'Ld_valid={value("decomposer_valid/rate")} '
            f'worker_match={value("worker_match/rate")} '
            f'comparison_valid={value("worker_match_valid/rate")} '
            f'single_subtask={value("rejected/single_subtask_rate")} '
            f'removed_correct={value("removed_correct_rate")}',
            flush=True,
        )

    def _attach_prefix_probe_signals(
        self,
        data_batch,
        outcome_scores,
        metrics,
        *,
        validation=False,
        probe_reward_fn=None,
    ):
        """Probe plans once per unique input and compare worker/final answers."""

        if not self.prefix_probe_enabled:
            return
        hierarchy = self._get_hierarchy_config()
        decomposer_role = hierarchy.get('decomposer_role', 'decomposer')
        stage_roles = hierarchy.get('stage_roles', [])
        focal_role = None if validation else self._current_train_agent
        histories = data_batch.non_tensor_batch.get('history')
        if histories is None:
            raise ValueError("Prefix probe requires rollout history metadata")
        size = len(data_batch)
        terminal_roles = data_batch.non_tensor_batch.get(
            'terminal_stage_role',
            np.array([hierarchy.get('score_role', '')] * size, dtype=object),
        )
        data_batch.non_tensor_batch['terminal_stage_role'] = np.asarray(terminal_roles, dtype=object)
        probe_reward_fn = probe_reward_fn or self.reward_fn
        requests = collect_prefix_probe_requests(
            histories, terminal_roles, focal_role=focal_role,
            decomposer_role=decomposer_role, stage_roles=stage_roles,
        )
        plan_scores = np.full(size, np.nan, dtype=object)
        plan_responses = np.full(size, '', dtype=object)
        plan_requested = np.zeros(size, dtype=bool)
        plan_nonempty = np.zeros(size, dtype=bool)
        plan_length = np.zeros(size, dtype=bool)

        # Shared C3 prefixes produce identical plans. Probe each unique plan
        # only once in this batch and broadcast the measurement to its group.
        unique_items, unique_indices, index_by_key = [], [], {}
        extra_infos = data_batch.non_tensor_batch.get(
            'extra_info', np.array([None] * size, dtype=object),
        )
        for request in requests:
            index = request.sample_index
            item = (
                request.message,
                data_batch.non_tensor_batch['data_source'][index],
                data_batch.non_tensor_batch['reward_model'][index]['ground_truth'],
                extra_infos[index],
            )
            key = self._prefix_probe_context_key(*item)
            if key not in index_by_key:
                index_by_key[key] = len(unique_items)
                unique_items.append(item)
            unique_indices.append(index_by_key[key])

        if unique_items:
            responses, tokens, stops = self._generate_prefix_probe_responses(
                [item[0] for item in unique_items]
            )
            if not (len(responses) == len(tokens) == len(stops) == len(unique_items)):
                raise RuntimeError("Prefix probe returned an incomplete generation batch")
            answers = [extract_complete_boxed_answer(response) for response in responses]
            unknown = [
                answer is not None and ''.join(answer.lower().split()) in {
                    'unknown', r'\text{unknown}', r'\mathrm{unknown}',
                }
                for answer in answers
            ]
            scores = [0.0 if missing else float('nan') for missing in unknown]
            valid_answers = [
                missing or parse_boxed_math_answer(response) is not None
                for missing, response in zip(unknown, responses)
            ]
            to_score = [i for i, valid in enumerate(valid_answers) if valid and not unknown[i]]
            if to_score:
                measured = probe_reward_fn.score_responses(
                    [unique_items[i][1] for i in to_score],
                    [responses[i] for i in to_score],
                    [unique_items[i][2] for i in to_score],
                    [unique_items[i][3] for i in to_score],
                    show_progress=False,
                    timeout_score=float('nan'),
                )
                if len(measured) != len(to_score):
                    raise RuntimeError("Prefix-probe scorer returned an incomplete score batch")
                for index, score in zip(to_score, measured):
                    scores[index] = float(score)
            diagnosed = 0
            for i, valid in enumerate(valid_answers):
                if not valid and diagnosed < int(self.prefix_probe_config.get('diagnostic_samples', 2)):
                    diagnosed += 1
                    print(
                        f'[prefix_probe/invalid] source=decomposer stop={stops[i]} '
                        f'tokens={tokens[i]} response={responses[i][:240]!r}', flush=True,
                    )
            for request, source_index in zip(requests, unique_indices):
                index = request.sample_index
                plan_scores[index] = scores[source_index]
                plan_responses[index] = responses[source_index]
                plan_requested[index] = True
                plan_nonempty[index] = bool(responses[source_index].strip())
                plan_length[index] = stops[source_index] == 'length'

        counts = []
        comparison_scores = np.full(size, np.nan, dtype=object)
        comparison_required = []
        comparison_records = np.empty(size, dtype=object)
        comparison_cache = {}
        for index, (history, terminal_role) in enumerate(zip(histories, terminal_roles)):
            records = answer_round_records(history, str(terminal_role), decomposer_role)
            counts.append(count_planned_subtasks(records, decomposer_role))
            executed = {
                record.get('role'): record for record in records
                if record.get('executed', True) is not False
            }
            terminal_output = executed.get(str(terminal_role), {}).get('content', '')
            if validation:
                roles = [
                    role for role in stage_roles
                    if role != str(terminal_role) and role in executed
                ]
            else:
                roles = (
                    [focal_role] if focal_role in stage_roles
                    and focal_role != str(terminal_role) else []
                )
            comparison_required.append(bool(roles))
            comparisons = []
            for role in roles:
                output = executed.get(role, {}).get('content', '')
                key = (output, terminal_output)
                if key not in comparison_cache:
                    comparison_cache[key] = compare_worker_final_answers(output, terminal_output)
                comparisons.append({'role': role, 'match': comparison_cache[key]})
            comparison_records[index] = comparisons
            if comparisons and all(item['match'] is not None for item in comparisons):
                comparison_scores[index] = float(any(item['match'] for item in comparisons))

        raw = outcome_scores.detach().float().cpu()
        gate = apply_prefix_probe_gate(
            raw.tolist(), plan_scores.tolist(), counts,
            comparison_scores.tolist(), comparison_required,
        )
        data_batch.non_tensor_batch.update({
            'prefix_probe_decomposer_score': plan_scores,
            'prefix_probe_decomposer_response': plan_responses,
            'prefix_probe_worker_match_score': comparison_scores,
            'prefix_probe_worker_comparisons': comparison_records,
            'prefix_probe_rejection_reason': np.asarray(gate.rejection_reasons, dtype=object),
        })
        tensor_values = {
            'prefix_probe_raw_outcome_score': raw,
            'prefix_probe_gated_outcome_score': torch.tensor(gate.outcome_scores, dtype=torch.float32),
            'prefix_probe_gate_valid': torch.tensor(gate.valid_mask, dtype=torch.bool),
            'prefix_probe_collaboration_eligible': torch.tensor(gate.collaboration_eligible_mask, dtype=torch.bool),
            'prefix_probe_plan_eligible': torch.tensor(gate.plan_eligible_mask, dtype=torch.bool),
            'prefix_probe_subtask_count': torch.tensor(counts, dtype=torch.long),
            'prefix_probe_decomposer_requested': torch.from_numpy(plan_requested),
            'prefix_probe_decomposer_valid': torch.tensor([bool(np.isfinite(float(v))) for v in plan_scores], dtype=torch.bool),
            'prefix_probe_decomposer_nonempty': torch.from_numpy(plan_nonempty),
            'prefix_probe_decomposer_length_stopped': torch.from_numpy(plan_length),
            'prefix_probe_worker_match_requested': torch.tensor(comparison_required, dtype=torch.bool),
            'prefix_probe_worker_match_valid': torch.tensor([bool(np.isfinite(float(v))) for v in comparison_scores], dtype=torch.bool),
        }
        for key, value in tensor_values.items():
            data_batch.batch[key] = value
        scope = 'val' if validation else 'all'
        prefix = 'val/leakage' if validation else 'reward/leakage/all'
        self._record_leakage_metrics(data_batch, metrics, prefix)
        self._print_leakage_summary(metrics, scope, metric_prefix=prefix)
        if not validation:
            indices = select_console_probe_indices(
                data_batch.non_tensor_batch['data_source'].tolist(),
                int(getattr(probe_reward_fn, 'num_examine', 1)),
            )
            if indices:
                self._print_validation_leakage_examples(data_batch[indices], scope='train')

    def _update_prefix_probe_batch_metrics(self, data_batch, metrics):
        self._record_leakage_metrics(data_batch, metrics, 'reward/leakage/train')
        self._print_leakage_summary(metrics, 'train')

    @staticmethod
    def _unpad_role_messages(messages):
        if isinstance(messages, np.ndarray):
            messages = messages.tolist()
        unpadded = [
            dict(message)
            for message in messages
            if isinstance(message, dict) and message.get('role') != 'padding'
        ]
        # Stored conversations include the sampled assistant response. C3
        # compares the role prefix before that response was generated.
        if unpadded and unpadded[-1].get('role') == 'assistant':
            unpadded = unpadded[:-1]
        return unpadded


    def _c3_action_present_mask(self, data_batch, role):
        token_key = f'{role}_action_token_ids'
        batch_size = len(data_batch)
        present = torch.zeros(batch_size, dtype=torch.bool)
        if token_key not in data_batch.non_tensor_batch:
            return present
        for sample_idx, token_ids in enumerate(
            data_batch.non_tensor_batch[token_key]
        ):
            if isinstance(token_ids, np.ndarray):
                token_ids = token_ids.tolist()
            present[sample_idx] = bool(token_ids)
        return present

    def _c3_exact_prefix_mask(self, data_batch, role, candidate_mask, metrics):
        """Verify that every C3 action group shares one factual role prefix."""

        batch_size = len(data_batch)
        exact_mask = torch.zeros(batch_size, dtype=torch.bool)
        chat_key = f'{role}_conversation_history'
        if chat_key not in data_batch.non_tensor_batch:
            return exact_mask

        uid_to_indices = defaultdict(list)
        for sample_idx, uid in enumerate(data_batch.non_tensor_batch['uid']):
            if bool(candidate_mask[sample_idx].item()):
                uid_to_indices[uid].append(sample_idx)

        exact_group_count = 0
        for indices in uid_to_indices.values():
            if len(indices) < 2:
                continue
            canonical_prompts = []
            for sample_idx in indices:
                messages = self._unpad_role_messages(
                    data_batch.non_tensor_batch[chat_key][sample_idx]
                )
                canonical_prompts.append(tuple(
                    (
                        str(message.get('role', '')),
                        str(message.get('content', '')),
                    )
                    for message in messages
                ))
            if len(set(canonical_prompts)) != 1:
                continue
            exact_mask[indices] = True
            exact_group_count += 1

        prefix = f'reward/c3/roles/{role}'
        group_count = len(uid_to_indices)
        metrics[f'{prefix}/exact_prefix_group_rate'] = (
            float(exact_group_count) / float(group_count)
            if group_count else 0.0
        )
        return exact_mask

    @staticmethod
    def _c3_mixed_group_mask(outcome_scores, group_ids, candidate_mask):
        mixed_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
        uid_to_indices = defaultdict(list)
        for sample_idx, uid in enumerate(group_ids):
            if bool(candidate_mask[sample_idx].item()):
                uid_to_indices[uid].append(sample_idx)
        for indices in uid_to_indices.values():
            if len(indices) < 2:
                continue
            scores = outcome_scores[indices]
            if (
                bool(torch.isfinite(scores).all().item())
                and float(scores.max().item()) > float(scores.min().item())
            ):
                mixed_mask[indices] = True
        return mixed_mask

    def _attach_scoped_c3_grpo_signals(
        self,
        data_batch,
        reward_tensor_map,
        metrics,
    ):
        """Attach exact fixed-prefix outcomes used by C3."""

        if not self.scoped_c3_grpo_enabled:
            return

        role = self._current_train_agent
        raw_outcome_scores = reward_tensor_map['acc'].float().cpu()
        outcome_scores = raw_outcome_scores
        prefix_gate_valid = torch.ones_like(
            raw_outcome_scores,
            dtype=torch.bool,
        )
        collaboration_eligible = torch.ones_like(
            raw_outcome_scores,
            dtype=torch.bool,
        )
        if self.prefix_probe_enabled:
            prefix_gate_valid = data_batch.batch[
                'prefix_probe_gate_valid'
            ].bool().cpu()
            collaboration_eligible = data_batch.batch[
                'prefix_probe_collaboration_eligible'
            ].bool().cpu()
        action_present = self._c3_action_present_mask(data_batch, role)
        action_turns = data_batch.non_tensor_batch.get('c3_action_turn')
        if action_turns is None:
            raise ValueError(
                "Scoped C3 rollout did not return c3_action_turn metadata"
            )
        # DataProto concatenation/slicing may preserve scalar metadata in an
        # object-dtype NumPy array. Normalize values before creating a tensor.
        action_turns = torch.tensor(
            [int(value) for value in action_turns],
            dtype=torch.long,
        )
        configured_branch_turn = self.scoped_c3_grpo_config.get(
            'branch_turn',
            'latest',
        )
        expected_branch_turn = (
            int(self.config.actor_rollout_ref.rollout.max_num_turns) - 1
            if configured_branch_turn == 'latest'
            else int(configured_branch_turn)
        )
        wrong_action_turn = action_present & (
            action_turns != expected_branch_turn
        )
        if bool(wrong_action_turn.any().item()):
            raise ValueError(
                "Scoped C3 selected an action from a different turn than "
                f"branch_turn={expected_branch_turn}"
            )
        exact_prefix = self._c3_exact_prefix_mask(
            data_batch,
            role,
            action_present,
            metrics,
        )
        # A failed gate removes gradients, never factual outcomes from the
        # leave-one-out baseline (including failed mathematical comparisons).
        baseline_valid = action_present & exact_prefix
        update_valid = baseline_valid & prefix_gate_valid & collaboration_eligible
        pre_mixed_valid = baseline_valid.clone()
        mixed_mask = torch.ones_like(baseline_valid)
        if bool(
            self.scoped_c3_grpo_config.get('mixed_groups_only', True)
        ):
            mixed_mask = self._c3_mixed_group_mask(
                outcome_scores,
                data_batch.non_tensor_batch['uid'],
                baseline_valid,
            )
            baseline_valid &= mixed_mask
            update_valid &= mixed_mask

        data_batch.batch['scoped_c3_raw_outcome_score'] = raw_outcome_scores
        data_batch.batch['scoped_c3_outcome_score'] = outcome_scores
        data_batch.batch['scoped_c3_causal_valid'] = baseline_valid
        data_batch.batch['scoped_c3_update_mask'] = update_valid

        prefix = f'reward/c3/roles/{role}'
        metrics[f'{prefix}/action_present_rate'] = float(
            action_present.float().mean().item()
        )
        metrics[f'{prefix}/rejected/missing_action_rate'] = float(
            (~action_present).float().mean().item()
        )
        metrics[f'{prefix}/rejected/prefix_mismatch_rate'] = float(
            (action_present & ~exact_prefix).float().mean().item()
        )
        metrics[f'{prefix}/rejected/leakage_gate_rate'] = float(
            (
                action_present
                & exact_prefix
                & prefix_gate_valid
                & ~collaboration_eligible
            )
            .float()
            .mean()
            .item()
        )
        metrics[f'{prefix}/rejected/probe_invalid_rate'] = float(
            (action_present & exact_prefix & ~prefix_gate_valid)
            .float()
            .mean()
            .item()
        )
        metrics[f'{prefix}/rejected/no_outcome_contrast_rate'] = float(
            (pre_mixed_valid & ~mixed_mask).float().mean().item()
        )
        metrics[f'{prefix}/causal_valid_rate'] = float(
            baseline_valid.float().mean().item()
        )
        metrics[f'{prefix}/update_eligible_rate'] = float(
            update_valid.float().mean().item()
        )

    def _compute_scoped_c3_grpo_advantage(self, data_batch, metrics):
        """Build role-local fixed-prefix C3 LOO advantages."""

        estimate = estimate_scoped_c3_grpo(
            data_batch.batch['scoped_c3_outcome_score'].float(),
            data_batch.non_tensor_batch['uid'],
            data_batch.batch['scoped_c3_causal_valid'].bool(),
            update_mask=data_batch.batch['scoped_c3_update_mask'].bool(),
            normalize=bool(
                self.scoped_c3_grpo_config.get(
                    'normalize_advantages',
                    True,
                )
            ),
            epsilon=float(
                self.scoped_c3_grpo_config.get(
                    'normalization_epsilon',
                    1e-6,
                )
            ),
        )
        step_mask = data_batch.batch['step_ids'] != -100
        advantages = (
            estimate.advantage.unsqueeze(-1)
            * step_mask.to(dtype=estimate.advantage.dtype)
        )
        data_batch.batch['advantages'] = advantages
        data_batch.batch['returns'] = advantages.clone()

        direct_rewards = torch.zeros_like(
            data_batch.batch['token_level_rewards'],
            dtype=torch.float32,
        )
        sequence_positions = torch.arange(
            step_mask.shape[1],
            device=step_mask.device,
        ).expand_as(step_mask)
        last_positions = torch.where(
            step_mask,
            sequence_positions,
            torch.full_like(sequence_positions, -1),
        ).max(dim=1).values
        reward_rows = torch.nonzero(
            estimate.effective_mask & (last_positions >= 0),
            as_tuple=False,
        ).flatten()
        if reward_rows.numel() > 0:
            direct_rewards[
                reward_rows,
                last_positions[reward_rows],
            ] = data_batch.batch['scoped_c3_outcome_score'][reward_rows]
        data_batch.batch['token_level_scores'] = direct_rewards
        data_batch.batch['token_level_rewards'] = direct_rewards

        ineffective_rows = ~estimate.effective_mask
        if ineffective_rows.any():
            data_batch.batch['labels'][ineffective_rows] = -100
            data_batch.batch['step_ids'][ineffective_rows] = -100
            data_batch.batch['advantages'][ineffective_rows] = 0.0
            data_batch.batch['returns'][ineffective_rows] = 0.0

        role = self._current_train_agent
        prefix = f'reward/c3/roles/{role}'
        effective = estimate.effective_mask
        advantage = estimate.advantage
        positive = effective & (advantage > 0)
        negative = effective & (advantage < 0)
        for sign, candidates in (('positive', advantage > 0), ('negative', advantage < 0)):
            metrics[f'{prefix}/{sign}_before_gate_count'] = float(candidates.sum().item())
            metrics[f'{prefix}/{sign}_after_gate_count'] = float((effective & candidates).sum().item())
            metrics[f'{prefix}/{sign}_removed_count'] = float((~effective & candidates).sum().item())
        metrics[f'{prefix}/effective_sample_rate'] = float(
            effective.float().mean().item()
        )
        metrics[f'{prefix}/effective_group_count'] = float(len({
            str(data_batch.non_tensor_batch['uid'][sample_idx])
            for sample_idx in torch.nonzero(
                effective,
                as_tuple=False,
            ).flatten().tolist()
        }))
        metrics[f'{prefix}/positive_advantage_rate'] = 0.0
        metrics[f'{prefix}/negative_advantage_rate'] = 0.0
        metrics[f'{prefix}/advantage_std'] = 0.0
        if bool(effective.any().item()):
            effective_count = effective.float().sum()
            metrics[f'{prefix}/positive_advantage_rate'] = float(
                positive.float().sum().div(effective_count).item()
            )
            metrics[f'{prefix}/negative_advantage_rate'] = float(
                negative.float().sum().div(effective_count).item()
            )
            metrics[f'{prefix}/advantage_std'] = (
                float(advantage[effective].std(unbiased=False).item())
                if int(effective.sum().item()) > 1 else 0.0
            )
        print(
            ' '.join([
                '[c3]',
                f'step={self.global_steps}',
                f'role={role}',
                f'effective={metrics[f"{prefix}/effective_sample_rate"]:.3f}',
                f'groups={int(metrics[f"{prefix}/effective_group_count"])}',
                f'adv_std={metrics.get(f"{prefix}/advantage_std", 0.0):.3f}',
            ]),
            flush=True,
        )
        return data_batch


    def _hierarchy_enabled(self) -> bool:
        return bool(self.config.algorithm.get('hierarchy', {}).get('enable', False))

    def _get_hierarchy_config(self) -> Dict:
        hierarchy_config = self.config.algorithm.get('hierarchy', {})
        hierarchy_config = OmegaConf.to_container(hierarchy_config, resolve=True) if hierarchy_config else {}
        if not hierarchy_config:
            return {}

        decomposer_role = hierarchy_config.get('decomposer_role', 'decomposer')
        selector_role = hierarchy_config.get('selector_role', 'selector')
        if 'stage_roles' not in hierarchy_config:
            num_worker_stages = int(hierarchy_config.get('num_worker_stages', 0))
            hierarchy_config['stage_roles'] = [
                f'worker_stage_{idx}'
                for idx in range(1, num_worker_stages + 1)
            ]
        stage_roles = hierarchy_config.get('stage_roles', [])
        hierarchy_config.setdefault(
            'agent_roles',
            [decomposer_role, selector_role] + stage_roles,
        )
        hierarchy_config.setdefault('train_agent_roles', hierarchy_config['agent_roles'])
        if stage_roles:
            hierarchy_config.setdefault('score_role', stage_roles[-1])
        return hierarchy_config

    def _get_rollout_agent_roles(self):
        if self._hierarchy_enabled():
            return self._get_hierarchy_config()['agent_roles']
        return self.config.algorithm.get('switch_agent', {}).get('agent_roles', ['meta_thinking', 'reasoning'])

    def _get_train_agent_roles(self):
        if self._hierarchy_enabled():
            hierarchy_config = self._get_hierarchy_config()
            return hierarchy_config.get('train_agent_roles', hierarchy_config['agent_roles'])
        return self.config.algorithm.get('switch_agent', {}).get('agent_roles', ['meta_thinking', 'reasoning'])

    def _get_start_agent(self):
        if self._hierarchy_enabled():
            hierarchy_config = self._get_hierarchy_config()
            return hierarchy_config.get('start_agent', self._get_train_agent_roles()[0])
        return self.config.algorithm.get('switch_agent', {}).get('start_agent', self._get_train_agent_roles()[0])

    def _get_score_role(self):
        if self._hierarchy_enabled():
            hierarchy_config = self._get_hierarchy_config()
            return hierarchy_config.get('score_role', hierarchy_config['agent_roles'][-1])
        return 'reasoning'

    def _get_agent12_curriculum_config(self) -> Dict:
        if not self._hierarchy_enabled():
            return {}
        hierarchy_config = self._get_hierarchy_config()
        return hierarchy_config.get('agent12_curriculum', {}) or {}

    def _get_agent12_curriculum_state(self) -> Agent12CurriculumState:
        curriculum = self._get_agent12_curriculum_config()
        return compute_agent12_curriculum_state(
            self.global_steps,
            worker_bootstrap_steps=curriculum.get(
                'worker_bootstrap_steps', 0
            ),
            decomposer_transfer_steps=curriculum.get(
                'decomposer_transfer_steps', 0
            ),
            worker_question_fade_steps=curriculum.get(
                'worker_question_fade_steps', 0
            ),
            worker_question_final_probability=curriculum.get(
                'worker_question_final_probability', 1.0
            ),
            worker_question_bootstrap_probability=curriculum.get(
                'worker_question_bootstrap_probability', 1.0
            ),
        )

    def _build_rollout_meta_info(self, max_num_turns: int) -> Dict:
        if self._hierarchy_enabled():
            from prompt.math.hierarchical_mamrp import build_hierarchical_system_prompts
            from prompt import FINISH_FLAG
            hierarchy_config = self._get_hierarchy_config()
            return {
                'agent_roles': hierarchy_config['agent_roles'],
                'finish_flag': FINISH_FLAG,
                'system_prompts': build_hierarchical_system_prompts(
                    hierarchy_config.get('stage_roles'),
                ),
                'max_num_turns': max_num_turns,
                'hierarchy': hierarchy_config,
            }

        if max_num_turns > 1:
            from prompt.math.multi_turn_subtask_mamrp import MTA_SYSTEM_PRMOPT, RA_SYSTEM_PRMOPT
            from prompt import FINISH_FLAG
            return {
                'agent_roles': self._get_rollout_agent_roles(),
                'finish_flag': FINISH_FLAG,
                'system_prompts': {
                    'meta_thinking': MTA_SYSTEM_PRMOPT,
                    'reasoning': RA_SYSTEM_PRMOPT,
                },
                'max_num_turns': max_num_turns,
            }

        from prompt.math.single_turn_mamrp import MTA_SYSTEM_PRMOPT, RA_SYSTEM_PRMOPT
        return {
            'agent_roles': self._get_rollout_agent_roles(),
            'finish_flag': None,
            'system_prompts': {
                'meta_thinking': MTA_SYSTEM_PRMOPT,
                'reasoning': RA_SYSTEM_PRMOPT,
            },
            'max_num_turns': max_num_turns,
        }

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        effective_train_prompt_batch_size = int(config.data.train_batch_size)

        if self._hierarchy_enabled():
            hierarchy_config = self._get_hierarchy_config()
            routing_mode = str(
                hierarchy_config.get('routing_mode', 'selector')
            ).lower()
            if routing_mode == 'derive_verify':
                stage_roles = list(hierarchy_config.get('stage_roles', []))
                if len(stage_roles) != 3:
                    raise ValueError(
                        "hierarchy.routing_mode=derive_verify requires exactly "
                        "three stages: derive, verify/repair, and final"
                    )
                if int(config.actor_rollout_ref.rollout.max_num_turns) != 1:
                    raise ValueError(
                        "hierarchy.routing_mode=derive_verify currently requires "
                        "actor_rollout_ref.rollout.max_num_turns=1"
                    )
                if bool(
                    hierarchy_config.get('accept_revise', {}).get('enable', False)
                ):
                    raise ValueError(
                        "hierarchy.accept_revise.enable must be False in the "
                        "single-round derive_verify protocol"
                    )
                selector_role = hierarchy_config.get('selector_role', 'selector')
                if selector_role in hierarchy_config.get('train_agent_roles', []):
                    raise ValueError(
                        "The selector is deterministic in derive_verify mode and "
                        "must not appear in hierarchy.train_agent_roles"
                    )
            elif routing_mode == 'sequential_plan':
                stage_roles = list(hierarchy_config.get('stage_roles', []))
                terminal_worker_as_answer = bool(
                    hierarchy_config.get('terminal_worker_as_answer', False)
                )
                worker_stage_count = (
                    len(stage_roles)
                    if terminal_worker_as_answer
                    else len(stage_roles) - 1
                )
                if worker_stage_count < 1:
                    raise ValueError(
                        "hierarchy.routing_mode=sequential_plan requires at "
                        + (
                            "least one worker stage"
                            if terminal_worker_as_answer
                            else "least one non-final worker and one final stage"
                        )
                    )
                max_planned_subtasks = int(
                    hierarchy_config.get('max_planned_subtasks', 0)
                )
                if max_planned_subtasks != worker_stage_count:
                    raise ValueError(
                        "hierarchy.max_planned_subtasks must equal the available "
                        "planned-worker capacity in sequential_plan mode"
                    )
                if int(config.actor_rollout_ref.rollout.max_num_turns) != 1:
                    raise ValueError(
                        "hierarchy.routing_mode=sequential_plan currently "
                        "requires actor_rollout_ref.rollout.max_num_turns=1"
                    )
                if bool(
                    hierarchy_config.get('accept_revise', {}).get(
                        'enable', False
                    )
                ):
                    raise ValueError(
                        "hierarchy.accept_revise.enable must be False in the "
                        "single-round sequential_plan protocol"
                    )
                selector_role = hierarchy_config.get(
                    'selector_role', 'selector'
                )
                if selector_role in hierarchy_config.get(
                    'train_agent_roles', []
                ):
                    raise ValueError(
                        "The selector is deterministic in sequential_plan mode "
                        "and must not appear in hierarchy.train_agent_roles"
                    )
            curriculum = hierarchy_config.get('agent12_curriculum', {}) or {}
            if bool(curriculum.get('enable', False)):
                if bool(
                    config.algorithm.get('final_worker_curriculum', {}).get(
                        'enable', False
                    )
                ):
                    raise ValueError(
                        "agent12_curriculum and final_worker_curriculum cannot "
                        "be enabled together"
                    )
                decomposer_role = hierarchy_config.get(
                    'decomposer_role', 'decomposer'
                )
                train_roles = list(
                    hierarchy_config.get('train_agent_roles', [])
                )
                train_decomposer = bool(
                    curriculum.get('train_decomposer', True)
                )
                if train_decomposer and decomposer_role not in train_roles:
                    raise ValueError(
                        "agent12_curriculum.train_decomposer=True requires "
                        "the decomposer in hierarchy.train_agent_roles"
                    )
                if not train_decomposer and decomposer_role in train_roles:
                    raise ValueError(
                        "agent12_curriculum.train_decomposer=False requires "
                        "the decomposer to be absent from "
                        "hierarchy.train_agent_roles"
                    )
                worker_train_roles = [
                    role for role in train_roles
                    if role not in {
                        decomposer_role,
                        hierarchy_config.get('selector_role', 'selector'),
                    }
                ]
                if not worker_train_roles:
                    raise ValueError(
                        "agent12_curriculum requires at least one trainable "
                        "worker stage"
                    )
                for key in (
                    'worker_bootstrap_steps',
                    'decomposer_transfer_steps',
                    'worker_question_fade_steps',
                ):
                    if int(curriculum.get(key, 0)) < 0:
                        raise ValueError(f"agent12_curriculum.{key} must be non-negative")
                for key in (
                    'worker_question_bootstrap_probability',
                    'worker_question_final_probability',
                    'worker_question_eval_probability',
                ):
                    probability = float(curriculum.get(key, 1.0))
                    if not 0.0 <= probability <= 1.0:
                        raise ValueError(
                            f"agent12_curriculum.{key} must be in [0, 1]"
                        )
                if not str(
                    curriculum.get('teacher_attempts_key', '')
                ).strip():
                    raise ValueError(
                        "agent12_curriculum.teacher_attempts_key cannot be empty"
                    )
                if not str(
                    curriculum.get('teacher_attempt_key', '')
                ).strip():
                    raise ValueError(
                        "agent12_curriculum.teacher_attempt_key cannot be empty"
                    )
                if not bool(
                    curriculum.get('expand_all_teacher_attempts', False)
                ):
                    raise ValueError(
                        "agent12_curriculum requires "
                        "expand_all_teacher_attempts=True"
                    )
                for key in (
                    'teacher_attempts_per_question',
                    'optimizer_prompt_batch_size',
                ):
                    if int(curriculum.get(key, 0)) <= 0:
                        raise ValueError(
                            f"agent12_curriculum.{key} must be positive"
                        )
                expected_prompt_batch_size = (
                    int(config.data.train_batch_size)
                    * int(curriculum.get('teacher_attempts_per_question', 0))
                )
                if int(curriculum.get('optimizer_prompt_batch_size', 0)) != (
                    expected_prompt_batch_size
                ):
                    raise ValueError(
                        "agent12_curriculum.optimizer_prompt_batch_size must "
                        "equal data.train_batch_size * "
                        "teacher_attempts_per_question so no teacher attempt "
                        "group is dropped"
                    )
                effective_train_prompt_batch_size = expected_prompt_batch_size
                if int(curriculum.get('teacher_attempt_max_chars', 0)) <= 0:
                    raise ValueError(
                        "agent12_curriculum.teacher_attempt_max_chars must be "
                        "positive"
                    )

        # 1. Check total batch size for data correctness
        real_train_batch_size = (
            effective_train_prompt_batch_size
            * config.actor_rollout_ref.rollout.n
        )
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert effective_train_prompt_batch_size >= (
                config.actor_rollout_ref.actor.ppo_mini_batch_size
            )
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert effective_train_prompt_batch_size >= (
                config.critic.ppo_mini_batch_size
            )
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"
        
        if config.algorithm.filter_groups.enable:
            assert config.actor_rollout_ref.rollout.n > 1
        final_worker_curriculum = config.algorithm.get('final_worker_curriculum', {})
        if final_worker_curriculum.get('enable', False):
            assert self._hierarchy_enabled(), \
                "algorithm.final_worker_curriculum requires algorithm.hierarchy.enable=True"
            score_role = self._get_score_role()
            assert score_role in self._get_train_agent_roles(), \
                f"score_role={score_role} must be in train_agent_roles for final worker curriculum"
        
        if config.actor_rollout_ref.actor.clip_mode == 'turn':
            assert config.actor_rollout_ref.actor.agg_mode != 'token'
        
        if config.reward_model.get('use_format_reward', False):
            assert config.actor_rollout_ref.rollout.max_num_turns == 1, \
                "use_format_reward only support max_num_turns==1"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                        #  tokenizer=self.tokenizer,
                                        #  processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                        #  image_key=self.config.data.get('image_key', 'images'),
                                        #  max_prompt_length=self.config.data.max_prompt_length,
                                        #  filter_prompts=True,
                                        #  return_raw_chat=self.config.data.get('return_raw_chat', False),
                                        #  truncation=self.config.data.get('truncation', 'error'),
                                        #  filter_overlong_prompts=self.config.data.filter_overlong_prompts
                                        )
        # TODO(ziyu): try to check data in dataset.
        #### UNUSED NOW
        # assert self.train_dataset.truncation == self.config.data.get(
        #     'truncation', 'error'
        # ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        #########################################################
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                    #    tokenizer=self.tokenizer,
                                    #    processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       #    image_key=self.config.data.get('image_key', 'images'),
                                       #    max_prompt_length=self.config.data.max_prompt_length,
                                       #    filter_prompts=True,
                                       #    return_raw_chat=self.config.data.get('return_raw_chat', False),
                                    #    truncation=self.config.data.get('truncation', 'error'),
                                    #    filter_overlong_prompts=self.config.data.filter_overlong_prompts
                                       )
        # TODO(ziyu): try to check data in dataset.     
        ##### UNUSED NOW
        # assert self.val_dataset.truncation == self.config.data.get(
        #     'truncation', 'error'
        # ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        #########################################################
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            # batch_size=len(self.val_dataset),
            batch_size=self.config.data.val_batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        # assert len(
        #     self.val_dataloader
        # ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores, groundtruths, histories):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, groundtruths, histories))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        acc_tensor_lst = []
        round_state_score_lst = []
        round_executed_lst = []
        data_source_lst = []
        num_turns_lst = []
        history_lst = []
        sample_groundtruths = []
        completion_tokens_lst = []
        accepted_lst = []
        decision_valid_lst = []
        attempted_round_count_lst = []
        candidate_source_round_lst = []
        probe_histories = []
        probe_terminal_roles = []
        probe_data_sources = []
        probe_reward_models = []
        probe_extra_infos = []
        probe_strata = []
        val_leakage_metrics = {}
        validation_max_samples = int(
            self.prefix_probe_config.get('validation_max_samples', 128)
        ) if self.prefix_probe_enabled else 0
        live_probe_indices = set()

        def as_object_array(values):
            result = np.empty(len(values), dtype=object)
            result[:] = values
            return result

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        max_num_turns = self.config.actor_rollout_ref.rollout.max_num_turns
        rollout_meta_info = self._build_rollout_meta_info(max_num_turns)
        score_role = self._get_score_role()
        score_tokenizer = self._tokenizer_for_role(score_role)
        accept_revise_config = (
            self._get_hierarchy_config().get('accept_revise', {}) or {}
        )
        accept_revise_enabled = bool(
            accept_revise_config.get('enable', False)
        )

        for test_data in self.val_dataloader:
            # test_batch = DataProto.from_single_dict(test_data)
            dummy_tensor = torch.arange(0, len(test_data['question']))
            test_data['batch_idx'] = dummy_tensor
            test_batch: DataProto = DataProto.from_single_dict(test_data, meta_info=rollout_meta_info)          

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            # input_ids = test_batch.batch['input_ids']
            # input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            input_texts = test_batch.non_tensor_batch['question']
            sample_inputs.extend(input_texts)

            # Store original ground truth if available
            ground_truths = [x['ground_truth'] for x in test_data['reward_model'].tolist()]

            sample_groundtruths.extend(ground_truths)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                raise NotImplementedError('validation is not implemented yet')
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.select(
                        batch_keys=['batch_idx'], 
                        non_tensor_batch_keys=['question'], 
                        meta_info_keys=['agent_roles', 'finish_flag', 'system_prompts', 'hierarchy'],
                        deepcopy=True
                    )
            
            test_gen_batch.meta_info.update({
                'eos_token_id': score_tokenizer.eos_token_id,
                'pad_token_id': score_tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            })
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size

            pad_role = rollout_meta_info['agent_roles'][0]
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg[pad_role].world_size)
            test_output_gen_batch_padded = self.multi_turn_generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            # output_ids = test_output_gen_batch.batch['responses']
            # output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            output_texts = test_output_gen_batch.non_tensor_batch['response']
            sample_outputs.extend(output_texts)
            accepted_lst.extend(
                bool(value)
                for value in test_output_gen_batch.non_tensor_batch.get(
                    'accepted',
                    np.zeros(len(test_output_gen_batch), dtype=bool),
                )
            )
            decision_valid_lst.extend(
                bool(value)
                for value in test_output_gen_batch.non_tensor_batch.get(
                    'accept_revise_decision_valid',
                    np.ones(len(test_output_gen_batch), dtype=bool),
                )
            )
            attempted_round_count_lst.extend(
                sum(bool(item) for item in values)
                for values in test_output_gen_batch.non_tensor_batch.get(
                    'round_attempted',
                    np.array([[] for _ in range(len(test_output_gen_batch))], dtype=object),
                )
            )
            candidate_source_round_lst.extend(
                int(value) + 1 if int(value) >= 0 else 0
                for value in test_output_gen_batch.non_tensor_batch.get(
                    'candidate_source_turn',
                    np.full(len(test_output_gen_batch), -1, dtype=np.int64),
                )
            )

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info['mask_unfinished_reward'] = self.config.reward_model.mask_unfinished_reward
            test_batch.meta_info['use_format_reward'] = self.config.reward_model.get('use_format_reward', False)
            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)
            dynamic_score_roles = test_output_gen_batch.non_tensor_batch.get(
                'terminal_stage_role'
            )
            score_reward_tensor = select_score_role_rewards(
                reward_tensor,
                score_role,
                rollout_meta_info['agent_roles'],
                dynamic_score_roles,
            )
            reward_tensor_lst.append(score_reward_tensor)
            acc_tensor_lst.append(reward_tensor['acc'])

            histories = test_output_gen_batch.non_tensor_batch['history'].tolist()
            if self.prefix_probe_enabled:
                probe_batch_offset = len(probe_histories)
                terminal_role_values = dynamic_score_roles
                if terminal_role_values is None:
                    terminal_role_values = np.array(
                        [score_role] * len(test_batch),
                        dtype=object,
                    )
                probe_histories.extend(histories)
                probe_terminal_roles.extend(
                    str(value) for value in terminal_role_values
                )
                probe_data_sources.extend(
                    test_batch.non_tensor_batch['data_source'].tolist()
                )
                probe_reward_models.extend(
                    test_batch.non_tensor_batch['reward_model'].tolist()
                )
                probe_extra_infos.extend(
                    test_batch.non_tensor_batch.get(
                        'extra_info',
                        np.array([None] * len(test_batch), dtype=object),
                    ).tolist()
                )
                probe_strata.extend(
                    str(value)
                    for value in test_batch.non_tensor_batch.get(
                        'subset',
                        test_batch.non_tensor_batch['data_source'],
                    )
                )

                # Reproduce the reward manager's first-N selection so the
                # leakage line immediately following its example describes
                # the same trajectory. These probes count toward the global
                # validation budget and are not generated again below.
                remaining_live_budget = (
                    validation_max_samples - len(live_probe_indices)
                )
                console_local_indices = select_console_probe_indices(
                    test_batch.non_tensor_batch['data_source'].tolist(),
                    int(getattr(self.val_reward_fn, 'num_examine', 1)),
                )[:max(remaining_live_budget, 0)]
                if console_local_indices:
                    console_global_indices = [
                        probe_batch_offset + index
                        for index in console_local_indices
                    ]
                    live_probe_indices.update(console_global_indices)
                    batch_extra_infos = test_batch.non_tensor_batch.get(
                        'extra_info',
                        np.array([None] * len(test_batch), dtype=object),
                    )
                    live_probe_batch = DataProto.from_dict(
                        tensors={
                            'batch_idx': torch.arange(
                                len(console_local_indices)
                            ),
                        },
                        non_tensors={
                            'question': as_object_array([
                                test_batch.non_tensor_batch['question'][index]
                                for index in console_local_indices
                            ]),
                            'history': as_object_array([
                                histories[index]
                                for index in console_local_indices
                            ]),
                            'terminal_stage_role': as_object_array([
                                terminal_role_values[index]
                                for index in console_local_indices
                            ]),
                            'data_source': as_object_array([
                                test_batch.non_tensor_batch['data_source'][index]
                                for index in console_local_indices
                            ]),
                            'reward_model': as_object_array([
                                test_batch.non_tensor_batch['reward_model'][index]
                                for index in console_local_indices
                            ]),
                            'extra_info': as_object_array([
                                batch_extra_infos[index]
                                for index in console_local_indices
                            ]),
                        },
                    )
                    live_outcome_scores = reward_tensor['acc'][
                        console_local_indices
                    ].detach().cpu()
                    self._attach_prefix_probe_signals(
                        live_probe_batch,
                        live_outcome_scores,
                        val_leakage_metrics,
                        validation=True,
                        probe_reward_fn=self.val_reward_fn,
                    )
                    self._print_validation_leakage_examples(
                        live_probe_batch
                    )
            turn_counts = [
                int(value)
                for value in test_output_gen_batch.non_tensor_batch['num_turns'].tolist()
            ]
            round_outputs, round_executed = extract_round_score_role_outputs(
                histories,
                turn_counts,
                rollout_meta_info['agent_roles'],
                score_role,
                max_num_turns,
                score_roles=dynamic_score_roles,
            )
            candidate_scores = torch.zeros(
                (len(test_batch), max_num_turns),
                dtype=torch.float32,
            )
            final_scores = reward_tensor['acc'].detach().cpu().float()
            candidate_source_turns = test_output_gen_batch.non_tensor_batch.get(
                'candidate_source_turn',
                np.array([count - 1 for count in turn_counts], dtype=np.int64),
            )
            for sample_idx, source_turn in enumerate(candidate_source_turns):
                source_turn = int(source_turn)
                if 0 <= source_turn < max_num_turns:
                    candidate_scores[
                        sample_idx,
                        source_turn,
                    ] = final_scores[sample_idx]

            data_sources_for_scoring = test_batch.non_tensor_batch['data_source']
            reward_models_for_scoring = test_batch.non_tensor_batch['reward_model']
            extra_infos_for_scoring = test_batch.non_tensor_batch.get(
                'extra_info',
                np.array([None] * len(test_batch), dtype=object),
            )
            flat_data_sources = []
            flat_responses = []
            flat_ground_truths = []
            flat_extra_infos = []
            flat_locations = []
            for turn_idx in range(max_num_turns):
                for sample_idx in range(len(test_batch)):
                    if not bool(round_executed[sample_idx, turn_idx].item()):
                        continue
                    if turn_idx == int(candidate_source_turns[sample_idx]):
                        continue
                    flat_data_sources.append(data_sources_for_scoring[sample_idx])
                    flat_responses.append(round_outputs[turn_idx][sample_idx])
                    flat_ground_truths.append(
                        reward_models_for_scoring[sample_idx]['ground_truth']
                    )
                    flat_extra_infos.append(extra_infos_for_scoring[sample_idx])
                    flat_locations.append((sample_idx, turn_idx))

            flat_round_scores = self.val_reward_fn.score_responses(
                flat_data_sources,
                flat_responses,
                flat_ground_truths,
                flat_extra_infos,
                show_progress=False,
            )
            for (sample_idx, turn_idx), score in zip(
                flat_locations, flat_round_scores
            ):
                candidate_scores[sample_idx, turn_idx] = float(score)

            round_state_score_lst.append(
                carry_forward_round_scores(candidate_scores, round_executed)
            )
            round_executed_lst.append(round_executed)

            # Store scores
            scores = score_reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            num_turns = torch.tensor(test_output_gen_batch.non_tensor_batch['num_turns'].tolist(), dtype=torch.float32, device="cpu")
            num_turns_lst.append(num_turns)
            turn_level_completion_tokens = None
            for role in rollout_meta_info['agent_roles']:
                role_tokens = test_output_gen_batch.batch[f'{role}_num_gen_tokens'].cpu()
                turn_level_completion_tokens = role_tokens if turn_level_completion_tokens is None else turn_level_completion_tokens + role_tokens
            completion_tokens = turn_level_completion_tokens.sum(dim=-1)
            completion_tokens_lst.append(completion_tokens)

            # not use `data_source`, use `subset` instead
            data_source_lst.append(test_batch.non_tensor_batch.get(
                'subset',
                ['unknown'] * score_reward_tensor.shape[0],
            ))
            
            history_lst.append(histories)

        remaining_probe_budget = max(
            validation_max_samples - len(live_probe_indices),
            0,
        )
        remaining_probe_indices = [
            index
            for index in range(len(probe_strata))
            if index not in live_probe_indices
        ]
        selected_remaining_offsets = select_stratified_probe_indices(
            [probe_strata[index] for index in remaining_probe_indices],
            remaining_probe_budget,
        )
        probe_indices = [
            remaining_probe_indices[offset]
            for offset in selected_remaining_offsets
        ]
        if probe_indices:
            probe_batch = DataProto.from_dict(
                tensors={
                    'batch_idx': torch.arange(len(probe_indices)),
                },
                non_tensors={
                    'history': as_object_array([
                        probe_histories[index] for index in probe_indices
                    ]),
                    'terminal_stage_role': as_object_array([
                        probe_terminal_roles[index] for index in probe_indices
                    ]),
                    'data_source': as_object_array([
                        probe_data_sources[index] for index in probe_indices
                    ]),
                    'reward_model': as_object_array([
                        probe_reward_models[index] for index in probe_indices
                    ]),
                    'extra_info': as_object_array([
                        probe_extra_infos[index] for index in probe_indices
                    ]),
                },
            )
            all_acc_scores = torch.cat(acc_tensor_lst, dim=0).cpu()
            probe_outcome_scores = all_acc_scores[
                torch.tensor(probe_indices, dtype=torch.long)
            ]
            self._attach_prefix_probe_signals(
                probe_batch,
                probe_outcome_scores,
                val_leakage_metrics,
                validation=True,
                probe_reward_fn=self.val_reward_fn,
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, groundtruths=sample_groundtruths, histories=history_lst)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        acc_tensor = torch.cat(acc_tensor_lst, dim=0).cpu() #(batch_size,)
        round_state_scores = torch.cat(round_state_score_lst, dim=0).cpu()
        round_executed = torch.cat(round_executed_lst, dim=0).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_acc = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
            if data_source not in data_source_acc:
                data_source_acc[data_source] = []
            data_source_acc[data_source].append(acc_tensor[i].item())


        metric_dict = {}
        if not self.scoped_c3_grpo_enabled:
            for data_source, rewards in data_source_reward.items():
                metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
        for data_source, accs in data_source_acc.items():
            metric_dict[f'val/acc/{data_source}'] = np.mean(accs)
        metric_dict.update(val_leakage_metrics)

        # ``round_N_acc`` is the correctness of the answer state after N
        # rounds. Samples that stopped earlier retain their latest candidate.
        if max_num_turns > 1:
            for turn_idx in range(max_num_turns):
                round_number = turn_idx + 1
                metric_dict[f'val/round_{round_number}_acc'] = (
                    round_state_scores[:, turn_idx].float().mean().item()
                )
                metric_dict[f'val/round_{round_number}_executed_rate'] = (
                    round_executed[:, turn_idx].float().mean().item()
                )
                for data_source in data_source_acc:
                    source_mask = torch.from_numpy(data_sources == data_source)
                    metric_dict[f'val/round_{round_number}_acc/{data_source}'] = (
                        round_state_scores[source_mask, turn_idx]
                        .float()
                        .mean()
                        .item()
                    )

            metric_dict.update(
                compute_round_transition_metrics(
                    round_state_scores,
                    round_executed,
                )
            )
        if accept_revise_enabled and accepted_lst:
            metric_dict['val/accept_revise/accept_rate'] = float(np.mean(accepted_lst))
            metric_dict['val/accept_revise/decision_valid_rate'] = float(
                np.mean(decision_valid_lst)
            )
            metric_dict['val/accept_revise/attempted_round_count'] = float(
                np.mean(attempted_round_count_lst)
            )
            metric_dict['val/accept_revise/candidate_source_round'] = float(
                np.mean(candidate_source_round_lst)
            )
            accepted_indices = [
                idx for idx, value in enumerate(accepted_lst) if value
            ]
            if accepted_indices:
                metric_dict['val/accept_revise/accepted_acc'] = float(
                    acc_tensor[accepted_indices].float().mean().item()
                )
        
        # Add num_turns and completion_tokens metrics
        if num_turns_lst:
            num_turns_tensor = torch.cat(num_turns_lst, dim=0)
            metric_dict['val/num_turns/mean'] = num_turns_tensor.float().mean().item()
            metric_dict['val/num_turns/max'] = num_turns_tensor.max().item()
            metric_dict['val/num_turns/min'] = num_turns_tensor.min().item()
        
        if completion_tokens_lst:
            completion_tokens_tensor = torch.cat(completion_tokens_lst, dim=0)
            metric_dict['val/completion_tokens/mean'] = completion_tokens_tensor.float().mean().item()
            metric_dict['val/completion_tokens/max'] = completion_tokens_tensor.max().item()
            metric_dict['val/completion_tokens/min'] = completion_tokens_tensor.min().item()

        # Save generation results to a JSON file
        if self.config.trainer.get('save_val_generations', False):
            output_dir = Path(self.config.trainer.default_local_dir) / 'eval_records'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f'val_step_{self.global_steps}.jsonl'
            
            # Concatenate history lists from different batches
            all_histories = []
            for history_batch in history_lst:
                all_histories.extend(history_batch)
            
            results_to_save = []
            for inp, outp, gt, hist, score in zip(sample_inputs, sample_outputs, sample_groundtruths, all_histories, sample_scores):
                unpad_history = [x for x in hist if x['role'] != 'padding']
                results_to_save.append({
                    'question': inp,
                    'answer': outp, 
                    'groundtruth': gt,
                    'history': unpad_history,
                    'score': score
                })
            
            with jsonlines.open(output_file, 'w') as writer:
                writer.write_all(results_to_save)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            switch_config = self.config.algorithm.get('switch_agent', {})
            default_model_path = self.config.actor_rollout_ref.model.path
            model_paths = list(switch_config.get('model_paths', [default_model_path, default_model_path]))
            default_remove_padding = self.config.actor_rollout_ref.model.get('use_remove_padding', False)
            model_remove_padding = switch_config.get('model_use_remove_padding', None)
            if model_remove_padding is not None and len(model_remove_padding) != len(model_paths):
                raise ValueError(
                    "algorithm.switch_agent.model_use_remove_padding must have "
                    "the same length as algorithm.switch_agent.model_paths"
                )

            def use_remove_padding_for(model_idx):
                if model_remove_padding is None:
                    return default_remove_padding
                return model_remove_padding[model_idx]

            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Agent0_ActorRollout)
            agent0_config = copy.deepcopy(self.config.actor_rollout_ref)
            agent0_config.model.path = model_paths[0]
            agent0_config.model.use_remove_padding = use_remove_padding_for(0)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Agent0_ActorRollout],
                                                     config=agent0_config,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['agent0_actor_rollout'] = actor_rollout_cls

            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Agent1_ActorRollout)
            agent1_config = copy.deepcopy(self.config.actor_rollout_ref)
            agent1_config.model.path = model_paths[1] if len(model_paths) > 1 else model_paths[0]
            agent1_config.model.use_remove_padding = use_remove_padding_for(1 if len(model_paths) > 1 else 0)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Agent1_ActorRollout],
                                                     config=agent1_config,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['agent1_actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            raise NotImplementedError
            # resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            # critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            # self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            raise NotImplementedError
            # resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            # ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
            #                                       config=self.config.actor_rollout_ref,
            #                                       role='ref')
            # self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()
        
        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg0 = all_wg['agent0_actor_rollout']
        self.actor_rollout_wg0.init_model()
        
        self.actor_rollout_wg1 = all_wg['agent1_actor_rollout']
        self.actor_rollout_wg1.init_model()

        from verl.utils import hf_tokenizer
        tokenizer_by_path = {str(default_model_path): self.tokenizer}
        model_tokenizers = []
        for model_path in model_paths:
            model_path = str(model_path)
            if model_path not in tokenizer_by_path:
                tokenizer_by_path[model_path] = hf_tokenizer(model_path)
            model_tokenizers.append(tokenizer_by_path[model_path])
        
        self.actor_rollout_wg = {
            'meta_thinking': self.actor_rollout_wg0,
            'reasoning': self.actor_rollout_wg1,
        }
        if self._hierarchy_enabled():
            hierarchy_config = self._get_hierarchy_config()
            decomposer_role = hierarchy_config.get('decomposer_role', 'decomposer')
            self.actor_rollout_wg = {}
            for role in hierarchy_config['agent_roles']:
                self.actor_rollout_wg[role] = self.actor_rollout_wg0 if role == decomposer_role else self.actor_rollout_wg1

        worker_tokenizer = model_tokenizers[1] if len(model_tokenizers) > 1 else model_tokenizers[0]
        self.role_tokenizers = {
            role: model_tokenizers[0] if worker_group is self.actor_rollout_wg0 else worker_tokenizer
            for role, worker_group in self.actor_rollout_wg.items()
        }

        self.multi_agent_rollout = MultiAgentRollout(
            self.config.actor_rollout_ref.rollout,
            self.role_tokenizers,
            self.actor_rollout_wg,
        )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')

        print(f'local_global_step_folder: {local_global_step_folder}')
        import gc; gc.collect()
        for role, wg in self.actor_rollout_wg.items():
            actor_local_path = os.path.join(local_global_step_folder, f'{role}/actor')

            actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', f'{role}/actor')
            wg.save_checkpoint(actor_local_path,
                               actor_remote_path,
                               self.global_steps,
                               remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            for role, wg in self.critic_wg:
                critic_local_path = os.path.join(local_global_step_folder, f'{role}/critic')
                critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                    self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', f'{role}/critic')
                wg.save_checkpoint(critic_local_path,
                                            critic_remote_path,
                                            self.global_steps,
                                            remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        # load actor
        for role, wg in self.actor_rollout_wg.items():
            actor_path = os.path.join(global_step_folder, f'{role}/actor')
            wg.load_checkpoint(actor_path,
                                del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            for role, wg in self.critic_wg.items():
                critic_path = os.path.join(global_step_folder, f'{role}/critic')
                wg.load_checkpoint(critic_path,
                                    del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # A chunked online run intentionally switches to a new immutable shard
        # between sessions, while retaining model and optimizer state.
        restore_dataloader = bool(
            self.config.trainer.get('restore_dataloader_on_resume', True)
        )
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if restore_dataloader and os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        elif restore_dataloader:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")
        else:
            print("Starting the current data shard from a fresh dataloader state")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Balance tokens and keep scoped updates collective-safe across ranks."""
        attention_mask = batch.batch['attention_mask']
        # meta_thinking_attention_mask = batch.batch['meta_thinking_attention_mask']
        # reasoning_attention_mask = batch.batch['reasoning_attention_mask']
        batch_size = attention_mask.shape[0]
        # global_seqlen_lst = (meta_thinking_attention_mask.view(batch_size, -1).sum(-1) + reasoning_attention_mask.view(batch_size, -1).sum(-1)).tolist()  # (train_batch_size,)
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg[self._current_train_agent].world_size
        trainable_mask = (
            batch.batch['labels'].ne(-100).any(dim=-1).tolist()
        )
        trainable_count = sum(bool(value) for value in trainable_mask)
        metrics[f'{logging_prefix}/trainable_sample_count'] = float(
            trainable_count
        )
        global_partition_lst = None
        if self.scoped_c3_grpo_enabled:
            global_partition_lst = build_trainable_rank_partitions(
                global_seqlen_lst,
                world_size,
                trainable_mask,
            )
            if global_partition_lst is None:
                metrics[f'{logging_prefix}/collective_safe'] = 0.0
                metrics[f'{logging_prefix}/trainable_rank_coverage'] = (
                    float(trainable_count) / float(world_size)
                )
                return False
            metrics[f'{logging_prefix}/collective_safe'] = 1.0
            metrics[f'{logging_prefix}/trainable_rank_coverage'] = 1.0
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst,
                k_partitions=world_size,
                equal_size=True,
            )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)
        return True
        
    
    def multi_turn_generate_sequences(self, gen_batch: DataProto):
        agent_roles = gen_batch.meta_info['agent_roles']
        dummy_batch = DataProto.from_dict(
            {'dummy_tensor': torch.arange(0, len(gen_batch.batch))}
        )
        for agent_role, wg in self.actor_rollout_wg.items():
            assert agent_role in agent_roles, f'{agent_roles=}, {agent_role=}'
            wg.enter_generate_context(dummy_batch)            
        
        
        try: 
            output = self.multi_agent_rollout.generate(gen_batch)
        except Exception as e:
            raise e
        finally:
            for agent_role, wg in self.actor_rollout_wg.items():
                wg.exit_generate_context(dummy_batch)      
    
        return output

    def _update_current_train_agent(self, epoch: int = None) -> None:
        """Update the current training agent based on switch config.
        
        Args:
            epoch (int, optional): Current epoch number. Only needed for epoch-level switching.
        """
        # agent_roles = ['meta_thinking', 'reasoning']
        switch_config = self.config.algorithm.get('switch_agent', {})
        agent_roles = self._get_train_agent_roles()
        start_agent = self._get_start_agent()
        agent12_curriculum = self._get_agent12_curriculum_config()
        if agent12_curriculum.get('enable', False):
            hierarchy_config = self._get_hierarchy_config()
            decomposer_role = hierarchy_config.get(
                'decomposer_role', 'decomposer'
            )
            selector_role = hierarchy_config.get('selector_role', 'selector')
            worker_roles = [
                role for role in agent_roles
                if role not in {decomposer_role, selector_role}
            ]
            curriculum_state = self._get_agent12_curriculum_state()
            switch_freq = max(int(switch_config.get('freq', 1)), 1)
            train_decomposer = bool(
                agent12_curriculum.get('train_decomposer', True)
            )
            new_agent = select_agent12_training_role(
                curriculum_state,
                decomposer_role=decomposer_role,
                worker_roles=worker_roles,
                switch_freq=switch_freq,
                train_decomposer=train_decomposer,
            )

            self._current_train_agent_idx = agent_roles.index(new_agent)
            if self._current_train_agent != new_agent:
                print(
                    "Agent 1/2 curriculum: "
                    f"phase={curriculum_state.phase}, "
                    f"training_role={new_agent}"
                )
                self._current_train_agent = new_agent
            return

        final_worker_curriculum = self.config.algorithm.get('final_worker_curriculum', {})
        final_worker_warmup_steps = int(final_worker_curriculum.get('warmup_steps', 0))
        if (
            final_worker_curriculum.get('enable', False)
            and self.global_steps < final_worker_warmup_steps
        ):
            score_role = self._get_score_role()
            self._current_train_agent_idx = agent_roles.index(score_role)
            if self._current_train_agent != score_role:
                print(
                    f'Training curriculum: using final worker {score_role} '
                    f'until step {final_worker_warmup_steps}'
                )
                self._current_train_agent = score_role
            return
        switch_level = switch_config.get('level', 'step')
        switch_freq = switch_config.get('freq', 1)
        effective_global_steps = self.global_steps
        if final_worker_curriculum.get('enable', False):
            effective_global_steps = max(0, self.global_steps - final_worker_warmup_steps)
        
        # Calculate new agent index based on switch level
        if switch_level == 'step':
            self._current_train_agent_idx = effective_global_steps // switch_freq \
                + agent_roles.index(start_agent)
        elif switch_level == 'epoch':
            if epoch is None:
                epoch_idx = self.global_steps // len(self.train_dataloader)
            else:
                epoch_idx = epoch
            self._current_train_agent_idx = epoch_idx // switch_freq \
                + agent_roles.index(start_agent)
        else:
            raise ValueError(f"Unknown switch_level: {switch_level}")
        
        # Apply modulo to keep index in valid range
        self._current_train_agent_idx %= len(agent_roles)
        
        # Update current agent if changed
        new_agent = agent_roles[self._current_train_agent_idx]
        if self._current_train_agent != new_agent:
            print(f'Training switching to {new_agent}')
            self._current_train_agent = new_agent

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        session_stop_step = int(
            self.config.trainer.get(
                'session_stop_step',
                self.total_training_steps,
            )
        )
        if session_stop_step > self.total_training_steps:
            raise ValueError(
                f'trainer.session_stop_step={session_stop_step} exceeds '
                f'total_training_steps={self.total_training_steps}'
            )
        if session_stop_step <= self.global_steps:
            print(
                f'Training session already complete at step {self.global_steps}; '
                f'target was {session_stop_step}'
            )
            return
        print(
            f'Training session target: {session_stop_step}; '
            f'global training target: {self.total_training_steps}'
        )

        if self.config.trainer.get('fork_wandb_id', None) is not None:
            fork_wandb_id = self.config.trainer.fork_wandb_id
            # wandb_kwargs = {'resume': 'must', 'id': fork_wandb_id}
            print(f'**[WANDB]: will fork run from wandb id: `{fork_wandb_id}` at step {self.global_steps} **')
            
            # e.g. fork_from="6yaq69uw?_step=200"
            wandb_kwargs = {'fork_from': f"{fork_wandb_id}?_step={self.global_steps}"}
        else:
            wandb_kwargs = {}
            wandb_run_id = self.config.trainer.get('wandb_run_id', None)
            if wandb_run_id:
                wandb_resume = self.config.trainer.get('wandb_resume', 'allow')
                wandb_kwargs.update(
                    id=str(wandb_run_id),
                    resume=str(wandb_resume),
                )
                print(
                    f'**[WANDB]: using run id `{wandb_run_id}` '
                    f'with resume=`{wandb_resume}` **'
                )
        
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True),
                          wandb_kwargs=wandb_kwargs
                          )

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        self._update_current_train_agent()
        print(f'Starting training with {self._current_train_agent}')

        max_num_turns = self.config.actor_rollout_ref.rollout.max_num_turns
        rollout_meta_info = self._build_rollout_meta_info(max_num_turns)
        agent12_curriculum = self._get_agent12_curriculum_config()
        agent12_curriculum_enabled = bool(
            agent12_curriculum.get('enable', False)
        )
        
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        total_prompt_cnt = 0 
        all_negative_cnt = 0
        all_positive_cnt = 0
        mixed_prompt_cnt = 0
        kept_traj_cnt = 0
        c3_trainable_prompt_cnt = 0
        c3_ineligible_prompt_cnt = 0

        for epoch in range(self.config.trainer.total_epochs):
            self._update_current_train_agent(epoch)
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                skip_filtered_actor_update = False

                curriculum_state = self._get_agent12_curriculum_state()
                teacher_attempts_key = str(
                    agent12_curriculum.get(
                        'teacher_attempts_key', 'teacher_attempts'
                    )
                )
                teacher_attempt_key = str(
                    agent12_curriculum.get(
                        'teacher_attempt_key', 'teacher_attempt'
                    )
                )
                base_question_batch_size = len(batch_dict['question'])
                if agent12_curriculum_enabled:
                    batch_dict, base_question_batch_size = (
                        expand_agent12_teacher_attempt_batch(
                            batch_dict,
                            attempts_key=teacher_attempts_key,
                            attempt_key=teacher_attempt_key,
                            expected_attempts=int(
                                agent12_curriculum.get(
                                    'teacher_attempts_per_question', 16
                                )
                            ),
                        )
                    )
                expanded_prompt_group_count = len(batch_dict['question'])

                # create a dummy tensor for the construction function
                dummy_tensor = torch.arange(0, len(batch_dict['question']))
                batch_dict['batch_idx'] = dummy_tensor
                # DataProto.union mutates meta_info in place. Keep per-rollout
                # metadata isolated so a focal role from an earlier optimizer
                # step cannot leak into the next role-switch cycle.
                new_batch: DataProto = DataProto.from_single_dict(
                    batch_dict,
                    meta_info=deepcopy(rollout_meta_info),
                )
                # Assign uid after attempt expansion: every (question, attempt)
                # is an independent GRPO group, while its N rollouts share uid.
                new_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
                                                             dtype=object)
                new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                num_gen_batches += 1

                # pop those keys for generation
                if 'multi_modal_inputs' in new_batch.non_tensor_batch.keys():
                    raise NotImplementedError('multi_modal_inputs is not implemented yet')
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    # because verl originally calls this 'chat'
                    generation_non_tensor_keys = ['question', 'uid']
                    for teacher_context_key in (
                        teacher_attempt_key,
                        'teacher_attempt_index',
                        'teacher_attempt_count',
                    ):
                        if teacher_context_key in new_batch.non_tensor_batch:
                            generation_non_tensor_keys.append(
                                teacher_context_key
                            )
                    gen_batch = new_batch.select(
                        batch_keys=['batch_idx'],
                        non_tensor_batch_keys=generation_non_tensor_keys,
                        meta_info_keys=['agent_roles', 'finish_flag', 'system_prompts', 'hierarchy'],
                        deepcopy=True
                    )
                gen_batch.meta_info['c3_focal_role'] = self._current_train_agent
                gen_batch.meta_info['validate'] = False
                if agent12_curriculum_enabled:
                    gen_batch.meta_info['curriculum_step'] = self.global_steps
                    gen_batch.meta_info['curriculum_phase'] = curriculum_state.phase
                    gen_batch.meta_info['teacher_attempt_key'] = teacher_attempt_key
                    gen_batch.meta_info['teacher_attempt_probability'] = (
                        curriculum_state.teacher_attempt_probability
                    )
                    gen_batch.meta_info['worker_question_probability'] = (
                        curriculum_state.worker_question_probability
                    )

                is_training_last_step = self.global_steps >= self.total_training_steps
                is_session_last_step = self.global_steps >= session_stop_step

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.multi_turn_generate_sequences(gen_batch)
                        
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        raise NotImplementedError('REMAX is not implemented yet')
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    if agent12_curriculum_enabled:
                        teacher_visible = new_batch.non_tensor_batch.get(
                            'teacher_attempt_visible',
                            np.zeros(len(new_batch), dtype=bool),
                        )
                        worker_question_visible = new_batch.non_tensor_batch.get(
                            'worker_question_visible',
                            np.zeros(len(new_batch), dtype=bool),
                        )
                        teacher_attempt_indices = new_batch.non_tensor_batch.get(
                            'teacher_attempt_index',
                            np.full(len(new_batch), -1, dtype=np.int64),
                        )
                        teacher_attempt_counts = new_batch.non_tensor_batch.get(
                            'teacher_attempt_count',
                            np.zeros(len(new_batch), dtype=np.int64),
                        )
                        teacher_visible_array = np.asarray(
                            teacher_visible,
                            dtype=bool,
                        )
                        metrics.update({
                            'curriculum/base_question_batch_size': float(
                                base_question_batch_size
                            ),
                            'curriculum/expanded_prompt_group_count': float(
                                expanded_prompt_group_count
                            ),
                            'curriculum/decomposer_trainable': float(
                                agent12_curriculum.get(
                                    'train_decomposer', True
                                )
                            ),
                            'curriculum/phase_id': float(
                                curriculum_state.phase_id
                            ),
                            'curriculum/phase_step': float(
                                curriculum_state.phase_step
                            ),
                            'curriculum/teacher_attempt_probability': float(
                                curriculum_state.teacher_attempt_probability
                            ),
                            'curriculum/teacher_attempt_visible_rate': float(
                                teacher_visible_array.mean()
                            ),
                            'curriculum/teacher_attempt_count_mean': float(
                                np.asarray(
                                    teacher_attempt_counts,
                                    dtype=float,
                                ).mean()
                            ),
                            'curriculum/teacher_attempt_index_mean': float(
                                np.asarray(
                                    teacher_attempt_indices,
                                    dtype=float,
                                )[teacher_visible_array].mean()
                                if teacher_visible_array.any()
                                else -1.0
                            ),
                            'curriculum/worker_question_probability': float(
                                curriculum_state.worker_question_probability
                            ),
                            'curriculum/worker_question_visible_rate': float(
                                np.asarray(
                                    worker_question_visible,
                                    dtype=float,
                                ).mean()
                            ),
                        })

                    
                    # compute global_valid tokens
                    global_attention_mask = None
                    for role in rollout_meta_info['agent_roles']:
                        role_attention_mask = new_batch.batch[f'{role}_attention_mask']
                        global_attention_mask = role_attention_mask if global_attention_mask is None else global_attention_mask + role_attention_mask
                    new_batch.meta_info['global_token_num'] = torch.sum(global_attention_mask, dim=-1).tolist()

                    # # recompute old_log_probs
                    # with _timer('old_log_prob', timing_raw):
                    #     old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    #     batch = batch.union(old_log_prob)


                    # compute values
                    if self.use_critic:
                        raise NotImplementedError('critic is not implemented yet')
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('reward', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            raise NotImplementedError('RM is not implemented yet')
                            # # we first compute reward model score
                            # reward_tensor = self.rm_wg.compute_rm_score(batch)
                            # batch = batch.union(reward_tensor)

                        # add mask_unfinished_reward to meta_info
                        new_batch.meta_info['mask_unfinished_reward'] = self.config.reward_model.mask_unfinished_reward
                        new_batch.meta_info['use_format_reward'] = self.config.reward_model.get('use_format_reward', False)
                        # rule-based rm build token-level reward_tensor_map for each agent
                        # {
                        #     "meta_thinking_turn_level_reward": tensor([...], device='cuda:0'),
                        #     "reasoning_turn_level_reward": tensor([...], device='cuda:0'),
                        # }
                        reward_tensor_map = self.reward_fn(new_batch)
                        if self.prefix_probe_enabled:
                            with _timer('prefix_probe', timing_raw):
                                self._attach_prefix_probe_signals(
                                    new_batch,
                                    reward_tensor_map['acc'],
                                    metrics,
                                )
                        self._attach_scoped_c3_grpo_signals(
                            new_batch,
                            reward_tensor_map,
                            metrics,
                        )
                        if self.scoped_c3_grpo_enabled:
                            # C3 consumes only terminal correctness. Do not
                            # attach or log legacy manual role rewards.
                            new_batch.batch['acc'] = reward_tensor_map['acc']
                        else:
                            penalty_names = sorted({
                                key[:-len('_applied')]
                                for key in reward_tensor_map
                                if key.endswith('_penalty_applied')
                            } | {
                                key[:-len('_value')]
                                for key in reward_tensor_map
                                if key.endswith('_penalty_value')
                            })
                            for penalty_name in penalty_names:
                                penalty_applied = reward_tensor_map.pop(f'{penalty_name}_applied', None)
                                penalty_value = reward_tensor_map.pop(f'{penalty_name}_value', None)
                                if penalty_applied is not None:
                                    metrics[f'reward/{penalty_name}_applied_count'] = penalty_applied.sum().item()
                                    metrics[f'reward/{penalty_name}_applied_rate'] = penalty_applied.float().mean().item()
                                    metrics[f'reward/penalties/{penalty_name}/applied_count'] = penalty_applied.sum().item()
                                    metrics[f'reward/penalties/{penalty_name}/applied_rate'] = penalty_applied.float().mean().item()
                                if penalty_value is not None:
                                    metrics[f'reward/{penalty_name}_avg_value'] = penalty_value.float().mean().item()
                                    metrics[f'reward/penalties/{penalty_name}/avg_value'] = penalty_value.float().mean().item()
                            new_batch.batch['acc'] = reward_tensor_map.pop('acc')
                            for key_reward, reward_tensor in reward_tensor_map.items():
                                new_batch.batch[key_reward] = reward_tensor
                                if not key_reward.endswith('_turn_level_reward'):
                                    continue
                                turn_mask = verl_F.get_turn_mask(
                                    reward_tensor,
                                    new_batch.non_tensor_batch['num_turns'],
                                )
                                key_return = key_reward.replace('reward', 'return')
                                new_batch.batch[key_return] = core_algos.compute_turn_level_return(
                                    reward_tensor,
                                    turn_mask,
                                    self.config.algorithm.gamma_turn_level,
                                )
                    
                    # statistics for group filter
                    if self.config.actor_rollout_ref.rollout.n > 1:
                        # key_reward = list(reward_tensor_map.keys())[0]
                        # one_agent_reward_tensor = reward_tensor_map[key_reward]
                        acc_tensor = new_batch.batch['acc']
                        id2acc = defaultdict(list)
                        id2indices = defaultdict(list)
                        for i_bsz, uid in enumerate(new_batch.non_tensor_batch['uid']):
                            id2acc[uid].append(acc_tensor[i_bsz])
                            id2indices[uid].append(i_bsz)

                        kept_prompt_uids = []
                        for key_uid, acc_this_uid in id2acc.items():
                            acc_this_uid = torch.stack([
                                value.detach().float().cpu()
                                for value in acc_this_uid
                            ])
                            if (acc_this_uid == 0).all():
                                all_negative_cnt += 1
                            elif (acc_this_uid == 1).all():
                                all_positive_cnt += 1
                            else:
                                # keep prompt with none-zero advantages
                                kept_prompt_uids.append(key_uid)
                                mixed_prompt_cnt += 1
                            total_prompt_cnt += 1

                        if self.scoped_c3_grpo_enabled:
                            causal_valid = new_batch.batch[
                                'scoped_c3_causal_valid'
                            ].bool().cpu()
                            c3_update_mask = new_batch.batch[
                                'scoped_c3_update_mask'
                            ].bool().cpu()
                            c3_kept_prompt_uids = []
                            for key_uid, sample_indices in id2indices.items():
                                causal_indices = [
                                    sample_idx
                                    for sample_idx in sample_indices
                                    if bool(causal_valid[sample_idx].item())
                                ]
                                if len(causal_indices) < 2:
                                    continue
                                causal_scores = (
                                    new_batch.batch[
                                        'scoped_c3_outcome_score'
                                    ][causal_indices]
                                    .detach()
                                    .float()
                                    .cpu()
                                )
                                if (
                                    bool(torch.isfinite(causal_scores).all().item())
                                    and float(causal_scores.max().item())
                                    > float(causal_scores.min().item())
                                    and any(
                                        bool(c3_update_mask[index].item())
                                        for index in causal_indices
                                    )
                                ):
                                    c3_kept_prompt_uids.append(key_uid)
                            kept_prompt_uids = c3_kept_prompt_uids
                            c3_trainable_prompt_cnt += len(kept_prompt_uids)
                            c3_ineligible_prompt_cnt += (
                                len(id2indices) - len(kept_prompt_uids)
                            )
                    
                    if not self.config.algorithm.filter_groups.enable:
                        # if not enable group filter, keep all data
                        batch = new_batch
                    else:
                        # filter data based on group filter statistics
                        num_prompt_in_batch += len(kept_prompt_uids)
                        # Keep one complete rollout batch for diagnostics when
                        # the capped search finds no trainable GRPO group.
                        unfiltered_new_batch = new_batch
                        # get kept data batch
                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch['uid']):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)
                        kept_traj_cnt += len(kept_traj_idxs)
                        new_batch = new_batch[kept_traj_idxs]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])
                        
                        # check if we have enough data
                        prompt_bsz = int(self.config.data.train_batch_size)
                        if agent12_curriculum_enabled:
                            prompt_bsz = int(
                                agent12_curriculum.get(
                                    'optimizer_prompt_batch_size',
                                    prompt_bsz,
                                )
                            )
                        selected_prompt_bsz = prompt_bsz
                        if num_prompt_in_batch < prompt_bsz:
                            # keep generating
                            print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
                            filter_config = self.config.algorithm.filter_groups
                            max_num_gen_batches = filter_config.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f'{num_gen_batches=}. Keep generating...')
                                continue
                            use_partial_batch = bool(
                                filter_config.get(
                                    'use_partial_batch_on_exhaustion',
                                    False,
                                )
                            )
                            prompt_minibatch_size = int(
                                self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                            )
                            allow_sub_minibatch = bool(
                                filter_config.get(
                                    'allow_sub_minibatch_on_exhaustion',
                                    False,
                                )
                            )
                            dynamic_batching_enabled = bool(
                                self.config.actor_rollout_ref.actor.use_dynamic_bsz
                            )
                            selected_prompt_bsz = compute_usable_filtered_prompt_count(
                                num_prompt_in_batch,
                                prompt_bsz,
                                prompt_minibatch_size,
                                allow_sub_minibatch=(
                                    allow_sub_minibatch
                                    and dynamic_batching_enabled
                                ),
                            )
                            skip_zero_trainable = bool(
                                filter_config.get(
                                    'skip_update_on_zero_trainable',
                                    False,
                                )
                                and num_prompt_in_batch == 0
                            )
                            if skip_zero_trainable:
                                skip_filtered_actor_update = True
                                batch = unfiltered_new_batch
                                print(
                                    'Group-filter cap reached with no trainable '
                                    f'mixed prompts for role='
                                    f'{self._current_train_agent!r}; skipping '
                                    'this actor update.'
                                )
                                metrics[
                                    'rollout/zero_trainable_filtered_update'
                                ] = 1.0
                                metrics[
                                    'rollout/skipped_zero_trainable_update'
                                ] = 1.0
                            elif not use_partial_batch or selected_prompt_bsz <= 0:
                                raise ValueError(
                                    f'{num_gen_batches=} >= {max_num_gen_batches=} '
                                    f'with only {num_prompt_in_batch} trainable mixed '
                                    f'prompts for role={self._current_train_agent!r}; '
                                    f'need at least one complete prompt minibatch of '
                                    f'{prompt_minibatch_size}, or enable dynamic '
                                    f'sub-minibatch fallback.'
                                )
                            if not skip_filtered_actor_update:
                                used_sub_minibatch = (
                                    selected_prompt_bsz < prompt_minibatch_size
                                )
                                print(
                                    'Group-filter cap reached; using a partial '
                                    f'batch of {selected_prompt_bsz}/'
                                    f'{num_prompt_in_batch} mixed prompts'
                                    + (' (sub-minibatch).' if used_sub_minibatch else '.')
                                )
                                metrics['rollout/partial_filtered_batch_used'] = 1.0
                                metrics['rollout/partial_filtered_prompt_count'] = float(
                                    selected_prompt_bsz
                                )
                                metrics['rollout/partial_filtered_available_count'] = float(
                                    num_prompt_in_batch
                                )
                                metrics[
                                    'rollout/partial_filtered_subminibatch_used'
                                ] = float(used_sub_minibatch)
                        # Keep complete rollout groups; dynamic batching can
                        # consume a final sub-minibatch when the cap is sparse.
                        if not skip_filtered_actor_update:
                            traj_bsz = (
                                selected_prompt_bsz
                                * self.config.actor_rollout_ref.rollout.n
                            )
                            batch = batch[:traj_bsz]

                    if self.config.actor_rollout_ref.rollout.n > 1:
                        if self.scoped_c3_grpo_enabled:
                            metrics.update({
                                'rollout/c3/generation_batches': float(
                                    num_gen_batches
                                ),
                                'rollout/c3/mixed_prompt_rate': (
                                    mixed_prompt_cnt / total_prompt_cnt
                                    if total_prompt_cnt > 0 else 0.0
                                ),
                                'rollout/c3/trainable_prompt_rate': (
                                    c3_trainable_prompt_cnt / total_prompt_cnt
                                    if total_prompt_cnt > 0 else 0.0
                                ),
                            })
                        else:
                            metrics.update({
                                'rollout/all_negative_cnt': all_negative_cnt,
                                'rollout/all_positive_cnt': all_positive_cnt,
                                'rollout/mixed_prompt_cnt': mixed_prompt_cnt,
                                'rollout/kept_traj_cnt': kept_traj_cnt,
                                'rollout/total_prompt_cnt': total_prompt_cnt,
                                'rollout/num_gen_batches': num_gen_batches,
                                'rollout/mixed_prompt_rate': (
                                    mixed_prompt_cnt / total_prompt_cnt
                                    if total_prompt_cnt > 0 else 0.0
                                ),
                            })
                    if self.prefix_probe_enabled:
                        self._update_prefix_probe_batch_metrics(batch, metrics)
                    if not self.scoped_c3_grpo_enabled:
                        metrics.update(compute_reward_diagnostic_metrics(batch))
                    
                    with _timer('save_train_generation', timing_raw):
                        # save train generation
                        if self.config.trainer.get('save_train_generations', False):
                            self._save_train_generations(batch)

                    

                    with _timer('adv', timing_raw):
                        # Merge different role data into a single DataProto
                        agents_batches: Dict[str, DataProto] = split_batch_for_agents(batch)
                        agent_batch = agents_batches[self._current_train_agent]
                        
                        # assign turn_level scores to the last token of each turn, w/ step_ids
                        #  and then i'll call compute_advantage to distribute the score to all
                        #  tokens of each step.
                        if self.scoped_c3_grpo_enabled:
                            token_level_scores = torch.zeros_like(
                                agent_batch.batch['labels'],
                                dtype=torch.float32,
                            )
                        else:
                            token_level_scores = compute_token_level_scores(
                                agent_batch
                            )
                        agent_batch.batch['token_level_scores'] = token_level_scores
                        batch = agent_batch
                        
                        # # compute rewards. apply_kl_penalty if available
                        # if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                          kl_ctrl=self.kl_ctrl,
                        #                                          kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                        
                        # XXX(ziyu): debug
                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # in this case, its usage is changed.
                        # for REINFORCE++, it's used to distribute the score from last token of each turn
                        # to all tokens of each step.
                        # for GRPO, we use turn_level_reward.sum(-1) as the outcome reward and then
                        # assign each label token the normalized advantage.
                        if self.scoped_c3_grpo_enabled:
                            batch = self._compute_scoped_c3_grpo_advantage(
                                batch,
                                metrics,
                            )
                        else:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma_token_level,
                                lam=self.config.algorithm.lam_token_level,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                            )

                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    collective_safe_update = not skip_filtered_actor_update
                    if collective_safe_update and self.config.trainer.balance_batch:
                        collective_safe_update = self._balance_batch(
                            batch,
                            metrics=metrics,
                        )
                    elif collective_safe_update and self.scoped_c3_grpo_enabled:
                        trainable_count = int(
                            batch.batch['labels'].ne(-100).any(dim=-1).sum().item()
                        )
                        world_size = self.actor_rollout_wg[
                            self._current_train_agent
                        ].world_size
                        collective_safe_update = trainable_count >= world_size

                    metrics['rollout/skipped_sparse_actor_update'] = float(
                        not collective_safe_update
                    )
                    metrics.setdefault(
                        'rollout/skipped_zero_trainable_update',
                        0.0,
                    )

                    
                    # recompute old_log_probs
                    if collective_safe_update:
                        with _timer('old_log_prob', timing_raw):
                            old_log_prob = self.actor_rollout_wg[
                                self._current_train_agent
                            ].compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                    if self.use_reference_policy and collective_safe_update:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg[self._current_train_agent].compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)


                    # update critic
                    if self.use_critic and collective_safe_update:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg[self._current_train_agent].update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if (
                        collective_safe_update
                        and self.config.trainer.critic_warmup <= self.global_steps
                    ):
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg[self._current_train_agent].update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_training_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_training_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if is_session_last_step or (
                        self.config.trainer.save_freq > 0
                        and self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                final_worker_curriculum = self.config.algorithm.get('final_worker_curriculum', {})
                final_worker_warmup_steps = int(final_worker_curriculum.get('warmup_steps', 0))
                metrics.update({
                    'train/current_agent_idx': self._current_train_agent_idx,
                    'train/final_worker_curriculum_active': float(
                        final_worker_curriculum.get('enable', False)
                        and self.global_steps < final_worker_warmup_steps
                    ),
                })
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                all_negative_cnt = 0
                all_positive_cnt = 0
                mixed_prompt_cnt = 0
                kept_traj_cnt = 0
                c3_trainable_prompt_cnt = 0
                c3_ineligible_prompt_cnt = 0
                total_prompt_cnt = 0

                if is_session_last_step:
                    if is_training_last_step:
                        pprint(f'Final validation metrics: {last_val_metrics}')
                    else:
                        print(
                            f'Chunked training session completed at step '
                            f'{self.global_steps}'
                        )
                    return

                self.global_steps += 1
                self._update_current_train_agent()

    def _save_train_generations(self, batch: DataProto):
        # save train generations
        output_dir = Path(self.config.trainer.default_local_dir) / 'replay_buffer'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'train_step_{self.global_steps}.jsonl'
        
        
        results_dict = {}
        for i, data_item in enumerate(batch):
            uid = data_item.non_tensor_batch['uid']
            if uid not in results_dict:
                result = {
                    "question": data_item.non_tensor_batch['question'],
                    "groundtruth": data_item.non_tensor_batch['reward_model']['ground_truth'],
                    "response": [],
                    "history": [],
                    "score": [],
                    "finish_reason": [],
                    "terminal_stage_role": [],
                }
                if self.prefix_probe_enabled:
                    result.update({
                        "prefix_probe_decomposer_score": [],
                        "prefix_probe_decomposer_response": [],
                        "prefix_probe_worker_match_score": [],
                        "prefix_probe_worker_comparisons": [],
                        "prefix_probe_rejection_reason": [],
                        "planned_subtask_count": [],
                        "prefix_probe_plan_eligible": [],
                        "raw_outcome_score": [],
                        "gated_outcome_score": [],
                        "prefix_probe_gate_valid": [],
                        "prefix_probe_collaboration_eligible": [],
                    })
                results_dict[uid] = result

            padded_history = data_item.non_tensor_batch['history']
            unpad_history = [x for x in padded_history if x['role'] != 'padding']
            results_dict[uid]['history'].append(unpad_history)
            results_dict[uid]['response'].append(data_item.non_tensor_batch['response'])
            score_role = data_item.non_tensor_batch.get(
                'terminal_stage_role', self._get_score_role()
            )
            if score_role not in self._get_rollout_agent_roles():
                score_role = self._get_score_role()
            results_dict[uid]['score'].append(
                data_item.batch[f'{score_role}_turn_level_reward'].sum().item()
            )
            results_dict[uid]['finish_reason'].append(
                data_item.non_tensor_batch['finish_reason']
            )
            results_dict[uid]['terminal_stage_role'].append(score_role)
            if self.prefix_probe_enabled:
                results_dict[uid]['prefix_probe_decomposer_score'].append(
                    float(data_item.non_tensor_batch.get(
                        'prefix_probe_decomposer_score', float('nan')
                    ))
                )
                results_dict[uid]['prefix_probe_decomposer_response'].append(
                    data_item.non_tensor_batch.get(
                        'prefix_probe_decomposer_response', ''
                    )
                )
                for key in (
                    'prefix_probe_worker_match_score',
                    'prefix_probe_worker_comparisons',
                    'prefix_probe_rejection_reason',
                ):
                    results_dict[uid][key].append(data_item.non_tensor_batch[key])
                results_dict[uid]['planned_subtask_count'].append(
                    int(data_item.batch['prefix_probe_subtask_count'].item())
                )
                results_dict[uid]['prefix_probe_plan_eligible'].append(
                    bool(data_item.batch['prefix_probe_plan_eligible'].item())
                )
                results_dict[uid]['raw_outcome_score'].append(float(
                    data_item.batch['scoped_c3_raw_outcome_score'].item()
                ))
                results_dict[uid]['gated_outcome_score'].append(float(
                    data_item.batch[
                        'prefix_probe_gated_outcome_score'
                    ].item()
                ))
                results_dict[uid]['prefix_probe_gate_valid'].append(bool(
                    data_item.batch['prefix_probe_gate_valid'].item()
                ))
                results_dict[uid][
                    'prefix_probe_collaboration_eligible'
                ].append(bool(
                    data_item.batch[
                        'prefix_probe_collaboration_eligible'
                    ].item()
                ))

        results_to_save = []
        for uid, result in results_dict.items():
            result['avg_score'] = sum(result['score']) / len(result['score'])
            results_to_save.append(result)
        with jsonlines.open(output_file, 'w') as writer:
            writer.write_all(results_to_save)

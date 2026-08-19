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

import pdb
import copy
import os
import zlib
from pathlib import Path
import uuid
import jsonlines
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type, Dict
from copy import deepcopy
from collections import defaultdict, deque

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
from verl.utils.model import compute_position_id_with_mask
from verl.rema_separated_trainer.ppo.cpcr import estimate_cpcr
from verl.rema_separated_trainer.ppo.direct_scoped_grpo import (
    build_dependency_removed_scope_counterfactual,
    build_full_task_scope_counterfactual,
    build_scope_assignment_counterfactual,
    build_verifier_scope_counterfactuals,
    estimate_direct_scoped_grpo,
)
from verl.rema_separated_trainer.ppo.multi_agent_rollout import MultiAgentRollout
from verl.rema_separated_trainer.ppo.scoped_c3_grpo import (
    estimate_scoped_c3_grpo,
)



WorkerType = Type[Worker]


@dataclass(frozen=True)
class Agent12CurriculumState:
    phase: str
    phase_id: int
    phase_step: int
    teacher_solution_probability: float
    worker_question_probability: float


def compute_agent12_curriculum_state(
    global_step: int,
    *,
    worker_bootstrap_steps: int,
    decomposer_transfer_steps: int,
    worker_question_fade_steps: int,
    worker_question_final_probability: float,
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

    if completed_steps < worker_bootstrap_steps:
        return Agent12CurriculumState(
            phase="worker_bootstrap",
            phase_id=0,
            phase_step=completed_steps,
            teacher_solution_probability=1.0,
            worker_question_probability=1.0,
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
            teacher_solution_probability=teacher_probability,
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
        teacher_solution_probability=0.0,
        worker_question_probability=worker_question_probability,
    )


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
):
    """Extract the scoring-role candidate produced in every executed round."""

    if score_role not in agent_roles:
        raise ValueError(f"score_role={score_role!r} is not present in agent_roles")
    if len(histories) != len(num_turns):
        raise ValueError("histories and num_turns must have equal lengths")

    batch_size = len(histories)
    outputs = [["" for _ in range(batch_size)] for _ in range(max_num_turns)]
    executed = torch.zeros((batch_size, max_num_turns), dtype=torch.bool)
    roles_per_round = len(agent_roles)

    for sample_idx, (sample_history, sample_num_turns) in enumerate(
        zip(histories, num_turns)
    ):
        sample_num_turns = min(int(sample_num_turns), max_num_turns)
        for turn_idx in range(sample_num_turns):
            start = turn_idx * roles_per_round
            turn_history = sample_history[start:start + roles_per_round]
            final_message = next(
                (
                    message
                    for message in turn_history
                    if isinstance(message, dict)
                    and message.get("role") == score_role
                ),
                None,
            )
            if final_message is None:
                raise ValueError(
                    f"Missing {score_role} history slot for sample={sample_idx}, "
                    f"turn={turn_idx + 1}"
                )
            content = final_message.get("content", "")
            outputs[turn_idx][sample_idx] = content if isinstance(content, str) else ""
            executed[sample_idx, turn_idx] = bool(
                final_message.get("executed", True)
            )

    return outputs, executed


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
from verl.workers.reward_manager.prd_composer import (
    PRD_REWARD_SOURCE_NAMES,
    PRD_ROLE_FEATURE_NAMES,
    ROLE_PRD_FEATURE_NAMES,
    PRDRewardComposer,
    RolePRDCreditRouter,
    build_prd_graph_prior_tensors,
    build_prd_role_feature_tensor,
    build_prd_source_tensor,
    build_role_prd_graph_prior_tensors,
)
from verl.workers.reward_manager.rema import _build_prd_reward_sources


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
        self._init_online_prd_composer()
        self._init_scoped_c3_grpo()
        self._init_direct_scoped_grpo()
        self._init_cpcr()

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
        if self.online_prd_enabled:
            raise ValueError(
                "Scoped C3 GRPO and online PRD cannot be enabled together"
            )
        if bool(
            hierarchy_config.get('direct_scoped_grpo', {}).get(
                'enable',
                False,
            )
        ):
            raise ValueError(
                "Scoped C3 GRPO and direct CPCR scoped GRPO are mutually exclusive"
            )
        if bool(hierarchy_config.get('cpcr', {}).get('enable', False)):
            raise ValueError(
                "Scoped C3 GRPO uses exact grouped rollouts; disable CPCR"
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
        tss_roles = train_agent_roles.intersection(stage_roles[:-1])
        worker_context_mode = str(
            hierarchy_config.get('worker_context_mode', 'subtask_context')
        )
        if (
            tss_roles
            and worker_context_mode not in {'full_question', 'question', 'full'}
        ):
            raise ValueError(
                "Scoped C3 TSS requires hierarchy.worker_context_mode="
                "'full_question' for non-final worker roles"
            )

    def _init_direct_scoped_grpo(self):
        hierarchy_config = self.config.algorithm.get('hierarchy', {})
        direct_config = (
            hierarchy_config.get('direct_scoped_grpo', {})
            if hierarchy_config else {}
        )
        self.direct_scoped_grpo_config = (
            OmegaConf.to_container(direct_config, resolve=True)
            if direct_config else {}
        )
        self.direct_scoped_grpo_enabled = bool(
            self.direct_scoped_grpo_config.get('enable', False)
        )
        if not self.direct_scoped_grpo_enabled:
            return
        if self.scoped_c3_grpo_enabled:
            raise ValueError(
                "Direct CPCR scoped GRPO and scoped C3 GRPO are mutually exclusive"
            )

        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise ValueError(
                "Direct scoped GRPO requires algorithm.adv_estimator='grpo'"
            )
        if not bool(hierarchy_config.get('enable', False)):
            raise ValueError(
                "Direct scoped GRPO requires algorithm.hierarchy.enable=True"
            )
        worker_context_mode = str(
            hierarchy_config.get('worker_context_mode', 'subtask_context')
        )
        if worker_context_mode not in {'full_question', 'question', 'full'}:
            raise ValueError(
                "Teacher-forced TSS requires hierarchy.worker_context_mode="
                "'full_question'"
            )
        if self.online_prd_enabled:
            raise ValueError(
                "Direct scoped GRPO and online PRD cannot be enabled together"
            )
        cpcr_config = hierarchy_config.get('cpcr', {})
        if (
            not bool(cpcr_config.get('enable', False))
            or str(cpcr_config.get('mode', '')) != 'direct_scoped_grpo'
        ):
            raise ValueError(
                "Direct scoped GRPO requires cpcr.enable=True and "
                "cpcr.mode='direct_scoped_grpo'"
            )

    def _init_cpcr(self):
        hierarchy_config = self.config.algorithm.get('hierarchy', {})
        cpcr_config = hierarchy_config.get('cpcr', {}) if hierarchy_config else {}
        self.cpcr_config = OmegaConf.to_container(cpcr_config, resolve=True) if cpcr_config else {}
        self.cpcr_enabled = bool(self.cpcr_config.get('enable', False))
        self._cpcr_last_scored_step = None
        self._cpcr_role_visit_counts = defaultdict(int)
        self._cpcr_role_last_visit_step = {}
        self._cpcr_role_probe_counts = defaultdict(int)
        if not self.cpcr_enabled:
            return

        mode = str(self.cpcr_config.get('mode', 'diagnostic'))
        estimator = str(self.cpcr_config.get('estimator', 'full_suffix'))
        outcome_mode = str(self.cpcr_config.get('outcome_mode', 'candidate_factual'))
        if mode not in {'diagnostic', 'prd_target', 'direct_scoped_grpo'}:
            raise ValueError(f"Unsupported CPCR mode: {mode}")
        if estimator != 'full_suffix':
            raise ValueError(
                "Only CPCR estimator='full_suffix' is implemented."
            )
        if outcome_mode != 'candidate_factual':
            raise ValueError(
                "Only CPCR outcome_mode='candidate_factual' is implemented. "
                "Reverified stitched outcomes require a verifier replay backend."
            )
        if mode == 'prd_target':
            if not self.online_prd_enabled:
                raise ValueError("CPCR prd_target mode requires reward_composer.online.enable=True")
            if self.online_prd_model_type != 'role':
                raise ValueError("CPCR prd_target mode requires online PRD model_type='role'")
        if mode == 'direct_scoped_grpo':
            if not self.direct_scoped_grpo_enabled:
                raise ValueError(
                    "CPCR direct_scoped_grpo mode requires "
                    "hierarchy.direct_scoped_grpo.enable=True"
                )
            if self.cpcr_config.get('roles', 'current') != 'current':
                raise ValueError(
                    "Direct scoped GRPO currently requires cpcr.roles='current'"
                )
            if int(self.cpcr_config.get('score_interval', 1)) != 1:
                raise ValueError(
                    "Direct scoped GRPO requires cpcr.score_interval=1 so every "
                    "role visit receives a direct training target"
                )
        rollout_config = self.config.actor_rollout_ref.rollout
        if int(rollout_config.get('n', 1)) < 2:
            raise ValueError("CPCR requires actor_rollout_ref.rollout.n >= 2")
        if not bool(rollout_config.get('do_sample', True)):
            raise ValueError("CPCR requires stochastic rollouts with do_sample=True")
        if float(rollout_config.get('temperature', 1.0)) <= 0.0:
            raise ValueError("CPCR requires rollout.temperature > 0")
        if float(rollout_config.get('top_p', 1.0)) != 1.0:
            raise ValueError("CPCR likelihood correction currently requires rollout.top_p=1")
        if int(rollout_config.get('top_k', -1)) not in {-1, 0}:
            raise ValueError("CPCR likelihood correction currently requires top_k to be disabled")
        min_p = rollout_config.get('min_p', 0.0)
        if min_p is not None and float(min_p) != 0.0:
            raise ValueError("CPCR likelihood correction currently requires rollout.min_p=0")
        presence_penalty = rollout_config.get('presence_penalty', 0.0)
        if presence_penalty is not None and float(presence_penalty) != 0.0:
            raise ValueError("CPCR does not reproduce rollout.presence_penalty")
        frequency_penalty = rollout_config.get('frequency_penalty', 0.0)
        if frequency_penalty is not None and float(frequency_penalty) != 0.0:
            raise ValueError("CPCR does not reproduce rollout.frequency_penalty")
        repetition_penalty = rollout_config.get('repetition_penalty', 1.0)
        if repetition_penalty is not None and float(repetition_penalty) != 1.0:
            raise ValueError("CPCR does not reproduce rollout.repetition_penalty")

    @staticmethod
    def _cpcr_unpad_messages(messages):
        if isinstance(messages, np.ndarray):
            messages = messages.tolist()
        unpadded = [
            dict(message)
            for message in messages
            if isinstance(message, dict) and message.get('role') != 'padding'
        ]
        # _build_tensor_dict appends the sampled response to every stored
        # conversation. Counterfactual scoring needs the state before that
        # action, not a prompt that already contains the action being scored.
        if unpadded and unpadded[-1].get('role') == 'assistant':
            unpadded = unpadded[:-1]
        return unpadded

    @staticmethod
    def _cpcr_latest_role_message(history, role):
        if isinstance(history, np.ndarray):
            history = history.tolist()
        for message in reversed(history):
            if (
                isinstance(message, dict)
                and message.get('role') == role
                and isinstance(message.get('content'), str)
            ):
                return dict(message)
        return None

    def _cpcr_encode_prompt_action(
        self,
        chat,
        action_ids,
        prompt_length,
        max_length,
    ):
        """Attach an already sampled token action to a stored role prompt."""

        query_ids = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            truncation=True,
            max_length=prompt_length,
        )
        action_ids = [int(token_id) for token_id in action_ids]
        full_ids = list(query_ids) + action_ids
        # The final ignore label keeps tensor lengths aligned. Every token that
        # vLLM actually returned is still scored exactly once.
        labels = [-100] * (len(query_ids) - 1) + action_ids + [-100]
        if (
            len(full_ids) != len(labels)
            or len(full_ids) > max_length
        ):
            return None
        return full_ids, labels

    def _cpcr_action_token_ids(
        self,
        data_batch,
        role,
        sample_idx,
        turn_idx,
        stop_reason,
    ):
        token_ids_key = f'{role}_action_token_ids'
        if token_ids_key in data_batch.non_tensor_batch:
            raw_token_ids = data_batch.non_tensor_batch[token_ids_key][sample_idx]
            if isinstance(raw_token_ids, np.ndarray):
                raw_token_ids = raw_token_ids.tolist()
            if isinstance(raw_token_ids, (list, tuple)) and raw_token_ids:
                return [int(token_id) for token_id in raw_token_ids]

        # Compatibility fallback for trajectories produced before raw sampled
        # token ids were stored in history.
        labels_key = f'{role}_labels'
        step_ids_key = f'{role}_step_ids'
        if labels_key not in data_batch.batch or step_ids_key not in data_batch.batch:
            return None
        action_labels = data_batch.batch[labels_key][sample_idx][
            data_batch.batch[step_ids_key][sample_idx] == turn_idx
        ]
        action_labels = action_labels[action_labels != -100]
        if stop_reason == 'stop':
            if (
                self.tokenizer.eos_token_id is None
                or action_labels.numel() == 0
                or int(action_labels[-1].item()) != self.tokenizer.eos_token_id
            ):
                return None
            action_labels = action_labels[:-1]
        elif stop_reason != 'length':
            return None
        if action_labels.numel() == 0:
            return None
        return action_labels.tolist()

    @staticmethod
    def _cpcr_replace_last_user_message(chat, content):
        if not chat or chat[-1].get('role') != 'user':
            return None
        return chat[:-1] + [{'role': 'user', 'content': content}]

    def _cpcr_build_stitched_suffix(
        self,
        data_batch,
        target_idx,
        candidate_idx,
        start_role,
    ):
        """Replay one hierarchical suffix using existing candidate actions.

        The first role receives its exact target-rollout prompt. Later prompts
        are rebuilt deterministically from the target prefix and the transported
        candidate outputs. No model decoding happens here.
        """

        agent_roles = list(data_batch.meta_info['agent_roles'])
        hierarchy = dict(data_batch.meta_info.get('hierarchy', {}))
        system_prompts = dict(data_batch.meta_info.get('system_prompts', {}))
        if start_role not in agent_roles:
            return None
        if str(data_batch.non_tensor_batch['question'][target_idx]) != str(
            data_batch.non_tensor_batch['question'][candidate_idx]
        ):
            return None

        decomposer_role = hierarchy.get('decomposer_role', 'decomposer')
        selector_role = hierarchy.get('selector_role', 'selector')
        stage_roles = list(hierarchy.get('stage_roles', []))
        worker_types = list(hierarchy.get('worker_roles', []))
        if not stage_roles or decomposer_role not in agent_roles or selector_role not in agent_roles:
            return None
        default_worker = hierarchy.get(
            'default_worker',
            worker_types[-1] if worker_types else selector_role,
        )
        worker_specs = hierarchy.get('worker_specs', {})
        worker_spec_text = MultiAgentRollout._format_worker_specs(worker_specs, worker_types)
        worker_context_mode = hierarchy.get(
            'worker_context_mode',
            'full_question' if hierarchy.get('pass_question_to_workers', False) else 'subtask_context',
        )
        pass_question_to_workers = worker_context_mode in {'full_question', 'question', 'full'}
        final_context_mode = hierarchy.get('final_context_mode', 'full_question')

        history_batch = data_batch.non_tensor_batch['history']
        target_history = history_batch[target_idx]
        candidate_history = history_batch[candidate_idx]
        target_messages = {
            role: self._cpcr_latest_role_message(target_history, role)
            for role in agent_roles
        }
        candidate_messages = {
            role: self._cpcr_latest_role_message(candidate_history, role)
            for role in agent_roles
        }
        start_position = agent_roles.index(start_role)
        target_start = target_messages.get(start_role)
        candidate_start = candidate_messages.get(start_role)
        if (
            target_start is None
            or candidate_start is None
            or not target_start['content'].strip()
            or not candidate_start['content'].strip()
        ):
            return None

        stitched_outputs = {}
        for position, role in enumerate(agent_roles):
            message = candidate_messages[role] if position >= start_position else target_messages[role]
            stitched_outputs[role] = message['content'] if message is not None else ''

        final_stage_role = stage_roles[-1]
        max_planned_subtasks = int(
            hierarchy.get(
                'max_planned_subtasks',
                max(len(stage_roles) - 1, 1),
            )
        )

        def parse_execution_stages(plan_text, selector_text):
            parsed_subtasks = MultiAgentRollout._extract_subtasks(
                plan_text,
                max_subtasks=max_planned_subtasks,
            )
            if str(hierarchy.get('routing_mode', 'selector')).lower() == 'derive_verify':
                _, parsed_stages = MultiAgentRollout._build_derive_verify_stages(
                    parsed_subtasks,
                    stage_roles,
                    default_worker,
                )
                return parsed_stages
            parsed_stages = MultiAgentRollout._parse_ordered_worker_stages(
                selector_text,
                parsed_subtasks,
                stage_roles,
                worker_types,
                default_worker,
            )
            if all(stage_role != final_stage_role for stage_role, _, _ in parsed_stages):
                parsed_stages.append((final_stage_role, default_worker, []))
            return parsed_stages

        plan = stitched_outputs.get(decomposer_role, '')
        assignments = stitched_outputs.get(selector_role, '')
        ordered_stages = parse_execution_stages(plan, assignments)

        candidate_plan = (
            candidate_messages.get(decomposer_role, {}).get('content', '')
            if candidate_messages.get(decomposer_role) is not None else ''
        )
        candidate_assignments = (
            candidate_messages.get(selector_role, {}).get('content', '')
            if candidate_messages.get(selector_role) is not None else ''
        )
        candidate_stages = parse_execution_stages(candidate_plan, candidate_assignments)

        deterministic_routing = (
            str(hierarchy.get('routing_mode', 'selector')).lower() == 'derive_verify'
        )
        control_roles = (
            [decomposer_role]
            if deterministic_routing
            else [decomposer_role, selector_role]
        )
        execution_roles = control_roles + [
            stage_role for stage_role, _, _ in ordered_stages
        ]
        candidate_execution_roles = control_roles + [
            stage_role for stage_role, _, _ in candidate_stages
        ]
        if start_role not in execution_roles or start_role not in candidate_execution_roles:
            return None
        suffix_roles = execution_roles[execution_roles.index(start_role):]
        candidate_suffix_roles = candidate_execution_roles[
            candidate_execution_roles.index(start_role):
        ]
        # MIS requires candidate l to denote one fixed suffix in every row k.
        # A changed activation path is therefore unsupported, not a new suffix.
        if suffix_roles != candidate_suffix_roles:
            return None
        stage_specs = {
            stage_role: (stage_idx, worker_type, assigned_subtasks)
            for stage_idx, (stage_role, worker_type, assigned_subtasks) in enumerate(ordered_stages)
        }

        target_chats = {}
        for role in agent_roles:
            key = f'{role}_conversation_history'
            if key not in data_batch.non_tensor_batch:
                return None
            target_chats[role] = self._cpcr_unpad_messages(
                data_batch.non_tensor_batch[key][target_idx]
            )
        if not target_chats.get(start_role):
            return None

        question = str(data_batch.non_tensor_batch['question'][target_idx])
        completed_results = []
        if start_role in stage_specs:
            start_stage_idx = stage_specs[start_role][0]
            for stage_role, worker_type, assigned_subtasks in ordered_stages[:start_stage_idx]:
                output = stitched_outputs.get(stage_role, '')
                if output.strip():
                    subtask_ids = ', '.join(subtask_id for subtask_id, _ in assigned_subtasks)
                    completed_results.append((stage_role, worker_type, subtask_ids, output))

        suffix_steps = []
        for role in suffix_roles:
            candidate_message = candidate_messages.get(role)
            if candidate_message is None or not candidate_message['content'].strip():
                return None

            if role == start_role:
                chat = target_chats[role]
            elif role == selector_role:
                selector_content = (
                    f"Question:\n{question}\n\n"
                    f"Plan:\n{plan}\n\n"
                    f"Available workers:\n{worker_spec_text}"
                )
                chat = self._cpcr_replace_last_user_message(
                    target_chats[selector_role],
                    selector_content,
                )
            elif role in stage_specs:
                stage_idx, worker_type, assigned_subtasks = stage_specs[role]
                is_final_stage = stage_idx == len(ordered_stages) - 1
                if is_final_stage:
                    question_block = (
                        ''
                        if final_context_mode in {'notes_only', 'notes', 'no_question'}
                        else f"Question:\n{question}\n\n"
                    )
                elif pass_question_to_workers:
                    question_block = (
                        f"Reference problem:\n{question}\n\n"
                        "Use the reference problem only to recover facts needed for the assigned subtask.\n\n"
                    )
                else:
                    question_block = (
                        "The assigned subtask is your task context and should contain the needed facts. "
                        "Use previous LOCAL_RESULTs when they help.\n\n"
                    )

                if is_final_stage:
                    if final_context_mode in {'worker_results_only', 'workers_only', 'local_results_only'}:
                        work_so_far = MultiAgentRollout._format_worker_results_for_final(
                            completed_results
                        )
                    else:
                        work_so_far = MultiAgentRollout._format_final_notes(
                            plan,
                            MultiAgentRollout._format_work_so_far(completed_results),
                        )
                else:
                    work_so_far = MultiAgentRollout._format_previous_local_results(
                        completed_results
                    )

                assigned_subtasks_text = MultiAgentRollout._format_subtasks(assigned_subtasks)
                if is_final_stage:
                    stage_instruction = (
                        "Synthesize the final answer from the worker results. "
                        "Check the worker results, repair mistakes if needed, and end with the final answer in \\boxed{}."
                    )
                    if not assigned_subtasks_text:
                        assigned_subtasks_text = "- Use the work above to synthesize the final answer."
                else:
                    stage_instruction = (
                        "Work on the assigned subtask above. "
                        "Reason step by step with concrete calculations, transformations, or checks. "
                        "Verify dependencies, warnings, boundary cases, signs, domains, units, and repair instructions. "
                        "Finish with one concise LOCAL_RESULT for this subtask. "
                        "Output exactly:\n"
                        "REASONING:\n"
                        "<step-by-step reasoning for this subtask>\n\n"
                        "LOCAL_RESULT: \\boxed{<useful result of this subtask>}"
                    )
                work_so_far_block = f"{work_so_far}\n\n" if work_so_far else ''
                dependency_instruction = (
                    "Use the PREVIOUS LOCAL RESULTS when the current task depends on them.\n\n"
                    if work_so_far and not is_final_stage else ''
                )
                user_content = (
                    f"{question_block}"
                    f"{work_so_far_block}"
                    f"{dependency_instruction}"
                    f"CURRENT TASK:\n{assigned_subtasks_text}\n\n"
                    f"{stage_instruction}\n\n"
                )
                system_prompt = (
                    system_prompts.get('finalizer')
                    if is_final_stage else None
                ) or system_prompts.get(worker_type, system_prompts.get(role, ''))
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_content},
                ]
            else:
                return None

            if not chat:
                return None
            suffix_steps.append((role, chat, candidate_message))
            if role in stage_specs:
                _, worker_type, assigned_subtasks = stage_specs[role]
                subtask_ids = ', '.join(subtask_id for subtask_id, _ in assigned_subtasks)
                completed_results.append((
                    role,
                    worker_type,
                    subtask_ids,
                    candidate_message['content'],
                ))

        return suffix_steps

    def _score_teacher_forced_actions(self, role, encoded_actions):
        """Score variable-length actions without diverging FSDP collectives."""

        if not encoded_actions:
            empty = torch.empty(0, dtype=torch.float32)
            return empty, empty, 0, 0, 0

        worker_group = self.actor_rollout_wg[role]
        scoring_config = (
            self.scoped_c3_grpo_config
            if self.scoped_c3_grpo_enabled
            else self.cpcr_config
        )
        micro_batch_size = max(
            int(
                scoring_config.get(
                    'teacher_forcing_micro_batch_size_per_gpu',
                    1,
                )
            ),
            1,
        )
        # One RPC contains at most one fixed microbatch per rank. Variable-length
        # dynamic batching can otherwise give FSDP ranks different forward counts.
        rpc_batch_size = worker_group.world_size * micro_batch_size
        expected_call_count = (
            len(encoded_actions) + rpc_batch_size - 1
        ) // rpc_batch_size
        print(
            'Teacher-forced scoring '
            f'role={role}, actions={len(encoded_actions)}, '
            f'actions_per_call={rpc_batch_size}, calls={expected_call_count}'
        )
        sum_log_probs = []
        mean_log_probs = []
        action_token_count = 0
        input_token_count = 0
        scoring_call_count = 0

        for start in range(0, len(encoded_actions), rpc_batch_size):
            action_chunk = encoded_actions[start:start + rpc_batch_size]
            sequence_length = max(
                len(input_ids)
                for input_ids, _ in action_chunk
            )
            batch_size = len(action_chunk)
            input_ids = torch.full(
                (batch_size, sequence_length),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            labels = torch.full(
                (batch_size, sequence_length),
                -100,
                dtype=torch.long,
            )
            attention_mask = torch.zeros(
                (batch_size, sequence_length),
                dtype=torch.long,
            )
            for item_idx, (item_input_ids, item_labels) in enumerate(action_chunk):
                item_length = len(item_input_ids)
                input_ids[item_idx, :item_length] = torch.tensor(
                    item_input_ids,
                    dtype=torch.long,
                )
                labels[item_idx, :item_length] = torch.tensor(
                    item_labels,
                    dtype=torch.long,
                )
                attention_mask[item_idx, :item_length] = 1

            scoring_batch = DataProto.from_dict(
                {
                    'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': attention_mask,
                    'position_ids': compute_position_id_with_mask(attention_mask),
                },
                meta_info={
                    'static_log_prob_micro_batch_size_per_gpu': micro_batch_size,
                },
            )
            scoring_batch, pad_size = pad_dataproto_to_divisor(
                scoring_batch,
                worker_group.world_size,
            )
            scored = worker_group.compute_log_prob(scoring_batch)
            scored = unpad_dataproto(scored, pad_size=pad_size)
            token_log_probs = scored.batch['old_log_probs'].float()
            label_mask = labels != -100
            sequence_sums = token_log_probs.masked_fill(
                ~label_mask,
                0.0,
            ).sum(dim=1)
            sum_log_probs.append(sequence_sums)
            mean_log_probs.append(
                sequence_sums / label_mask.sum(dim=1).clamp_min(1)
            )
            action_token_count += int(label_mask.sum().item())
            input_token_count += int(attention_mask.sum().item())
            scoring_call_count += 1

        return (
            torch.cat(sum_log_probs),
            torch.cat(mean_log_probs),
            action_token_count,
            input_token_count,
            scoring_call_count,
        )

    def _cpcr_score_full_suffix_pairs(
        self,
        data_batch,
        role,
        selected_uids=None,
    ):
        """Score transported hierarchical suffixes without decoding."""

        uid_to_indices = defaultdict(list)
        for idx, uid in enumerate(data_batch.non_tensor_batch['uid']):
            uid_to_indices[uid].append(idx)

        turn_counts = [int(value) for value in data_batch.non_tensor_batch['num_turns']]
        prompt_length = int(self.config.actor_rollout_ref.rollout.prompt_length)
        max_length = prompt_length + int(
            self.config.actor_rollout_ref.rollout.response_length
        )

        encoded_by_role = defaultdict(list)
        expected_steps = {}
        skipped_pair_count = 0
        factual_prompt_check_count = 0
        factual_prompt_match_count = 0
        active_target_count = 0
        group_specs = {}
        for group_idx, (uid, indices) in enumerate(uid_to_indices.items()):
            if len(indices) < 2 or (selected_uids is not None and uid not in selected_uids):
                continue
            group_specs[uid] = indices
            candidate_limit = int(self.cpcr_config.get('max_candidates_per_group', 0))
            positions_by_turn = defaultdict(list)
            for position, global_idx in enumerate(indices):
                factual_suffix = self._cpcr_build_stitched_suffix(
                    data_batch,
                    global_idx,
                    global_idx,
                    role,
                )
                if factual_suffix:
                    positions_by_turn[turn_counts[global_idx]].append(position)
            active_positions = {
                position
                for positions in positions_by_turn.values()
                for position in positions
            }
            active_target_count += len(active_positions)
            candidates_by_turn = {}
            for turn_count, positions in positions_by_turn.items():
                if candidate_limit <= 0 or len(positions) <= candidate_limit:
                    candidates_by_turn[turn_count] = positions
                    continue
                # Uniform sampling is independent of outcome, so the common
                # inclusion probability cancels in the self-normalized MIS ratio.
                seed = (
                    int(getattr(self, 'global_steps', 0)) * 1_000_003
                    + group_idx * 10_007
                    + turn_count * 101
                ) % (2**32)
                rng = np.random.default_rng(seed)
                candidates_by_turn[turn_count] = sorted(
                    rng.choice(positions, size=candidate_limit, replace=False).tolist()
                )
            for target_pos, target_idx in enumerate(indices):
                if target_pos not in active_positions:
                    continue
                target_turn = turn_counts[indices[target_pos]]
                for candidate_pos in candidates_by_turn.get(target_turn, []):
                    candidate_idx = indices[candidate_pos]
                    suffix_steps = self._cpcr_build_stitched_suffix(
                        data_batch,
                        target_idx,
                        candidate_idx,
                        role,
                    )
                    if not suffix_steps:
                        skipped_pair_count += 1
                        continue
                    if target_idx == candidate_idx:
                        for suffix_role, chat, _ in suffix_steps:
                            factual_prompt_check_count += 1
                            factual_chat = self._cpcr_unpad_messages(
                                data_batch.non_tensor_batch[
                                    f'{suffix_role}_conversation_history'
                                ][candidate_idx]
                            )
                            factual_prompt_match_count += int(chat == factual_chat)
                    candidate_turn = turn_counts[candidate_idx] - 1
                    if candidate_turn < 0:
                        skipped_pair_count += 1
                        continue
                    pair_key = (uid, target_pos, candidate_pos)
                    pair_encoded = []
                    for suffix_role, chat, candidate_message in suffix_steps:
                        stop_reason = candidate_message.get('stop_reason', 'stop')
                        action_token_ids = self._cpcr_action_token_ids(
                            data_batch,
                            suffix_role,
                            candidate_idx,
                            candidate_turn,
                            stop_reason,
                        )
                        if action_token_ids is None:
                            pair_encoded = []
                            break
                        encoded = self._cpcr_encode_prompt_action(
                            chat,
                            action_token_ids,
                            prompt_length,
                            max_length,
                        )
                        if encoded is None:
                            pair_encoded = []
                            break
                        pair_encoded.append((suffix_role, encoded))
                    if not pair_encoded:
                        skipped_pair_count += 1
                        continue
                    expected_steps[pair_key] = len(pair_encoded)
                    for suffix_role, encoded in pair_encoded:
                        encoded_by_role[suffix_role].append((pair_key, encoded))

        if factual_prompt_match_count != factual_prompt_check_count:
            raise RuntimeError(
                "CPCR full-suffix replay diverged from factual rollout prompts "
                f"for role={role}: matched {factual_prompt_match_count}/"
                f"{factual_prompt_check_count}"
            )

        if not expected_steps:
            return {
                uid: torch.full((len(indices), len(indices)), -torch.inf)
                for uid, indices in group_specs.items()
            }, skipped_pair_count, 0, 0, 0, 0, 0, factual_prompt_match_count, factual_prompt_check_count, active_target_count

        pair_log_probs = defaultdict(float)
        scored_steps_per_pair = defaultdict(int)
        scored_action_tokens = 0
        scored_input_tokens = 0
        scored_suffix_actions = 0
        teacher_forcing_calls = 0
        for suffix_role, role_items in encoded_by_role.items():
            encoded_actions = [encoded for _, encoded in role_items]
            (
                sequence_log_probs,
                _,
                action_token_count,
                input_token_count,
                scoring_call_count,
            ) = self._score_teacher_forced_actions(
                suffix_role,
                encoded_actions,
            )
            scored_action_tokens += action_token_count
            scored_input_tokens += input_token_count
            scored_suffix_actions += len(role_items)
            teacher_forcing_calls += scoring_call_count
            for item_idx, (pair_key, _) in enumerate(role_items):
                pair_log_probs[pair_key] = pair_log_probs[pair_key] + sequence_log_probs[item_idx]
                scored_steps_per_pair[pair_key] += 1

        matrices = {
            uid: torch.full((len(indices), len(indices)), -torch.inf)
            for uid, indices in group_specs.items()
        }
        scored_pair_count = 0
        for pair_key, expected_count in expected_steps.items():
            if scored_steps_per_pair[pair_key] != expected_count:
                skipped_pair_count += 1
                continue
            uid, target_pos, candidate_pos = pair_key
            matrices[uid][target_pos, candidate_pos] = pair_log_probs[pair_key]
            scored_pair_count += 1
        return (
            matrices,
            skipped_pair_count,
            scored_pair_count,
            scored_suffix_actions,
            scored_action_tokens,
            scored_input_tokens,
            teacher_forcing_calls,
            factual_prompt_match_count,
            factual_prompt_check_count,
            active_target_count,
        )

    def _compute_cpcr_targets(self, data_batch, reward_tensor_map, metrics):
        if not self.cpcr_enabled:
            return {}

        role_setting = self.cpcr_config.get('roles', 'current')
        if role_setting == 'current':
            roles = [self._current_train_agent]
        elif role_setting == 'all':
            roles = list(data_batch.meta_info['agent_roles'])
        elif isinstance(role_setting, str):
            roles = [role_setting] if role_setting in data_batch.meta_info['agent_roles'] else []
        else:
            roles = [role for role in role_setting if role in data_batch.meta_info['agent_roles']]
        roles = [role for role in roles if role is not None and role in self.actor_rollout_wg]

        current_step = int(getattr(self, 'global_steps', 0))
        score_interval = max(int(self.cpcr_config.get('score_interval', 1)), 1)
        due_roles = []
        for role in roles:
            if self._cpcr_role_last_visit_step.get(role) != current_step:
                self._cpcr_role_visit_counts[role] += 1
                self._cpcr_role_last_visit_step[role] = current_step
            metrics[f'reward/cpcr/roles/{role}/visit_count'] = float(
                self._cpcr_role_visit_counts[role]
            )
            if self._cpcr_role_visit_counts[role] % score_interval == 0:
                due_roles.append(role)
        roles = due_roles
        if not roles:
            metrics['reward/cpcr/enabled'] = 1.0
            metrics.setdefault('reward/cpcr/scored_this_step', 0.0)
            return {}

        uid_to_indices = defaultdict(list)
        for idx, uid in enumerate(data_batch.non_tensor_batch['uid']):
            uid_to_indices[uid].append(idx)
        raw_scores = reward_tensor_map['acc'].float()
        min_ess = float(self.cpcr_config.get('min_effective_sample_size', 2.0))
        log_weight_clip = self.cpcr_config.get('log_weight_clip', 20.0)
        fallback = str(self.cpcr_config.get('fallback', 'group_loo'))

        metrics['reward/cpcr/enabled'] = 1.0
        mode = str(self.cpcr_config.get('mode', 'diagnostic'))
        metrics['reward/cpcr/prd_target_mode'] = float(
            mode == 'prd_target'
        )
        metrics['reward/cpcr/direct_scoped_grpo_mode'] = float(
            mode == 'direct_scoped_grpo'
        )
        targets = {}
        selected_uids = []
        mixed_groups_only = bool(self.cpcr_config.get('mixed_groups_only', True))
        for uid, indices in uid_to_indices.items():
            if len(indices) < 2:
                continue
            group_scores = raw_scores[torch.tensor(indices, dtype=torch.long)]
            is_mixed = bool((group_scores > 0).any() and (group_scores <= 0).any())
            if mixed_groups_only and not is_mixed:
                continue
            selected_uids.append(uid)
        group_limit = int(self.cpcr_config.get('max_groups_per_step', 0))
        if group_limit > 0 and len(selected_uids) > group_limit:
            rng = np.random.default_rng((current_step * 1_000_003 + 17) % (2**32))
            selected_positions = sorted(
                rng.choice(len(selected_uids), size=group_limit, replace=False).tolist()
            )
            selected_uids = [selected_uids[position] for position in selected_positions]
        selected_uids = set(selected_uids)
        metrics['reward/cpcr/selected_group_count'] = max(
            metrics.get('reward/cpcr/selected_group_count', 0.0),
            float(len(selected_uids)),
        )
        selected_target_count = sum(
            len(indices) for uid, indices in uid_to_indices.items() if uid in selected_uids
        )
        if not selected_uids or self._cpcr_last_scored_step == current_step:
            metrics.setdefault('reward/cpcr/scored_this_step', 0.0)
            return {}
        metrics.setdefault('reward/cpcr/scored_this_step', 0.0)
        total_scored_pairs = 0
        for role in roles:
            (
                matrices,
                skipped_pairs,
                scored_pairs,
                scored_suffix_actions,
                scored_action_tokens,
                scored_input_tokens,
                teacher_forcing_calls,
                factual_prompt_matches,
                factual_prompt_checks,
                active_targets,
            ) = self._cpcr_score_full_suffix_pairs(
                data_batch,
                role,
                selected_uids=selected_uids,
            )
            role_advantages = torch.zeros_like(raw_scores)
            role_valid = torch.zeros_like(raw_scores, dtype=torch.bool)
            role_ess = torch.zeros_like(raw_scores)
            baselines = []
            advantages = []
            ess_values = []
            fallback_values = []
            max_weights = []
            supported_target_count = 0
            for uid, indices in uid_to_indices.items():
                if uid not in matrices or len(indices) < 2:
                    continue
                idx_tensor = torch.tensor(indices, dtype=torch.long)
                estimate = estimate_cpcr(
                    matrices[uid],
                    raw_scores[idx_tensor],
                    min_effective_sample_size=min_ess,
                    log_weight_clip=log_weight_clip,
                    fallback=fallback,
                )
                use_fallback_targets = bool(self.cpcr_config.get('use_fallback_targets', False))
                reliable_mask = estimate.valid_target_mask
                if not use_fallback_targets:
                    reliable_mask = reliable_mask & ~estimate.fallback_mask
                role_advantages[idx_tensor] = estimate.advantage
                role_ess[idx_tensor] = estimate.effective_sample_size
                role_valid[idx_tensor] = reliable_mask
                supported = estimate.valid_target_mask
                if supported.any():
                    supported_target_count += int(supported.sum().item())
                    baselines.append(estimate.baseline[supported])
                    advantages.append(estimate.advantage[supported])
                    ess_values.append(estimate.effective_sample_size[supported])
                    fallback_values.append(estimate.fallback_mask[supported].float())
                    max_weights.append(
                        estimate.normalized_weights.max(dim=1).values[supported]
                    )

            prefix = f'reward/cpcr/roles/{role}'
            metrics[f'{prefix}/probe_ordinal'] = float(
                self._cpcr_role_probe_counts[role]
            )
            total_scored_pairs += scored_pairs
            metrics[f'{prefix}/pairs_scored'] = float(scored_pairs)
            metrics[f'{prefix}/pairs_skipped'] = float(skipped_pairs)
            metrics[f'{prefix}/active_target_count'] = float(active_targets)
            metrics[f'{prefix}/suffix_actions_scored'] = float(scored_suffix_actions)
            metrics[f'{prefix}/mean_suffix_length'] = (
                float(scored_suffix_actions) / float(scored_pairs)
                if scored_pairs > 0 else 0.0
            )
            metrics[f'{prefix}/pair_coverage'] = (
                float(scored_pairs) / float(scored_pairs + skipped_pairs)
                if scored_pairs + skipped_pairs > 0 else 0.0
            )
            metrics[f'{prefix}/factual_prompt_check_count'] = float(factual_prompt_checks)
            metrics[f'{prefix}/factual_prompt_match_rate'] = (
                float(factual_prompt_matches) / float(factual_prompt_checks)
                if factual_prompt_checks > 0 else 0.0
            )
            metrics[f'{prefix}/action_tokens_scored'] = float(scored_action_tokens)
            metrics[f'{prefix}/input_tokens_scored'] = float(scored_input_tokens)
            metrics[f'{prefix}/teacher_forcing_call_count'] = float(
                teacher_forcing_calls
            )
            metrics[f'{prefix}/supported_target_count'] = float(supported_target_count)
            metrics[f'{prefix}/reliable_target_count'] = float(role_valid.sum().item())
            metrics[f'{prefix}/target_coverage'] = float(role_valid.float().mean().item())
            metrics[f'{prefix}/selected_target_coverage'] = (
                float(role_valid.sum().item()) / float(selected_target_count)
                if selected_target_count > 0 else 0.0
            )
            if baselines:
                metrics[f'{prefix}/baseline_mean'] = float(torch.cat(baselines).mean().item())
                metrics[f'{prefix}/advantage_mean'] = float(torch.cat(advantages).mean().item())
                metrics[f'{prefix}/advantage_std'] = self._safe_tensor_std(torch.cat(advantages))
                metrics[f'{prefix}/ess_mean'] = float(torch.cat(ess_values).mean().item())
                metrics[f'{prefix}/ess_min'] = float(torch.cat(ess_values).min().item())
                metrics[f'{prefix}/fallback_rate'] = float(torch.cat(fallback_values).mean().item())
                metrics[f'{prefix}/max_weight_mean'] = float(torch.cat(max_weights).mean().item())

            targets[role] = {
                'advantage': role_advantages,
                'valid_mask': role_valid,
                'effective_sample_size': role_ess,
                'probe_ordinal': self._cpcr_role_probe_counts[role],
            }
            if role_valid.any():
                self._cpcr_role_probe_counts[role] += 1

        if total_scored_pairs > 0:
            self._cpcr_last_scored_step = current_step
            metrics['reward/cpcr/scored_this_step'] = 1.0
        return targets

    def _direct_scoped_role_requires_tss(self, data_batch, role):
        hierarchy = dict(data_batch.meta_info.get('hierarchy', {}))
        stage_roles = list(hierarchy.get('stage_roles', []))
        routing_mode = str(
            hierarchy.get('routing_mode', 'selector')
        ).lower()
        if routing_mode == 'derive_verify':
            return len(stage_roles) >= 2 and role == stage_roles[1]
        if routing_mode == 'sequential_plan':
            return role in stage_roles[:-1]
        return role in stage_roles[:-1]

    def _build_direct_scope_prompt_bundle(self, data_batch, role, sample_idx):
        """Return the factual prompt and role-appropriate TSS contrasts."""

        hierarchy = dict(data_batch.meta_info.get('hierarchy', {}))
        decomposer_role = hierarchy.get('decomposer_role', 'decomposer')
        max_planned_subtasks = int(
            hierarchy.get(
                'max_planned_subtasks',
                max(len(hierarchy.get('stage_roles', [])) - 1, 1),
            )
        )
        c3_turns = data_batch.non_tensor_batch.get('c3_action_turn')
        use_c3_snapshot = bool(
            c3_turns is not None
            and int(c3_turns[sample_idx]) >= 0
            and data_batch.meta_info.get('c3_focal_role') == role
        )
        if use_c3_snapshot:
            raw_assigned_subtasks = data_batch.non_tensor_batch[
                'c3_action_assigned_subtasks'
            ][sample_idx]
            if isinstance(raw_assigned_subtasks, np.ndarray):
                raw_assigned_subtasks = raw_assigned_subtasks.tolist()
            assigned_subtasks = []
            for item in raw_assigned_subtasks:
                if isinstance(item, np.ndarray):
                    item = item.tolist()
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    return None
                subtask_id, description = item
                if str(subtask_id).strip() and str(description).strip():
                    assigned_subtasks.append(
                        (str(subtask_id).upper(), str(description))
                    )
            if not assigned_subtasks:
                return None
            raw_plan_subtasks = data_batch.non_tensor_batch.get(
                'c3_action_plan_subtasks'
            )
            if raw_plan_subtasks is None:
                plan_subtasks = list(assigned_subtasks)
            else:
                raw_plan_subtasks = raw_plan_subtasks[sample_idx]
                if isinstance(raw_plan_subtasks, np.ndarray):
                    raw_plan_subtasks = raw_plan_subtasks.tolist()
                plan_subtasks = []
                for item in raw_plan_subtasks:
                    if isinstance(item, np.ndarray):
                        item = item.tolist()
                    if not isinstance(item, (list, tuple)) or len(item) != 2:
                        return None
                    subtask_id, description = item
                    if str(subtask_id).strip() and str(description).strip():
                        plan_subtasks.append(
                            (str(subtask_id).upper(), str(description))
                        )
                if not plan_subtasks:
                    plan_subtasks = list(assigned_subtasks)
            worker_message = {
                'assigned_subtasks': [
                    subtask_id for subtask_id, _ in assigned_subtasks
                ],
                'stop_reason': data_batch.non_tensor_batch[
                    'c3_action_stop_reason'
                ][sample_idx],
            }
        else:
            history = data_batch.non_tensor_batch['history'][sample_idx]
            plan_message = self._cpcr_latest_role_message(
                history,
                decomposer_role,
            )
            worker_message = self._cpcr_latest_role_message(history, role)
            if plan_message is None or worker_message is None:
                return None

            assigned_subtask_ids = worker_message.get(
                'assigned_subtasks',
                [],
            )
            if isinstance(assigned_subtask_ids, np.ndarray):
                assigned_subtask_ids = assigned_subtask_ids.tolist()
            if not isinstance(assigned_subtask_ids, (list, tuple)):
                return None
            assigned_subtask_ids = [
                str(subtask_id).upper()
                for subtask_id in assigned_subtask_ids
                if str(subtask_id).strip()
            ]
            if not assigned_subtask_ids:
                return None

            parsed_subtasks = MultiAgentRollout._extract_subtasks(
                plan_message['content'],
                max_subtasks=max_planned_subtasks,
            )
            plan_subtasks = list(parsed_subtasks)
            subtask_map = {
                str(subtask_id).upper(): description
                for subtask_id, description in parsed_subtasks
            }
            if any(
                subtask_id not in subtask_map
                for subtask_id in assigned_subtask_ids
            ):
                return None
            assigned_subtasks = [
                (subtask_id, subtask_map[subtask_id])
                for subtask_id in assigned_subtask_ids
            ]
        assigned_subtasks_text = MultiAgentRollout._format_subtasks(
            assigned_subtasks
        )

        chat_key = f'{role}_conversation_history'
        if chat_key not in data_batch.non_tensor_batch:
            return None
        factual_chat = self._cpcr_unpad_messages(
            data_batch.non_tensor_batch[chat_key][sample_idx]
        )
        routing_mode = str(
            hierarchy.get('routing_mode', 'selector')
        ).lower()
        if routing_mode == 'derive_verify':
            question = str(data_batch.non_tensor_batch['question'][sample_idx])
            verifier_counterfactuals = build_verifier_scope_counterfactuals(
                factual_chat,
                question,
                assigned_subtasks_text,
            )
            if verifier_counterfactuals is None:
                return None
            return {
                'factual': factual_chat,
                'contrasts': list(verifier_counterfactuals.items()),
                'tss_mode': 'verifier_role_dependency',
                'worker_message': worker_message,
            }

        counterfactual_chat = build_full_task_scope_counterfactual(
            factual_chat,
            assigned_subtasks_text,
        )
        if counterfactual_chat is None:
            return None
        assigned_ids = {
            str(subtask_id).upper()
            for subtask_id, _ in assigned_subtasks
        }
        wrong_subtask_chats = []
        for subtask_id, description in plan_subtasks:
            subtask_id = str(subtask_id).upper()
            if subtask_id in assigned_ids:
                continue
            replacement_text = MultiAgentRollout._format_subtasks([
                (subtask_id, description)
            ])
            wrong_chat = build_scope_assignment_counterfactual(
                factual_chat,
                assigned_subtasks_text,
                replacement_text,
            )
            if wrong_chat is None:
                return None
            wrong_subtask_chats.append((subtask_id, wrong_chat))
        if routing_mode == 'sequential_plan':
            contrasts = [('full_task', counterfactual_chat)]
            contrasts.extend([
                (f'wrong_subtask_{subtask_id}', wrong_chat)
                for subtask_id, wrong_chat in wrong_subtask_chats
            ])
            stage_roles = list(hierarchy.get('stage_roles', []))
            role_stage_idx = (
                stage_roles.index(role)
                if role in stage_roles else -1
            )
            if role_stage_idx > 0:
                question = str(
                    data_batch.non_tensor_batch['question'][sample_idx]
                )
                no_dependency_chat = (
                    build_dependency_removed_scope_counterfactual(
                        factual_chat,
                        question,
                        assigned_subtasks_text,
                    )
                )
                if no_dependency_chat is None:
                    return None
                contrasts.append(('dependency_removed', no_dependency_chat))
            return {
                'factual': factual_chat,
                'contrasts': contrasts,
                'tss_mode': 'sequential_scope_dependency',
                'worker_message': worker_message,
            }
        return {
            'factual': factual_chat,
            'full_task': counterfactual_chat,
            'wrong_subtasks': wrong_subtask_chats,
            'tss_mode': 'subtask_scope',
            'worker_message': worker_message,
        }

    def _score_direct_scope_tss(
        self,
        data_batch,
        role,
        candidate_mask,
        metrics,
        *,
        metric_namespace='direct_scoped_grpo',
        scope_threshold=None,
    ):
        """Score an action against role-appropriate teacher-forced contrasts."""

        batch_size = len(data_batch)
        margins = torch.zeros(batch_size, dtype=torch.float32)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool)
        factual_scores = torch.zeros(batch_size, dtype=torch.float32)
        primary_contrast_scores = torch.zeros(batch_size, dtype=torch.float32)
        contrast_scores = defaultdict(dict)
        if not bool(candidate_mask.any().item()):
            return margins, valid_mask, factual_scores, primary_contrast_scores

        prompt_length = int(self.config.actor_rollout_ref.rollout.prompt_length)
        max_length = prompt_length + int(
            self.config.actor_rollout_ref.rollout.response_length
        )
        encoded_items = []
        expected_prompt_kinds = {}
        scoring_config = (
            self.scoped_c3_grpo_config
            if self.scoped_c3_grpo_enabled
            else self.direct_scoped_grpo_config
        )
        contrast_plan_subtasks = bool(
            scoring_config.get('contrast_plan_subtasks', True)
        )
        max_wrong_contrasts = int(
            scoring_config.get('max_wrong_subtask_contrasts', 0)
        )
        verifier_role_contrast = bool(
            scoring_config.get('verifier_role_contrast', True)
        )
        verifier_dependency_contrast = bool(
            scoring_config.get('verifier_dependency_contrast', True)
        )
        sequential_full_task_contrast = bool(
            scoring_config.get('sequential_full_task_contrast', True)
        )
        sequential_identity_contrast = bool(
            scoring_config.get('sequential_identity_contrast', True)
        )
        sequential_dependency_contrast = bool(
            scoring_config.get('sequential_dependency_contrast', True)
        )
        for sample_idx in torch.nonzero(candidate_mask, as_tuple=False).flatten().tolist():
            prompt_bundle = self._build_direct_scope_prompt_bundle(
                data_batch,
                role,
                sample_idx,
            )
            if prompt_bundle is None:
                continue
            factual_chat = prompt_bundle['factual']
            worker_message = prompt_bundle['worker_message']
            tss_mode = prompt_bundle.get('tss_mode', 'subtask_scope')
            if tss_mode == 'verifier_role_dependency':
                contrasts = []
                for contrast_name, contrast_chat in prompt_bundle['contrasts']:
                    if contrast_name == 'solve_from_scratch' and not verifier_role_contrast:
                        continue
                    if contrast_name == 'dependency_removed' and not verifier_dependency_contrast:
                        continue
                    contrasts.append((contrast_name, contrast_chat))
            elif tss_mode == 'sequential_scope_dependency':
                contrasts = []
                named_contrasts = list(prompt_bundle['contrasts'])
                wrong_contrasts = [
                    item for item in named_contrasts
                    if item[0].startswith('wrong_subtask_')
                ]
                if wrong_contrasts:
                    rotation = sample_idx % len(wrong_contrasts)
                    wrong_contrasts = (
                        wrong_contrasts[rotation:]
                        + wrong_contrasts[:rotation]
                    )
                    if max_wrong_contrasts > 0:
                        wrong_contrasts = wrong_contrasts[
                            :max_wrong_contrasts
                        ]
                for contrast_name, contrast_chat in named_contrasts:
                    if contrast_name == 'full_task':
                        if sequential_full_task_contrast:
                            contrasts.append((contrast_name, contrast_chat))
                    elif contrast_name == 'dependency_removed':
                        if sequential_dependency_contrast:
                            contrasts.append((contrast_name, contrast_chat))
                if sequential_identity_contrast and contrast_plan_subtasks:
                    contrasts.extend(wrong_contrasts)
            else:
                contrasts = [('full_task', prompt_bundle['full_task'])]
                wrong_chats = (
                    list(prompt_bundle['wrong_subtasks'])
                    if contrast_plan_subtasks else []
                )
                if max_wrong_contrasts > 0:
                    wrong_chats = wrong_chats[:max_wrong_contrasts]
                contrasts.extend([
                    (f'wrong_subtask_{subtask_id}', wrong_chat)
                    for subtask_id, wrong_chat in wrong_chats
                ])
            if not contrasts:
                continue
            c3_action_turns = data_batch.non_tensor_batch.get('c3_action_turn')
            if (
                c3_action_turns is not None
                and int(c3_action_turns[sample_idx]) >= 0
                and data_batch.meta_info.get('c3_focal_role') == role
            ):
                turn_idx = int(c3_action_turns[sample_idx])
            else:
                turn_idx = (
                    int(data_batch.non_tensor_batch['num_turns'][sample_idx])
                    - 1
                )
            if turn_idx < 0:
                continue
            action_token_ids = self._cpcr_action_token_ids(
                data_batch,
                role,
                sample_idx,
                turn_idx,
                worker_message.get('stop_reason', 'stop'),
            )
            if action_token_ids is None:
                continue
            factual_encoded = self._cpcr_encode_prompt_action(
                factual_chat,
                action_token_ids,
                prompt_length,
                max_length,
            )
            if factual_encoded is None:
                continue
            sample_items = [
                (sample_idx, 'factual', None, factual_encoded),
            ]
            contrast_encoding_failed = False
            for contrast_name, contrast_chat in contrasts:
                contrast_encoded = self._cpcr_encode_prompt_action(
                    contrast_chat,
                    action_token_ids,
                    prompt_length,
                    max_length,
                )
                if contrast_encoded is None:
                    contrast_encoding_failed = True
                    break
                sample_items.append((
                    sample_idx,
                    'contrast',
                    contrast_name,
                    contrast_encoded,
                ))
            if contrast_encoding_failed:
                continue
            encoded_items.extend(sample_items)
            expected_prompt_kinds[sample_idx] = {
                'contrast_names': [name for name, _ in contrasts],
            }

        if not encoded_items:
            return margins, valid_mask, factual_scores, primary_contrast_scores

        (
            _,
            mean_log_probs,
            _,
            _,
            teacher_forcing_calls,
        ) = self._score_teacher_forced_actions(
            role,
            [encoded for _, _, _, encoded in encoded_items],
        )

        factual_seen = set()
        for item_idx, (
            sample_idx,
            prompt_kind,
            contrast_name,
            _,
        ) in enumerate(encoded_items):
            score = mean_log_probs[item_idx]
            if prompt_kind == 'factual':
                factual_scores[sample_idx] = score
                factual_seen.add(sample_idx)
            else:
                contrast_scores[sample_idx][contrast_name] = score

        contrast_margins = defaultdict(list)
        contrast_log_probs = defaultdict(list)
        contrast_wins = defaultdict(int)
        contrast_counts = defaultdict(int)
        total_contrast_count = 0
        samples_with_all_contrasts = 0
        for sample_idx, expected in expected_prompt_kinds.items():
            if sample_idx not in factual_seen:
                continue
            expected_names = list(expected['contrast_names'])
            sample_scores = contrast_scores[sample_idx]
            if any(name not in sample_scores for name in expected_names):
                continue
            sample_margins = []
            for contrast_name in expected_names:
                contrast_margin = (
                    factual_scores[sample_idx] - sample_scores[contrast_name]
                )
                sample_margins.append(contrast_margin)
                contrast_margins[contrast_name].append(contrast_margin)
                contrast_log_probs[contrast_name].append(
                    sample_scores[contrast_name]
                )
                contrast_wins[contrast_name] += int(
                    contrast_margin.item() > 0.0
                )
                contrast_counts[contrast_name] += 1
                total_contrast_count += 1
            margins[sample_idx] = torch.stack(sample_margins).min()
            primary_name = (
                'solve_from_scratch'
                if 'solve_from_scratch' in sample_scores
                else 'full_task'
                if 'full_task' in sample_scores
                else expected_names[0]
            )
            primary_contrast_scores[sample_idx] = sample_scores[primary_name]
            valid_mask[sample_idx] = True
            samples_with_all_contrasts += 1

        valid_margins = margins[valid_mask]
        prefix = f'reward/{metric_namespace}/tss/roles/{role}'
        hierarchy = dict(data_batch.meta_info.get('hierarchy', {}))
        verifier_tss_mode = (
            str(hierarchy.get('routing_mode', 'selector')).lower()
            == 'derive_verify'
        )
        sequential_tss_mode = (
            str(hierarchy.get('routing_mode', 'selector')).lower()
            == 'sequential_plan'
        )
        metrics[f'{prefix}/verifier_role_dependency_mode'] = float(
            verifier_tss_mode
        )
        metrics[f'{prefix}/role_contrast_enabled'] = float(
            verifier_tss_mode and verifier_role_contrast
        )
        metrics[f'{prefix}/dependency_contrast_enabled'] = float(
            (verifier_tss_mode and verifier_dependency_contrast)
            or (sequential_tss_mode and sequential_dependency_contrast)
        )
        metrics[f'{prefix}/sequential_scope_dependency_mode'] = float(
            sequential_tss_mode
        )
        metrics[f'{prefix}/full_task_contrast_enabled'] = float(
            sequential_tss_mode and sequential_full_task_contrast
        )
        metrics[f'{prefix}/identity_contrast_enabled'] = float(
            sequential_tss_mode
            and sequential_identity_contrast
            and contrast_plan_subtasks
        )
        metrics[f'{prefix}/candidate_count'] = float(candidate_mask.sum().item())
        metrics[f'{prefix}/valid_count'] = float(valid_mask.sum().item())
        metrics[f'{prefix}/teacher_forcing_call_count'] = float(
            teacher_forcing_calls
        )
        metrics[f'{prefix}/contrast_count'] = float(total_contrast_count)
        metrics[f'{prefix}/samples_with_all_contrasts_count'] = float(
            samples_with_all_contrasts
        )
        metrics[f'{prefix}/coverage'] = (
            float(valid_mask.sum().item()) / float(candidate_mask.sum().item())
            if candidate_mask.any() else 0.0
        )
        if valid_margins.numel() > 0:
            threshold = (
                float(scope_threshold)
                if scope_threshold is not None
                else float(
                    scoring_config.get(
                        'scope_threshold',
                        0.0,
                    )
                )
            )
            metrics[f'{prefix}/margin_mean'] = float(valid_margins.mean().item())
            metrics[f'{prefix}/margin_std'] = (
                float(valid_margins.std(unbiased=False).item())
                if valid_margins.numel() > 1 else 0.0
            )
            metrics[f'{prefix}/gate_open_rate'] = float(
                (valid_margins > threshold).float().mean().item()
            )
            metrics[f'{prefix}/positive_wins_all_rate'] = float(
                (valid_margins > threshold).float().mean().item()
            )
            metrics[f'{prefix}/factual_log_prob_per_token_mean'] = float(
                factual_scores[valid_mask].mean().item()
            )
            metrics[f'{prefix}/all_contrast_win_rate'] = float(
                (valid_margins > 0.0).float().mean().item()
            )
            for contrast_name, margin_values in contrast_margins.items():
                if not margin_values:
                    continue
                metric_name = contrast_name.lower()
                metrics[f'{prefix}/{metric_name}_margin_mean'] = float(
                    torch.stack(margin_values).mean().item()
                )
                metrics[f'{prefix}/{metric_name}_log_prob_per_token_mean'] = float(
                    torch.stack(contrast_log_probs[contrast_name]).mean().item()
                )
                metrics[f'{prefix}/{metric_name}_win_rate'] = (
                    float(contrast_wins[contrast_name])
                    / float(contrast_counts[contrast_name])
                )
            if contrast_margins.get('solve_from_scratch'):
                metrics[f'{prefix}/role_margin_mean'] = metrics[
                    f'{prefix}/solve_from_scratch_margin_mean'
                ]
                metrics[f'{prefix}/role_win_rate'] = metrics[
                    f'{prefix}/solve_from_scratch_win_rate'
                ]
            if contrast_margins.get('dependency_removed'):
                metrics[f'{prefix}/dependency_margin_mean'] = metrics[
                    f'{prefix}/dependency_removed_margin_mean'
                ]
                metrics[f'{prefix}/dependency_win_rate'] = metrics[
                    f'{prefix}/dependency_removed_win_rate'
                ]

            # Preserve legacy metric names for the general subtask protocol.
            if contrast_margins.get('full_task'):
                metrics[f'{prefix}/full_task_log_prob_per_token_mean'] = float(
                    primary_contrast_scores[valid_mask].mean().item()
                )
                metrics[f'{prefix}/local_vs_full_margin_mean'] = metrics[
                    f'{prefix}/full_task_margin_mean'
                ]
            wrong_names = [
                name for name in contrast_margins
                if name.startswith('wrong_subtask_')
            ]
            if wrong_names:
                wrong_values = [
                    value
                    for name in wrong_names
                    for value in contrast_margins[name]
                ]
                metrics[f'{prefix}/wrong_subtask_contrast_count'] = float(
                    len(wrong_values)
                )
                metrics[f'{prefix}/local_vs_wrong_subtask_margin_mean'] = float(
                    torch.stack(wrong_values).mean().item()
                )
        return margins, valid_mask, factual_scores, primary_contrast_scores

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
                messages = self._cpcr_unpad_messages(
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

        prefix = f'reward/scoped_c3_grpo/roles/{role}'
        group_count = len(uid_to_indices)
        metrics[f'{prefix}/candidate_group_count'] = float(group_count)
        metrics[f'{prefix}/exact_prefix_group_count'] = float(
            exact_group_count
        )
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

    @staticmethod
    def _c3_positive_effect_mask(outcome_scores, group_ids, candidate_mask):
        """Mark alternatives whose outcome beats their fixed-prefix peers."""

        positive_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
        uid_to_indices = defaultdict(list)
        for sample_idx, uid in enumerate(group_ids):
            if bool(candidate_mask[sample_idx].item()):
                uid_to_indices[uid].append(sample_idx)
        for indices in uid_to_indices.values():
            if len(indices) < 2:
                continue
            scores = outcome_scores[indices]
            peer_means = (
                scores.sum() - scores
            ) / float(len(indices) - 1)
            positive_mask[indices] = scores > peer_means
        return positive_mask

    def _attach_scoped_c3_grpo_signals(
        self,
        data_batch,
        reward_tensor_map,
        metrics,
    ):
        """Attach exact fixed-prefix outcomes and the existing stitched TSS."""

        if not self.scoped_c3_grpo_enabled:
            return

        role = self._current_train_agent
        batch_size = len(data_batch)
        outcome_scores = reward_tensor_map['acc'].float().cpu()
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
        causal_valid = action_present & exact_prefix
        pre_mixed_valid = causal_valid.clone()
        mixed_mask = torch.ones_like(causal_valid)
        if bool(
            self.scoped_c3_grpo_config.get('mixed_groups_only', True)
        ):
            mixed_mask = self._c3_mixed_group_mask(
                outcome_scores,
                data_batch.non_tensor_batch['uid'],
                causal_valid,
            )
            causal_valid &= mixed_mask

        scope_margin = torch.zeros(batch_size, dtype=torch.float32)
        scope_gate = torch.ones(batch_size, dtype=torch.bool)
        tss_valid = torch.zeros(batch_size, dtype=torch.bool)
        positive_effect = self._c3_positive_effect_mask(
            outcome_scores,
            data_batch.non_tensor_batch['uid'],
            causal_valid,
        )
        tss_required = self._direct_scoped_role_requires_tss(data_batch, role)
        if tss_required:
            threshold = float(
                self.scoped_c3_grpo_config.get('scope_threshold', 0.0)
            )
            (
                scope_margin,
                tss_valid,
                _,
                _,
            ) = self._score_direct_scope_tss(
                data_batch,
                role,
                positive_effect,
                metrics,
                metric_namespace='scoped_c3_grpo',
                scope_threshold=threshold,
            )
            scope_gate[positive_effect] = (
                tss_valid[positive_effect]
                & (scope_margin[positive_effect] > threshold)
            )
            target_valid = causal_valid & (
                ~positive_effect | tss_valid
            )
        else:
            target_valid = causal_valid

        data_batch.batch['scoped_c3_outcome_score'] = outcome_scores
        data_batch.batch['scoped_c3_causal_valid'] = causal_valid
        data_batch.batch['scoped_c3_scope_margin'] = scope_margin
        data_batch.batch['scoped_c3_scope_gate'] = scope_gate
        data_batch.batch['scoped_c3_tss_valid'] = tss_valid
        data_batch.batch['scoped_c3_positive_effect'] = positive_effect
        data_batch.batch['scoped_c3_target_valid'] = target_valid

        prefix = f'reward/scoped_c3_grpo/roles/{role}'
        metrics['reward/scoped_c3_grpo/enabled'] = 1.0
        metrics['reward/scoped_c3_grpo/manual_role_rewards_used'] = 0.0
        metrics['reward/scoped_c3_grpo/outcome_is_raw_acc'] = 1.0
        metrics[f'{prefix}/tss_required'] = float(tss_required)
        metrics[f'{prefix}/action_present_count'] = float(
            action_present.sum().item()
        )
        metrics[f'{prefix}/selected_action_turn'] = float(
            expected_branch_turn + 1
        )
        metrics[f'{prefix}/selected_action_turn_mismatch_count'] = float(
            wrong_action_turn.sum().item()
        )
        metrics[f'{prefix}/exact_prefix_sample_count'] = float(
            exact_prefix.sum().item()
        )
        metrics[f'{prefix}/pre_mixed_valid_count'] = float(
            pre_mixed_valid.sum().item()
        )
        metrics[f'{prefix}/mixed_valid_count'] = float(
            causal_valid.sum().item()
        )
        metrics[f'{prefix}/rejected_missing_action_count'] = float(
            (~action_present).sum().item()
        )
        metrics[f'{prefix}/rejected_prefix_mismatch_count'] = float(
            (action_present & ~exact_prefix).sum().item()
        )
        metrics[f'{prefix}/rejected_no_outcome_contrast_count'] = float(
            (pre_mixed_valid & ~mixed_mask).sum().item()
        )
        metrics[f'{prefix}/causal_valid_count'] = float(
            causal_valid.sum().item()
        )
        metrics[f'{prefix}/target_valid_count'] = float(
            target_valid.sum().item()
        )
        metrics[f'{prefix}/target_coverage'] = float(
            target_valid.float().mean().item()
        )
        if bool(causal_valid.any().item()):
            valid_scores = outcome_scores[causal_valid]
            metrics[f'{prefix}/outcome_mean'] = float(
                valid_scores.mean().item()
            )
            metrics[f'{prefix}/outcome_std'] = (
                float(valid_scores.std(unbiased=False).item())
                if valid_scores.numel() > 1 else 0.0
            )
        if tss_required:
            metrics[f'{prefix}/rejected_tss_unavailable_count'] = float(
                (positive_effect & ~tss_valid).sum().item()
            )
            metrics[f'{prefix}/scope_gate_open_count'] = float(
                (positive_effect & tss_valid & scope_gate).sum().item()
            )

    def _attach_direct_scoped_grpo_signals(
        self,
        data_batch,
        cpcr_targets,
        metrics,
    ):
        """Persist CPCR/TSS scalars so filtering and concatenation preserve them."""

        if not self.direct_scoped_grpo_enabled:
            return

        batch_size = len(data_batch)
        cpcr_score = torch.zeros(batch_size, dtype=torch.float32)
        cpcr_valid = torch.zeros(batch_size, dtype=torch.bool)
        scope_margin = torch.zeros(batch_size, dtype=torch.float32)
        scope_gate = torch.ones(batch_size, dtype=torch.bool)
        tss_valid = torch.zeros(batch_size, dtype=torch.bool)
        role = self._current_train_agent

        target_data = cpcr_targets.get(role)
        if target_data is not None:
            cpcr_score = target_data['advantage'].float().cpu()
            cpcr_valid = target_data['valid_mask'].bool().cpu()

        tss_required = self._direct_scoped_role_requires_tss(data_batch, role)
        if tss_required:
            (
                scope_margin,
                tss_valid,
                _,
                _,
            ) = self._score_direct_scope_tss(
                data_batch,
                role,
                cpcr_valid,
                metrics,
            )
            threshold = float(
                self.direct_scoped_grpo_config.get('scope_threshold', 0.0)
            )
            scope_gate = tss_valid & (scope_margin > threshold)
            target_valid = cpcr_valid & tss_valid
        else:
            target_valid = cpcr_valid

        data_batch.batch['direct_scoped_cpcr_score'] = cpcr_score
        data_batch.batch['direct_scoped_cpcr_valid'] = cpcr_valid
        data_batch.batch['direct_scoped_scope_margin'] = scope_margin
        data_batch.batch['direct_scoped_scope_gate'] = scope_gate
        data_batch.batch['direct_scoped_tss_valid'] = tss_valid
        data_batch.batch['direct_scoped_target_valid'] = target_valid

        prefix = f'reward/direct_scoped_grpo/roles/{role}'
        metrics['reward/direct_scoped_grpo/enabled'] = 1.0
        metrics[f'{prefix}/tss_required'] = float(tss_required)
        metrics[f'{prefix}/cpcr_valid_count'] = max(
            metrics.get(f'{prefix}/cpcr_valid_count', 0.0),
            float(cpcr_valid.sum().item()),
        )
        metrics[f'{prefix}/target_valid_count'] = max(
            metrics.get(f'{prefix}/target_valid_count', 0.0),
            float(target_valid.sum().item()),
        )
        metrics[f'{prefix}/target_coverage'] = max(
            metrics.get(f'{prefix}/target_coverage', 0.0),
            float(target_valid.float().mean().item()),
        )
        if bool(cpcr_valid.any().item()):
            valid_scores = cpcr_score[cpcr_valid]
            metrics[f'{prefix}/cpcr_score_mean'] = float(
                valid_scores.mean().item()
            )
            metrics[f'{prefix}/cpcr_score_std'] = (
                float(valid_scores.std(unbiased=False).item())
                if valid_scores.numel() > 1 else 0.0
            )

    def _compute_direct_scoped_grpo_advantage(self, data_batch, metrics):
        """Group-normalize CPCR, then suppress only out-of-scope positive credit."""

        estimate = estimate_direct_scoped_grpo(
            data_batch.batch['direct_scoped_cpcr_score'].float(),
            data_batch.non_tensor_batch['uid'],
            data_batch.batch['direct_scoped_target_valid'].bool(),
            data_batch.batch['direct_scoped_scope_gate'].float(),
            epsilon=float(
                self.direct_scoped_grpo_config.get('normalization_epsilon', 1e-6)
            ),
        )
        step_mask = data_batch.batch['step_ids'] != -100
        advantages = (
            estimate.scoped_advantage.unsqueeze(-1)
            * step_mask.to(dtype=estimate.scoped_advantage.dtype)
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
            ] = data_batch.batch['direct_scoped_cpcr_score'][reward_rows]
        data_batch.batch['token_level_scores'] = direct_rewards
        data_batch.batch['token_level_rewards'] = direct_rewards

        # Invalid counterfactual estimates must not contribute policy or entropy
        # gradients. Keeping the rows preserves rollout-group and DP batch shapes.
        ineffective_rows = ~estimate.effective_mask
        if ineffective_rows.any():
            data_batch.batch['labels'][ineffective_rows] = -100
            data_batch.batch['step_ids'][ineffective_rows] = -100
            data_batch.batch['advantages'][ineffective_rows] = 0.0
            data_batch.batch['returns'][ineffective_rows] = 0.0

        role = self._current_train_agent
        prefix = f'reward/direct_scoped_grpo/roles/{role}'
        effective = estimate.effective_mask
        base = estimate.base_advantage
        scoped = estimate.scoped_advantage
        gate = data_batch.batch['direct_scoped_scope_gate'].bool()
        positive = effective & (base > 0)
        negative = effective & (base < 0)
        gated_positive = positive & ~gate
        metrics[f'{prefix}/effective_sample_count'] = float(
            effective.sum().item()
        )
        metrics[f'{prefix}/effective_sample_rate'] = float(
            effective.float().mean().item()
        )
        metrics[f'{prefix}/effective_group_count'] = float(len({
            str(data_batch.non_tensor_batch['uid'][sample_idx])
            for sample_idx in torch.nonzero(effective, as_tuple=False).flatten().tolist()
        }))
        metrics[f'{prefix}/positive_advantage_count'] = float(
            positive.sum().item()
        )
        metrics[f'{prefix}/negative_advantage_count'] = float(
            negative.sum().item()
        )
        metrics[f'{prefix}/gated_positive_count'] = float(
            gated_positive.sum().item()
        )
        metrics[f'{prefix}/gated_positive_rate'] = (
            float(gated_positive.sum().item()) / float(positive.sum().item())
            if bool(positive.any().item()) else 0.0
        )
        if bool(effective.any().item()):
            metrics[f'{prefix}/base_advantage_mean'] = float(
                base[effective].mean().item()
            )
            metrics[f'{prefix}/base_advantage_std'] = (
                float(base[effective].std(unbiased=False).item())
                if int(effective.sum().item()) > 1 else 0.0
            )
            metrics[f'{prefix}/scoped_advantage_mean'] = float(
                scoped[effective].mean().item()
            )
            metrics[f'{prefix}/scoped_advantage_std'] = (
                float(scoped[effective].std(unbiased=False).item())
                if int(effective.sum().item()) > 1 else 0.0
            )
        return data_batch

    def _compute_scoped_c3_grpo_advantage(self, data_batch, metrics):
        """Build role-local C3 LOO advantages and apply TSS after comparison."""

        estimate = estimate_scoped_c3_grpo(
            data_batch.batch['scoped_c3_outcome_score'].float(),
            data_batch.non_tensor_batch['uid'],
            data_batch.batch['scoped_c3_causal_valid'].bool(),
            data_batch.batch['scoped_c3_scope_gate'].float(),
            update_mask=data_batch.batch['scoped_c3_target_valid'].bool(),
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
            estimate.scoped_advantage.unsqueeze(-1)
            * step_mask.to(dtype=estimate.scoped_advantage.dtype)
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
        prefix = f'reward/scoped_c3_grpo/roles/{role}'
        effective = estimate.effective_mask
        causal = estimate.causal_advantage
        scoped = estimate.scoped_advantage
        gate = data_batch.batch['scoped_c3_scope_gate'].bool()
        causal_valid = data_batch.batch['scoped_c3_causal_valid'].bool()
        update_valid = data_batch.batch['scoped_c3_target_valid'].bool()
        candidate_positive = causal_valid & (causal > 0)
        update_valid_positive = update_valid & (causal > 0)
        positive = effective & (causal > 0)
        negative = effective & (causal < 0)
        gated_positive = candidate_positive & (
            ~update_valid | ~gate
        )
        metrics[f'{prefix}/effective_sample_count'] = float(
            effective.sum().item()
        )
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
        metrics[f'{prefix}/positive_advantage_count'] = float(
            positive.sum().item()
        )
        metrics[f'{prefix}/candidate_positive_count'] = float(
            candidate_positive.sum().item()
        )
        metrics[f'{prefix}/update_valid_positive_count'] = float(
            update_valid_positive.sum().item()
        )
        metrics[f'{prefix}/negative_advantage_count'] = float(
            negative.sum().item()
        )
        metrics[f'{prefix}/gated_positive_count'] = float(
            gated_positive.sum().item()
        )
        metrics[f'{prefix}/gated_positive_rate'] = (
            float(gated_positive.sum().item())
            / float(candidate_positive.sum().item())
            if bool(candidate_positive.any().item()) else 0.0
        )
        if bool(effective.any().item()):
            metrics[f'{prefix}/causal_advantage_mean'] = float(
                causal[effective].mean().item()
            )
            metrics[f'{prefix}/causal_advantage_std'] = (
                float(causal[effective].std(unbiased=False).item())
                if int(effective.sum().item()) > 1 else 0.0
            )
            metrics[f'{prefix}/scoped_advantage_mean'] = float(
                scoped[effective].mean().item()
            )
            metrics[f'{prefix}/scoped_advantage_std'] = (
                float(scoped[effective].std(unbiased=False).item())
                if int(effective.sum().item()) > 1 else 0.0
            )
        return data_batch

    def _get_reward_composer_config(self) -> Dict:
        hierarchy_config = self.config.algorithm.get('hierarchy', {})
        reward_composer_config = hierarchy_config.get('reward_composer', {}) if hierarchy_config else {}
        return OmegaConf.to_container(reward_composer_config, resolve=True) if reward_composer_config else {}

    def _init_online_prd_composer(self):
        reward_composer_config = self._get_reward_composer_config()
        online_config = reward_composer_config.get('online', {})
        self.online_prd_config = online_config
        self.online_prd_enabled = bool(online_config.get('enable', False))
        self.online_prd_train = None
        self.online_prd_active = None
        self.online_prd_optimizer = None
        self.online_prd_training_objective = str(
            online_config.get('training_objective', 'group_ranking')
        )
        if self.online_prd_training_objective not in {'group_ranking', 'cpcr_replay'}:
            raise ValueError(
                f'Unsupported online PRD training objective: '
                f'{self.online_prd_training_objective}'
            )
        self.online_prd_replay_config = dict(online_config.get('cpcr_replay', {}))
        replay_capacity = int(self.online_prd_replay_config.get('capacity', 8192))
        self.online_prd_cpcr_train_buffer = deque(maxlen=max(replay_capacity, 1))
        self.online_prd_cpcr_validation_buffer = deque(maxlen=max(replay_capacity, 1))
        self.online_prd_validation_correlation_ema = None
        self.online_prd_validation_sign_accuracy_ema = None
        if not self.online_prd_enabled:
            return

        self.online_prd_model_type = str(online_config.get('model_type', 'source'))
        hidden_dim = int(online_config.get('hidden_dim', 128))
        source_embed_dim = int(online_config.get('source_embed_dim', 32))
        global_context_dim = online_config.get('global_context_dim', None)
        routing_activation_value = online_config.get('routing_activation', None)
        routing_activation = str(
            routing_activation_value
            or ('softmax' if self.online_prd_model_type == 'role' else 'sigmoid')
        )
        routing_floor = float(online_config.get('routing_floor', 0.0))
        score_activation = str(online_config.get('score_activation', 'identity'))
        if self.online_prd_training_objective == 'cpcr_replay':
            if self.online_prd_model_type != 'role':
                raise ValueError("training_objective='cpcr_replay' requires model_type='role'")
            hierarchy_config = self.config.algorithm.get('hierarchy', {})
            cpcr_config = hierarchy_config.get('cpcr', {}) if hierarchy_config else {}
            if not bool(cpcr_config.get('enable', False)) or str(
                cpcr_config.get('mode', 'diagnostic')
            ) != 'prd_target':
                raise ValueError(
                    "training_objective='cpcr_replay' requires CPCR mode='prd_target'"
                )
        init_checkpoint_path = online_config.get('init_checkpoint_path')
        if init_checkpoint_path and self.online_prd_model_type == 'role':
            self.online_prd_train = RolePRDCreditRouter.from_checkpoint(init_checkpoint_path, map_location='cpu')
            self.online_prd_train.score_activation = score_activation
        elif init_checkpoint_path:
            self.online_prd_train = PRDRewardComposer.from_checkpoint(init_checkpoint_path, map_location='cpu')
        elif self.online_prd_model_type == 'role':
            self.online_prd_train = RolePRDCreditRouter(
                role_feature_dim=len(ROLE_PRD_FEATURE_NAMES),
                hidden_dim=hidden_dim,
                routing_activation=routing_activation,
                routing_floor=routing_floor,
                score_activation=score_activation,
            )
        else:
            self.online_prd_train = PRDRewardComposer(
                num_sources=len(PRD_REWARD_SOURCE_NAMES),
                role_feature_dim=len(PRD_ROLE_FEATURE_NAMES),
                hidden_dim=hidden_dim,
                source_embed_dim=source_embed_dim,
                global_context_dim=global_context_dim,
                routing_activation=routing_activation,
                routing_floor=routing_floor,
            )
        self.online_prd_active = copy.deepcopy(self.online_prd_train)
        self.online_prd_active.eval()
        self.online_prd_optimizer = torch.optim.AdamW(
            self.online_prd_train.parameters(),
            lr=float(online_config.get('lr', 3e-4)),
            weight_decay=float(online_config.get('weight_decay', 0.0)),
        )

    def _get_online_prd_alpha(self) -> float:
        if not self.online_prd_enabled:
            return 0.0
        warmup_steps = int(self.online_prd_config.get('warmup_steps', 0))
        if getattr(self, 'global_steps', 0) <= warmup_steps:
            return 0.0
        alpha = float(self.online_prd_config.get('blend_alpha', 0.0))
        alpha_max = float(self.online_prd_config.get('blend_alpha_max', alpha))
        ramp_steps = int(self.online_prd_config.get('blend_ramp_steps', 0))
        if ramp_steps > 0:
            progress = min(1.0, max(0.0, (self.global_steps - warmup_steps) / float(ramp_steps)))
            alpha = alpha * progress
        return max(0.0, min(alpha_max, alpha))

    def _sync_online_prd_active(self):
        if not self.online_prd_enabled or self.online_prd_active is None:
            return False
        sync_interval = int(self.online_prd_config.get('sync_interval', 1))
        if sync_interval <= 0 or self.global_steps % sync_interval != 0:
            return False
        ema_beta = float(self.online_prd_config.get('ema_beta', 0.0))
        with torch.no_grad():
            active_state = self.online_prd_active.state_dict()
            train_state = self.online_prd_train.state_dict()
            for name, active_tensor in active_state.items():
                active_tensor.copy_(ema_beta * active_tensor + (1.0 - ema_beta) * train_state[name])
        return True

    @staticmethod
    def _reward_scalar(reward_tensor_map: Dict[str, torch.Tensor], name: str, batch_idx: int, default: float = 0.0) -> float:
        tensor = reward_tensor_map.get(name)
        if tensor is None:
            return float(default)
        return float(tensor[batch_idx].item())

    def _build_role_prd_features_for_sample(
        self,
        reward_tensor_map: Dict[str, torch.Tensor],
        batch_idx: int,
        agent_roles,
        score_role,
        worker_roles,
    ):
        worker_roles = set(worker_roles)
        denom = max(len(agent_roles) - 1, 1)
        rows = []
        activities = []

        decomposer_activity = (
            self._reward_scalar(reward_tensor_map, 'decomposer_plan_parseable_gate', batch_idx)
            * max(
                self._reward_scalar(reward_tensor_map, 'hierarchy_utilization_gate', batch_idx),
                self._reward_scalar(reward_tensor_map, 'decomposer_dependency_usage_rate', batch_idx),
            )
        )
        selector_activity = (
            self._reward_scalar(reward_tensor_map, 'selector_assignment_completeness', batch_idx)
            * self._reward_scalar(reward_tensor_map, 'selector_assignment_precision', batch_idx)
            * self._reward_scalar(reward_tensor_map, 'selector_assignment_final_present', batch_idx)
            * (1.0 - self._reward_scalar(reward_tensor_map, 'selector_empty_output', batch_idx))
        )
        worker_activity = (
            self._reward_scalar(reward_tensor_map, 'worker_unique_local_result_rate', batch_idx)
            * self._reward_scalar(reward_tensor_map, 'worker_downstream_used_rate', batch_idx)
            * (1.0 - self._reward_scalar(reward_tensor_map, 'worker_duplicate_result_penalty_applied', batch_idx))
        )
        final_activity = (
            self._reward_scalar(reward_tensor_map, 'final_worker_result_usage_rate', batch_idx)
            * self._reward_scalar(reward_tensor_map, 'final_consistency_with_worker_results', batch_idx)
            * (1.0 - self._reward_scalar(reward_tensor_map, 'final_ignores_worker_results_penalty_applied', batch_idx))
        )

        for idx, role in enumerate(agent_roles):
            is_decomposer = 1.0 if role == 'decomposer' else 0.0
            is_selector = 1.0 if role == 'selector' else 0.0
            is_worker = 1.0 if role in worker_roles else 0.0
            is_final = 1.0 if role == score_role else 0.0

            if is_decomposer:
                activity = decomposer_activity
            elif is_selector:
                activity = selector_activity
            elif is_final:
                activity = final_activity
            else:
                activity = worker_activity
            activities.append(float(max(0.0, min(1.0, activity))))

            rows.append([
                is_decomposer,
                is_selector,
                is_worker,
                is_final,
                float(idx) / float(denom),
                self._reward_scalar(reward_tensor_map, 'decomposer_plan_parseable_gate', batch_idx) if is_decomposer else 0.0,
                self._reward_scalar(reward_tensor_map, 'hierarchy_utilization_gate', batch_idx) if is_decomposer else 0.0,
                self._reward_scalar(reward_tensor_map, 'decomposer_dependency_usage_rate', batch_idx) if is_decomposer else 0.0,
                self._reward_scalar(reward_tensor_map, 'decomposer_repair_success', batch_idx) if is_decomposer else 0.0,
                self._reward_scalar(reward_tensor_map, 'planner_repeat_penalty_applied', batch_idx) if is_decomposer else 0.0,
                self._reward_scalar(reward_tensor_map, 'planner_excess_subtask_penalty_applied', batch_idx) if is_decomposer else 0.0,
                self._reward_scalar(reward_tensor_map, 'selector_assignment_completeness', batch_idx) if is_selector else 0.0,
                self._reward_scalar(reward_tensor_map, 'selector_assignment_precision', batch_idx) if is_selector else 0.0,
                self._reward_scalar(reward_tensor_map, 'selector_assignment_recall', batch_idx) if is_selector else 0.0,
                self._reward_scalar(reward_tensor_map, 'selector_assignment_final_present', batch_idx) if is_selector else 0.0,
                self._reward_scalar(reward_tensor_map, 'selector_empty_output', batch_idx) if is_selector else 0.0,
                self._reward_scalar(reward_tensor_map, 'selector_extra_assignment_penalty_applied', batch_idx) if is_selector else 0.0,
                self._reward_scalar(reward_tensor_map, 'selector_missing_assignment_penalty_applied', batch_idx) if is_selector else 0.0,
                self._reward_scalar(reward_tensor_map, 'selector_missing_final_penalty_applied', batch_idx) if is_selector else 0.0,
                self._reward_scalar(reward_tensor_map, 'worker_unique_local_result_rate', batch_idx) if is_worker else 0.0,
                self._reward_scalar(reward_tensor_map, 'worker_downstream_used_rate', batch_idx) if is_worker else 0.0,
                self._reward_scalar(reward_tensor_map, 'worker_later_worker_used_rate', batch_idx) if is_worker else 0.0,
                self._reward_scalar(reward_tensor_map, 'worker_missing_local_result_penalty_applied', batch_idx) if is_worker else 0.0,
                self._reward_scalar(reward_tensor_map, 'worker_duplicate_result_penalty_applied', batch_idx) if is_worker else 0.0,
                self._reward_scalar(reward_tensor_map, 'worker_subtask_overreach_penalty_applied', batch_idx) if is_worker else 0.0,
                self._reward_scalar(reward_tensor_map, 'final_worker_result_usage_rate', batch_idx) if is_final else 0.0,
                self._reward_scalar(reward_tensor_map, 'final_consistency_with_worker_results', batch_idx) if is_final else 0.0,
                self._reward_scalar(reward_tensor_map, 'final_ignores_worker_results_penalty_applied', batch_idx) if is_final else 0.0,
            ])
        return (
            torch.tensor(rows, dtype=torch.float32),
            torch.tensor(activities, dtype=torch.float32),
        )

    def _build_online_prd_batch(self, data_batch: DataProto, reward_tensor_map: Dict[str, torch.Tensor]):
        hierarchy_config = self._get_hierarchy_config()
        agent_roles = hierarchy_config.get('agent_roles', self._get_rollout_agent_roles())
        score_role = hierarchy_config.get('score_role', agent_roles[-1] if agent_roles else None)
        worker_roles = set(hierarchy_config.get('stage_roles', []))
        worker_roles.update(set(hierarchy_config.get('worker_roles', [])).intersection(set(agent_roles)))
        if score_role is not None:
            worker_roles.discard(score_role)
        use_manual_role_features = bool(self.online_prd_config.get('use_manual_role_features', False))
        model_type = str(self.online_prd_config.get('model_type', 'source'))

        if model_type == 'role':
            role_feature_rows = []
            role_activity_rows = []
            for i_bsz in range(len(data_batch)):
                role_features, role_activity = self._build_role_prd_features_for_sample(
                    reward_tensor_map,
                    i_bsz,
                    agent_roles,
                    score_role,
                    worker_roles,
                )
                role_feature_rows.append(role_features)
                role_activity_rows.append(role_activity)
            return {
                'model_type': 'role',
                'role_features': torch.stack(role_feature_rows, dim=0),
                'role_activity': torch.stack(role_activity_rows, dim=0),
                'agent_roles': agent_roles,
                'score_role': score_role,
                'graph_prior': build_role_prd_graph_prior_tensors(
                    agent_roles,
                    score_role,
                    worker_roles,
                    mode=str(self.online_prd_config.get('graph_prior_mode', 'none')),
                    soft_distance_penalty=float(self.online_prd_config.get('graph_prior_soft_distance_penalty', 1.0)),
                    reverse_distance_penalty=float(self.online_prd_config.get('graph_prior_reverse_distance_penalty', 3.0)),
                ),
            }

        source_rows = []
        role_feature_rows = []
        acc_tensor = reward_tensor_map['acc']
        for i_bsz in range(len(data_batch)):
            source_values = _build_prd_reward_sources(
                reward_tensor_map,
                i_bsz,
                float(acc_tensor[i_bsz].item()),
            )
            source_rows.append(build_prd_source_tensor(source_values, PRD_REWARD_SOURCE_NAMES))
            manual_role_scores = {}
            if use_manual_role_features:
                for role in agent_roles:
                    manual_role_scores[role] = float(
                        reward_tensor_map[f'{role}_turn_level_reward'][i_bsz].sum().item()
                    )
            role_feature_rows.append(
                build_prd_role_feature_tensor(
                    agent_roles,
                    score_role,
                    worker_roles,
                    manual_role_scores=manual_role_scores if use_manual_role_features else None,
                )
            )
        return {
            'model_type': 'source',
            'source_values': torch.stack(source_rows, dim=0),
            'role_features': torch.stack(role_feature_rows, dim=0),
            'agent_roles': agent_roles,
            'score_role': score_role,
            'graph_prior': build_prd_graph_prior_tensors(
                agent_roles,
                score_role,
                worker_roles,
                PRD_REWARD_SOURCE_NAMES,
                mode=str(self.online_prd_config.get('graph_prior_mode', 'none')),
                soft_distance_penalty=float(self.online_prd_config.get('graph_prior_soft_distance_penalty', 1.0)),
                reverse_distance_penalty=float(self.online_prd_config.get('graph_prior_reverse_distance_penalty', 3.0)),
            ),
        }

    @staticmethod
    def _online_prd_group_ranking_loss(role_scores, raw_scores, role_mask=None):
        if role_mask is None:
            rollout_scores = role_scores.sum(dim=1)
        else:
            rollout_scores = (role_scores * role_mask).sum(dim=1)
        positive = raw_scores.clamp_min(0.0)
        if float(positive.sum().item()) <= 0.0:
            return rollout_scores.sum() * 0.0
        target = positive / positive.sum().clamp_min(1e-8)
        return -(target * torch.log_softmax(rollout_scores, dim=0)).sum()

    @staticmethod
    def _online_prd_role_activity_ranking_loss(
        role_scores: torch.Tensor,
        raw_scores: torch.Tensor,
        role_activity: torch.Tensor,
        uid_list,
        *,
        activity_margin: float = 0.05,
        temperature: float = 1.0,
    ):
        uid_to_indices = defaultdict(list)
        for idx, uid in enumerate(uid_list):
            uid_to_indices[uid].append(idx)

        losses = []
        pair_count = 0
        eps = float(activity_margin)
        temp = max(float(temperature), 1e-6)
        for indices in uid_to_indices.values():
            if len(indices) < 2:
                continue
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=role_scores.device)
            group_raw = raw_scores.to(role_scores.device)[idx_tensor]
            pos_idx = idx_tensor[group_raw > 0]
            neg_idx = idx_tensor[group_raw <= 0]
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            pos_scores = role_scores[pos_idx]
            neg_scores = role_scores[neg_idx]
            pos_activity = role_activity.to(role_scores.device)[pos_idx]
            neg_activity = role_activity.to(role_scores.device)[neg_idx]
            score_diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
            activity_diff = pos_activity.unsqueeze(1) - neg_activity.unsqueeze(0)
            mask = activity_diff > eps
            if not bool(mask.any().item()):
                continue
            losses.append(-torch.nn.functional.logsigmoid(score_diff[mask] / temp).mean())
            pair_count += int(mask.sum().item())

        if not losses:
            return role_scores.sum() * 0.0, 0
        return torch.stack(losses).mean(), pair_count

    @staticmethod
    def _online_prd_implicit_counterfactual_loss(
        role_scores: torch.Tensor,
        raw_scores: torch.Tensor,
        role_features: torch.Tensor,
        uid_list,
        *,
        role_distance_margin: float = 0.05,
        huber_delta: float = 1.0,
    ):
        """Estimate a separate implicit counterfactual baseline per role.

        For each factual rollout and role, other rollouts of the same prompt
        form a weighted counterfactual baseline. A candidate receives high
        weight when the target role changed while the remaining roles stayed
        similar. The predicted role-score effect is regressed toward the
        resulting role-specific raw-score effect.
        """

        uid_to_indices = defaultdict(list)
        for idx, uid in enumerate(uid_list):
            uid_to_indices[uid].append(idx)

        num_roles = role_scores.shape[1]
        # Identity flags and role position are constant across rollouts and do
        # not describe behavioral changes, so exclude the first five fields.
        dynamic_features = role_features[..., 5:].to(role_scores.device)
        raw_scores = raw_scores.to(role_scores.device)
        margin = max(float(role_distance_margin), 0.0)
        delta = max(float(huber_delta), 1e-6)

        prompt_losses = []
        target_confidences = []
        selected_role_distances = []
        selected_other_distances = []
        candidate_pair_count = 0
        matched_pair_count = 0
        candidate_target_count = 0
        target_count = 0
        per_role_pair_count = [0 for _ in range(num_roles)]
        per_role_weight_sum = [0.0 for _ in range(num_roles)]
        per_role_target_count = [0 for _ in range(num_roles)]
        per_role_confidence_sum = [0.0 for _ in range(num_roles)]
        per_role_target_effect_sum = [0.0 for _ in range(num_roles)]
        per_role_target_effect_abs_sum = [0.0 for _ in range(num_roles)]
        target_effects = []

        for indices in uid_to_indices.values():
            if len(indices) < 2:
                continue
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=role_scores.device)
            group_raw = raw_scores[idx_tensor]
            if not bool(((group_raw > 0).any() & (group_raw <= 0).any()).item()):
                continue
            group_size = int(idx_tensor.numel())
            candidate_pair_count += group_size * (group_size - 1) * num_roles
            candidate_target_count += group_size * num_roles

            group_features = dynamic_features[idx_tensor]
            # [factual, candidate, role]: behavioral distance of each role.
            role_distances = (
                group_features.unsqueeze(1) - group_features.unsqueeze(0)
            ).abs().sum(dim=-1)
            if num_roles > 1:
                other_distances = (
                    role_distances.sum(dim=-1, keepdim=True) - role_distances
                ) / float(num_roles - 1)
            else:
                other_distances = torch.zeros_like(role_distances)

            role_dominance = role_distances / (
                role_distances + other_distances + 1e-8
            )
            other_similarity = 1.0 / (1.0 + other_distances)
            weights = role_dominance * other_similarity
            diagonal = torch.eye(
                group_size,
                dtype=torch.bool,
                device=role_scores.device,
            ).unsqueeze(-1)
            valid_pairs = (role_distances > margin) & ~diagonal
            weights = weights * valid_pairs.to(weights.dtype)
            if not bool(valid_pairs.any().item()):
                continue
            matched_pair_count += int(valid_pairs.sum().item())

            weight_sums = weights.sum(dim=1)
            valid_targets = weight_sums > 1e-8
            if not bool(valid_targets.any().item()):
                continue
            normalized_weights = weights / weight_sums.unsqueeze(1).clamp_min(1e-8)

            # Each (factual rollout, role) gets its own counterfactual outcome
            # and role-score baseline, averaged over matched candidate rollouts.
            counterfactual_raw = (
                normalized_weights * group_raw.view(1, group_size, 1)
            ).sum(dim=1)
            group_role_scores = role_scores[idx_tensor]
            counterfactual_role_scores = (
                normalized_weights * group_role_scores.unsqueeze(0)
            ).sum(dim=1)
            target_effect = group_raw.unsqueeze(-1) - counterfactual_raw
            predicted_effect = group_role_scores - counterfactual_role_scores
            target_losses = torch.nn.functional.huber_loss(
                predicted_effect,
                target_effect,
                reduction='none',
                delta=delta,
            )

            # The best available match determines confidence, while all valid
            # candidates contribute to the role-specific baseline.
            confidence = weights.max(dim=1).values
            prompt_loss = (
                target_losses[valid_targets] * confidence[valid_targets]
            ).sum() / confidence[valid_targets].sum().clamp_min(1e-8)
            prompt_losses.append(prompt_loss)
            target_confidences.append(confidence[valid_targets].detach())
            target_effects.append(target_effect[valid_targets].detach())
            selected_role_distances.append(role_distances[valid_pairs].detach())
            selected_other_distances.append(other_distances[valid_pairs].detach())
            target_count += int(valid_targets.sum().item())

            for role_idx in range(num_roles):
                role_pair_valid = valid_pairs[..., role_idx]
                role_pair_count = int(role_pair_valid.sum().item())
                if role_pair_count > 0:
                    per_role_pair_count[role_idx] += role_pair_count
                    per_role_weight_sum[role_idx] += float(
                        weights[..., role_idx][role_pair_valid].sum().detach().item()
                    )
                role_target_valid = valid_targets[..., role_idx]
                role_target_count = int(role_target_valid.sum().item())
                if role_target_count > 0:
                    per_role_target_count[role_idx] += role_target_count
                    per_role_confidence_sum[role_idx] += float(
                        confidence[..., role_idx][role_target_valid].sum().detach().item()
                    )
                    role_effects = target_effect[..., role_idx][role_target_valid]
                    per_role_target_effect_sum[role_idx] += float(
                        role_effects.sum().detach().item()
                    )
                    per_role_target_effect_abs_sum[role_idx] += float(
                        role_effects.abs().sum().detach().item()
                    )

        if not prompt_losses:
            return role_scores.sum() * 0.0, {
                'pair_count': 0,
                'candidate_pair_count': candidate_pair_count,
                'pair_coverage': 0.0,
                'target_count': 0,
                'candidate_target_count': candidate_target_count,
                'target_coverage': 0.0,
                'weight_sum': 0.0,
                'weight_mean': 0.0,
                'confidence_mean': 0.0,
                'target_effect_mean': 0.0,
                'target_effect_abs_mean': 0.0,
                'role_distance_mean': 0.0,
                'other_distance_mean': 0.0,
                'per_role_pair_count': per_role_pair_count,
                'per_role_weight_mean': [0.0 for _ in range(num_roles)],
                'per_role_target_count': per_role_target_count,
                'per_role_confidence_mean': [0.0 for _ in range(num_roles)],
                'per_role_target_effect_mean': [0.0 for _ in range(num_roles)],
                'per_role_target_effect_abs_mean': [0.0 for _ in range(num_roles)],
            }

        loss = torch.stack(prompt_losses).mean()
        all_target_confidences = torch.cat(target_confidences)
        all_target_effects = torch.cat(target_effects)
        all_role_distances = torch.cat(selected_role_distances)
        all_other_distances = torch.cat(selected_other_distances)
        per_role_weight_mean = [
            per_role_weight_sum[idx] / per_role_pair_count[idx]
            if per_role_pair_count[idx] > 0 else 0.0
            for idx in range(num_roles)
        ]
        per_role_confidence_mean = [
            per_role_confidence_sum[idx] / per_role_target_count[idx]
            if per_role_target_count[idx] > 0 else 0.0
            for idx in range(num_roles)
        ]
        per_role_target_effect_mean = [
            per_role_target_effect_sum[idx] / per_role_target_count[idx]
            if per_role_target_count[idx] > 0 else 0.0
            for idx in range(num_roles)
        ]
        per_role_target_effect_abs_mean = [
            per_role_target_effect_abs_sum[idx] / per_role_target_count[idx]
            if per_role_target_count[idx] > 0 else 0.0
            for idx in range(num_roles)
        ]
        return loss, {
            'pair_count': matched_pair_count,
            'candidate_pair_count': candidate_pair_count,
            'pair_coverage': (
                float(matched_pair_count) / float(candidate_pair_count)
                if candidate_pair_count > 0 else 0.0
            ),
            'target_count': target_count,
            'candidate_target_count': candidate_target_count,
            'target_coverage': (
                float(target_count) / float(candidate_target_count)
                if candidate_target_count > 0 else 0.0
            ),
            'weight_sum': float(all_target_confidences.sum().item()),
            'weight_mean': float(all_target_confidences.mean().item()),
            'confidence_mean': float(all_target_confidences.mean().item()),
            'target_effect_mean': float(all_target_effects.mean().item()),
            'target_effect_abs_mean': float(all_target_effects.abs().mean().item()),
            'role_distance_mean': float(all_role_distances.mean().item()),
            'other_distance_mean': float(all_other_distances.mean().item()),
            'per_role_pair_count': per_role_pair_count,
            'per_role_weight_mean': per_role_weight_mean,
            'per_role_target_count': per_role_target_count,
            'per_role_confidence_mean': per_role_confidence_mean,
            'per_role_target_effect_mean': per_role_target_effect_mean,
            'per_role_target_effect_abs_mean': per_role_target_effect_abs_mean,
        }

    @staticmethod
    def _safe_tensor_std(values: torch.Tensor) -> float:
        if values.numel() <= 1:
            return 0.0
        return float(values.float().std(unbiased=False).item())

    @staticmethod
    def _safe_binary_separation(scores: torch.Tensor, raw_scores: torch.Tensor):
        positives = scores[raw_scores > 0]
        negatives = scores[raw_scores <= 0]
        if positives.numel() == 0 or negatives.numel() == 0:
            return None
        return float((positives.mean() - negatives.mean()).item())

    @staticmethod
    def _safe_group_top1_accuracy(uid_list, rollout_scores: torch.Tensor, raw_scores: torch.Tensor):
        uid_to_indices = defaultdict(list)
        for idx, uid in enumerate(uid_list):
            uid_to_indices[uid].append(idx)
        correct = 0
        total = 0
        for indices in uid_to_indices.values():
            if len(indices) < 2:
                continue
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            group_raw = raw_scores[idx_tensor]
            if not ((group_raw > 0).any() and (group_raw <= 0).any()):
                continue
            group_scores = rollout_scores[idx_tensor]
            top_idx = idx_tensor[int(torch.argmax(group_scores).item())]
            correct += int(raw_scores[top_idx].item() > 0)
            total += 1
        if total == 0:
            return None
        return correct / total

    @staticmethod
    def _cpcr_replay_is_validation(
        problem_key,
        validation_fraction,
        role=None,
        probe_ordinal=None,
        split_mode='problem_hash',
    ):
        if validation_fraction <= 0.0:
            return False
        if validation_fraction >= 1.0:
            return True
        if (
            split_mode == 'role_probe_stratified'
            and role is not None
            and probe_ordinal is not None
        ):
            # Keep a complete role/problem probe together while guaranteeing
            # periodic validation support for every trained role.
            validation_period = max(int(round(1.0 / validation_fraction)), 1)
            return int(probe_ordinal) % validation_period == 0
        # Keep every role, rollout, and round for one problem in one split.
        split_key = str(problem_key).encode('utf-8')
        split_value = zlib.crc32(split_key) / float(2**32)
        return split_value < validation_fraction

    @staticmethod
    def _cpcr_replay_coverage(records, agent_roles):
        """Summarize independent replay support instead of only row count."""

        records = list(records)
        unique_problems = set()
        unique_probes = set()
        per_role = {
            role: {
                'record_count': 0,
                'unique_problem_count': 0,
                'unique_probe_group_count': 0,
            }
            for role in agent_roles
        }
        role_problems = defaultdict(set)
        role_probes = defaultdict(set)
        for record_idx, record in enumerate(records):
            role = str(record.get('role', ''))
            step = int(record.get('step', -1))
            problem_key = record.get('problem_key')
            if problem_key is None:
                # Older checkpoints were collected with one selected group per
                # step and did not persist the problem key. Grouping by step is
                # conservative and prevents rollout rows from posing as probes.
                problem_key = (
                    f'legacy-step-{step}'
                    if step >= 0 else f'legacy-record-{record_idx}'
                )
            problem_key = str(problem_key)
            probe_key = (problem_key, role, step)
            unique_problems.add(problem_key)
            unique_probes.add(probe_key)
            if role not in per_role:
                continue
            per_role[role]['record_count'] += 1
            role_problems[role].add(problem_key)
            role_probes[role].add(probe_key)

        for role in agent_roles:
            per_role[role]['unique_problem_count'] = len(role_problems[role])
            per_role[role]['unique_probe_group_count'] = len(role_probes[role])
        return {
            'record_count': len(records),
            'unique_problem_count': len(unique_problems),
            'unique_probe_group_count': len(unique_probes),
            'per_role': per_role,
        }

    def _append_online_prd_cpcr_records(
        self,
        cpcr_targets,
        role_features,
        problem_keys,
        agent_roles,
    ):
        """Store reliable CPCR labels for amortized, role-local supervision."""

        validation_fraction = float(
            self.online_prd_replay_config.get('validation_fraction', 0.1)
        )
        validation_split_mode = str(
            self.online_prd_replay_config.get(
                'validation_split_mode',
                'problem_hash',
            )
        )
        added_train = 0
        added_validation = 0
        per_role_added = defaultdict(int)
        for role, target_data in cpcr_targets.items():
            if role not in agent_roles:
                continue
            role_idx = agent_roles.index(role)
            valid_mask = target_data['valid_mask'].bool()
            advantages = target_data['advantage'].float()
            ess_values = target_data.get(
                'effective_sample_size',
                torch.zeros_like(advantages),
            ).float()
            probe_ordinal = target_data.get('probe_ordinal')
            for sample_idx in torch.where(valid_mask)[0].tolist():
                target_value = float(advantages[sample_idx].item())
                if not -1.0001 <= target_value <= 1.0001:
                    raise ValueError(
                        f'CPCR advantage outside [-1, 1]: role={role}, value={target_value}'
                    )
                record = {
                    'role_features': role_features[sample_idx].detach().cpu().float().clone(),
                    'role_idx': role_idx,
                    'role': role,
                    'problem_key': str(problem_keys[sample_idx]),
                    'target': max(-1.0, min(1.0, target_value)),
                    'ess': float(ess_values[sample_idx].item()),
                    'step': int(getattr(self, 'global_steps', 0)),
                    'probe_ordinal': (
                        int(probe_ordinal)
                        if probe_ordinal is not None else None
                    ),
                }
                if self._cpcr_replay_is_validation(
                    problem_keys[sample_idx],
                    validation_fraction,
                    role=role,
                    probe_ordinal=probe_ordinal,
                    split_mode=validation_split_mode,
                ):
                    self.online_prd_cpcr_validation_buffer.append(record)
                    added_validation += 1
                else:
                    self.online_prd_cpcr_train_buffer.append(record)
                    added_train += 1
                per_role_added[role] += 1
        return {
            'train': added_train,
            'validation': added_validation,
            'per_role': dict(per_role_added),
        }

    @staticmethod
    def _sample_cpcr_replay_records(records, batch_size, seed, balance_roles=True):
        records = list(records)
        if not records or batch_size <= 0:
            return []
        if len(records) <= batch_size:
            return records

        rng = np.random.default_rng(seed)
        if not balance_roles:
            indices = rng.choice(len(records), size=batch_size, replace=False)
            return [records[int(idx)] for idx in indices]

        indices_by_role = defaultdict(list)
        for idx, record in enumerate(records):
            indices_by_role[int(record['role_idx'])].append(idx)
        selected = []
        selected_set = set()
        per_role_quota = max(batch_size // max(len(indices_by_role), 1), 1)
        for role_idx in sorted(indices_by_role):
            role_indices = indices_by_role[role_idx]
            take = min(per_role_quota, len(role_indices), batch_size - len(selected))
            if take <= 0:
                break
            chosen = rng.choice(role_indices, size=take, replace=False).tolist()
            selected.extend(int(idx) for idx in chosen)
            selected_set.update(int(idx) for idx in chosen)

        if len(selected) < batch_size:
            remaining = [idx for idx in range(len(records)) if idx not in selected_set]
            take = min(batch_size - len(selected), len(remaining))
            if take > 0:
                selected.extend(
                    int(idx)
                    for idx in rng.choice(remaining, size=take, replace=False).tolist()
                )
        return [records[idx] for idx in selected]

    @staticmethod
    def _online_prd_cpcr_replay_loss(
        model,
        records,
        agent_roles,
        routing_bias,
        routing_mask,
        *,
        huber_delta=1.0,
    ):
        if not records:
            return None, {}
        device = next(model.parameters()).device
        role_features = torch.stack(
            [record['role_features'] for record in records],
            dim=0,
        ).to(device)
        role_indices = torch.tensor(
            [record['role_idx'] for record in records],
            dtype=torch.long,
            device=device,
        )
        targets = torch.tensor(
            [record['target'] for record in records],
            dtype=role_features.dtype,
            device=device,
        )
        output = model(
            role_features,
            attention_bias=routing_bias,
            attention_mask=routing_mask,
        )
        row_indices = torch.arange(len(records), device=device)
        predictions = output['role_scores'][row_indices, role_indices]
        delta = max(float(huber_delta), 1e-6)
        loss = torch.nn.functional.huber_loss(
            predictions,
            targets,
            reduction='mean',
            delta=delta,
        )

        detached_predictions = predictions.detach().float()
        detached_targets = targets.detach().float()
        errors = detached_predictions - detached_targets
        pred_centered = detached_predictions - detached_predictions.mean()
        target_centered = detached_targets - detached_targets.mean()
        corr_denom = torch.sqrt(
            pred_centered.square().sum() * target_centered.square().sum()
        )
        correlation = (
            float((pred_centered * target_centered).sum().item() / corr_denom.item())
            if float(corr_denom.item()) > 1e-12 else 0.0
        )
        nonzero_targets = detached_targets.abs() > 1e-6
        sign_accuracy = (
            float(
                (
                    torch.sign(detached_predictions[nonzero_targets])
                    == torch.sign(detached_targets[nonzero_targets])
                ).float().mean().item()
            )
            if bool(nonzero_targets.any().item()) else 0.0
        )
        stats = {
            'count': len(records),
            'loss': float(loss.detach().item()),
            'mae': float(errors.abs().mean().item()),
            'rmse': float(torch.sqrt(errors.square().mean()).item()),
            'correlation': correlation,
            'sign_accuracy': sign_accuracy,
            'prediction_mean': float(detached_predictions.mean().item()),
            'prediction_std': RayReMASeparatedTrainer._safe_tensor_std(detached_predictions),
            'target_mean': float(detached_targets.mean().item()),
            'target_std': RayReMASeparatedTrainer._safe_tensor_std(detached_targets),
            'ess_mean': float(np.mean([record['ess'] for record in records])),
            'per_role': {},
        }
        for role_idx, role in enumerate(agent_roles):
            mask = role_indices == role_idx
            if not bool(mask.any().item()):
                continue
            role_predictions = detached_predictions[mask]
            role_targets = detached_targets[mask]
            role_errors = role_predictions - role_targets
            role_pred_centered = role_predictions - role_predictions.mean()
            role_target_centered = role_targets - role_targets.mean()
            role_corr_denom = torch.sqrt(
                role_pred_centered.square().sum()
                * role_target_centered.square().sum()
            )
            role_nonzero = role_targets.abs() > 1e-6
            stats['per_role'][role] = {
                'count': int(mask.sum().item()),
                'mae': float(role_errors.abs().mean().item()),
                'rmse': float(torch.sqrt(role_errors.square().mean()).item()),
                'correlation': (
                    float(
                        (role_pred_centered * role_target_centered).sum().item()
                        / role_corr_denom.item()
                    )
                    if float(role_corr_denom.item()) > 1e-12 else 0.0
                ),
                'sign_accuracy': (
                    float(
                        (
                            torch.sign(role_predictions[role_nonzero])
                            == torch.sign(role_targets[role_nonzero])
                        ).float().mean().item()
                    )
                    if bool(role_nonzero.any().item()) else 0.0
                ),
                'prediction_mean': float(role_predictions.mean().item()),
                'target_mean': float(role_targets.mean().item()),
            }
        return loss, stats

    @staticmethod
    def _online_prd_cpcr_supervision_loss(
        role_scores,
        cpcr_targets,
        uid_list,
        agent_roles,
        turn_counts,
        *,
        huber_delta=1.0,
    ):
        """Fit PRD role scores to absolute CPCR role advantages."""

        boundary_to_indices = defaultdict(list)
        for idx, (uid, turn_count) in enumerate(zip(uid_list, turn_counts)):
            boundary_to_indices[(uid, int(turn_count))].append(idx)

        losses = []
        role_stats = {}
        delta = max(float(huber_delta), 1e-6)
        for role, target in cpcr_targets.items():
            if role not in agent_roles:
                continue
            role_idx = agent_roles.index(role)
            target_advantage = target['advantage'].to(role_scores.device).detach()
            valid_mask = target['valid_mask'].to(role_scores.device)
            role_losses = []
            target_count = 0
            group_count = 0
            no_contrast_group_count = 0
            for indices in boundary_to_indices.values():
                idx_tensor = torch.tensor(indices, dtype=torch.long, device=role_scores.device)
                group_valid = valid_mask[idx_tensor]
                if int(group_valid.sum().item()) < 2:
                    continue
                valid_indices = idx_tensor[group_valid]
                predicted = role_scores[valid_indices, role_idx]
                target_values = target_advantage[valid_indices]
                if float(target_values.abs().max().item()) <= 1e-8:
                    no_contrast_group_count += 1
                    continue
                role_losses.append(torch.nn.functional.huber_loss(
                    predicted,
                    target_values,
                    reduction='mean',
                    delta=delta,
                ))
                target_count += int(valid_indices.numel())
                group_count += 1
            if role_losses:
                role_loss = torch.stack(role_losses).mean()
                losses.append(role_loss)
                role_stats[role] = {
                    'loss': float(role_loss.detach().item()),
                    'target_count': target_count,
                    'group_count': group_count,
                    'no_contrast_group_count': no_contrast_group_count,
                }
            elif no_contrast_group_count > 0:
                role_stats[role] = {
                    'loss': 0.0,
                    'target_count': 0,
                    'group_count': 0,
                    'no_contrast_group_count': no_contrast_group_count,
                }

        if not losses:
            return None, role_stats
        return torch.stack(losses).mean(), role_stats

    def _apply_online_prd_cpcr_replay_rewards(
        self,
        data_batch,
        reward_tensor_map,
        metrics,
        cpcr_targets,
        prd_batch,
    ):
        """Train a lagged PRD model only from reliable full-suffix CPCR labels."""

        role_features = prd_batch['role_features']
        agent_roles = prd_batch['agent_roles']
        routing_bias, routing_mask = prd_batch['graph_prior']
        raw_scores = reward_tensor_map['acc'].float()
        uid_list = list(data_batch.non_tensor_batch['uid'])
        problem_keys = list(
            data_batch.non_tensor_batch.get('question', uid_list)
        )
        turn_counts = list(data_batch.non_tensor_batch['num_turns'])

        max_age_steps = int(self.online_prd_replay_config.get('max_age_steps', 500))
        if max_age_steps > 0:
            oldest_allowed_step = int(getattr(self, 'global_steps', 0)) - max_age_steps
            for replay_buffer in (
                self.online_prd_cpcr_train_buffer,
                self.online_prd_cpcr_validation_buffer,
            ):
                while replay_buffer and int(replay_buffer[0]['step']) < oldest_allowed_step:
                    replay_buffer.popleft()

        added = self._append_online_prd_cpcr_records(
            cpcr_targets or {},
            role_features,
            problem_keys,
            agent_roles,
        )
        replay_prefix = 'reward/prd_online/cpcr_replay'
        metrics['reward/prd_online/training_objective_id'] = 1.0
        metrics['reward/prd_online/group_ranking_effective_weight'] = 0.0
        metrics['reward/prd_online/implicit_cf_effective_weight'] = 0.0
        metrics['reward/prd_online/cpcr_supervision_configured'] = 1.0
        metrics[f'{replay_prefix}/validation_split_mode_id'] = {
            'problem_hash': 0.0,
            'role_probe_stratified': 1.0,
        }.get(
            str(
                self.online_prd_replay_config.get(
                    'validation_split_mode',
                    'problem_hash',
                )
            ),
            -1.0,
        )
        metrics[f'{replay_prefix}/added_train_count'] = (
            metrics.get(f'{replay_prefix}/added_train_count', 0.0) + float(added['train'])
        )
        metrics[f'{replay_prefix}/added_validation_count'] = (
            metrics.get(f'{replay_prefix}/added_validation_count', 0.0)
            + float(added['validation'])
        )
        metrics[f'{replay_prefix}/train_buffer_size'] = float(
            len(self.online_prd_cpcr_train_buffer)
        )
        metrics[f'{replay_prefix}/validation_buffer_size'] = float(
            len(self.online_prd_cpcr_validation_buffer)
        )
        train_coverage = self._cpcr_replay_coverage(
            self.online_prd_cpcr_train_buffer,
            agent_roles,
        )
        validation_coverage = self._cpcr_replay_coverage(
            self.online_prd_cpcr_validation_buffer,
            agent_roles,
        )
        for split, coverage in (
            ('train', train_coverage),
            ('validation', validation_coverage),
        ):
            metrics[f'{replay_prefix}/{split}/unique_problem_count'] = float(
                coverage['unique_problem_count']
            )
            metrics[f'{replay_prefix}/{split}/unique_probe_group_count'] = float(
                coverage['unique_probe_group_count']
            )
        for role in agent_roles:
            added_key = f'{replay_prefix}/roles/{role}/added_count'
            metrics[added_key] = (
                metrics.get(added_key, 0.0) + float(added['per_role'].get(role, 0))
            )
            metrics[f'{replay_prefix}/roles/{role}/train_buffer_count'] = float(
                train_coverage['per_role'][role]['record_count']
            )
            metrics[f'{replay_prefix}/roles/{role}/validation_buffer_count'] = float(
                validation_coverage['per_role'][role]['record_count']
            )
            metrics[f'{replay_prefix}/roles/{role}/train_unique_problem_count'] = float(
                train_coverage['per_role'][role]['unique_problem_count']
            )
            metrics[f'{replay_prefix}/roles/{role}/validation_unique_problem_count'] = float(
                validation_coverage['per_role'][role]['unique_problem_count']
            )
            metrics[f'{replay_prefix}/roles/{role}/train_unique_probe_group_count'] = float(
                train_coverage['per_role'][role]['unique_probe_group_count']
            )
            metrics[f'{replay_prefix}/roles/{role}/validation_unique_probe_group_count'] = float(
                validation_coverage['per_role'][role]['unique_probe_group_count']
            )

        min_train_size = int(self.online_prd_replay_config.get('min_train_size', 64))
        min_train_probes = int(
            self.online_prd_replay_config.get(
                'min_unique_train_probe_groups_for_training',
                0,
            )
        )
        replay_batch_size = int(self.online_prd_replay_config.get('batch_size', 256))
        balance_roles = bool(self.online_prd_replay_config.get('balance_roles', True))
        train_records = []
        train_data_ready = (
            len(self.online_prd_cpcr_train_buffer) >= min_train_size
            and train_coverage['unique_probe_group_count'] >= min_train_probes
        )
        metrics[f'{replay_prefix}/train_data_ready'] = float(train_data_ready)
        if train_data_ready:
            train_records = self._sample_cpcr_replay_records(
                self.online_prd_cpcr_train_buffer,
                replay_batch_size,
                seed=int(getattr(self, 'global_steps', 0)) * 1_000_003 + 29,
                balance_roles=balance_roles,
            )

        prd_updated = False
        train_replay_stats = {}
        if train_records:
            self.online_prd_train.train()
            replay_loss, train_replay_stats = self._online_prd_cpcr_replay_loss(
                self.online_prd_train,
                train_records,
                agent_roles,
                routing_bias,
                routing_mask,
                huber_delta=float(self.cpcr_config.get('prd_huber_delta', 1.0)),
            )
            self.online_prd_optimizer.zero_grad(set_to_none=True)
            replay_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.online_prd_train.parameters(),
                float(self.online_prd_config.get('grad_clip', 1.0)),
            )
            self.online_prd_optimizer.step()
            prd_updated = True
            metrics['reward/prd_online/loss'] = float(replay_loss.detach().item())
            metrics['reward/prd_online/cpcr_supervision_loss'] = float(
                replay_loss.detach().item()
            )
        else:
            metrics['reward/prd_online/loss'] = 0.0
        metrics['reward/prd_online/cpcr_supervision_active'] = float(prd_updated)
        metrics[f'{replay_prefix}/train_batch_size'] = float(len(train_records))
        metrics[f'{replay_prefix}/optimizer_step'] = float(prd_updated)
        for name, value in train_replay_stats.items():
            if name == 'per_role':
                continue
            metrics[f'{replay_prefix}/train/{name}'] = float(value)
        for role, role_stats in train_replay_stats.get('per_role', {}).items():
            for name, value in role_stats.items():
                metrics[f'{replay_prefix}/train/roles/{role}/{name}'] = float(value)

        validation_min_size = int(
            self.online_prd_replay_config.get('validation_min_size', 32)
        )
        validation_batch_size = int(
            self.online_prd_replay_config.get('validation_batch_size', 512)
        )
        validation_records = []
        if len(self.online_prd_cpcr_validation_buffer) >= validation_min_size:
            validation_records = self._sample_cpcr_replay_records(
                self.online_prd_cpcr_validation_buffer,
                validation_batch_size,
                seed=int(getattr(self, 'global_steps', 0)) * 1_000_003 + 43,
                balance_roles=balance_roles,
            )
        validation_stats = {}
        if validation_records:
            self.online_prd_active.eval()
            with torch.no_grad():
                _, validation_stats = self._online_prd_cpcr_replay_loss(
                    self.online_prd_active,
                    validation_records,
                    agent_roles,
                    routing_bias,
                    routing_mask,
                    huber_delta=float(self.cpcr_config.get('prd_huber_delta', 1.0)),
                )
        metrics[f'{replay_prefix}/validation_batch_size'] = float(len(validation_records))
        for name, value in validation_stats.items():
            if name == 'per_role':
                continue
            metrics[f'{replay_prefix}/validation/{name}'] = float(value)
        for role, role_stats in validation_stats.get('per_role', {}).items():
            for name, value in role_stats.items():
                metrics[f'{replay_prefix}/validation/roles/{role}/{name}'] = float(value)

        quality_ema_beta = float(
            self.online_prd_replay_config.get('validation_quality_ema_beta', 0.9)
        )
        quality_ema_beta = min(max(quality_ema_beta, 0.0), 1.0)
        if validation_stats:
            validation_correlation = float(validation_stats.get('correlation', 0.0))
            validation_sign_accuracy = float(validation_stats.get('sign_accuracy', 0.0))
            if self.online_prd_validation_correlation_ema is None:
                self.online_prd_validation_correlation_ema = validation_correlation
                self.online_prd_validation_sign_accuracy_ema = validation_sign_accuracy
            else:
                self.online_prd_validation_correlation_ema = (
                    quality_ema_beta * self.online_prd_validation_correlation_ema
                    + (1.0 - quality_ema_beta) * validation_correlation
                )
                self.online_prd_validation_sign_accuracy_ema = (
                    quality_ema_beta * self.online_prd_validation_sign_accuracy_ema
                    + (1.0 - quality_ema_beta) * validation_sign_accuracy
                )
        validation_correlation_ema = (
            float(self.online_prd_validation_correlation_ema)
            if self.online_prd_validation_correlation_ema is not None else 0.0
        )
        validation_sign_accuracy_ema = (
            float(self.online_prd_validation_sign_accuracy_ema)
            if self.online_prd_validation_sign_accuracy_ema is not None else 0.0
        )
        metrics[f'{replay_prefix}/validation/correlation_ema'] = validation_correlation_ema
        metrics[f'{replay_prefix}/validation/sign_accuracy_ema'] = validation_sign_accuracy_ema

        # Log current-batch predictions diagnostically. They do not contribute
        # to the replay-only loss above.
        self.online_prd_train.eval()
        with torch.no_grad():
            train_output = self.online_prd_train(
                role_features,
                attention_bias=routing_bias,
                attention_mask=routing_mask,
            )
        train_role_scores = train_output['role_scores'].detach()
        train_rollout_scores = train_role_scores.sum(dim=1)
        train_routing = train_output['routing'].detach()
        metrics['reward/prd_online/model_type_id'] = 1.0
        metrics['reward/prd_online/graph_prior_mode_id'] = {
            'none': 0.0,
            'soft': 1.0,
            'hard': 2.0,
        }.get(str(self.online_prd_config.get('graph_prior_mode', 'none')), -1.0)
        if routing_bias is not None:
            metrics['reward/prd_online/graph_prior_bias_mean'] = float(
                routing_bias.float().mean().item()
            )
            metrics['reward/prd_online/graph_prior_bias_std'] = self._safe_tensor_std(
                routing_bias.float()
            )
        if routing_mask is not None:
            metrics['reward/prd_online/graph_prior_mask_mean'] = float(
                routing_mask.float().mean().item()
            )
        metrics['reward/prd_online/train_routing_mean'] = float(train_routing.mean().item())
        metrics['reward/prd_online/train_routing_std'] = self._safe_tensor_std(train_routing)
        metrics['reward/prd_online/train_rollout_score_mean'] = float(
            train_rollout_scores.mean().item()
        )
        metrics['reward/prd_online/train_rollout_score_std'] = self._safe_tensor_std(
            train_rollout_scores
        )
        metrics['reward/prd_online/train_role_score_mean'] = float(
            train_role_scores.mean().item()
        )
        metrics['reward/prd_online/train_role_score_std'] = self._safe_tensor_std(
            train_role_scores
        )
        top1_acc = self._safe_group_top1_accuracy(uid_list, train_rollout_scores, raw_scores)
        if top1_acc is not None:
            metrics['reward/prd_online/train_group_top1_acc'] = top1_acc
        for role_idx, role in enumerate(agent_roles):
            metrics[f'reward/prd_online/roles/{role}/train_score_mean'] = float(
                train_role_scores[:, role_idx].mean().item()
            )
            metrics[f'reward/prd_online/roles/{role}/train_score_std'] = self._safe_tensor_std(
                train_role_scores[:, role_idx]
            )

        configured_alpha = self._get_online_prd_alpha()
        record_count_ready = len(self.online_prd_cpcr_train_buffer) >= int(
            self.online_prd_replay_config.get('min_size_for_blend', min_train_size)
        )
        probe_count_ready = train_coverage['unique_probe_group_count'] >= int(
            self.online_prd_replay_config.get(
                'min_unique_train_probe_groups_for_blend',
                0,
            )
        )
        min_train_records_per_role = int(
            self.online_prd_replay_config.get(
                'min_train_records_per_role_for_blend',
                0,
            )
        )
        train_roles_ready = all(
            train_coverage['per_role'][role]['record_count'] >= min_train_records_per_role
            for role in agent_roles
        )
        min_train_probes_per_role = int(
            self.online_prd_replay_config.get(
                'min_train_probe_groups_per_role_for_blend',
                0,
            )
        )
        train_role_probes_ready = all(
            train_coverage['per_role'][role]['unique_probe_group_count']
            >= min_train_probes_per_role
            for role in agent_roles
        )
        min_validation_records_per_role = int(
            self.online_prd_replay_config.get(
                'min_validation_records_per_role_for_blend',
                0,
            )
        )
        validation_roles_ready = all(
            validation_coverage['per_role'][role]['record_count']
            >= min_validation_records_per_role
            for role in agent_roles
        )
        min_validation_probes_per_role = int(
            self.online_prd_replay_config.get(
                'min_validation_probe_groups_per_role_for_blend',
                0,
            )
        )
        validation_role_probes_ready = all(
            validation_coverage['per_role'][role]['unique_probe_group_count']
            >= min_validation_probes_per_role
            for role in agent_roles
        )
        min_validation_correlation = float(
            self.online_prd_replay_config.get(
                'min_validation_correlation_for_blend',
                -1.0,
            )
        )
        validation_correlation_ready = (
            min_validation_correlation < 0.0
            or (
                bool(validation_stats)
                and validation_correlation_ema >= min_validation_correlation
            )
        )
        min_validation_sign_accuracy = float(
            self.online_prd_replay_config.get(
                'min_validation_sign_accuracy_for_blend',
                -1.0,
            )
        )
        validation_sign_ready = (
            min_validation_sign_accuracy < 0.0
            or (
                bool(validation_stats)
                and validation_sign_accuracy_ema >= min_validation_sign_accuracy
            )
        )
        blend_ready = (
            record_count_ready
            and probe_count_ready
            and train_roles_ready
            and train_role_probes_ready
            and validation_roles_ready
            and validation_role_probes_ready
            and validation_correlation_ready
            and validation_sign_ready
        )
        alpha = configured_alpha if blend_ready else 0.0
        metrics['reward/prd_online/configured_blend_alpha'] = configured_alpha
        metrics['reward/prd_online/blend_ready'] = float(blend_ready)
        metrics['reward/prd_online/blend_alpha'] = alpha
        metrics[f'{replay_prefix}/blend_gate/record_count_ready'] = float(record_count_ready)
        metrics[f'{replay_prefix}/blend_gate/probe_count_ready'] = float(probe_count_ready)
        metrics[f'{replay_prefix}/blend_gate/train_roles_ready'] = float(train_roles_ready)
        metrics[f'{replay_prefix}/blend_gate/train_role_probes_ready'] = float(
            train_role_probes_ready
        )
        metrics[f'{replay_prefix}/blend_gate/validation_roles_ready'] = float(
            validation_roles_ready
        )
        metrics[f'{replay_prefix}/blend_gate/validation_role_probes_ready'] = float(
            validation_role_probes_ready
        )
        metrics[f'{replay_prefix}/blend_gate/validation_correlation_ready'] = float(
            validation_correlation_ready
        )
        metrics[f'{replay_prefix}/blend_gate/validation_sign_ready'] = float(
            validation_sign_ready
        )

        if alpha > 0.0:
            self.online_prd_active.eval()
            with torch.no_grad():
                active_output = self.online_prd_active(
                    role_features,
                    attention_bias=routing_bias,
                    attention_mask=routing_mask,
                )
            prd_role_scores = active_output['role_scores']
            active_routing = active_output['routing']
            active_rollout_scores = prd_role_scores.sum(dim=1)
            metrics['reward/prd_online/active_routing_mean'] = float(
                active_routing.mean().item()
            )
            metrics['reward/prd_online/active_routing_std'] = self._safe_tensor_std(
                active_routing
            )
            metrics['reward/prd_online/active_role_score_mean'] = float(
                prd_role_scores.mean().item()
            )
            metrics['reward/prd_online/active_role_score_std'] = self._safe_tensor_std(
                prd_role_scores
            )
            metrics['reward/prd_online/active_rollout_score_mean'] = float(
                active_rollout_scores.mean().item()
            )
            metrics['reward/prd_online/active_rollout_score_std'] = self._safe_tensor_std(
                active_rollout_scores
            )

            last_turn_indices = torch.tensor(
                [max(int(num_turn) - 1, 0) for num_turn in turn_counts],
                dtype=torch.long,
            )
            for role_idx, role in enumerate(agent_roles):
                key = f'{role}_turn_level_reward'
                if key not in reward_tensor_map:
                    continue
                reward_tensor = reward_tensor_map[key].clone()
                batch_indices = torch.arange(reward_tensor.shape[0], dtype=torch.long)
                turn_indices = last_turn_indices.to(reward_tensor.device)
                batch_indices_device = batch_indices.to(reward_tensor.device)
                manual_scores = reward_tensor[
                    batch_indices_device,
                    turn_indices,
                ].detach()
                prd_scores = prd_role_scores[:, role_idx].to(
                    device=reward_tensor.device,
                    dtype=reward_tensor.dtype,
                )
                blended_scores = (1.0 - alpha) * manual_scores + alpha * prd_scores
                reward_tensor[batch_indices_device, turn_indices] = blended_scores
                reward_tensor_map[key] = reward_tensor
                blend_delta = (blended_scores - manual_scores).detach()
                metrics[f'reward/prd_online/roles/{role}/manual_score_mean'] = float(
                    manual_scores.mean().item()
                )
                metrics[f'reward/prd_online/roles/{role}/active_score_mean'] = float(
                    prd_scores.mean().item()
                )
                metrics[f'reward/prd_online/roles/{role}/blended_score_mean'] = float(
                    blended_scores.mean().item()
                )
                metrics[f'reward/prd_online/roles/{role}/blend_delta_mean'] = float(
                    blend_delta.mean().item()
                )
                metrics[f'reward/prd_online/roles/{role}/blend_delta_abs_max'] = float(
                    blend_delta.abs().max().item()
                )

        # The current batch always uses the pre-update active network. EMA is
        # applied only afterwards, preventing same-batch target leakage.
        ema_updated = self._sync_online_prd_active() if prd_updated else False
        metrics[f'{replay_prefix}/ema_update_applied'] = float(ema_updated)
        return reward_tensor_map

    def _apply_online_prd_rewards(
        self,
        data_batch: DataProto,
        reward_tensor_map: Dict[str, torch.Tensor],
        metrics: Dict,
        cpcr_targets=None,
    ):
        if not self.online_prd_enabled:
            return reward_tensor_map
        prd_batch = self._build_online_prd_batch(data_batch, reward_tensor_map)
        if self.online_prd_training_objective == 'cpcr_replay':
            return self._apply_online_prd_cpcr_replay_rewards(
                data_batch,
                reward_tensor_map,
                metrics,
                cpcr_targets,
                prd_batch,
            )
        model_type = prd_batch.get('model_type', 'source')
        source_values = prd_batch.get('source_values')
        role_features = prd_batch['role_features']
        agent_roles = prd_batch['agent_roles']
        routing_bias, routing_mask = prd_batch['graph_prior']

        # Train PRD_train from current rollout groups.
        self.online_prd_train.train()
        if model_type == 'role':
            train_output = self.online_prd_train(
                role_features,
                attention_bias=routing_bias,
                attention_mask=routing_mask,
            )
        else:
            train_output = self.online_prd_train(
                source_values,
                role_features,
                routing_bias=routing_bias,
                routing_mask=routing_mask,
            )
        raw_scores = reward_tensor_map['acc'].float()
        uid_list = list(data_batch.non_tensor_batch['uid'])
        uid_to_indices = defaultdict(list)
        for idx, uid in enumerate(uid_list):
            uid_to_indices[uid].append(idx)

        losses = []
        mixed_group_count = 0
        skipped_group_count = 0
        mixed_only = bool(self.online_prd_config.get('mixed_groups_only', True))
        for indices in uid_to_indices.values():
            if len(indices) < int(self.online_prd_config.get('min_group_size', 2)):
                skipped_group_count += 1
                continue
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            group_raw_scores = raw_scores[idx_tensor]
            is_mixed = bool((group_raw_scores > 0).any() and (group_raw_scores <= 0).any())
            if mixed_only and not is_mixed:
                skipped_group_count += 1
                continue
            if is_mixed:
                mixed_group_count += 1
            losses.append(
                self._online_prd_group_ranking_loss(
                    train_output['role_scores'][idx_tensor],
                    group_raw_scores,
                )
            )

        role_rank_loss = None
        role_rank_pair_count = 0
        if model_type == 'role' and float(self.online_prd_config.get('role_rank_loss_weight', 0.0)) > 0.0:
            role_rank_loss, role_rank_pair_count = self._online_prd_role_activity_ranking_loss(
                train_output['role_scores'],
                raw_scores,
                prd_batch['role_activity'],
                uid_list,
                activity_margin=float(self.online_prd_config.get('role_activity_margin', 0.05)),
                temperature=float(self.online_prd_config.get('role_rank_temperature', 1.0)),
            )

        cpcr_targets = cpcr_targets or {}
        cpcr_mode = str(self.cpcr_config.get('mode', 'diagnostic')) if self.cpcr_enabled else 'diagnostic'
        cpcr_supervision_configured = cpcr_mode == 'prd_target'
        use_cpcr_supervision = bool(cpcr_targets) and cpcr_supervision_configured
        metrics['reward/prd_online/cpcr_supervision_configured'] = float(
            cpcr_supervision_configured
        )

        implicit_cf_loss = None
        implicit_cf_stats = None
        implicit_cf_weight = float(self.online_prd_config.get('implicit_cf_loss_weight', 0.0))
        if cpcr_supervision_configured and not bool(self.cpcr_config.get('combine_with_implicit_cf', False)):
            implicit_cf_weight = 0.0
        metrics['reward/prd_online/implicit_cf_effective_weight'] = implicit_cf_weight
        if model_type == 'role' and implicit_cf_weight > 0.0:
            implicit_cf_loss, implicit_cf_stats = self._online_prd_implicit_counterfactual_loss(
                train_output['role_scores'],
                raw_scores,
                role_features,
                uid_list,
                role_distance_margin=float(
                    self.online_prd_config.get('implicit_cf_role_distance_margin', 0.05)
                ),
                huber_delta=float(self.online_prd_config.get('implicit_cf_huber_delta', 1.0)),
            )

        cpcr_loss = None
        cpcr_role_stats = {}
        if model_type == 'role' and use_cpcr_supervision:
            cpcr_loss, cpcr_role_stats = self._online_prd_cpcr_supervision_loss(
                train_output['role_scores'],
                cpcr_targets,
                uid_list,
                agent_roles,
                data_batch.non_tensor_batch['num_turns'],
                huber_delta=float(self.cpcr_config.get('prd_huber_delta', 1.0)),
            )
        metrics['reward/prd_online/cpcr_supervision_active'] = float(cpcr_loss is not None)

        if losses or role_rank_loss is not None or implicit_cf_loss is not None or cpcr_loss is not None:
            prd_loss = torch.stack(losses).mean() if losses else train_output['role_scores'].sum() * 0.0
            if role_rank_loss is not None:
                role_rank_weight = float(self.online_prd_config.get('role_rank_loss_weight', 0.0))
                prd_loss = prd_loss + role_rank_weight * role_rank_loss
                metrics['reward/prd_online/role_rank_loss'] = float(role_rank_loss.item())
                metrics['reward/prd_online/role_rank_pair_count'] = role_rank_pair_count
            if implicit_cf_loss is not None:
                prd_loss = prd_loss + implicit_cf_weight * implicit_cf_loss
                metrics['reward/prd_online/implicit_cf_loss'] = float(implicit_cf_loss.item())
                metrics['reward/prd_online/implicit_cf_pair_count'] = implicit_cf_stats['pair_count']
                metrics['reward/prd_online/implicit_cf_candidate_pair_count'] = (
                    implicit_cf_stats['candidate_pair_count']
                )
                metrics['reward/prd_online/implicit_cf_pair_coverage'] = (
                    implicit_cf_stats['pair_coverage']
                )
                metrics['reward/prd_online/implicit_cf_target_count'] = (
                    implicit_cf_stats['target_count']
                )
                metrics['reward/prd_online/implicit_cf_candidate_target_count'] = (
                    implicit_cf_stats['candidate_target_count']
                )
                metrics['reward/prd_online/implicit_cf_target_coverage'] = (
                    implicit_cf_stats['target_coverage']
                )
                metrics['reward/prd_online/implicit_cf_weight_sum'] = implicit_cf_stats['weight_sum']
                metrics['reward/prd_online/implicit_cf_weight_mean'] = implicit_cf_stats['weight_mean']
                metrics['reward/prd_online/implicit_cf_confidence_mean'] = (
                    implicit_cf_stats['confidence_mean']
                )
                metrics['reward/prd_online/implicit_cf_target_effect_mean'] = (
                    implicit_cf_stats['target_effect_mean']
                )
                metrics['reward/prd_online/implicit_cf_target_effect_abs_mean'] = (
                    implicit_cf_stats['target_effect_abs_mean']
                )
                metrics['reward/prd_online/implicit_cf_role_distance_mean'] = (
                    implicit_cf_stats['role_distance_mean']
                )
                metrics['reward/prd_online/implicit_cf_other_distance_mean'] = (
                    implicit_cf_stats['other_distance_mean']
                )
                for role_idx, role in enumerate(agent_roles):
                    metrics[f'reward/prd_online/roles/{role}/implicit_cf_pair_count'] = (
                        implicit_cf_stats['per_role_pair_count'][role_idx]
                    )
                    metrics[f'reward/prd_online/roles/{role}/implicit_cf_weight_mean'] = (
                        implicit_cf_stats['per_role_weight_mean'][role_idx]
                    )
                    metrics[f'reward/prd_online/roles/{role}/implicit_cf_target_count'] = (
                        implicit_cf_stats['per_role_target_count'][role_idx]
                    )
                    metrics[f'reward/prd_online/roles/{role}/implicit_cf_confidence_mean'] = (
                        implicit_cf_stats['per_role_confidence_mean'][role_idx]
                    )
                    metrics[f'reward/prd_online/roles/{role}/implicit_cf_target_effect_mean'] = (
                        implicit_cf_stats['per_role_target_effect_mean'][role_idx]
                    )
                    metrics[f'reward/prd_online/roles/{role}/implicit_cf_target_effect_abs_mean'] = (
                        implicit_cf_stats['per_role_target_effect_abs_mean'][role_idx]
                    )
            if cpcr_loss is not None:
                cpcr_loss_weight = float(self.cpcr_config.get('prd_loss_weight', 1.0))
                prd_loss = prd_loss + cpcr_loss_weight * cpcr_loss
                metrics['reward/prd_online/cpcr_supervision_loss'] = float(cpcr_loss.item())
                metrics['reward/prd_online/cpcr_supervision_weight'] = cpcr_loss_weight
                for role, stats in cpcr_role_stats.items():
                    metrics[f'reward/prd_online/roles/{role}/cpcr_loss'] = stats['loss']
                    metrics[f'reward/prd_online/roles/{role}/cpcr_target_count'] = stats['target_count']
                    metrics[f'reward/prd_online/roles/{role}/cpcr_group_count'] = stats['group_count']
                    metrics[f'reward/prd_online/roles/{role}/cpcr_no_contrast_group_count'] = (
                        stats['no_contrast_group_count']
                    )
            self.online_prd_optimizer.zero_grad(set_to_none=True)
            prd_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.online_prd_train.parameters(),
                float(self.online_prd_config.get('grad_clip', 1.0)),
            )
            self.online_prd_optimizer.step()
            metrics['reward/prd_online/loss'] = float(prd_loss.item())
        else:
            metrics['reward/prd_online/loss'] = 0.0
        train_role_scores = train_output['role_scores'].detach()
        train_rollout_scores = train_role_scores.sum(dim=1)
        train_routing = train_output['routing'].detach()
        metrics['reward/prd_online/mixed_group_count'] = mixed_group_count
        metrics['reward/prd_online/skipped_group_count'] = skipped_group_count
        metrics['reward/prd_online/group_count'] = len(uid_to_indices)
        metrics['reward/prd_online/model_type_id'] = 1.0 if model_type == 'role' else 0.0
        metrics['reward/prd_online/graph_prior_mode_id'] = {
            'none': 0.0,
            'soft': 1.0,
            'hard': 2.0,
        }.get(str(self.online_prd_config.get('graph_prior_mode', 'none')), -1.0)
        if routing_bias is not None:
            metrics['reward/prd_online/graph_prior_bias_mean'] = float(routing_bias.float().mean().item())
            metrics['reward/prd_online/graph_prior_bias_std'] = self._safe_tensor_std(routing_bias.float())
        if routing_mask is not None:
            metrics['reward/prd_online/graph_prior_mask_mean'] = float(routing_mask.float().mean().item())
        metrics['reward/prd_online/train_routing_mean'] = float(train_routing.mean().item())
        metrics['reward/prd_online/train_routing_std'] = self._safe_tensor_std(train_routing)
        metrics['reward/prd_online/train_rollout_score_mean'] = float(train_rollout_scores.mean().item())
        metrics['reward/prd_online/train_rollout_score_std'] = self._safe_tensor_std(train_rollout_scores)
        metrics['reward/prd_online/train_role_score_mean'] = float(train_role_scores.mean().item())
        metrics['reward/prd_online/train_role_score_std'] = self._safe_tensor_std(train_role_scores)
        separation = self._safe_binary_separation(train_rollout_scores, raw_scores)
        if separation is not None:
            metrics['reward/prd_online/train_pos_neg_score_gap'] = separation
        top1_acc = self._safe_group_top1_accuracy(uid_list, train_rollout_scores, raw_scores)
        if top1_acc is not None:
            metrics['reward/prd_online/train_group_top1_acc'] = top1_acc
        for role_idx, role in enumerate(agent_roles):
            metrics[f'reward/prd_online/roles/{role}/train_score_mean'] = float(
                train_role_scores[:, role_idx].mean().item()
            )
            metrics[f'reward/prd_online/roles/{role}/train_score_std'] = self._safe_tensor_std(
                train_role_scores[:, role_idx]
            )
            if model_type == 'role':
                role_activity = prd_batch['role_activity']
                metrics[f'reward/prd_online/roles/{role}/activity_mean'] = float(
                    role_activity[:, role_idx].mean().item()
                )
                metrics[f'reward/prd_online/roles/{role}/activity_std'] = self._safe_tensor_std(
                    role_activity[:, role_idx]
                )

        self._sync_online_prd_active()

        alpha = self._get_online_prd_alpha()
        metrics['reward/prd_online/blend_alpha'] = alpha
        if alpha <= 0.0:
            return reward_tensor_map

        self.online_prd_active.eval()
        with torch.no_grad():
            if model_type == 'role':
                active_output = self.online_prd_active(
                    role_features,
                    attention_bias=routing_bias,
                    attention_mask=routing_mask,
                )
            else:
                active_output = self.online_prd_active(
                    source_values,
                    role_features,
                    routing_bias=routing_bias,
                    routing_mask=routing_mask,
                )
        prd_role_scores = active_output['role_scores']
        active_routing = active_output['routing']
        active_rollout_scores = prd_role_scores.sum(dim=1)
        metrics['reward/prd_online/active_routing_mean'] = float(active_routing.mean().item())
        metrics['reward/prd_online/active_routing_std'] = self._safe_tensor_std(active_routing)
        metrics['reward/prd_online/active_role_score_mean'] = float(prd_role_scores.mean().item())
        metrics['reward/prd_online/active_role_score_std'] = self._safe_tensor_std(prd_role_scores)
        metrics['reward/prd_online/active_rollout_score_mean'] = float(active_rollout_scores.mean().item())
        metrics['reward/prd_online/active_rollout_score_std'] = self._safe_tensor_std(active_rollout_scores)
        separation = self._safe_binary_separation(active_rollout_scores, raw_scores)
        if separation is not None:
            metrics['reward/prd_online/active_pos_neg_score_gap'] = separation
        top1_acc = self._safe_group_top1_accuracy(uid_list, active_rollout_scores, raw_scores)
        if top1_acc is not None:
            metrics['reward/prd_online/active_group_top1_acc'] = top1_acc
        for role_idx, role in enumerate(agent_roles):
            metrics[f'reward/prd_online/roles/{role}/active_score_mean'] = float(
                prd_role_scores[:, role_idx].mean().item()
            )
            metrics[f'reward/prd_online/roles/{role}/active_score_std'] = self._safe_tensor_std(
                prd_role_scores[:, role_idx]
            )

        num_turns = data_batch.non_tensor_batch['num_turns']
        last_turn_indices = torch.tensor(
            [max(int(num_turn) - 1, 0) for num_turn in num_turns],
            dtype=torch.long,
        )
        for role_idx, role in enumerate(agent_roles):
            key = f'{role}_turn_level_reward'
            if key not in reward_tensor_map:
                continue
            reward_tensor = reward_tensor_map[key].clone()
            batch_indices = torch.arange(reward_tensor.shape[0], dtype=torch.long)
            last_turn_indices_on_device = last_turn_indices.to(reward_tensor.device)
            batch_indices_on_device = batch_indices.to(reward_tensor.device)

            manual_sequence_scores = reward_tensor.sum(dim=1).detach()
            manual_scores = reward_tensor[
                batch_indices_on_device,
                last_turn_indices_on_device,
            ].detach()
            prd_scores = prd_role_scores[:, role_idx].to(
                device=reward_tensor.device,
                dtype=reward_tensor.dtype,
            )
            blended_scores = (
                (1.0 - alpha) * manual_scores
                + alpha * prd_scores
            )
            reward_tensor[
                batch_indices_on_device,
                last_turn_indices_on_device,
            ] = blended_scores

            blended_sequence_scores = reward_tensor.sum(dim=1).detach()
            actual_delta = (blended_scores - manual_scores).detach()
            expected_delta = (alpha * (prd_scores - manual_scores)).detach()
            reward_tensor_map[key] = reward_tensor
            metrics[f'reward/prd_online/roles/{role}/manual_score_mean'] = float(manual_scores.mean().item())
            metrics[f'reward/prd_online/roles/{role}/blended_score_mean'] = float(blended_scores.mean().item())
            metrics[f'reward/prd_online/roles/{role}/blend_delta_mean'] = float(
                actual_delta.mean().item()
            )
            metrics[f'reward/prd_online/roles/{role}/expected_blend_delta_mean'] = float(
                expected_delta.mean().item()
            )
            metrics[f'reward/prd_online/roles/{role}/blend_delta_abs_max'] = float(
                actual_delta.abs().max().item()
            )
            metrics[f'reward/prd_online/roles/{role}/blend_changed_count'] = float(
                (actual_delta.abs() > 1e-8).sum().item()
            )
            metrics[f'reward/prd_online/roles/{role}/manual_sequence_score_mean'] = float(
                manual_sequence_scores.mean().item()
            )
            metrics[f'reward/prd_online/roles/{role}/blended_sequence_score_mean'] = float(
                blended_sequence_scores.mean().item()
            )
            metrics[f'reward/prd_online/roles/{role}/sequence_blend_delta_mean'] = float(
                (blended_sequence_scores - manual_sequence_scores).mean().item()
            )
        return reward_tensor_map

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
                tss_config = hierarchy_config.get('scoped_c3_grpo', {})
                if bool(tss_config.get('enable', False)) and not (
                    bool(tss_config.get('verifier_role_contrast', True))
                    or bool(tss_config.get('verifier_dependency_contrast', True))
                ):
                    raise ValueError(
                        "derive_verify TSS requires verifier_role_contrast or "
                        "verifier_dependency_contrast to be enabled"
                    )
            elif routing_mode == 'sequential_plan':
                stage_roles = list(hierarchy_config.get('stage_roles', []))
                worker_stage_count = len(stage_roles) - 1
                if worker_stage_count < 1:
                    raise ValueError(
                        "hierarchy.routing_mode=sequential_plan requires at "
                        "least one non-final worker and one final stage"
                    )
                max_planned_subtasks = int(
                    hierarchy_config.get('max_planned_subtasks', 0)
                )
                if max_planned_subtasks != worker_stage_count:
                    raise ValueError(
                        "hierarchy.max_planned_subtasks must equal the available "
                        "non-final stage capacity in sequential_plan mode"
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
                tss_config = hierarchy_config.get('scoped_c3_grpo', {})
                worker_scope_contrast = bool(tss_config.get(
                    'sequential_full_task_contrast', True
                )) or (
                    bool(tss_config.get(
                        'sequential_identity_contrast', True
                    ))
                    and bool(tss_config.get('contrast_plan_subtasks', True))
                )
                if (
                    bool(tss_config.get('enable', False))
                    and not worker_scope_contrast
                ):
                    raise ValueError(
                        "sequential_plan TSS requires a full-task contrast or "
                        "an enabled planned-subtask identity contrast so W1 "
                        "has at least one counterfactual"
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
                if decomposer_role not in train_roles:
                    raise ValueError(
                        "agent12_curriculum requires the decomposer in "
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
                    'worker_question_final_probability',
                    'worker_question_eval_probability',
                ):
                    probability = float(curriculum.get(key, 1.0))
                    if not 0.0 <= probability <= 1.0:
                        raise ValueError(
                            f"agent12_curriculum.{key} must be in [0, 1]"
                        )
                if not str(
                    curriculum.get('teacher_solution_key', '')
                ).strip():
                    raise ValueError(
                        "agent12_curriculum.teacher_solution_key cannot be empty"
                    )
                if int(curriculum.get('teacher_solution_max_chars', 0)) <= 0:
                    raise ValueError(
                        "agent12_curriculum.teacher_solution_max_chars must be "
                        "positive"
                    )

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
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
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
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

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        max_num_turns = self.config.actor_rollout_ref.rollout.max_num_turns
        rollout_meta_info = self._build_rollout_meta_info(max_num_turns)
        score_role = self._get_score_role()

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
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
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
            score_reward_key = f'{score_role}_turn_level_reward'
            reward_tensor_lst.append(reward_tensor[score_reward_key])
            acc_tensor_lst.append(reward_tensor['acc'])

            histories = test_output_gen_batch.non_tensor_batch['history'].tolist()
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
            scores = reward_tensor[score_reward_key].sum(-1).cpu().tolist()
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
            data_source_lst.append(test_batch.non_tensor_batch.get('subset', ['unknown'] * reward_tensor[score_reward_key].shape[0]))
            
            history_lst.append(histories)

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
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
        for data_source, accs in data_source_acc.items():
            metric_dict[f'val/acc/{data_source}'] = np.mean(accs)

        # ``round_N_acc`` is the correctness of the answer state after N
        # rounds. Samples that stopped earlier retain their latest candidate.
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
                    round_state_scores[source_mask, turn_idx].float().mean().item()
                )

        metric_dict.update(
            compute_round_transition_metrics(round_state_scores, round_executed)
        )
        if accepted_lst:
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
            model_paths = switch_config.get('model_paths', [default_model_path, default_model_path])
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Agent0_ActorRollout)
            agent0_config = copy.deepcopy(self.config.actor_rollout_ref)
            agent0_config.model.path = model_paths[0]
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Agent0_ActorRollout],
                                                     config=agent0_config,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['agent0_actor_rollout'] = actor_rollout_cls

            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Agent1_ActorRollout)
            agent1_config = copy.deepcopy(self.config.actor_rollout_ref)
            agent1_config.model.path = model_paths[1] if len(model_paths) > 1 else model_paths[0]
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

        self.multi_agent_rollout = MultiAgentRollout(
            self.config.actor_rollout_ref.rollout,
            {role: self.tokenizer for role in self.actor_rollout_wg.keys()},
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

        if self.online_prd_enabled and self.online_prd_train is not None:
            prd_local_path = os.path.join(local_global_step_folder, 'prd_composer.pt')
            torch.save(
                {
                    'train': self.online_prd_train.checkpoint_payload(),
                    'active': self.online_prd_active.checkpoint_payload(),
                    'optimizer': self.online_prd_optimizer.state_dict(),
                    'cpcr_train_buffer': list(self.online_prd_cpcr_train_buffer),
                    'cpcr_validation_buffer': list(
                        self.online_prd_cpcr_validation_buffer
                    ),
                    'cpcr_role_probe_counts': dict(
                        self._cpcr_role_probe_counts
                    ),
                    'validation_correlation_ema': (
                        self.online_prd_validation_correlation_ema
                    ),
                    'validation_sign_accuracy_ema': (
                        self.online_prd_validation_sign_accuracy_ema
                    ),
                    'global_steps': self.global_steps,
                },
                prd_local_path,
            )

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

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        prd_local_path = os.path.join(global_step_folder, 'prd_composer.pt')
        if self.online_prd_enabled and os.path.exists(prd_local_path):
            prd_state = torch.load(prd_local_path, map_location='cpu')
            self.online_prd_train.load_state_dict(prd_state['train']['state_dict'])
            self.online_prd_active.load_state_dict(prd_state['active']['state_dict'])
            self.online_prd_optimizer.load_state_dict(prd_state['optimizer'])
            self.online_prd_cpcr_train_buffer.extend(
                prd_state.get('cpcr_train_buffer', [])
            )
            self.online_prd_cpcr_validation_buffer.extend(
                prd_state.get('cpcr_validation_buffer', [])
            )
            restored_probe_counts = prd_state.get(
                'cpcr_role_probe_counts',
                {},
            )
            if restored_probe_counts:
                self._cpcr_role_probe_counts.update({
                    str(role): int(count)
                    for role, count in restored_probe_counts.items()
                })
            else:
                for record in (
                    list(self.online_prd_cpcr_train_buffer)
                    + list(self.online_prd_cpcr_validation_buffer)
                ):
                    role = str(record.get('role', ''))
                    probe_ordinal = record.get('probe_ordinal')
                    if role and probe_ordinal is not None:
                        self._cpcr_role_probe_counts[role] = max(
                            self._cpcr_role_probe_counts[role],
                            int(probe_ordinal) + 1,
                        )
            self.online_prd_validation_correlation_ema = prd_state.get(
                'validation_correlation_ema'
            )
            self.online_prd_validation_sign_accuracy_ema = prd_state.get(
                'validation_sign_accuracy_ema'
            )
        elif self.online_prd_enabled:
            print(f"Warning: No online PRD composer state found at {prd_local_path}, using initialized PRD state")

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
        if self.scoped_c3_grpo_enabled or self.direct_scoped_grpo_enabled:
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

            if curriculum_state.phase == 'worker_bootstrap':
                role_index = curriculum_state.phase_step // switch_freq
                new_agent = worker_roles[role_index % len(worker_roles)]
            elif curriculum_state.phase == 'decomposer_transfer':
                new_agent = decomposer_role
            else:
                # Alternate model pools 1:1. Worker stage prompts still rotate
                # within the shared Agent 2 model on its update blocks.
                block_index = curriculum_state.phase_step // switch_freq
                if block_index % 2 == 0:
                    new_agent = decomposer_role
                else:
                    worker_index = (block_index // 2) % len(worker_roles)
                    new_agent = worker_roles[worker_index]

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

        if self.config.trainer.get('fork_wandb_id', None) is not None:
            fork_wandb_id = self.config.trainer.fork_wandb_id
            # wandb_kwargs = {'resume': 'must', 'id': fork_wandb_id}
            print(f'**[WANDB]: will fork run from wandb id: `{fork_wandb_id}` at step {self.global_steps} **')
            
            # e.g. fork_from="6yaq69uw?_step=200"
            wandb_kwargs = {'fork_from': f"{fork_wandb_id}?_step={self.global_steps}"}
        else:
            wandb_kwargs = {}
        
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
                new_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
                                                             dtype=object)
                new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                num_gen_batches += 1

                curriculum_state = self._get_agent12_curriculum_state()
                teacher_solution_key = str(
                    agent12_curriculum.get(
                        'teacher_solution_key', 'teacher_solution'
                    )
                )
                if (
                    agent12_curriculum_enabled
                    and curriculum_state.teacher_solution_probability > 0.0
                    and teacher_solution_key not in new_batch.non_tensor_batch
                ):
                    raise ValueError(
                        f"Agent 1/2 curriculum phase={curriculum_state.phase!r} "
                        f"requires dataset column {teacher_solution_key!r}"
                    )

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
                    if teacher_solution_key in new_batch.non_tensor_batch:
                        generation_non_tensor_keys.append(teacher_solution_key)
                    for teacher_metadata_key in (
                        'teacher_score',
                        'teacher_solution_correct',
                    ):
                        if teacher_metadata_key in new_batch.non_tensor_batch:
                            generation_non_tensor_keys.append(
                                teacher_metadata_key
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
                    gen_batch.meta_info['teacher_solution_key'] = teacher_solution_key
                    gen_batch.meta_info['teacher_solution_probability'] = (
                        curriculum_state.teacher_solution_probability
                    )
                    gen_batch.meta_info['worker_question_probability'] = (
                        curriculum_state.worker_question_probability
                    )

                is_last_step = self.global_steps >= self.total_training_steps

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
                            'teacher_solution_visible',
                            np.zeros(len(new_batch), dtype=bool),
                        )
                        worker_question_visible = new_batch.non_tensor_batch.get(
                            'worker_question_visible',
                            np.zeros(len(new_batch), dtype=bool),
                        )
                        teacher_correct = new_batch.non_tensor_batch.get(
                            'teacher_solution_correct',
                            np.ones(len(new_batch), dtype=bool),
                        )
                        teacher_visible_array = np.asarray(
                            teacher_visible,
                            dtype=bool,
                        )
                        teacher_correct_array = np.asarray(
                            teacher_correct,
                            dtype=bool,
                        )
                        visible_teacher_count = int(
                            teacher_visible_array.sum()
                        )
                        metrics.update({
                            'curriculum/phase_id': float(
                                curriculum_state.phase_id
                            ),
                            'curriculum/phase_step': float(
                                curriculum_state.phase_step
                            ),
                            'curriculum/teacher_solution_probability': float(
                                curriculum_state.teacher_solution_probability
                            ),
                            'curriculum/teacher_solution_visible_rate': float(
                                teacher_visible_array.mean()
                            ),
                            'curriculum/teacher_solution_correct_rate': float(
                                teacher_correct_array.mean()
                            ),
                            'curriculum/visible_teacher_correct_rate': float(
                                (
                                    teacher_correct_array
                                    & teacher_visible_array
                                ).sum()
                                / max(visible_teacher_count, 1)
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
                        self._attach_scoped_c3_grpo_signals(
                            new_batch,
                            reward_tensor_map,
                            metrics,
                        )
                        cpcr_targets = self._compute_cpcr_targets(
                            new_batch,
                            reward_tensor_map,
                            metrics,
                        )
                        reliable_cpcr_target_count = sum(
                            int(target_data['valid_mask'].bool().sum().item())
                            for target_data in cpcr_targets.values()
                        )
                        if reliable_cpcr_target_count > 0:
                            metrics['reward/cpcr/scored_this_step'] = 1.0
                        metrics['reward/cpcr/reliable_target_count_this_step'] = max(
                            float(reliable_cpcr_target_count),
                            metrics.get(
                                'reward/cpcr/reliable_target_count_this_step',
                                0.0,
                            ),
                        )
                        self._attach_direct_scoped_grpo_signals(
                            new_batch,
                            cpcr_targets,
                            metrics,
                        )
                        reward_tensor_map = self._apply_online_prd_rewards(
                            new_batch,
                            reward_tensor_map,
                            metrics,
                            cpcr_targets=cpcr_targets,
                        )
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
                        # batch.batch['token_level_scores'] = reward_tensor
                        new_batch.batch['acc'] = reward_tensor_map.pop('acc')
                        for key_reward, reward_tensor in reward_tensor_map.items():
                            new_batch.batch[key_reward] = reward_tensor
                            if not key_reward.endswith('_turn_level_reward'):
                                continue
                            # get_turn_mask, shape (bsz, max_num_turns), 1 for valid turn, 0 for invalid turn
                            turn_mask = verl_F.get_turn_mask(reward_tensor, new_batch.non_tensor_batch['num_turns'])
                            key_return = key_reward.replace('reward', 'return')
                            # compute turn_level return with turn_level_gamma
                            new_batch.batch[key_return] = core_algos.compute_turn_level_return(
                                reward_tensor, turn_mask, self.config.algorithm.gamma_turn_level)
                    
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
                            target_valid = new_batch.batch[
                                'scoped_c3_target_valid'
                            ].bool().cpu()
                            scope_gate = new_batch.batch[
                                'scoped_c3_scope_gate'
                            ].bool().cpu()
                            scoped_kept_prompt_uids = []
                            for key_uid, sample_indices in id2indices.items():
                                causal_indices = [
                                    sample_idx
                                    for sample_idx in sample_indices
                                    if bool(causal_valid[sample_idx].item())
                                ]
                                if (
                                    len(causal_indices) < 2
                                    or not any(
                                        bool(target_valid[sample_idx].item())
                                        for sample_idx in sample_indices
                                    )
                                ):
                                    continue
                                causal_scores = (
                                    acc_tensor[causal_indices]
                                    .detach()
                                    .float()
                                    .cpu()
                                )
                                if (
                                    bool(torch.isfinite(causal_scores).all().item())
                                    and float(causal_scores.max().item())
                                    > float(causal_scores.min().item())
                                ):
                                    has_train_signal = False
                                    for sample_idx in sample_indices:
                                        if not bool(target_valid[sample_idx].item()):
                                            continue
                                        peer_indices = [
                                            peer_idx
                                            for peer_idx in causal_indices
                                            if peer_idx != sample_idx
                                        ]
                                        if not peer_indices:
                                            continue
                                        causal_delta = float(
                                            acc_tensor[sample_idx].item()
                                        ) - float(
                                            acc_tensor[peer_indices]
                                            .float()
                                            .mean()
                                            .item()
                                        )
                                        if (
                                            causal_delta < 0.0
                                            or (
                                                causal_delta > 0.0
                                                and bool(
                                                    scope_gate[sample_idx].item()
                                                )
                                            )
                                        ):
                                            has_train_signal = True
                                            break
                                    if has_train_signal:
                                        scoped_kept_prompt_uids.append(key_uid)
                            kept_prompt_uids = scoped_kept_prompt_uids
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
                            'rollout/c3_trainable_prompt_cnt': (
                                c3_trainable_prompt_cnt
                            ),
                            'rollout/c3_ineligible_prompt_cnt': (
                                c3_ineligible_prompt_cnt
                            ),
                            'rollout/c3_trainable_prompt_rate': (
                                c3_trainable_prompt_cnt / total_prompt_cnt
                                if total_prompt_cnt > 0 else 0.0
                            ),
                        })
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
                        if (
                            self.scoped_c3_grpo_enabled
                            or self.direct_scoped_grpo_enabled
                        ):
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
                        elif self.direct_scoped_grpo_enabled:
                            batch = self._compute_direct_scoped_grpo_advantage(
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
                    elif collective_safe_update and (
                        self.scoped_c3_grpo_enabled
                        or self.direct_scoped_grpo_enabled
                    ):
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
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
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

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
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
                results_dict[uid] = {
                    "question": data_item.non_tensor_batch['question'],
                    "groundtruth": data_item.non_tensor_batch['reward_model']['ground_truth'],
                    "response": [],
                    "history": [],
                    "score": [],
                    "finish_reason": [],
                }

            padded_history = data_item.non_tensor_batch['history']
            unpad_history = [x for x in padded_history if x['role'] != 'padding']
            results_dict[uid]['history'].append(unpad_history)
            results_dict[uid]['response'].append(data_item.non_tensor_batch['response'])
            score_role = self._get_score_role()
            results_dict[uid]['score'].append(
                data_item.batch[f'{score_role}_turn_level_reward'].sum().item()
            )
            results_dict[uid]['finish_reason'].append(
                data_item.non_tensor_batch['finish_reason']
            )

        results_to_save = []
        for uid, result in results_dict.items():
            result['avg_score'] = sum(result['score']) / len(result['score'])
            results_to_save.append(result)
        with jsonlines.open(output_file, 'w') as writer:
            writer.write_all(results_to_save)

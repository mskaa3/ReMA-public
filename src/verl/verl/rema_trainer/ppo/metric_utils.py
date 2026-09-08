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
Metrics related to the PPO trainer.
"""

import torch
from typing import Any, Dict, List
import numpy as np
from verl import DataProto


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _log_tensor_metric(metrics: Dict[str, Any], batch: DataProto, source_key: str, target_key: str) -> None:
    if source_key not in batch.batch:
        return
    tensor = batch.batch[source_key]
    if not torch.is_tensor(tensor):
        return
    metrics[target_key] = tensor.float().mean().detach().item()


def _log_tensor_summary(metrics: Dict[str, Any], tensor: torch.Tensor, prefix: str) -> None:
    tensor = tensor.float()
    metrics[f'{prefix}/mean'] = tensor.mean().detach().item()
    metrics[f'{prefix}/max'] = tensor.max().detach().item()
    metrics[f'{prefix}/min'] = tensor.min().detach().item()


def compute_reward_diagnostic_metrics(batch: DataProto) -> Dict[str, Any]:
    """Export reward diagnostics under stable W&B-friendly namespaces."""
    metrics: Dict[str, Any] = {}

    scalar_aliases = {
        'acc': 'reward/global/acc',
        'positive_role_bonus_gate': 'reward/global/positive_bonus_gate',
        'upstream_global_correctness_bonus': 'reward/global/upstream_global_correctness_bonus',
        'upstream_hierarchical_correctness_bonus': 'reward/global/upstream_hierarchical_correctness_bonus',
        'hierarchy_utilization_gate': 'reward/hierarchy/utilization_gate',
        'distinct_worker_result_gate': 'reward/hierarchy/distinct_worker_result_gate',
        'valid_nonfinal_worker_count': 'reward/hierarchy/valid_nonfinal_worker_count',
        'planned_subtask_count': 'reward/hierarchy/planned_subtask_count',
        'executed_subtask_count': 'reward/hierarchy/executed_subtask_count',
        'decomposer_unique_local_result_rate': 'reward/decomposer/unique_local_result_rate',
        'decomposer_dependency_usage_rate': 'reward/decomposer/dependency_usage_rate',
        'decomposer_repair_success': 'reward/decomposer/repair_success',
        'decomposer_plan_parseable_gate': 'reward/decomposer/plan_parseable_gate',
        'decomposer_global_correctness_bonus': 'reward/decomposer/global_correctness_bonus',
        'decomposer_local_bonus_raw': 'reward/decomposer/local_bonus_raw',
        'decomposer_local_bonus': 'reward/decomposer/local_bonus',
        'selector_assignment_completeness': 'reward/selector/assignment/completeness',
        'selector_assignment_precision': 'reward/selector/assignment/precision',
        'selector_assignment_recall': 'reward/selector/assignment/recall',
        'selector_assignment_final_present': 'reward/selector/assignment/final_present',
        'selector_assignment_extra_count': 'reward/selector/assignment/extra_count',
        'selector_assignment_missing_count': 'reward/selector/assignment/missing_count',
        'selector_assignment_duplicate_count': 'reward/selector/assignment/duplicate_count',
        'selector_empty_output': 'reward/selector/empty_output_rate',
        'selector_worker_valid_local_result_rate': 'reward/selector/worker_valid_local_result_rate',
        'selector_global_correctness_bonus': 'reward/selector/global_correctness_bonus',
        'selector_local_bonus_raw': 'reward/selector/local_bonus_raw',
        'selector_local_bonus': 'reward/selector/local_bonus',
        'worker_unique_local_result_rate': 'reward/workers/unique_local_result_rate',
        'worker_downstream_used_rate': 'reward/workers/downstream_used_by_final_rate',
        'worker_later_worker_used_rate': 'reward/workers/used_by_later_worker_rate',
        'worker_global_correctness_bonus_mean': 'reward/workers/global_correctness_bonus_mean',
        'worker_local_bonus_mean': 'reward/workers/local_bonus_mean',
        'final_worker_result_usage_rate': 'reward/final/worker_result_usage_rate',
        'final_consistency_with_worker_results': 'reward/final/consistency_with_worker_results',
        'final_worker_usage_gate': 'reward/final/worker_usage_gate',
        'final_local_bonus_raw': 'reward/final/local_bonus_raw',
        'final_local_bonus': 'reward/final/local_bonus',
        'accept_revise_enabled': 'rollout/accept_revise/enabled',
        'accept_revise_accepted': 'rollout/accept_revise/accept_rate',
        'accept_revise_decision_valid': 'rollout/accept_revise/decision_valid_rate',
        'accept_revise_attempted_round_count': 'rollout/accept_revise/attempted_round_count',
        'accept_revise_candidate_source_round': 'rollout/accept_revise/candidate_source_round',
        'accept_revise_accepted_correct': 'rollout/accept_revise/accepted_correct_rate',
    }
    for source_key, target_key in scalar_aliases.items():
        _log_tensor_metric(metrics, batch, source_key, target_key)

    agent_roles = batch.meta_info.get('agent_roles', [])
    for role in agent_roles:
        _log_tensor_metric(
            metrics,
            batch,
            f'{role}_prompt_truncated',
            f'rollout/context/{role}/prompt_truncated_rate',
        )
        _log_tensor_metric(
            metrics,
            batch,
            f'{role}_response_retokenized_truncated',
            f'rollout/context/{role}/response_retokenized_truncated_rate',
        )
        reward_key = f'{role}_turn_level_reward'
        if reward_key not in batch.batch:
            continue
        sequence_reward = batch.batch[reward_key].sum(-1)
        _log_tensor_summary(metrics, sequence_reward, f'reward/roles/{role}/sequence_reward')

    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def _compute_sequence_score_and_reward(batch: DataProto):
    """Read legacy turn rewards or scoped token rewards for logging."""

    if 'turn_level_reward' in batch.batch:
        sequence_reward = batch.batch['turn_level_reward'].sum(-1)
        return sequence_reward, sequence_reward
    return (
        batch.batch['token_level_scores'].sum(-1),
        batch.batch['token_level_rewards'].sum(-1),
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    # TODO: add response length
    sequence_score, sequence_reward = _compute_sequence_score_and_reward(batch)
    num_turns = batch.batch['num_turns'].to(torch.float32)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']
    turn_finished = batch.batch['turn_finished']

    # max_response_length = batch.batch['responses'].shape[-1]
    label_mask = batch.batch['labels'] != -100

    # prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    # response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    # max_prompt_length = prompt_mask.size(-1)

    # response_info = _compute_response_info(batch)
    # prompt_length = response_info['prompt_length']
    # response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, label_mask)
    valid_returns = torch.masked_select(returns, label_mask)
    # Direct counterfactual credit can legitimately reject every sample in a
    # step. Report a zero-valued no-op update instead of reducing empty tensors.
    if valid_adv.numel() == 0:
        valid_adv = advantages.new_zeros(1)
        valid_returns = returns.new_zeros(1)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, label_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)
    
    completion_tokens = batch.batch['num_gen_tokens'].sum(-1).float()
    completion_tokens_per_turn = completion_tokens / num_turns

    metrics = {
        # acc:
        'critic/acc': 
            torch.mean(batch.batch['acc']).detach().item(),
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # # response length
        # 'response_length/mean':
        #     torch.mean(response_length).detach().item(),
        # 'response_length/max':
        #     torch.max(response_length).detach().item(),
        # 'response_length/min':
        #     torch.min(response_length).detach().item(),
        # 'response_length/clip_ratio':
        #     torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),

        # # prompt length
        # 'prompt_length/mean':
        #     torch.mean(prompt_length).detach().item(),
        # 'prompt_length/max':
        #     torch.max(prompt_length).detach().item(),
        # 'prompt_length/min':
        #     torch.min(prompt_length).detach().item(),
        # 'prompt_length/clip_ratio':
        #     torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        # num turns
        'num_turns/mean':
            torch.mean(num_turns).detach().item(),
        'num_turns/max':
            torch.max(num_turns).detach().item(),
        'num_turns/min':
            torch.min(num_turns).detach().item(),
        # reach max turns ratio
        'num_turns/reach_max_turn':
            torch.mean((turn_finished == 1).float()).detach().item(),
        # reach max tokens ratio
        'num_turns/reach_max_tokens':
            torch.mean((turn_finished == 2).float()).detach().item(),
        'num_turns/stop_when_truncated':
            torch.mean((turn_finished == 3).float()).detach().item(),
        'num_turns/decomposer_accept':
            torch.mean((turn_finished == 5).float()).detach().item(),
        'completion_tokens/mean':
            torch.mean(completion_tokens).detach().item(),
        'completion_tokens/max':
            torch.max(completion_tokens).detach().item(),
        'completion_tokens/min':
            torch.min(completion_tokens).detach().item(),
        'completion_tokens_per_turn/mean':
            torch.mean(completion_tokens_per_turn).detach().item(),
        'completion_tokens_per_turn/max':
            torch.max(completion_tokens_per_turn).detach().item(),
        'completion_tokens_per_turn/min':
            torch.min(completion_tokens_per_turn).detach().item(),
        'completion_tokens_per_turn/clip_ratio':
            (batch.batch['stop_reasons'].float().sum() / num_turns.sum()).detach().item()
        
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    # response_info = _compute_response_info(batch)
    # num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    # num_response_tokens = torch.sum(response_info['response_length']).item()
    # num_overall_tokens = num_prompt_tokens + num_response_tokens
    attention_mask = batch.batch['attention_mask']
    num_overall_tokens = torch.sum(attention_mask).item()

    num_tokens_of_section = {
        # 'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info['global_token_num'])
    time = timing_raw['step']
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        'perf/total_num_tokens': total_num_tokens,
        'perf/time_per_step': time,
        'perf/throughput': total_num_tokens / (time * n_gpus),
    }

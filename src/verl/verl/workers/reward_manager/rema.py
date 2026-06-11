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

from functools import partial
from typing import Dict
import re

from tqdm import tqdm
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from math_verify.errors import TimeoutException

META_BOXED_PENALTY = 0.25
WORKER_BOXED_PENALTY = 0.05
WORKER_FINISH_PENALTY = 0.05
PLANNER_REPEAT_PENALTY = 0.10
PLANNER_EXCESS_SUBTASK_PENALTY = 0.10
PLANNER_SUBTASK_TARGET_MIN = 3
PLANNER_SUBTASK_TARGET_MAX = 5
PLANNER_SUBTASK_COUNT_PENALTY_PER_TASK = 0.20
PLANNER_SUBTASK_COUNT_MAX_PENALTY = 0.80
WORKER_EMPTY_ASSIGNED_PENALTY = 0.05
WORKER_MISSING_LOCAL_RESULT_PENALTY = 0.10
WORKER_SUBTASK_OVERREACH_PENALTY = 0.10
WORKER_DUPLICATE_RESULT_PENALTY = 0.03
FINAL_IGNORES_WORKER_RESULTS_PENALTY = 0.10
FINAL_IGNORES_MIN_LOCAL_RESULTS = 2
FINAL_IGNORES_MIN_WORDS = 80
DECOMPOSER_UNIQUE_LOCAL_RESULT_BONUS = 0.05
DECOMPOSER_DEPENDENCY_USAGE_BONUS = 0.05
DECOMPOSER_REPAIR_SUCCESS_BONUS = 0.10
SELECTOR_ASSIGNMENT_COMPLETENESS_BONUS = 0.05
SELECTOR_WORKER_VALID_LOCAL_RESULT_BONUS = 0.05
SELECTOR_EXTRA_ASSIGNMENT_PENALTY_PER_TASK = 0.05
SELECTOR_EXTRA_ASSIGNMENT_MAX_PENALTY = 0.50
SELECTOR_MISSING_ASSIGNMENT_PENALTY_PER_TASK = 0.05
SELECTOR_MISSING_ASSIGNMENT_MAX_PENALTY = 0.30
SELECTOR_DUPLICATE_ASSIGNMENT_PENALTY_PER_TASK = 0.05
SELECTOR_DUPLICATE_ASSIGNMENT_MAX_PENALTY = 0.30
SELECTOR_MISSING_FINAL_PENALTY = 0.05
SELECTOR_EMPTY_OUTPUT_PENALTY = 0.10
WORKER_UNIQUE_LOCAL_RESULT_BONUS = 0.03
WORKER_DOWNSTREAM_USED_BONUS = 0.03
FINAL_WORKER_RESULT_USAGE_BONUS = 0.05
FINAL_CONSISTENCY_WITH_WORKER_RESULTS_BONUS = 0.05
MIN_NEGATIVE_SHAPED_REWARD = 1e-6


def compute_score_fn(compute_score, params):
    data_source, response, ground_truth, extra_info = params
    return compute_score(data_source, response, ground_truth, extra_info)


def _rema_math_format_reward_fn(role, response_str):
    if 'boxed' in response_str:
        if role == 'meta_thinking':
            return -0.25
        elif role == 'reasoning':
            return 0.25
        else:
            raise ValueError(f"Unknown {role=}") 
    else: return 0.0

def _rema_laaj_format_reward_fn(role, response_str):
    from verl.utils.reward_score.pairwise_laaj import extract_final_verdict
    ans = extract_final_verdict(response_str)
    if ans is not None:
        if role == 'meta_thinking':
            return -0.25
        elif role == 'reasoning':
            return 0.25
        else:
            raise ValueError(f"Unknown {role=}") 
    else: return 0.0

def compute_format_r(data_source, role, response_str):
    if data_source == "ReMA-math":
        return _rema_math_format_reward_fn(role, response_str)
    elif data_source == 'ReMA-laaj':
        return _rema_laaj_format_reward_fn(role, response_str)
    else:
        raise ValueError(f'Unknown {data_source=} for format reward.')


def _normalize_role_output(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().strip().split())


def _extract_worker_local_results(text):
    """Return explicit worker LOCAL_RESULT values, accepting minor formatting variants."""
    if not isinstance(text, str):
        return []

    local_results = re.findall(r"(?im)^\s*LOCAL[_ ]RESULT\s*:\s*(.+?)\s*$", text)
    return [result.strip() for result in local_results if result.strip()]


def _is_valid_worker_local_result(text):
    normalized = _normalize_role_output(text)
    return normalized not in {"", "none", "n/a", "na", "null", "unknown"}


def _extract_worker_local_result_signature(text):
    """Return a comparable signature only for explicit worker LOCAL_RESULT lines."""
    local_results = [
        result for result in _extract_worker_local_results(text)
        if _is_valid_worker_local_result(result)
    ]
    if local_results:
        return _normalize_role_output(" | ".join(local_results))

    return ""


def _word_count(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b\w+\b", text))


def _compute_turn_worker_metrics(turn_history, worker_roles):
    active_workers = []
    for msg in turn_history:
        if not isinstance(msg, dict) or msg.get('role') not in worker_roles:
            continue
        assigned_subtasks = msg.get('assigned_subtasks') or []
        if not assigned_subtasks:
            continue
        content = msg.get('content') if isinstance(msg.get('content'), str) else ''
        signature = _extract_worker_local_result_signature(content)
        active_workers.append({
            'role': msg.get('role'),
            'content': content,
            'signature': signature,
        })

    valid_signatures = [worker['signature'] for worker in active_workers if worker['signature']]
    unique_signatures = set(valid_signatures)
    unique_local_result_rate = (
        len(unique_signatures) / len(valid_signatures)
        if valid_signatures else 0.0
    )

    dependency_hits = 0
    dependency_checks = 0
    previous_signatures = []
    for worker in active_workers:
        normalized_content = _normalize_role_output(worker['content'])
        if previous_signatures:
            dependency_checks += 1
            if any(signature in normalized_content for signature in previous_signatures):
                dependency_hits += 1
        if worker['signature']:
            previous_signatures.append(worker['signature'])

    dependency_usage_rate = (
        dependency_hits / dependency_checks
        if dependency_checks else 0.0
    )

    return {
        'active_worker_count': len(active_workers),
        'valid_local_result_count': len(valid_signatures),
        'unique_local_result_rate': unique_local_result_rate,
        'dependency_usage_rate': dependency_usage_rate,
        'duplicate_result_count': max(len(valid_signatures) - len(unique_signatures), 0),
        'missing_local_result_count': sum(1 for worker in active_workers if not worker['signature']),
        'empty_assigned_count': sum(1 for worker in active_workers if not worker['content'].strip()),
    }


def _compute_repair_success(previous_turn_metrics, current_turn_metrics):
    if not previous_turn_metrics:
        return 0.0

    previous_issue_count = (
        previous_turn_metrics['duplicate_result_count']
        + previous_turn_metrics['missing_local_result_count']
        + previous_turn_metrics['empty_assigned_count']
    )
    current_issue_count = (
        current_turn_metrics['duplicate_result_count']
        + current_turn_metrics['missing_local_result_count']
        + current_turn_metrics['empty_assigned_count']
    )

    if previous_issue_count <= 0:
        return 0.0

    return max(previous_issue_count - current_issue_count, 0) / previous_issue_count


def _compute_final_stage_bonus_stats(turn_history, worker_roles, score_role):
    previous_signatures = []
    final_output = ""
    for msg in turn_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role')
        content = msg.get('content') if isinstance(msg.get('content'), str) else ''
        if role == score_role:
            final_output = content
            continue
        if role not in worker_roles:
            continue
        assigned_subtasks = msg.get('assigned_subtasks') or []
        if not assigned_subtasks:
            continue
        signature = _extract_worker_local_result_signature(content)
        if signature:
            previous_signatures.append(signature)

    if not previous_signatures:
        return {
            'worker_result_usage_rate': 0.0,
            'consistency_with_worker_results': 0.0,
        }

    unique_signatures = []
    seen = set()
    for signature in previous_signatures:
        if signature not in seen:
            unique_signatures.append(signature)
            seen.add(signature)

    normalized_final_output = _normalize_role_output(final_output)
    used_count = sum(1 for signature in unique_signatures if signature in normalized_final_output)
    usage_rate = used_count / len(unique_signatures)
    if usage_rate == 1.0:
        consistency = 1.0
    elif usage_rate > 0.0:
        consistency = 0.5
    else:
        consistency = 0.0

    return {
        'worker_result_usage_rate': usage_rate,
        'consistency_with_worker_results': consistency,
    }


def _compute_selector_turn_bonus_stats(turn_history, worker_roles):
    decomposer_output = ""
    selector_output = ""
    for msg in turn_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role')
        content = msg.get('content') if isinstance(msg.get('content'), str) else ''
        if role == 'decomposer':
            decomposer_output = content
        elif role == 'selector':
            selector_output = content

    planned_subtasks = {
        subtask.upper()
        for subtask in re.findall(r"\bS\d+\b", decomposer_output, re.IGNORECASE)
    }
    assigned_subtasks = [
        subtask.upper()
        for subtask in re.findall(r"\bS\d+\b", selector_output, re.IGNORECASE)
    ]
    final_present = 1.0 if "final" in selector_output.lower() else 0.0
    if planned_subtasks:
        assigned_counts = {
            subtask: assigned_subtasks.count(subtask)
            for subtask in planned_subtasks
        }
        matched_once = sum(1 for count in assigned_counts.values() if count == 1)
        missing_assigned_count = sum(1 for count in assigned_counts.values() if count == 0)
        duplicate_assigned_count = sum(max(count - 1, 0) for count in assigned_counts.values())
        extra_assigned_count = sum(
            1 for subtask in assigned_subtasks if subtask not in planned_subtasks
        )
        precision = (
            matched_once / len(assigned_subtasks)
            if assigned_subtasks else 0.0
        )
        recall = matched_once / len(planned_subtasks)
        assignment_completeness = precision * recall * final_present
    else:
        precision = 0.0
        recall = 0.0
        extra_assigned_count = len(assigned_subtasks)
        missing_assigned_count = 0
        duplicate_assigned_count = 0
        assignment_completeness = 0.0

    worker_metrics = _compute_turn_worker_metrics(turn_history, worker_roles)
    active_worker_count = worker_metrics['active_worker_count']
    worker_valid_local_result_rate = (
        worker_metrics['valid_local_result_count'] / active_worker_count
        if active_worker_count > 0 else 0.0
    )

    return {
        'assignment_completeness': assignment_completeness,
        'assignment_precision': precision,
        'assignment_recall': recall,
        'assignment_final_present': final_present,
        'assignment_extra_count': float(extra_assigned_count),
        'assignment_missing_count': float(missing_assigned_count),
        'assignment_duplicate_count': float(duplicate_assigned_count),
        'selector_empty_output': 1.0 if not selector_output.strip() else 0.0,
        'worker_valid_local_result_rate': worker_valid_local_result_rate,
    }


def _compute_hierarchy_bonus_gates(turn_history, worker_roles, score_role):
    decomposer_output = ""
    active_nonfinal_workers = []

    for msg in turn_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role')
        content = msg.get('content') if isinstance(msg.get('content'), str) else ''
        if role == 'decomposer':
            decomposer_output = content
            continue
        if role not in worker_roles or role == score_role:
            continue
        assigned_subtasks = msg.get('assigned_subtasks') or []
        if not assigned_subtasks:
            continue
        active_nonfinal_workers.append({
            'content': content,
            'has_valid_local_result': bool(_extract_worker_local_result_signature(content)),
        })

    normalized_decomposer = _normalize_role_output(decomposer_output)
    has_plan_header = 'plan:' in normalized_decomposer
    planned_subtasks = {
        subtask.upper()
        for subtask in re.findall(r"\bS\d+\b", decomposer_output, re.IGNORECASE)
    }
    contains_forbidden_solution_markers = (
        'boxed' in normalized_decomposer
        or '[finish]' in normalized_decomposer
        or 'final answer' in normalized_decomposer
        or 'final result' in normalized_decomposer
    )
    plan_parseable_gate = 1.0 if (
        has_plan_header
        and len(planned_subtasks) > 0
        and not contains_forbidden_solution_markers
    ) else 0.0

    valid_nonfinal_worker_count = sum(
        1 for worker in active_nonfinal_workers if worker['has_valid_local_result']
    )
    executed_subtasks = set()
    for msg in turn_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role')
        if role not in worker_roles or role == score_role:
            continue
        assigned_subtasks = [
            subtask.upper()
            for subtask in (msg.get('assigned_subtasks') or [])
            if isinstance(subtask, str)
        ]
        if not assigned_subtasks:
            continue
        content = msg.get('content') if isinstance(msg.get('content'), str) else ''
        if not _extract_worker_local_result_signature(content):
            continue
        for subtask in assigned_subtasks:
            if subtask in planned_subtasks:
                executed_subtasks.add(subtask)

    planned_subtask_count = len(planned_subtasks)
    executed_subtask_count = len(executed_subtasks)
    hierarchy_utilization_gate = (
        executed_subtask_count / planned_subtask_count
        if planned_subtask_count > 0 else 0.0
    )

    return {
        'decomposer_plan_parseable_gate': plan_parseable_gate,
        'hierarchy_utilization_gate': hierarchy_utilization_gate,
        'valid_nonfinal_worker_count': valid_nonfinal_worker_count,
        'planned_subtask_count': planned_subtask_count,
        'executed_subtask_count': executed_subtask_count,
    }


def _compute_turn_worker_role_bonus_stats(turn_history, worker_roles, score_role):
    active_workers = []
    final_output = ""
    for msg in turn_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role')
        content = msg.get('content') if isinstance(msg.get('content'), str) else ''
        if role == score_role:
            final_output = content
        if role not in worker_roles or role == score_role:
            continue
        assigned_subtasks = msg.get('assigned_subtasks') or []
        if not assigned_subtasks:
            continue
        signature = _extract_worker_local_result_signature(content)
        active_workers.append({
            'role': role,
            'content': content,
            'signature': signature,
        })

    signature_counts = {}
    for worker in active_workers:
        if worker['signature']:
            signature_counts[worker['signature']] = signature_counts.get(worker['signature'], 0) + 1

    per_role = {}
    unique_hits = 0
    downstream_hits = 0
    final_output_norm = _normalize_role_output(final_output)
    valid_worker_count = 0
    for idx, worker in enumerate(active_workers):
        signature = worker['signature']
        unique_local_result = 0.0
        downstream_used = 0.0
        if signature:
            valid_worker_count += 1
            if signature_counts.get(signature, 0) == 1:
                unique_local_result = 1.0
                unique_hits += 1
            if final_output_norm and signature in final_output_norm:
                downstream_used = 1.0
                downstream_hits += 1
        per_role[worker['role']] = {
            'unique_local_result': unique_local_result,
            'downstream_used': downstream_used,
        }

    denom = valid_worker_count if valid_worker_count > 0 else 1
    return {
        'per_role': per_role,
        'unique_local_result_rate': unique_hits / denom if valid_worker_count > 0 else 0.0,
        'downstream_used_rate': downstream_hits / denom if valid_worker_count > 0 else 0.0,
    }

class ReMARewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto)-> Dict[str, torch.Tensor]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
        
        batch_size = len(data)
        max_num_turns = data.meta_info['max_num_turns']

        
        agent_roles = data.meta_info['agent_roles']
        hierarchy_config = data.meta_info.get('hierarchy', {})
        stage_roles = hierarchy_config.get('stage_roles')
        if stage_roles is None and hierarchy_config.get('num_worker_stages'):
            stage_roles = [
                f'worker_stage_{idx}'
                for idx in range(1, int(hierarchy_config.get('num_worker_stages')) + 1)
            ]
        if stage_roles is not None:
            worker_roles = set(stage_roles)
        else:
            worker_roles = set(hierarchy_config.get(
                'worker_roles',
                [role for role in agent_roles if role.endswith('_worker')],
            ))
        worker_type_roles = set(hierarchy_config.get(
            'worker_roles',
            [role for role in agent_roles if role.endswith('_worker')],
        ))
        # Backward compatibility for older hierarchical configs where worker
        # types themselves were the rollout roles.
        worker_roles.update(worker_type_roles.intersection(agent_roles))
        score_role = hierarchy_config.get(
            'score_role',
            agent_roles[-1] if agent_roles else None,
        )
        planner_roles = {'decomposer', 'selector'}
        reward_tensor_map = {
            f'{role}_turn_level_reward': torch.zeros(batch_size, max_num_turns, dtype=torch.float32) for role in agent_roles
        }
        reward_tensor_map['meta_boxed_penalty_applied'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['meta_boxed_penalty_value'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['worker_boxed_penalty_applied'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['worker_boxed_penalty_value'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['worker_finish_penalty_applied'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['worker_finish_penalty_value'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['planner_repeat_penalty_applied'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['planner_repeat_penalty_value'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['planner_excess_subtask_penalty_applied'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['planner_excess_subtask_penalty_value'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['planner_subtask_count_penalty_applied'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['planner_subtask_count_penalty_value'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_extra_assignment_penalty_applied'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_extra_assignment_penalty_value'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_missing_assignment_penalty_applied'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_missing_assignment_penalty_value'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_duplicate_assignment_penalty_applied'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_duplicate_assignment_penalty_value'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_missing_final_penalty_applied'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_missing_final_penalty_value'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_empty_output_penalty_applied'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_empty_output_penalty_value'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['worker_empty_assigned_penalty_applied'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['worker_empty_assigned_penalty_value'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['worker_missing_local_result_penalty_applied'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['worker_missing_local_result_penalty_value'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['worker_subtask_overreach_penalty_applied'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['worker_subtask_overreach_penalty_value'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['worker_duplicate_result_penalty_applied'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['worker_duplicate_result_penalty_value'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['final_ignores_worker_results_penalty_applied'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['final_ignores_worker_results_penalty_value'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['decomposer_unique_local_result_rate'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['decomposer_dependency_usage_rate'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['decomposer_repair_success'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['decomposer_plan_parseable_gate'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['hierarchy_utilization_gate'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['valid_nonfinal_worker_count'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['planned_subtask_count'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['executed_subtask_count'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['positive_role_bonus_gate'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['decomposer_local_bonus_raw'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['decomposer_local_bonus'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_assignment_completeness'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_assignment_precision'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_assignment_recall'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_assignment_final_present'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_assignment_extra_count'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_assignment_missing_count'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_assignment_duplicate_count'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_empty_output'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_worker_valid_local_result_rate'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_local_bonus_raw'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['selector_local_bonus'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['worker_unique_local_result_rate'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['worker_downstream_used_rate'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['worker_local_bonus_mean'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['final_worker_result_usage_rate'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['final_consistency_with_worker_results'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['final_local_bonus_raw'] = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor_map['final_local_bonus'] = torch.zeros(batch_size, dtype=torch.float32)
        
        already_print_data_sources = {}

        params = [
            (data[i].non_tensor_batch['data_source'],
             data[i].non_tensor_batch['response'],
             data[i].non_tensor_batch['reward_model']['ground_truth'],
             data[i].non_tensor_batch.get('extra_info', None),
             )
            for i in range(len(data))
        ]

        scores = []
        with ProcessPool(max_workers=1) as pool:
            future = pool.map(partial(compute_score_fn, self.compute_score), params, timeout=10)
            iterator = future.result()
            with tqdm(total=len(data), desc="Computing scores") as pbar:
                while True:
                    try:
                        result = next(iterator)
                        scores.append(result)
                    except TimeoutError:
                        print('Time Out')
                        scores.append(0.0)
                    except TimeoutException:
                        print('Math verify internal timeout')
                        scores.append(0.0)
                    except StopIteration:
                        break
                    except Exception as e:
                        print(f"Error: {e}")
                        raise e
                    pbar.update(1)
        
        assert len(scores) == len(data)
        accuracy = torch.tensor(scores, dtype=torch.float32) # bsz
        reward_tensor_map['acc'] = accuracy
        for i_bsz in range(len(data)):
            data_item = data[i_bsz]  # DataProtoItem
            response_str = data_item.non_tensor_batch['response']
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            # extra_info = data_item.non_tensor_batch.get('extra_info', None)
            # score = self.compute_score(
            #     data_source=data_source,
            #     solution_str=response_str,
            #     ground_truth=ground_truth,
            #     extra_info=extra_info,
            # )
            raw_score = scores[i_bsz]
            
            num_turns = data_item.non_tensor_batch['num_turns']
            full_history = data_item.non_tensor_batch.get('history', [])
            valid_history = full_history[:num_turns * len(agent_roles)]
            turn_histories = [
                valid_history[i_turn * len(agent_roles):(i_turn + 1) * len(agent_roles)]
                for i_turn in range(num_turns)
            ]
            meta_roles = {'meta_thinking', 'decomposer'}
            role_penalties = {role: 0.0 for role in agent_roles}
            role_bonuses = {role: 0.0 for role in agent_roles}
            active_penalties = []
            positive_role_bonus_gate = 1.0 if float(raw_score) > 0.0 else 0.0
            reward_tensor_map['positive_role_bonus_gate'][i_bsz] = positive_role_bonus_gate

            if 'decomposer' in agent_roles and turn_histories:
                hierarchy_bonus_gates = _compute_hierarchy_bonus_gates(
                    turn_histories[-1], worker_roles, score_role
                )
                current_turn_metrics = _compute_turn_worker_metrics(turn_histories[-1], worker_roles)
                previous_turn_metrics = (
                    _compute_turn_worker_metrics(turn_histories[-2], worker_roles)
                    if len(turn_histories) >= 2 else None
                )
                decomposer_unique_local_result_rate = current_turn_metrics['unique_local_result_rate']
                decomposer_dependency_usage_rate = current_turn_metrics['dependency_usage_rate']
                decomposer_repair_success = _compute_repair_success(
                    previous_turn_metrics,
                    current_turn_metrics,
                )
                decomposer_local_bonus = (
                    DECOMPOSER_UNIQUE_LOCAL_RESULT_BONUS * decomposer_unique_local_result_rate
                    + DECOMPOSER_DEPENDENCY_USAGE_BONUS * decomposer_dependency_usage_rate
                    + DECOMPOSER_REPAIR_SUCCESS_BONUS * decomposer_repair_success
                )
                decomposer_local_bonus *= hierarchy_bonus_gates['decomposer_plan_parseable_gate']
                decomposer_local_bonus *= hierarchy_bonus_gates['hierarchy_utilization_gate']
                decomposer_local_bonus *= positive_role_bonus_gate
                reward_tensor_map['decomposer_unique_local_result_rate'][i_bsz] = decomposer_unique_local_result_rate
                reward_tensor_map['decomposer_dependency_usage_rate'][i_bsz] = decomposer_dependency_usage_rate
                reward_tensor_map['decomposer_repair_success'][i_bsz] = decomposer_repair_success
                reward_tensor_map['decomposer_plan_parseable_gate'][i_bsz] = hierarchy_bonus_gates['decomposer_plan_parseable_gate']
                reward_tensor_map['hierarchy_utilization_gate'][i_bsz] = hierarchy_bonus_gates['hierarchy_utilization_gate']
                reward_tensor_map['valid_nonfinal_worker_count'][i_bsz] = hierarchy_bonus_gates['valid_nonfinal_worker_count']
                reward_tensor_map['planned_subtask_count'][i_bsz] = hierarchy_bonus_gates['planned_subtask_count']
                reward_tensor_map['executed_subtask_count'][i_bsz] = hierarchy_bonus_gates['executed_subtask_count']
                reward_tensor_map['decomposer_local_bonus_raw'][i_bsz] = (
                    DECOMPOSER_UNIQUE_LOCAL_RESULT_BONUS * decomposer_unique_local_result_rate
                    + DECOMPOSER_DEPENDENCY_USAGE_BONUS * decomposer_dependency_usage_rate
                    + DECOMPOSER_REPAIR_SUCCESS_BONUS * decomposer_repair_success
                )
                reward_tensor_map['decomposer_local_bonus'][i_bsz] = decomposer_local_bonus
                role_bonuses['decomposer'] += decomposer_local_bonus

            if 'selector' in agent_roles and turn_histories:
                hierarchy_bonus_gates = _compute_hierarchy_bonus_gates(
                    turn_histories[-1], worker_roles, score_role
                )
                selector_stats = _compute_selector_turn_bonus_stats(
                    turn_histories[-1], worker_roles
                )
                selector_local_bonus_raw = (
                    SELECTOR_ASSIGNMENT_COMPLETENESS_BONUS * selector_stats['assignment_completeness']
                    + SELECTOR_WORKER_VALID_LOCAL_RESULT_BONUS * selector_stats['worker_valid_local_result_rate']
                )
                selector_local_bonus_raw *= hierarchy_bonus_gates['hierarchy_utilization_gate']
                selector_local_bonus = selector_local_bonus_raw * positive_role_bonus_gate
                reward_tensor_map['selector_assignment_completeness'][i_bsz] = selector_stats['assignment_completeness']
                reward_tensor_map['selector_assignment_precision'][i_bsz] = selector_stats['assignment_precision']
                reward_tensor_map['selector_assignment_recall'][i_bsz] = selector_stats['assignment_recall']
                reward_tensor_map['selector_assignment_final_present'][i_bsz] = selector_stats['assignment_final_present']
                reward_tensor_map['selector_assignment_extra_count'][i_bsz] = selector_stats['assignment_extra_count']
                reward_tensor_map['selector_assignment_missing_count'][i_bsz] = selector_stats['assignment_missing_count']
                reward_tensor_map['selector_assignment_duplicate_count'][i_bsz] = selector_stats['assignment_duplicate_count']
                reward_tensor_map['selector_empty_output'][i_bsz] = selector_stats['selector_empty_output']
                reward_tensor_map['selector_worker_valid_local_result_rate'][i_bsz] = selector_stats['worker_valid_local_result_rate']
                reward_tensor_map['selector_local_bonus_raw'][i_bsz] = selector_local_bonus_raw
                reward_tensor_map['selector_local_bonus'][i_bsz] = selector_local_bonus
                role_bonuses['selector'] += selector_local_bonus
                selector_extra_assignment_penalty = min(
                    SELECTOR_EXTRA_ASSIGNMENT_MAX_PENALTY,
                    selector_stats['assignment_extra_count'] * SELECTOR_EXTRA_ASSIGNMENT_PENALTY_PER_TASK,
                )
                selector_missing_assignment_penalty = min(
                    SELECTOR_MISSING_ASSIGNMENT_MAX_PENALTY,
                    selector_stats['assignment_missing_count'] * SELECTOR_MISSING_ASSIGNMENT_PENALTY_PER_TASK,
                )
                selector_duplicate_assignment_penalty = min(
                    SELECTOR_DUPLICATE_ASSIGNMENT_MAX_PENALTY,
                    selector_stats['assignment_duplicate_count'] * SELECTOR_DUPLICATE_ASSIGNMENT_PENALTY_PER_TASK,
                )
                selector_missing_final_penalty = (
                    SELECTOR_MISSING_FINAL_PENALTY
                    if (
                        selector_stats['assignment_final_present'] == 0.0
                        and selector_stats['selector_empty_output'] == 0.0
                    ) else 0.0
                )
                selector_empty_output_penalty = (
                    SELECTOR_EMPTY_OUTPUT_PENALTY
                    if selector_stats['selector_empty_output'] > 0.0 else 0.0
                )
                selector_assignment_penalties = [
                    ('selector_extra_assignment', selector_extra_assignment_penalty),
                    ('selector_missing_assignment', selector_missing_assignment_penalty),
                    ('selector_duplicate_assignment', selector_duplicate_assignment_penalty),
                    ('selector_missing_final', selector_missing_final_penalty),
                    ('selector_empty_output', selector_empty_output_penalty),
                ]
                for penalty_name, penalty_value in selector_assignment_penalties:
                    if penalty_value <= 0.0:
                        continue
                    reward_tensor_map[f'{penalty_name}_penalty_applied'][i_bsz] = 1.0
                    reward_tensor_map[f'{penalty_name}_penalty_value'][i_bsz] = penalty_value
                    active_penalties.append((penalty_name, penalty_value, ['selector']))
                    role_penalties['selector'] += penalty_value

            if turn_histories:
                worker_bonus_stats = _compute_turn_worker_role_bonus_stats(
                    turn_histories[-1], worker_roles, score_role
                )
                reward_tensor_map['worker_unique_local_result_rate'][i_bsz] = worker_bonus_stats['unique_local_result_rate']
                reward_tensor_map['worker_downstream_used_rate'][i_bsz] = worker_bonus_stats['downstream_used_rate']
                worker_bonus_values = []
                for role, stats in worker_bonus_stats['per_role'].items():
                    worker_bonus = stats['downstream_used'] * (
                        WORKER_UNIQUE_LOCAL_RESULT_BONUS * stats['unique_local_result']
                        + WORKER_DOWNSTREAM_USED_BONUS * stats['downstream_used']
                    )
                    worker_bonus *= positive_role_bonus_gate
                    role_bonuses[role] += worker_bonus
                    worker_bonus_values.append(worker_bonus)
                if worker_bonus_values:
                    reward_tensor_map['worker_local_bonus_mean'][i_bsz] = sum(worker_bonus_values) / len(worker_bonus_values)

            if score_role in agent_roles and turn_histories:
                final_bonus_stats = _compute_final_stage_bonus_stats(
                    turn_histories[-1], worker_roles, score_role
                )
                final_local_bonus_raw = (
                    FINAL_WORKER_RESULT_USAGE_BONUS * final_bonus_stats['worker_result_usage_rate']
                    + FINAL_CONSISTENCY_WITH_WORKER_RESULTS_BONUS * final_bonus_stats['consistency_with_worker_results']
                )
                final_local_bonus = final_local_bonus_raw * positive_role_bonus_gate
                reward_tensor_map['final_worker_result_usage_rate'][i_bsz] = final_bonus_stats['worker_result_usage_rate']
                reward_tensor_map['final_consistency_with_worker_results'][i_bsz] = final_bonus_stats['consistency_with_worker_results']
                reward_tensor_map['final_local_bonus_raw'][i_bsz] = final_local_bonus_raw
                reward_tensor_map['final_local_bonus'][i_bsz] = final_local_bonus
                role_bonuses[score_role] += final_local_bonus

            meta_has_boxed = any(
                isinstance(msg, dict)
                and msg.get('role') in meta_roles
                and isinstance(msg.get('content'), str)
                and 'boxed' in msg.get('content').lower()
                for msg in valid_history
            )
            if meta_has_boxed:
                reward_tensor_map['meta_boxed_penalty_applied'][i_bsz] = 1.0
                reward_tensor_map['meta_boxed_penalty_value'][i_bsz] = META_BOXED_PENALTY
                active_penalties.append(('meta_boxed', META_BOXED_PENALTY, sorted(meta_roles.intersection(agent_roles))))

            worker_boxed_roles = set()
            for msg in valid_history:
                content = msg.get('content') if isinstance(msg, dict) else None
                if (
                    isinstance(msg, dict)
                    and msg.get('role') in worker_roles
                    and msg.get('role') != score_role
                    and isinstance(content, str)
                    and 'boxed' in content.lower()
                    and content != response_str
                ):
                    worker_boxed_roles.add(msg.get('role'))
            if worker_boxed_roles:
                reward_tensor_map['worker_boxed_penalty_applied'][i_bsz] = 1.0
                reward_tensor_map['worker_boxed_penalty_value'][i_bsz] = WORKER_BOXED_PENALTY
                active_penalties.append(('worker_boxed', WORKER_BOXED_PENALTY, sorted(worker_boxed_roles)))
                for role in worker_boxed_roles:
                    role_penalties[role] += WORKER_BOXED_PENALTY

            finish_flag = data.meta_info.get('finish_flag')
            worker_finish_roles = set()
            if finish_flag:
                for msg in valid_history:
                    content = msg.get('content') if isinstance(msg, dict) else None
                    if (
                        isinstance(msg, dict)
                        and msg.get('role') in worker_roles
                        and isinstance(content, str)
                        and finish_flag in content
                        and content != response_str
                    ):
                        worker_finish_roles.add(msg.get('role'))
            if worker_finish_roles:
                reward_tensor_map['worker_finish_penalty_applied'][i_bsz] = 1.0
                reward_tensor_map['worker_finish_penalty_value'][i_bsz] = WORKER_FINISH_PENALTY
                active_penalties.append(('worker_finish', WORKER_FINISH_PENALTY, sorted(worker_finish_roles)))
                for role in worker_finish_roles:
                    role_penalties[role] += WORKER_FINISH_PENALTY

            repeated_planner_roles = set()
            for role in planner_roles.intersection(agent_roles):
                role_outputs = [
                    _normalize_role_output(msg.get('content', ''))
                    for msg in valid_history
                    if isinstance(msg, dict) and msg.get('role') == role
                ]
                role_outputs = [output for output in role_outputs if output]
                if len(role_outputs) > len(set(role_outputs)):
                    repeated_planner_roles.add(role)
            if {'decomposer', 'selector'}.issubset(set(agent_roles)):
                for i_turn in range(num_turns):
                    turn_history = valid_history[
                        i_turn * len(agent_roles):(i_turn + 1) * len(agent_roles)
                    ]
                    planner_outputs = {
                        msg.get('role'): _normalize_role_output(msg.get('content', ''))
                        for msg in turn_history
                        if isinstance(msg, dict) and msg.get('role') in {'decomposer', 'selector'}
                    }
                    if (
                        planner_outputs.get('decomposer')
                        and planner_outputs.get('decomposer') == planner_outputs.get('selector')
                    ):
                        repeated_planner_roles.update({'decomposer', 'selector'})
            if repeated_planner_roles:
                reward_tensor_map['planner_repeat_penalty_applied'][i_bsz] = 1.0
                reward_tensor_map['planner_repeat_penalty_value'][i_bsz] = PLANNER_REPEAT_PENALTY
                active_penalties.append(('planner_repeat', PLANNER_REPEAT_PENALTY, sorted(repeated_planner_roles)))

            excess_subtask_roles = set()
            if 'decomposer' in agent_roles:
                max_worker_subtasks = max(
                    len(stage_roles or []) - 1,
                    1,
                )
                for msg in valid_history:
                    if not (
                        isinstance(msg, dict)
                        and msg.get('role') == 'decomposer'
                        and isinstance(msg.get('content'), str)
                    ):
                        continue
                    subtask_ids = set(re.findall(r"\bS\d+\b", msg.get('content'), re.IGNORECASE))
                    if len(subtask_ids) > max_worker_subtasks:
                        excess_subtask_roles.add('decomposer')
                        break
            if excess_subtask_roles:
                reward_tensor_map['planner_excess_subtask_penalty_applied'][i_bsz] = 1.0
                reward_tensor_map['planner_excess_subtask_penalty_value'][i_bsz] = PLANNER_EXCESS_SUBTASK_PENALTY
                active_penalties.append(('planner_excess_subtask', PLANNER_EXCESS_SUBTASK_PENALTY, sorted(excess_subtask_roles)))

            subtask_count_penalty = 0.0
            if 'decomposer' in agent_roles:
                for msg in valid_history:
                    if not (
                        isinstance(msg, dict)
                        and msg.get('role') == 'decomposer'
                        and isinstance(msg.get('content'), str)
                    ):
                        continue
                    subtask_ids = {
                        subtask.upper()
                        for subtask in re.findall(r"\bS\d+\b", msg.get('content'), re.IGNORECASE)
                    }
                    subtask_count = len(subtask_ids)
                    if subtask_count < PLANNER_SUBTASK_TARGET_MIN:
                        distance = PLANNER_SUBTASK_TARGET_MIN - subtask_count
                    elif subtask_count > PLANNER_SUBTASK_TARGET_MAX:
                        distance = subtask_count - PLANNER_SUBTASK_TARGET_MAX
                    else:
                        distance = 0
                    if distance:
                        subtask_count_penalty = max(
                            subtask_count_penalty,
                            min(
                                PLANNER_SUBTASK_COUNT_MAX_PENALTY,
                                distance * PLANNER_SUBTASK_COUNT_PENALTY_PER_TASK,
                            ),
                        )
            if subtask_count_penalty > 0.0:
                reward_tensor_map['planner_subtask_count_penalty_applied'][i_bsz] = 1.0
                reward_tensor_map['planner_subtask_count_penalty_value'][i_bsz] = subtask_count_penalty
                active_penalties.append(('planner_subtask_count', subtask_count_penalty, ['decomposer']))

            empty_assigned_roles = set()
            missing_local_result_roles = set()
            overreach_roles = set()
            for msg in valid_history:
                if not isinstance(msg, dict) or msg.get('role') not in worker_roles:
                    continue
                assigned_subtasks = msg.get('assigned_subtasks') or []
                content = msg.get('content') if isinstance(msg.get('content'), str) else ''
                if assigned_subtasks and not content.strip():
                    empty_assigned_roles.add(msg.get('role'))
                if assigned_subtasks and not _extract_worker_local_result_signature(content):
                    missing_local_result_roles.add(msg.get('role'))
                if assigned_subtasks:
                    mentioned_subtasks = {
                        subtask.upper()
                        for subtask in re.findall(r"\bS\d+\b", content, re.IGNORECASE)
                    }
                    allowed_subtasks = {subtask.upper() for subtask in assigned_subtasks}
                    if mentioned_subtasks - allowed_subtasks:
                        overreach_roles.add(msg.get('role'))
            if empty_assigned_roles:
                reward_tensor_map['worker_empty_assigned_penalty_applied'][i_bsz] = 1.0
                reward_tensor_map['worker_empty_assigned_penalty_value'][i_bsz] = WORKER_EMPTY_ASSIGNED_PENALTY
                active_penalties.append(('worker_empty_assigned', WORKER_EMPTY_ASSIGNED_PENALTY, sorted(empty_assigned_roles)))
                for role in empty_assigned_roles:
                    role_penalties[role] += WORKER_EMPTY_ASSIGNED_PENALTY
            if missing_local_result_roles:
                reward_tensor_map['worker_missing_local_result_penalty_applied'][i_bsz] = 1.0
                reward_tensor_map['worker_missing_local_result_penalty_value'][i_bsz] = WORKER_MISSING_LOCAL_RESULT_PENALTY
                active_penalties.append(('worker_missing_local_result', WORKER_MISSING_LOCAL_RESULT_PENALTY, sorted(missing_local_result_roles)))
                for role in missing_local_result_roles:
                    role_penalties[role] += WORKER_MISSING_LOCAL_RESULT_PENALTY
            if overreach_roles:
                reward_tensor_map['worker_subtask_overreach_penalty_applied'][i_bsz] = 1.0
                reward_tensor_map['worker_subtask_overreach_penalty_value'][i_bsz] = WORKER_SUBTASK_OVERREACH_PENALTY
                active_penalties.append(('worker_subtask_overreach', WORKER_SUBTASK_OVERREACH_PENALTY, sorted(overreach_roles)))

            duplicate_worker_roles = set()
            for i_turn in range(num_turns):
                turn_history = valid_history[
                    i_turn * len(agent_roles):(i_turn + 1) * len(agent_roles)
                ]
                signature_to_roles = {}
                for msg in turn_history:
                    if not isinstance(msg, dict) or msg.get('role') not in worker_roles:
                        continue
                    signature = _extract_worker_local_result_signature(msg.get('content', ''))
                    if not signature:
                        continue
                    signature_to_roles.setdefault(signature, set()).add(msg.get('role'))
                for roles_with_same_signature in signature_to_roles.values():
                    if len(roles_with_same_signature) > 1:
                        duplicate_worker_roles.update(roles_with_same_signature)
            if duplicate_worker_roles:
                reward_tensor_map['worker_duplicate_result_penalty_applied'][i_bsz] = 1.0
                reward_tensor_map['worker_duplicate_result_penalty_value'][i_bsz] = WORKER_DUPLICATE_RESULT_PENALTY
                active_penalties.append(('worker_duplicate_result', WORKER_DUPLICATE_RESULT_PENALTY, sorted(duplicate_worker_roles)))
                for role in duplicate_worker_roles:
                    role_penalties[role] += WORKER_DUPLICATE_RESULT_PENALTY

            final_ignores_worker_results = False
            if score_role in worker_roles:
                previous_local_results = []
                final_output = ""
                for msg in valid_history:
                    if not isinstance(msg, dict) or msg.get('role') not in worker_roles:
                        continue
                    content = msg.get('content') if isinstance(msg.get('content'), str) else ''
                    if msg.get('role') == score_role:
                        final_output = content
                        continue
                    assigned_subtasks = msg.get('assigned_subtasks') or []
                    if not assigned_subtasks:
                        continue
                    previous_local_results.extend([
                        result for result in _extract_worker_local_results(content)
                        if _is_valid_worker_local_result(result)
                    ])

                if (
                    len(previous_local_results) >= FINAL_IGNORES_MIN_LOCAL_RESULTS
                    and _word_count(final_output) >= FINAL_IGNORES_MIN_WORDS
                ):
                    normalized_final_output = _normalize_role_output(final_output)
                    uses_any_local_result = any(
                        _normalize_role_output(local_result) in normalized_final_output
                        for local_result in previous_local_results
                    )
                    final_ignores_worker_results = not uses_any_local_result
            if final_ignores_worker_results:
                reward_tensor_map['final_ignores_worker_results_penalty_applied'][i_bsz] = 1.0
                reward_tensor_map['final_ignores_worker_results_penalty_value'][i_bsz] = FINAL_IGNORES_WORKER_RESULTS_PENALTY
                active_penalties.append(('final_ignores_worker_results', FINAL_IGNORES_WORKER_RESULTS_PENALTY, [score_role]))
                if score_role in role_penalties:
                    role_penalties[score_role] += FINAL_IGNORES_WORKER_RESULTS_PENALTY

            role_shaped_scores = {}
            
            for i_role, role in enumerate(agent_roles):
                turn_finished = data_item.batch[f'{role}_turn_finished'].item()
                # Only the final scoring role receives the global task reward.
                # Other roles are shaped only by their own penalties or future
                # local credit mechanisms.
                effective_score = raw_score if role == score_role else 0.0
                effective_score += role_bonuses.get(role, 0.0)
                if role == score_role and data_item.meta_info['mask_unfinished_reward']:
                    # For the final scoring role, zero the global correctness reward
                    # only when generation was actually interrupted/truncated.
                    # Reaching max turns can still contain a correct final answer and
                    # should not automatically wipe out the reward.
                    if turn_finished in {2, 3}:  # completion_token_exceeded / stop_when_truncated
                        effective_score = 0.0

                # Legacy format reward path disabled for cleaner experiments.
                # We now use explicit role-level penalties/bonuses (e.g. META_BOXED_PENALTY)
                # instead of single-turn-only format shaping.
                # if turn_finished == 0 and data_item.meta_info['use_format_reward'] and max_num_turns == 1:
                #     # XXX(ziyu): only add format reward for normally finished 1-turn conversation
                #     last_round_msg = data_item.non_tensor_batch['history'][i_role]
                #     assert last_round_msg['role'] == role, role
                #
                #     format_r = compute_format_r(data_source, role, last_round_msg['content'])
                #     score += format_r

                role_penalty = role_penalties.get(role, 0.0)
                if role in meta_roles and meta_has_boxed:
                    role_penalty += META_BOXED_PENALTY

                if role == score_role:
                    role_score = effective_score - role_penalty
                elif role_penalty > 0.0:
                    role_score = -max(role_penalty, MIN_NEGATIVE_SHAPED_REWARD)
                else:
                    role_score = effective_score

                reward_tensor_map[f'{role}_turn_level_reward'][i_bsz, num_turns - 1] = role_score
                role_shaped_scores[role] = float(role_score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                prompt_str = data_item.non_tensor_batch['question']
                padded_history = data_item.non_tensor_batch['history']
                history = padded_history[:num_turns * len(agent_roles)]
                already_print_data_sources[data_source] += 1
                print("[question]", prompt_str)
                print("[ground_truth]", ground_truth)
                print("[answer]", response_str)
                print("[score]", raw_score)
                print("[raw_score]", raw_score)
                if score_role is not None and score_role in role_shaped_scores:
                    print("[shaped_score]", role_shaped_scores[score_role])
                    print("[score_role]", score_role)
                if 'decomposer' in agent_roles:
                    print("[decomposer_metrics]", {
                        'unique_local_result_rate': float(reward_tensor_map['decomposer_unique_local_result_rate'][i_bsz]),
                        'dependency_usage_rate': float(reward_tensor_map['decomposer_dependency_usage_rate'][i_bsz]),
                        'repair_success': float(reward_tensor_map['decomposer_repair_success'][i_bsz]),
                        'plan_parseable_gate': float(reward_tensor_map['decomposer_plan_parseable_gate'][i_bsz]),
                        'hierarchy_utilization_gate': float(reward_tensor_map['hierarchy_utilization_gate'][i_bsz]),
                        'valid_nonfinal_worker_count': float(reward_tensor_map['valid_nonfinal_worker_count'][i_bsz]),
                        'planned_subtask_count': float(reward_tensor_map['planned_subtask_count'][i_bsz]),
                        'executed_subtask_count': float(reward_tensor_map['executed_subtask_count'][i_bsz]),
                        'positive_bonus_gate': float(reward_tensor_map['positive_role_bonus_gate'][i_bsz]),
                        'local_bonus_raw': float(reward_tensor_map['decomposer_local_bonus_raw'][i_bsz]),
                        'local_bonus': float(reward_tensor_map['decomposer_local_bonus'][i_bsz]),
                    })
                if 'selector' in agent_roles:
                    print("[selector_metrics]", {
                        'hierarchy_utilization_gate': float(reward_tensor_map['hierarchy_utilization_gate'][i_bsz]),
                        'valid_nonfinal_worker_count': float(reward_tensor_map['valid_nonfinal_worker_count'][i_bsz]),
                        'planned_subtask_count': float(reward_tensor_map['planned_subtask_count'][i_bsz]),
                        'executed_subtask_count': float(reward_tensor_map['executed_subtask_count'][i_bsz]),
                        'positive_bonus_gate': float(reward_tensor_map['positive_role_bonus_gate'][i_bsz]),
                        'assignment_completeness': float(reward_tensor_map['selector_assignment_completeness'][i_bsz]),
                        'assignment_precision': float(reward_tensor_map['selector_assignment_precision'][i_bsz]),
                        'assignment_recall': float(reward_tensor_map['selector_assignment_recall'][i_bsz]),
                        'assignment_final_present': float(reward_tensor_map['selector_assignment_final_present'][i_bsz]),
                        'assignment_extra_count': float(reward_tensor_map['selector_assignment_extra_count'][i_bsz]),
                        'assignment_missing_count': float(reward_tensor_map['selector_assignment_missing_count'][i_bsz]),
                        'assignment_duplicate_count': float(reward_tensor_map['selector_assignment_duplicate_count'][i_bsz]),
                        'empty_output': float(reward_tensor_map['selector_empty_output'][i_bsz]),
                        'worker_valid_local_result_rate': float(reward_tensor_map['selector_worker_valid_local_result_rate'][i_bsz]),
                        'local_bonus_raw': float(reward_tensor_map['selector_local_bonus_raw'][i_bsz]),
                        'local_bonus': float(reward_tensor_map['selector_local_bonus'][i_bsz]),
                    })
                if worker_roles:
                    print("[worker_bonus_metrics]", {
                        'positive_bonus_gate': float(reward_tensor_map['positive_role_bonus_gate'][i_bsz]),
                        'unique_local_result_rate': float(reward_tensor_map['worker_unique_local_result_rate'][i_bsz]),
                        'downstream_used_rate': float(reward_tensor_map['worker_downstream_used_rate'][i_bsz]),
                        'local_bonus_mean': float(reward_tensor_map['worker_local_bonus_mean'][i_bsz]),
                    })
                if score_role in agent_roles:
                    print("[final_stage_metrics]", {
                        'positive_bonus_gate': float(reward_tensor_map['positive_role_bonus_gate'][i_bsz]),
                        'worker_result_usage_rate': float(reward_tensor_map['final_worker_result_usage_rate'][i_bsz]),
                        'consistency_with_worker_results': float(reward_tensor_map['final_consistency_with_worker_results'][i_bsz]),
                        'local_bonus_raw': float(reward_tensor_map['final_local_bonus_raw'][i_bsz]),
                        'local_bonus': float(reward_tensor_map['final_local_bonus'][i_bsz]),
                    })
                if active_penalties:
                    print("[penalties]", [
                        {
                            'name': penalty_name,
                            'value': penalty_value,
                            'roles': roles,
                        }
                        for penalty_name, penalty_value, roles in active_penalties
                    ])
                    print("[role_shaped_scores]", role_shaped_scores)
                else:
                    print("[penalties]", [])
                print("[history]", history)

        # Return both reward tensors in a dictionary
        return reward_tensor_map

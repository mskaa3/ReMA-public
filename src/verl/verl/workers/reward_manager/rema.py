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
WORKER_BOXED_PENALTY = 0.10
WORKER_FINISH_PENALTY = 0.10
PLANNER_REPEAT_PENALTY = 0.10
PLANNER_EXCESS_SUBTASK_PENALTY = 0.10
PLANNER_SUBTASK_TARGET_MIN = 3
PLANNER_SUBTASK_TARGET_MAX = 5
PLANNER_SUBTASK_COUNT_PENALTY_PER_TASK = 0.20
PLANNER_SUBTASK_COUNT_MAX_PENALTY = 0.80
WORKER_EMPTY_ASSIGNED_PENALTY = 0.10
WORKER_MISSING_LOCAL_RESULT_PENALTY = 0.20
WORKER_SUBTASK_OVERREACH_PENALTY = 0.10
WORKER_DUPLICATE_RESULT_PENALTY = 0.10
FINAL_IGNORES_WORKER_RESULTS_PENALTY = 0.20
FINAL_IGNORES_MIN_LOCAL_RESULTS = 2
FINAL_IGNORES_MIN_WORDS = 80
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
            meta_roles = {'meta_thinking', 'decomposer'}
            role_penalties = {role: 0.0 for role in agent_roles}
            active_penalties = []

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
                for role in repeated_planner_roles:
                    role_penalties[role] += PLANNER_REPEAT_PENALTY

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
                for role in excess_subtask_roles:
                    role_penalties[role] += PLANNER_EXCESS_SUBTASK_PENALTY

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
                role_penalties['decomposer'] += subtask_count_penalty

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
                for role in overreach_roles:
                    role_penalties[role] += WORKER_SUBTASK_OVERREACH_PENALTY

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

            global_penalty_value = sum(penalty_value for _, penalty_value, _ in active_penalties)
            role_shaped_scores = {}
            
            for i_role, role in enumerate(agent_roles):
                turn_finished = data_item.batch[f'{role}_turn_finished'].item()
                effective_score = raw_score
                if data_item.meta_info['mask_unfinished_reward']:
                    # if conversation is not finised normally, i.e. with ['FINISH']
                    #  the reward should be zero.
                    # `turn_finished` is 0 means finished normally.
                    effective_score = effective_score if turn_finished == 0 else 0.0

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

                if global_penalty_value > 0.0:
                    # A correct final answer should not hide protocol failures in
                    # the trajectory. Make every role's shaped reward negative
                    # when any structured penalty fired, so switching the trained
                    # agent cannot still positively reinforce a bad trajectory.
                    role_penalty = max(role_penalty, global_penalty_value)

                if role_penalty > 0.0:
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

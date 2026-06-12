import numpy as np
from omegaconf import DictConfig
from verl import DataProto
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizer
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.model import compute_position_id_with_mask
from verl.protocol import collate_fn as data_proto_collate_fn, pad_dataproto_to_divisor, unpad_dataproto
import torch
import unicodedata
import re

def normalize_text(text):
    return unicodedata.normalize('NFKC', text)

def _pad_history(input_historys: List[List[Dict[str, str]]],
                 max_length: int,
                 pad_value={
                     "role": "padding",
                     "content": "<PAD>"
                 }):
    padded_history = []
    for history in input_historys:
        current_length = len(history)
        pad_length = max_length - current_length
        assert pad_length >= 0, f"current_length: {current_length}, max_length: {max_length}"
        padded_history.append(history + [pad_value] * pad_length)
    return padded_history


def _encode_conversation(
    conversation: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    num_gen_tokens: List[int],
    stop_reasons: List[Optional[str]],
):
    IGNORE_INDEX = -100
    labels = []
    step_ids = []
    cur_len = 0
    cur_hist = []
    i_step = 0
    for i, msg in enumerate(conversation):
        if msg["role"] in ["system", "user"]:
            pass
        elif msg["role"] == "assistant":
            # query string
            query = tokenizer.apply_chat_template(cur_hist,
                                                  add_generation_prompt=True,
                                                  tokenize=False)
            # response string
            response = msg["content"]
            query_ids = tokenizer.encode(query, add_special_tokens=True)
            query_response_ids = tokenizer.encode(query + response,
                                                  add_special_tokens=True)
            response_ids = query_response_ids[len(query_ids):]
            input_ids = query_response_ids

            ################################################################
            # input_ids:
            # | this | is | a | test | <im_end> | <im_start> | <assistant> | this | is | a | response | <im_end> |
            # query_ids:
            # | this | is | a | test | <im_end> | <im_start> | <assistant> |
            # response_ids:
            # | this | is | a | response | <im_end> |
            # step_ids:
            # |IGNORE| IG |IG | IG   | IG       | IG         | i_step      |i_step| ... |i_step| IGNORE |
            # labels:
            # |IGNORE| IG |IG | IG   | IG       | IG         | this | is   | a | response   | <im_end> | IGNORE
            #################################################################
            step_ids.extend([IGNORE_INDEX] * (len(query_ids) - cur_len - 1))
            labels.extend([IGNORE_INDEX] * (len(query_ids) - cur_len - 1))

            stop_reason = stop_reasons[i_step]
            # if stop normally, add eos token
            if stop_reason == "stop":
                labels.extend(response_ids + [tokenizer.eos_token_id])
                step_ids.extend([i_step] * (len(response_ids) + 1))
                num_gen_tokens[i_step] = len(response_ids) + 1
            # if truncated, do not add eos token as label
            elif stop_reason == "length":
                # print("# STOP REASON:", stop_reasons[i_step])
                labels.extend(response_ids + [IGNORE_INDEX])
                step_ids.extend([i_step] * len(response_ids) + [IGNORE_INDEX])
                num_gen_tokens[i_step] = len(response_ids)
            elif stop_reason in [
                    "stop_when_truncated", "completion_token_exceeded"
            ]:
                # special case for dummy response
                # XXX: in this case, response == ""
                assert response == ""
                labels.extend(response_ids + [IGNORE_INDEX])
                step_ids.extend([IGNORE_INDEX] * (len(response_ids) + 1))
                num_gen_tokens[i_step] = 0
                break

            i_step += 1
            cur_len = len(query_response_ids)
        else:
            raise ValueError(f"Unknown message role: {msg['role']}")
        cur_hist.append(msg)

    assert len(input_ids) == len(labels), f"{len(input_ids)} != {len(labels)}"
    return input_ids, labels, step_ids


class MultiAgentRollout:

    def __init__(
        self, 
        config: DictConfig,
        tokenizers: Dict[str, PreTrainedTokenizer],
        rollout_wg_dict: Dict[str, RayWorkerGroup]
    ):
        self.config = config
        self.tokenizers = tokenizers
        self.rollout_wg_dict = rollout_wg_dict

    def _apply_chat_template(self, chat_lst: List[List[Dict[str, str]]],
                             tokenizer: PreTrainedTokenizer):
        """Apply chat template and encode"""
        return tokenizer.apply_chat_template(
            chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=self.config.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )

    def _initialize_conversation_state(self, batch_size):
        """Initialize conversation state variables"""
        history = [[] for _ in range(batch_size)]
        finish_flags = np.zeros(batch_size, dtype=bool)
        finish_reason = [None for _ in range(batch_size)]
        return history, finish_flags, finish_reason

    def _build_chat_list_for_role(
        self,
        role: str,
        history_list: List[List[Dict[str, str]]],
        questions: List[str],
        system_prompts: Dict[str, str],
        agent_roles: List[str],
    ):
        """Build chat list for a specific role"""

        chat_lst = [[{
            "role": "system",
            "content": system_prompts[role]
        }] for _ in range(len(history_list))]

        for i, (hist, question) in enumerate(zip(history_list, questions)):
            if role == agent_roles[0]: # meta-thinking
                chat_lst[i].append({"role": "user", "content": question})
                for j in range(len(hist)):
                    if j % 2 == 0:
                        chat_lst[i].append({
                            "role": "assistant",
                            "content": hist[j]["content"]
                        })
                    else:
                        chat_lst[i].append({
                            "role": "user",
                            "content": hist[j]["content"]
                        })
            else: # reasoning
                # Ablation: reasoning receives only the meta_thinking plan/instruction,
                # without the original question.
                chat_lst[i].append({
                    "role":
                    "user",
                    "content":
                    f'Plan:\n{hist[0]["content"]}',
                })
                for j in range(1, len(hist)):
                    if (j + 1) % 2 == 0:
                        chat_lst[i].append({
                            "role": "assistant",
                            "content": hist[j]["content"]
                        })
                    else:
                        chat_lst[i].append({
                            "role": "user",
                            "content": hist[j]["content"]
                        })

        return chat_lst

    def _prepare_role_prompts(
        self,
        role: str,
        unfinished_indices: np.ndarray,
        history: List[List[Dict[str, str]]],
        questions: List[str],
        agent_roles: List[str],
        system_prompts: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer],
    ) -> Tuple[DataProto, List[List[Dict[str, str]]]]:
        """Prepare prompts for a specific role"""

        # Prepare history and questions for currently unfinished samples
        current_history = [history[idx] for idx in unfinished_indices]
        current_questions = [questions[idx] for idx in unfinished_indices]

        # Build chat list
        chat_lst = self._build_chat_list_for_role(
            role,
            current_history,
            current_questions,
            system_prompts,
            agent_roles,
        )

        # Apply chat template and encode
        inputs = self._apply_chat_template(chat_lst, tokenizers[role])
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data = DataProto.from_dict(batch_dict)
        return data, chat_lst

    def _prepare_chat_prompts(
        self,
        role: str,
        chat_lst: List[List[Dict[str, str]]],
        tokenizers: Dict[str, PreTrainedTokenizer],
    ) -> DataProto:
        inputs = self._apply_chat_template(chat_lst, tokenizers[role])
        attention_mask = inputs["attention_mask"]
        data = DataProto.from_dict({
            "input_ids": inputs["input_ids"],
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
        })
        return data

    def _generate_from_chat_list(
        self,
        role: str,
        chat_lst: List[List[Dict[str, str]]],
        tokenizers: Dict[str, PreTrainedTokenizer],
        meta_info: Dict,
        response_length: int,
    ):
        prompt_proto = self._prepare_chat_prompts(role, chat_lst, tokenizers)
        prompt_proto.meta_info.update(meta_info)
        return self._generate_role_responses(
            rollout=self.rollout_wg_dict[role],
            prompt_proto=prompt_proto,
            tokenizer=tokenizers[role],
            response_length=response_length,
        )

    def _filter_truncated_prompts_before_generation(
        self,
        prompt_proto: DataProto,
        chat_lst: List[List[Dict[str, str]]],
        role: str,
        agent_roles: List[str],
        history: List[List[Dict[str, str]]],
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
        tokenizer: PreTrainedTokenizer,
        unfinished_indices: np.ndarray,
        finish_flags: np.ndarray,
        finish_reason: List[Optional[str]],
        i_turn: int,
    ):
        # check current state length
        non_trunc_input = tokenizer.apply_chat_template(
            chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=False,
            max_length=None,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        # state length
        seq_lens = non_trunc_input["attention_mask"].sum(dim=1).tolist()
        # if state length is larger than prompt length, the trajectory is terminated
        if not all([l <= self.config.prompt_length for l in seq_lens]):
            # drop the terminated trajectories
            new_seq_lens = []
            new_unfinished_indices = []
            new_prompt_protos = []
            new_chat_lst = []
            for i, idx in enumerate(unfinished_indices):
                if seq_lens[i] <= self.config.prompt_length:
                    new_unfinished_indices.append(idx)
                    new_prompt_protos.append(prompt_proto[i])
                    new_seq_lens.append(seq_lens[i])
                    new_chat_lst.append(chat_lst[i])
                else:
                    # set finish flag and finish reason
                    finish_flags[idx] = True
                    finish_reason[idx] = "completion_token_exceeded"
                    print(f"idx={idx}, completion_token_exceeded")
                    # if the next gen is for reasoning agent, we need to add a dummy response in history
                    if role == agent_roles[1]:
                        history[idx].append({
                            "role":
                            agent_roles[1],
                            "content":
                            "",
                            "num_gen_tokens":
                            0,
                            "stop_reason":
                            "completion_token_exceeded",
                        })
                        # update conversation history for reasoning agent
                        conversation_history[agent_roles[1]][idx] = chat_lst[i]
                    else:
                        if i_turn == 0:
                            raise RuntimeError(
                                f"1st round prompt larger than prompt length: {seq_lens[i]} > {self.config.prompt_length}"
                            )

            if len(new_prompt_protos):
                # collate prompt needed to generate this round
                new_prompt_proto = data_proto_collate_fn(new_prompt_protos)
                new_prompt_proto.meta_info = prompt_proto.meta_info
            else:
                new_prompt_proto = None
            return new_prompt_proto, new_chat_lst, new_unfinished_indices
        else:
            return prompt_proto, chat_lst, unfinished_indices

    def _generate_role_responses(
        self,
        rollout: RayWorkerGroup,
        prompt_proto: DataProto,
        tokenizer: PreTrainedTokenizer,
        response_length: int,
    ):
        """Generate responses for the current role"""
        pad_prompt_proto, pad_size = pad_dataproto_to_divisor(prompt_proto, rollout.world_size)
        output = rollout.raw_generate_sequences(pad_prompt_proto)
        unpad_output = unpad_dataproto(output, pad_size=pad_size)
        resp_lens = (unpad_output.batch["attention_mask"][:, -response_length:].sum(
            dim=1).tolist())
        vllm_output_text = unpad_output.non_tensor_batch["text"].tolist()

        # output_text = tokenizer.batch_decode(
        #     output.batch["input_ids"][:, -response_length:],
        #     skip_special_tokens=False,
        # )

        # # Remove padding and EOS tokens from the output in one pass
        # pad_token = tokenizer.pad_token
        # eos_token = tokenizer.eos_token
        # output_text_clean = [
        #     text.replace(pad_token, "").replace(eos_token, "")
        #     for text in output_text
        # ]

        # for i, (decode_txt, vllm_txt) in enumerate(zip(output_text_clean, vllm_output_text)):
        #     if decode_txt != vllm_txt:
        #         print(f"i={i}, decode_txt={decode_txt}, vllm_txt={vllm_txt}")

        num_gen_tokens = unpad_output.non_tensor_batch[
            "gen_response_lengths"].tolist()
        stop_reasons = unpad_output.non_tensor_batch["stop_reasons"].tolist()

        # return output_text_clean, num_gen_tokens, stop_reasons, resp_lens
        return vllm_output_text, num_gen_tokens, stop_reasons, resp_lens

    def _update_history_and_check_finish(
        self,
        role: str,
        current_outputs: List[str],
        unfinished_indices: np.ndarray,
        history: List[List[Dict[str, str]]],
        finish_flags: np.ndarray,
        finish_reason: List[Optional[str]],
        finish_flag: str,
        agent_roles: List[str],
        num_gen_tokens: List[int],
        stop_reasons: List[Optional[str]],
        questions: List[str],
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
        system_prompts: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer],
    ):
        """Update conversation history and check completion flags"""
        # Update history
        assert len(current_outputs) == len(
            unfinished_indices
        ), f"{len(current_outputs)} != {len(unfinished_indices)}"
        for i, idx in enumerate(unfinished_indices):
            history[idx].append({
                "role": role,
                "content": current_outputs[i],
                "num_gen_tokens": num_gen_tokens[i],
                "stop_reason": stop_reasons[i],
            })

        # Update finish flags
        # Check completion flags
        if role == agent_roles[1]:
            for i, idx in enumerate(unfinished_indices):
                last_output = history[idx][-2]
                assert last_output["role"] == agent_roles[0]
                response = last_output["content"]
                if finish_flag and finish_flag in response:
                    finish_flags[idx] = True
                    finish_reason[idx] = None

        if self.config.stop_when_truncated:
            for i, stop_reason in enumerate(stop_reasons):
                # if stop_reason == "length" and not finish_flags[unfinished_indices[i]]:
                # XXX: even if stop by finish_flag, if current output is truncated, we need
                #  mark this trajectory as terminated
                if stop_reason == "length":
                    idx = unfinished_indices[i]
                    print(f"idx={idx}, stop_when_truncated")
                    finish_flags[idx] = True
                    finish_reason[idx] = "stop_when_truncated"
                    if role == agent_roles[0]:
                        # update conversation for reasoning agent
                        _, new_conversation = self._prepare_role_prompts(
                            agent_roles[1],
                            [idx],
                            history,
                            questions,
                            agent_roles,
                            system_prompts,
                            tokenizers,
                        )
                        conversation_history[
                            agent_roles[1]][idx] = new_conversation[0]

                        # add dummy history of reasoning agent
                        history[idx].append({
                            "role":
                            agent_roles[1],
                            "content":
                            "",
                            "num_gen_tokens":
                            0,
                            "stop_reason":
                            "stop_when_truncated",
                        })

    def _run_multi_turn_conversation(
        self,
        prompts: DataProto,
        tokenizers: Dict[str, PreTrainedTokenizer],
        max_num_turns: int,
        agent_roles: List[str],
        system_prompts: Dict[str, str],
        finish_flag: str,
        history: List[List[Dict[str, str]]],
        finish_flags: np.ndarray,
        finish_reason: List[Optional[str]],
        response_length: int,
    ):
        questions = prompts.non_tensor_batch["question"]
        assert len(finish_flags) == len(
            questions), f"{finish_flags.shape} != {len(questions)}"

        conversation_history = {
            role: [None for _ in range(len(questions))]
            for role in agent_roles
        }

        for i_turn in range(max_num_turns):
            # Get indices of unfinished samples
            unfinished_indices = np.where(~finish_flags)[0]
            print(f"turn {i_turn+1} of {max_num_turns}, \
                    {len(unfinished_indices)}/{len(questions)} unfinished")

            if len(unfinished_indices) == 0:
                break
            # Each role takes turns generating in every round
            for i_role, role in enumerate(agent_roles):
                print(f"role: {role}")
                # Prepare prompts for current role
                prompt_proto, chat_lst = self._prepare_role_prompts(
                    role,
                    unfinished_indices,
                    history,
                    questions,
                    agent_roles,
                    system_prompts,
                    tokenizers,
                )

                # side effect on convsersation_history and history
                prompt_proto, chat_lst, unfinished_indices = (
                    self._filter_truncated_prompts_before_generation(
                        prompt_proto=prompt_proto,
                        chat_lst=chat_lst,
                        role=role,
                        agent_roles=agent_roles,
                        history=history,
                        conversation_history=conversation_history,
                        tokenizer=tokenizers[role],
                        unfinished_indices=unfinished_indices,
                        finish_flags=finish_flags,
                        finish_reason=finish_reason,
                        i_turn=i_turn,
                    ))
                if len(unfinished_indices) == 0:
                    break

                prompt_proto.meta_info.update(prompts.meta_info)
                for i, chat in enumerate(chat_lst):
                    idx = unfinished_indices[i]
                    conversation_history[role][idx] = chat

                # Generate responses for current role
                current_outputs, num_gen_tokens, stop_reasons, resp_lens = (
                    self._generate_role_responses(
                        rollout=self.rollout_wg_dict[role],
                        prompt_proto=prompt_proto,
                        tokenizer=tokenizers[role],
                        response_length=response_length,
                    ))

                # XXX(ziyu): remove finish flag in output for reasoning agent here
                #  consider move to a post-processing function
                if role == agent_roles[1] and finish_flag:
                    current_outputs = [
                        output.replace(finish_flag, "").rstrip()
                        for output in current_outputs
                    ]

                # XXX(ziyu): side effect on `history`
                self._update_history_and_check_finish(
                    role,
                    current_outputs,
                    unfinished_indices,
                    history,
                    finish_flags,
                    finish_reason,
                    finish_flag,
                    agent_roles,
                    num_gen_tokens,
                    stop_reasons,
                    questions,
                    conversation_history,
                    system_prompts,
                    tokenizers,
                )
                unfinished_indices = np.where(~finish_flags)[0]
                if len(unfinished_indices) == 0:
                    break
        # use the last output of each agent as latest output response
        latest_outputs = [h[-1]["content"] for h in history]
        return latest_outputs, conversation_history

    @staticmethod
    def _extract_subtasks(plan_text: str) -> List[Tuple[str, str]]:
        subtasks = []
        seen_subtasks = set()
        for line in plan_text.splitlines():
            match = re.match(
                r"\s*(?:[-*]\s*)?(?:\d+\.\s*)?(S\d+)\s*[:.)-]\s*(.+?)\s*$",
                line,
                re.IGNORECASE,
            )
            if match:
                subtask_id = match.group(1).upper()
                if subtask_id in seen_subtasks:
                    continue
                subtasks.append((subtask_id, match.group(2).strip()))
                seen_subtasks.add(subtask_id)
        return subtasks

    @staticmethod
    def _parse_assignments(
        selector_text: str,
        subtasks: List[Tuple[str, str]],
        worker_roles: List[str],
        default_worker: str,
    ) -> Dict[str, str]:
        assignments = {}
        lowered_workers = {worker.lower(): worker for worker in worker_roles}
        for subtask_id, _ in subtasks:
            selected_worker = None
            for line in selector_text.splitlines():
                if subtask_id.lower() not in line.lower():
                    continue
                for worker_lower, worker in lowered_workers.items():
                    if worker_lower in line.lower():
                        selected_worker = worker
                        break
                if selected_worker is not None:
                    break
            assignments[subtask_id] = selected_worker or default_worker
        return assignments

    @staticmethod
    def _parse_ordered_worker_stages(
        selector_text: str,
        subtasks: List[Tuple[str, str]],
        stage_roles: List[str],
        worker_types: List[str],
        default_worker: str,
    ) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
        """Parse selector output into ordered worker stages.

        Stage roles provide fixed tensor slots. Worker types provide the skill
        specialization used inside a stage. This lets the selector route:
        worker_stage_1 as algebra, worker_stage_2 as general math,
        worker_stage_3 as algebra again without repeating a role in history.
        """
        subtask_map = {subtask_id: description for subtask_id, description in subtasks}
        lowered_workers = {}
        for worker in worker_types:
            lowered_workers[worker.lower()] = worker
            if worker.endswith("_worker"):
                lowered_workers[worker[:-len("_worker")].lower()] = worker
        lowered_stages = {stage.lower(): stage for stage in stage_roles}
        stage_items = []
        seen_subtasks = set()
        next_stage_idx = 0
        worker_stage_roles = stage_roles[:-1] if len(stage_roles) > 1 else stage_roles
        final_stage_role = stage_roles[-1] if stage_roles else None

        for line in selector_text.splitlines():
            worker = None
            stage_role = None
            line_lower = line.lower()
            subtask_ids = [match.upper() for match in re.findall(r"\bS\d+\b", line, re.IGNORECASE)]
            is_final_line = "final" in line_lower
            for stage_lower, stage_name in lowered_stages.items():
                if stage_lower in line_lower:
                    stage_role = stage_name
                    break
            for worker_lower, worker_name in lowered_workers.items():
                if worker_lower in line_lower:
                    worker = worker_name
                    break
            if worker is None:
                worker = default_worker

            if stage_role is None and not subtask_ids and not is_final_line:
                continue

            if stage_role is None and is_final_line and final_stage_role:
                stage_role = final_stage_role
            elif stage_role is None:
                if next_stage_idx >= len(worker_stage_roles):
                    stage_role = worker_stage_roles[-1]
                else:
                    stage_role = worker_stage_roles[next_stage_idx]
                    next_stage_idx += 1
            elif subtask_ids and stage_role == final_stage_role and worker_stage_roles and not is_final_line:
                stage_role = worker_stage_roles[-1]

            stage_subtasks = []
            for subtask_id in subtask_ids:
                if subtask_id in subtask_map and subtask_id not in seen_subtasks:
                    stage_subtasks.append((subtask_id, subtask_map[subtask_id]))
                    seen_subtasks.add(subtask_id)
            if stage_subtasks:
                stage_items.append((stage_role, worker, stage_subtasks))
            elif final_stage_role and stage_role == final_stage_role and is_final_line:
                stage_items.append((stage_role, worker, []))

        if not stage_items:
            seen_subtasks = set()
            for i, (subtask_id, description) in enumerate(subtasks):
                stage_role = worker_stage_roles[min(i, len(worker_stage_roles) - 1)]
                worker = default_worker
                stage_items.append((stage_role, worker, [(subtask_id, description)]))
                seen_subtasks.add(subtask_id)

        for subtask_id, _ in subtasks:
            if subtask_id not in seen_subtasks:
                stage_role = worker_stage_roles[min(len(stage_items), len(worker_stage_roles) - 1)]
                stage_items.append((stage_role, default_worker, [(subtask_id, subtask_map[subtask_id])]))
                seen_subtasks.add(subtask_id)

        merged_by_stage = {}
        for stage_role, worker, stage_subtasks in stage_items:
            if stage_role not in merged_by_stage:
                merged_by_stage[stage_role] = [worker, []]
            merged_by_stage[stage_role][1].extend(stage_subtasks)

        return [
            (stage_role, merged_by_stage[stage_role][0], merged_by_stage[stage_role][1])
            for stage_role in stage_roles
            if stage_role in merged_by_stage
        ]

    @staticmethod
    def _format_worker_specs(worker_specs: Dict[str, str], worker_roles: List[str]) -> str:
        return "\n".join([
            f"- {role}: {worker_specs.get(role, '')}"
            for role in worker_roles
        ])

    @staticmethod
    def _format_subtasks(subtasks: List[Tuple[str, str]]) -> str:
        return "\n".join([f"- {subtask_id}: {description}" for subtask_id, description in subtasks])

    @staticmethod
    def _format_completed_worker_results(completed_results: List[Tuple[str, str, str, str]]) -> str:
        if not completed_results:
            return "- None yet."
        return "\n\n".join([
            f"{stage_role} as {worker_type} ({subtask_ids}):\n{output}"
            for stage_role, worker_type, subtask_ids, output in completed_results
        ])

    @staticmethod
    def _format_work_so_far(completed_results: List[Tuple[str, str, str, str]]) -> str:
        return "\n\n".join([
            output.strip()
            for _, _, _, output in completed_results
            if output and output.strip()
        ])

    @staticmethod
    def _extract_local_result(output: str) -> str:
        if not output:
            return ""
        match = re.search(
            r"local[_ ]result\s*:\s*(.*?)(?:\n\s*reasoning\s*:|\n\s*subtask\b|\Z)",
            output,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        first_line = output.strip().splitlines()[0] if output.strip() else ""
        return first_line[:240]

    @staticmethod
    def _extract_reasoning(output: str) -> str:
        if not output:
            return ""
        match = re.search(
            r"reasoning\s*:\s*(.*?)(?:\n\s*local[_ ]result\s*:|\n\s*subtask\b|\n\s*final\b|\Z)",
            output,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return output.strip()

    def _format_hierarchical_feedback(
        self,
        plan: str,
        assignments: str,
        worker_results: Dict[str, str],
        last_worker_output: str,
        worker_roles: List[str],
    ) -> str:
        fragments = []
        for worker_role in worker_roles:
            output = worker_results.get(worker_role, "")
            reasoning = self._extract_reasoning(output)
            local_result = self._extract_local_result(output)
            if reasoning:
                fragments.append(reasoning)
            if local_result and local_result not in reasoning:
                fragments.append(local_result)
        if last_worker_output and last_worker_output not in "\n\n".join(fragments):
            fragments.append(last_worker_output)
        return "\n\n".join(fragment for fragment in fragments if fragment.strip())

    def _run_hierarchical_conversation(
        self,
        prompts: DataProto,
        tokenizers: Dict[str, PreTrainedTokenizer],
        max_num_turns: int,
        agent_roles: List[str],
        system_prompts: Dict[str, str],
        hierarchy_config: Dict,
        history: List[List[Dict[str, str]]],
        finish_flags: np.ndarray,
        finish_reason: List[Optional[str]],
        response_length: int,
        finish_flag: Optional[str],
    ):
        questions = prompts.non_tensor_batch["question"]
        batch_size = len(questions)
        decomposer_role = hierarchy_config.get("decomposer_role", "decomposer")
        selector_role = hierarchy_config.get("selector_role", "selector")
        worker_types = hierarchy_config.get("worker_roles", [])
        stage_roles = hierarchy_config.get("stage_roles")
        if stage_roles is None:
            stage_roles = [
                f"worker_stage_{idx}"
                for idx in range(1, int(hierarchy_config.get("num_worker_stages", 0)) + 1)
            ] or worker_types
        default_worker = hierarchy_config.get("default_worker", worker_types[-1] if worker_types else selector_role)
        worker_specs = hierarchy_config.get("worker_specs", {})
        pass_question_to_workers = hierarchy_config.get("pass_question_to_workers", False)
        worker_spec_text = self._format_worker_specs(worker_specs, worker_types)

        conversation_history = {
            role: [None for _ in range(batch_size)]
            for role in agent_roles
        }
        running_conversation = {
            role: [[{"role": "system", "content": system_prompts[role]}]
                   for _ in range(batch_size)]
            for role in agent_roles
        }
        previous_feedback = [None for _ in range(batch_size)]
        latest_outputs = ["" for _ in range(batch_size)]

        def append_history(idx, role, content, num_gen_tokens, stop_reason):
            history[idx].append({
                "role": role,
                "content": content,
                "num_gen_tokens": num_gen_tokens,
                "stop_reason": stop_reason,
            })

        def build_prompt(role, idx, content):
            return running_conversation[role][idx] + [{"role": "user", "content": content}]

        def build_selected_worker_prompt(stage_role, worker_type, idx, content):
            system_prompt = system_prompts.get(worker_type, system_prompts[stage_role])
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]

        def record_prompt_and_output(idx, role, chat, output, num_gen_tokens, stop_reason):
            conversation_history[role][idx] = chat
            append_history(idx, role, output, num_gen_tokens, stop_reason)
            running_conversation[role][idx] = chat + [{"role": "assistant", "content": output}]

        for i_turn in range(max_num_turns):
            unfinished_indices = np.where(~finish_flags)[0]
            print(f"hierarchical turn {i_turn+1} of {max_num_turns}, "
                  f"{len(unfinished_indices)}/{batch_size} unfinished")
            if len(unfinished_indices) == 0:
                break

            # 1. Decompose or revise the plan using previous round feedback.
            decomposer_chats = []
            for idx in unfinished_indices:
                content = f"Question:\n{questions[idx]}"
                if previous_feedback[idx]:
                    content += f"\n\n{previous_feedback[idx]}"
                decomposer_chats.append(build_prompt(decomposer_role, idx, content))
            decomposer_outputs, decomposer_tokens, decomposer_stops, _ = self._generate_from_chat_list(
                decomposer_role, decomposer_chats, tokenizers, prompts.meta_info, response_length)
            current_plan = {}
            for local_idx, idx in enumerate(unfinished_indices):
                output = decomposer_outputs[local_idx]
                current_plan[idx] = output
                record_prompt_and_output(
                    idx, decomposer_role, decomposer_chats[local_idx], output,
                    decomposer_tokens[local_idx], decomposer_stops[local_idx])

            # 2. Select workers for each subtask.
            selector_chats = []
            parsed_subtasks = {}
            for idx in unfinished_indices:
                subtasks = self._extract_subtasks(current_plan[idx])
                parsed_subtasks[idx] = subtasks
                selector_chats.append(build_prompt(
                    selector_role,
                    idx,
                    (
                        f"Question:\n{questions[idx]}\n\n"
                        f"Plan:\n{current_plan[idx]}\n\n"
                        f"Available workers:\n{worker_spec_text}"
                    ),
                ))
            selector_outputs, selector_tokens, selector_stops, _ = self._generate_from_chat_list(
                selector_role, selector_chats, tokenizers, prompts.meta_info, response_length)
            ordered_stages_by_idx = {}
            selector_output_by_idx = {}
            for local_idx, idx in enumerate(unfinished_indices):
                output = selector_outputs[local_idx]
                selector_output_by_idx[idx] = output
                record_prompt_and_output(
                    idx, selector_role, selector_chats[local_idx], output,
                    selector_tokens[local_idx], selector_stops[local_idx])
                ordered_stages_by_idx[idx] = self._parse_ordered_worker_stages(
                    output, parsed_subtasks[idx], stage_roles, worker_types, default_worker)
                if stage_roles:
                    final_stage_role = stage_roles[-1]
                    if all(stage_role != final_stage_role
                           for stage_role, _, _ in ordered_stages_by_idx[idx]):
                        ordered_stages_by_idx[idx].append(
                            (final_stage_role, default_worker, []))

            # 3. Execute selected worker stages sequentially. Stage roles encode
            # the order; each later worker sees previous results.
            worker_results = {idx: {role: "" for role in stage_roles} for idx in unfinished_indices}
            worker_records = {
                idx: {
                    role: None
                    for role in stage_roles
                }
                for idx in unfinished_indices
            }
            completed_results_by_idx = {idx: [] for idx in unfinished_indices}
            max_stage_count = max(
                [len(ordered_stages_by_idx[idx]) for idx in unfinished_indices],
                default=0,
            )
            for stage_idx in range(max_stage_count):
                for stage_role in stage_roles:
                    stage_indices = [
                        idx for idx in unfinished_indices
                        if (
                            stage_idx < len(ordered_stages_by_idx[idx])
                            and ordered_stages_by_idx[idx][stage_idx][0] == stage_role
                        )
                    ]
                    if not stage_indices:
                        continue

                    worker_chats = []
                    worker_chats_by_idx = {}
                    stage_subtasks_by_idx = {}
                    worker_type_by_idx = {}
                    for idx in stage_indices:
                        _, worker_type, assigned_subtasks = ordered_stages_by_idx[idx][stage_idx]
                        stage_subtasks_by_idx[idx] = assigned_subtasks
                        worker_type_by_idx[idx] = worker_type
                        is_final_stage = stage_idx == len(ordered_stages_by_idx[idx]) - 1
                        if is_final_stage:
                            question_block = f"Question:\n{questions[idx]}\n\n"
                        elif pass_question_to_workers:
                            question_block = f"{questions[idx]}\n\n"
                        else:
                            question_block = ""
                        work_so_far = self._format_work_so_far(completed_results_by_idx[idx])
                        assigned_subtasks_text = self._format_subtasks(assigned_subtasks)
                        stage_instruction = (
                            "Write the final answer using the work above. "
                            "Include the exact token [FINISH] and put the final answer in \\boxed{}. "
                            "Do not write [FINISH] unless the final answer is present in \\boxed{}."
                            if is_final_stage else
                            "Solve only the step above. "
                            "Do not write [FINISH], \\boxed{}, or Final Answer. "
                            "Output exactly:\n"
                            "REASONING:\n"
                            "<brief reasoning for this subtask only>\n\n"
                            "LOCAL_RESULT:\n"
                            "<the final result of this subtask only>"
                        )
                        if is_final_stage and not assigned_subtasks_text:
                            assigned_subtasks_text = (
                                "- Use the work above to answer the original question."
                            )
                        work_so_far_block = f"{work_so_far}\n\n" if work_so_far else ""
                        chat = build_selected_worker_prompt(
                            stage_role,
                            worker_type,
                            idx,
                            (
                                f"{question_block}"
                                f"{work_so_far_block}"
                                f"{assigned_subtasks_text}\n\n"
                                f"{stage_instruction}\n\n"
                            ),
                        )
                        worker_chats.append(chat)
                        worker_chats_by_idx[idx] = chat
                    outputs, tokens, stops, _ = self._generate_from_chat_list(
                        stage_role, worker_chats, tokenizers, prompts.meta_info, response_length)
                    for local_idx, idx in enumerate(stage_indices):
                        output = outputs[local_idx]
                        worker_results[idx][stage_role] = output
                        worker_records[idx][stage_role] = (
                            worker_chats_by_idx[idx],
                            output,
                            tokens[local_idx],
                            stops[local_idx],
                            [subtask_id for subtask_id, _ in stage_subtasks_by_idx[idx]],
                        )
                        subtask_ids = ", ".join([subtask_id for subtask_id, _ in stage_subtasks_by_idx[idx]])
                        completed_results_by_idx[idx].append((stage_role, worker_type_by_idx[idx], subtask_ids, output))

                        latest_outputs[idx] = output
                        is_final_stage = stage_idx == len(ordered_stages_by_idx[idx]) - 1
                        if is_final_stage:
                            final_worker_has_answer = "[FINISH]" in output and "\\boxed" in output
                            if final_worker_has_answer:
                                finish_flags[idx] = True
                                finish_reason[idx] = None
                            if self.config.stop_when_truncated and stops[local_idx] == "length":
                                finish_flags[idx] = True
                                finish_reason[idx] = "stop_when_truncated"

            # Keep exactly one history slot per role per hierarchical turn.
            for idx in unfinished_indices:
                for stage_role in stage_roles:
                    record = worker_records[idx].get(stage_role)
                    if record is None:
                        chat = build_prompt(stage_role, idx, "No subtasks were assigned to this worker stage.")
                        record_prompt_and_output(idx, stage_role, chat, "", 0, "stop")
                    else:
                        chat, output, num_gen_tokens, stop_reason, assigned_subtask_ids = record
                        record_prompt_and_output(idx, stage_role, chat, output, num_gen_tokens, stop_reason)
                        if history[idx] and history[idx][-1].get("role") == stage_role:
                            history[idx][-1]["assigned_subtasks"] = assigned_subtask_ids

            for idx in unfinished_indices:
                previous_feedback[idx] = self._format_hierarchical_feedback(
                    current_plan[idx], selector_output_by_idx[idx], worker_results[idx],
                    latest_outputs[idx], stage_roles)

        return latest_outputs, conversation_history

    def _mark_unfinished_as_max_turns(self, finish_flags: np.ndarray,
                                      finish_reason: List[Optional[str]]):
        """Mark unfinished samples as reaching maximum turns"""
        for i in range(len(finish_flags)):
            if not finish_flags[i]:
                finish_reason[i] = "reach_max_turn"

    def _build_tensor_dict(
        self,
        last_round_responses: List[Dict[str, str]],
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
        tokenizers: Dict[str, PreTrainedTokenizer],
        num_gen_token_lst: Dict[str, List[List[int]]],
        stop_reason_lst: Dict[str, List[List[Optional[str]]]],
        max_num_turns: int,
        finish_reason: List[Optional[str]],
    ):
        # add last round output to make full conversation
        for i_batch in range(len(last_round_responses)):
            for role in last_round_responses[i_batch]:
                conversation_history[role][i_batch].append({
                    "role":
                    "assistant",
                    "content":
                    last_round_responses[i_batch][role],
                })

        input_ids_lst = {role: [] for role in conversation_history.keys()}
        labels_lst = {role: [] for role in conversation_history.keys()}
        step_ids_lst = {role: [] for role in conversation_history.keys()}

        # build tensors for training
        for i_batch in range(len(last_round_responses)):
            for role in conversation_history.keys():
                # encode conversation into input_ids, labels, step_ids
                # XXX(ziyu): need to consider stop reason here ?
                input_ids, labels, step_ids = _encode_conversation(
                    conversation_history[role][i_batch],
                    tokenizers[role],
                    num_gen_token_lst[role][i_batch],
                    stop_reason_lst[role][i_batch],
                )
                input_ids_lst[role].append(input_ids)
                labels_lst[role].append(labels)
                step_ids_lst[role].append(step_ids)

        # Apply padding to create tensors
        batch_size = len(last_round_responses)
        tensor_dict = {}
        finish_reason_array = []
        for fr in finish_reason:
            if fr == "reach_max_turn":
                finish_reason_array.append(1)
            elif fr == "completion_token_exceeded":
                finish_reason_array.append(2)
            elif fr == "stop_when_truncated":
                finish_reason_array.append(3)
            elif fr is None:
                finish_reason_array.append(0)
            else:
                raise ValueError(f"Unknown finish reason: {fr}")

        for role in conversation_history.keys():
            # Find max length for padding
            max_length = max([len(ids) for ids in input_ids_lst[role]])
            if max_length > self.config.response_length + self.config.prompt_length:
                print(
                    f"role: {role}, max_length={max_length} > {self.config.response_length + self.config.prompt_length}"
                )
                # raise RuntimeError(f"max_length={max_length} > {self.config.response_length + self.config.prompt_length}")

            # Use max length for padding and gathering
            max_length = self.config.response_length + self.config.prompt_length

            # Pad and convert to tensors
            padded_input_ids = torch.full((batch_size, max_length),
                                          tokenizers[role].pad_token_id,
                                          dtype=torch.long)
            padded_labels = torch.full(
                (batch_size, max_length),
                -100,
                dtype=torch.long  # IGNORE_INDEX
            )
            padded_step_ids = torch.full(
                (batch_size, max_length),
                -100,
                dtype=torch.long  # IGNORE_INDEX
            )
            attention_mask = torch.zeros((batch_size, max_length),
                                         dtype=torch.long)

            # Fill in the actual values
            for i, (input_ids, labels, step_ids) in enumerate(
                    zip(input_ids_lst[role], labels_lst[role],
                        step_ids_lst[role])):
                seq_len = min(len(input_ids), max_length)
                padded_input_ids[i, :seq_len] = torch.tensor(
                    input_ids[:seq_len], dtype=torch.long)
                padded_labels[i, :seq_len] = torch.tensor(labels[:seq_len],
                                                          dtype=torch.long)
                padded_step_ids[i, :seq_len] = torch.tensor(step_ids[:seq_len],
                                                            dtype=torch.long)
                attention_mask[i, :seq_len] = 1

            # Compute position ids from attention mask
            position_ids = compute_position_id_with_mask(attention_mask)

            padded_num_gen_tokens = torch.full((batch_size, max_num_turns),
                                               0,
                                               dtype=torch.long)
            for i, num_gen_tokens in enumerate(num_gen_token_lst[role]):
                padded_num_gen_tokens[i, :len(num_gen_tokens)] = torch.tensor(
                    num_gen_tokens, dtype=torch.long)
            padded_stop_reasons = torch.full((batch_size, max_num_turns),
                                             0,
                                             dtype=torch.bool)

            for i, stop_reasons in enumerate(stop_reason_lst[role]):
                stop_reason_array = np.array(
                    [0 if r == "stop" else 1 for r in stop_reasons])
                padded_stop_reasons[i, :len(stop_reason_array)] = torch.tensor(
                    stop_reason_array, dtype=torch.bool)

            # Create a separate tensor dict for each role
            tensor_dict[role] = dict(
                {
                    "input_ids": padded_input_ids,
                    "labels": padded_labels,
                    "step_ids": padded_step_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "num_gen_tokens": padded_num_gen_tokens,
                    "stop_reasons": padded_stop_reasons,
                    "turn_finished": torch.tensor(finish_reason_array),
                }, )

        # remove side effect
        for i_batch in range(len(last_round_responses)):
            for role in last_round_responses[i_batch]:
                conversation_history[role][i_batch].pop()

        return tensor_dict

    def _prepare_final_output(
        self,
        tensor_dict: Dict[str, Dict[str, torch.Tensor]],
        latest_outputs: List[str],
        history: List[List[Dict[str, str]]],
        finish_reason: List[Optional[str]],
        agent_roles: List[str],
        prompts: DataProto,
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
    ):
        """Prepare final output"""

        non_tensor_batch = prompts.non_tensor_batch
        hierarchy_config = prompts.meta_info.get("hierarchy", {})
        if hierarchy_config.get("enable", False):
            score_role = hierarchy_config.get("score_role")
            if score_role:
                latest_outputs = [
                    next(
                        (
                            msg.get("content", "")
                            for msg in reversed(sample_history)
                            if isinstance(msg, dict) and msg.get("role") == score_role
                        ),
                        output,
                    )
                    for sample_history, output in zip(history, latest_outputs)
                ]
        non_tensor_batch["finish_reason"] = finish_reason
        non_tensor_batch["num_turns"] = [
            len(h) // len(agent_roles) for h in history
        ]
        non_tensor_batch["response"] = latest_outputs

        max_history_length = max(2 * self.config.max_num_turns,
                                 len(agent_roles) * self.config.max_num_turns)
        padded_history = _pad_history(history, max_history_length)
        padded_conversation_history = {
            role:
            _pad_history(conversation_history[role],
                         max_history_length)
            for role in agent_roles
        }

        non_tensor_batch["history"] = padded_history
        for role in agent_roles:
            non_tensor_batch[
                f"{role}_conversation_history"] = padded_conversation_history[
                    role]

        flat_tensor_dict = {}
        for role in tensor_dict.keys():
            for key in tensor_dict[role].keys():
                flat_tensor_dict[f"{role}_{key}"] = tensor_dict[role][key]

        return DataProto.from_dict(
            tensors=flat_tensor_dict,
            non_tensors=non_tensor_batch,
            meta_info=prompts.meta_info,
        )
    
    def _checking(
        self,
        history: List[List[Dict[str, str]]],
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
        agent_roles: List[str],
        last_round_responses: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        tensor_dict: Dict[str, torch.tensor],
        final_output: DataProto,
    ):
        ###################### TESTING ######################
        # 1. test lengths of history and conversation_history
        #  len(history[i]) == len(conversation_history[role][i]) * len(agent_roles)
        for i in range(len(history)):
            assert len(history[i]) == len(conversation_history[agent_roles[0]][i]), \
                f"len(history[i]) = {len(history[i])} != len(conversation_history[agent_roles[0]][i]) = {len(conversation_history[agent_roles[0]][i])}"
            assert len(conversation_history[agent_roles[0]][i]) == len(conversation_history[agent_roles[1]][i]), \
                f"len(conversation_history[agent_roles[0]][i]) = {len(conversation_history[agent_roles[0]][i])} != len(conversation_history[agent_roles[1]][i]) = {len(conversation_history[agent_roles[1]][i])}"

        # 2. check history role name order
        for i in range(len(history)):
            for j in range(len(history[i])):
                assert history[i][j]['role'] == agent_roles[j % len(agent_roles)], \
                    f"history[i][j]['role'] = {history[i][j]['role']} != agent_roles[j % len(agent_roles)] = {agent_roles[j % len(agent_roles)]}"
                
            # 2.1 check last round response
            for i_role, role in enumerate(agent_roles):
                assert history[i][-len(agent_roles) + i_role]['role'] == role, \
                    f"history[i][-len(agent_roles) + i_role]['role'] = {history[i][-len(agent_roles) + i_role]['role']} != role = {role}"
                assert history[i][-len(agent_roles) + i_role]['content'] == last_round_responses[i][role], \
                    f"history[i][-1]['content'] = {history[i][-1]['content']} != last_round_responses[i][role] = {last_round_responses[i][role]}"
            
        # 3. check conversation_history role name order
        for i_role, role in enumerate(conversation_history.keys()):
            for i in range(len(conversation_history[role])):
                for j in range(len(conversation_history[role][i])):
                    if j == 0:
                        assert conversation_history[role][i][j]['role'] == "system", \
                            f"conversation_history[role][i][j]['role'] = {conversation_history[role][i][j]['role']} != 'system'"
                    elif j % 2 == 1:
                        assert conversation_history[role][i][j]['role'] == "user", \
                            f"conversation_history[role][i][j]['role'] = {conversation_history[role][i][j]['role']} != 'user'"
                    else:
                        assert conversation_history[role][i][j]['role'] == "assistant", \
                            f"conversation_history[role][i][j]['role'] = {conversation_history[role][i][j]['role']} != 'assistant'"
                        # check history string equals to conversation_string
                        assert conversation_history[role][i][j]['content'] == history[i][i_role + j - 2]['content'], \
                            f"'{[conversation_history[role][i][j]['content']]}' != '{[history[i][i_role + j - 2]['content']]}'"

        # 4. check input_ids
        for i_role, role in enumerate(agent_roles):
            role_tensor_dict = tensor_dict[role]
            for i in range(len(role_tensor_dict["input_ids"])):
                input_ids = role_tensor_dict["input_ids"][i]
                labels = role_tensor_dict["labels"][i]
                attention_mask = role_tensor_dict["attention_mask"][i]
                step_ids = role_tensor_dict["step_ids"][i]
                stop_reasons = role_tensor_dict["stop_reasons"][i]
                num_turn = final_output.non_tensor_batch["num_turns"][i]

                query_response = tokenizer.decode(input_ids[attention_mask == 1].tolist())
                raw_query_response = tokenizer.apply_chat_template(
                    conversation_history[role][i], 
                    add_generation_prompt=True, 
                    padding=True, 
                    truncation=False, 
                    max_length=None, 
                    tokenize=False, 
                ) + last_round_responses[i][role]
                
                assert step_ids.max() == num_turn - 1 or stop_reasons[num_turn - 1] != 0, \
                    f"{step_ids.max()} != {num_turn - 1} or {stop_reasons[num_turn - 1]} != 0"

                # FIXME: tokenizer has some issues on decode and encode unicode chars.

                assert normalize_text(query_response) == normalize_text(raw_query_response), \
                    f"'{query_response}' != '{raw_query_response}'"
                for i_turn in range(num_turn):
                    turn_labels = labels[step_ids == i_turn]
                    if stop_reasons[i_turn] == 0:
                        assert turn_labels[-1] == tokenizer.eos_token_id
                        turn_labels = turn_labels[:-1] # drop eos
                    response = tokenizer.decode(turn_labels.tolist())
                    assert normalize_text(response) == normalize_text(history[i][i_role + i_turn * len(agent_roles)]['content']), \
                        f"'{response}' != '{history[i][i_role + i_turn * len(agent_roles)]['content']}'"
        

    def generate(self, prompts: DataProto):
        agent_roles = prompts.meta_info["agent_roles"]
        system_prompts = prompts.meta_info["system_prompts"]
        finish_flag = prompts.meta_info["finish_flag"]
        max_num_turns = self.config.max_num_turns

        rollout_wg = self.rollout_wg_dict

        # tokenizers = {role: wg.tokenizer for role, wg in rollout_wg.items()}
        tokenizers = self.tokenizers
        for role in rollout_wg.keys():
            tokenizers[role].padding_side = "left"
            if tokenizers[role].pad_token is None:
                tokenizers[role].pad_token = tokenizers[role].eos_token

        prompts.meta_info['is_multi_turn'] = True

        questions = prompts.non_tensor_batch["question"]
        batch_size = len(questions)
        history, finish_flags, finish_reason = self._initialize_conversation_state(
            batch_size)

        # Multi-turn dialogue generation
        # this will change the history, finish_flags, finish_reason
        hierarchy_config = prompts.meta_info.get("hierarchy", {})
        if hierarchy_config.get("enable", False):
            latest_outputs, conversation_history = self._run_hierarchical_conversation(
                prompts=prompts,
                tokenizers=tokenizers,
                max_num_turns=max_num_turns,
                agent_roles=agent_roles,
                system_prompts=system_prompts,
                hierarchy_config=hierarchy_config,
                history=history,
                finish_flags=finish_flags,
                finish_reason=finish_reason,
                response_length=self.config.response_length,
                finish_flag=finish_flag,
            )
        else:
            latest_outputs, conversation_history = self._run_multi_turn_conversation(
                prompts=prompts,
                tokenizers=tokenizers,
                max_num_turns=max_num_turns,
                agent_roles=agent_roles,
                system_prompts=system_prompts,
                finish_flag=finish_flag,
                history=history,
                finish_flags=finish_flags,
                finish_reason=finish_reason,
                response_length=self.config.response_length,
            )

        # Mark completion reasons
        # this will change the finish_reason
        if max_num_turns > 1:
            self._mark_unfinished_as_max_turns(finish_flags, finish_reason)

        last_round_responses = [{
            m["role"]: m["content"]
            for m in h[-len(agent_roles):]
        } for h in history]

        # extract information from history record
        num_gen_token_lst = {role: [] for role in agent_roles}
        stop_reason_lst = {role: [] for role in agent_roles}
        for h in history:
            _num_gen_tokens = {role: [] for role in agent_roles}
            _stop_reasons = {role: [] for role in agent_roles}
            for m in h:
                _num_gen_tokens[m["role"]].append(m["num_gen_tokens"])
                _stop_reasons[m["role"]].append(m["stop_reason"])
            for role in agent_roles:
                num_gen_token_lst[role].append(_num_gen_tokens[role])
                stop_reason_lst[role].append(_stop_reasons[role])

        tensor_dict = self._build_tensor_dict(
            last_round_responses,
            conversation_history,
            tokenizers,
            num_gen_token_lst,
            stop_reason_lst,
            max_num_turns,
            finish_reason,
        )

        # Prepare return results
        final_output = self._prepare_final_output(
            tensor_dict=tensor_dict,
            latest_outputs=latest_outputs,
            history=history,
            finish_reason=finish_reason,
            agent_roles=agent_roles,
            prompts=prompts,
            conversation_history=conversation_history,
        )
        
        if self.config.add_checking and not hierarchy_config.get("enable", False):
            try:
                self._checking(
                    history=history,
                    conversation_history=conversation_history,
                    agent_roles=agent_roles,
                    last_round_responses=last_round_responses,
                    tokenizer=tokenizers[agent_roles[0]],
                    tensor_dict=tensor_dict,
                    final_output=final_output,
                )
            except AssertionError as e:
                print("Error during checking:", e)
        
        return final_output

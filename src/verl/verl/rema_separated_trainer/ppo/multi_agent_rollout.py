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
import hashlib


def curriculum_context_is_visible(
    uid: object,
    step: int,
    probability: float,
    *,
    salt: str,
) -> bool:
    """Sample context dropout deterministically for an entire GRPO group."""
    probability = min(max(float(probability), 0.0), 1.0)
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    payload = f"{salt}:{step}:{uid}".encode("utf-8")
    sample = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
    return sample < probability

def normalize_text(text):
    return unicodedata.normalize('NFKC', text)


def _has_usable_final_boxed_answer(text: str) -> bool:
    """Return True when the final stage produced a boxed answer, not a refusal."""
    if not isinstance(text, str) or "\\boxed" not in text:
        return False

    normalized = " ".join(text.lower().split())
    missing_info_markers = (
        "not enough information",
        "insufficient information",
        "missing information",
        "missing notes",
        "cannot synthesize",
        "cannot proceed",
        "cannot provide",
        "please provide",
        "no final answer",
    )
    return not any(marker in normalized for marker in missing_info_markers)


def _parse_decomposer_decision(text: str, has_candidate: bool):
    """Parse the small ACCEPT/REVISE protocol, defaulting safely to revision."""
    match = re.search(
        r"(?im)^\s*DECISION\s*:\s*(ACCEPT|REVISE)\b",
        text if isinstance(text, str) else "",
    )
    requested = match.group(1).upper() if match else "REVISE"
    format_valid = match is not None
    forced_revise = requested == "ACCEPT" and not has_candidate
    effective = "REVISE" if forced_revise else requested
    return requested, effective, format_valid, forced_revise


def _extract_local_result_values(text: str) -> List[str]:
    if not isinstance(text, str):
        return []

    return [
        match.strip()
        for match in re.findall(r"(?im)^\s*LOCAL[_ ]RESULT\s*:\s*(.+?)\s*$", text)
        if match.strip()
    ]


def _final_uses_worker_local_result(final_output: str, completed_results: List[Tuple[str, str, str, str]]) -> bool:
    if not isinstance(final_output, str) or not final_output.strip():
        return False

    normalized_final = " ".join(final_output.lower().split())
    for _, _, _, worker_output in completed_results:
        for local_result in _extract_local_result_values(worker_output):
            normalized_result = " ".join(local_result.lower().split())
            if normalized_result and normalized_result in normalized_final:
                return True
    return False

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


def _select_group_generation_sources(
    sample_indices,
    group_ids,
    coupled_group_ids,
):
    """Choose one generation source for each coupled rollout group."""

    coupled_group_ids = set(coupled_group_ids)
    anchor_by_group = {}
    generation_indices = []
    source_by_sample = {}
    for raw_idx in sample_indices:
        idx = int(raw_idx)
        group_id = group_ids[idx]
        if group_id in coupled_group_ids:
            source_idx = anchor_by_group.get(group_id)
            if source_idx is None:
                source_idx = idx
                anchor_by_group[group_id] = idx
                generation_indices.append(idx)
        else:
            source_idx = idx
            generation_indices.append(idx)
        source_by_sample[idx] = source_idx
    return generation_indices, source_by_sample


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


def _encode_latest_conversation(
    conversation: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    stop_reason: Optional[str],
    turn_idx: int,
    max_prompt_length: int,
    max_total_length: int,
):
    """Encode one exact hierarchical prompt/action pair.

    Hierarchical stages are generated from independent role-specific prompts,
    so concatenating several rounds into one chat changes the conditioning
    distribution. Keep the selected round as one factual PPO action instead.
    """

    ignore_index = -100
    assistant_indices = [
        idx for idx, message in enumerate(conversation)
        if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("Latest hierarchical conversation has no assistant response")

    assistant_idx = assistant_indices[-1]
    prompt_messages = conversation[:assistant_idx]
    response = conversation[assistant_idx].get("content", "")
    query = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    full_query_ids = tokenizer.encode(query, add_special_tokens=True)
    query_response_ids = tokenizer.encode(
        query + response,
        add_special_tokens=True,
    )
    response_ids = query_response_ids[len(full_query_ids):]

    prompt_was_truncated = len(full_query_ids) > max_prompt_length
    if prompt_was_truncated:
        if tokenizer.truncation_side == "left":
            query_ids = full_query_ids[-max_prompt_length:]
        else:
            query_ids = full_query_ids[:max_prompt_length]
    else:
        query_ids = full_query_ids

    max_response_length = max(max_total_length - len(query_ids), 0)
    response_was_truncated = len(response_ids) > max_response_length
    response_ids = response_ids[:max_response_length]
    input_ids = query_ids + response_ids
    encoded_stop_reason = (
        "length"
        if response_was_truncated and stop_reason == "stop"
        else stop_reason
    )

    prefix_ignore = [ignore_index] * max(len(query_ids) - 1, 0)
    if encoded_stop_reason == "stop":
        labels = prefix_ignore + response_ids + [tokenizer.eos_token_id]
        step_ids = (
            [ignore_index] * len(prefix_ignore)
            + [turn_idx] * (len(response_ids) + 1)
        )
        encoded_num_gen_tokens = len(response_ids) + 1
    elif encoded_stop_reason == "length":
        labels = prefix_ignore + response_ids + [ignore_index]
        step_ids = (
            [ignore_index] * len(prefix_ignore)
            + [turn_idx] * len(response_ids)
            + [ignore_index]
        )
        encoded_num_gen_tokens = len(response_ids)
    elif encoded_stop_reason in {"stop_when_truncated", "completion_token_exceeded"}:
        labels = [ignore_index] * len(input_ids)
        step_ids = [ignore_index] * len(input_ids)
        encoded_num_gen_tokens = 0
    else:
        raise ValueError(f"Unknown hierarchical stop reason: {encoded_stop_reason}")

    assert len(input_ids) == len(labels), f"{len(input_ids)} != {len(labels)}"
    return (
        input_ids,
        labels,
        step_ids,
        encoded_num_gen_tokens,
        encoded_stop_reason,
        prompt_was_truncated,
        response_was_truncated,
    )


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
        max_new_tokens: Optional[int] = None,
    ):
        prompt_proto = self._prepare_chat_prompts(role, chat_lst, tokenizers)
        prompt_proto.meta_info.update(meta_info)
        if max_new_tokens is not None:
            prompt_proto.meta_info["max_new_tokens"] = max_new_tokens
        return self._generate_role_responses(
            rollout=self.rollout_wg_dict[role],
            prompt_proto=prompt_proto,
            tokenizer=tokenizers[role],
            response_length=response_length,
        )

    def _generate_from_hierarchical_chat_map(
        self,
        role: str,
        sample_indices,
        chat_by_idx,
        tokenizers: Dict[str, PreTrainedTokenizer],
        meta_info: Dict,
        response_length: int,
        max_new_tokens: Optional[int],
        group_ids,
        coupled_group_ids,
    ):
        """Generate once per coupled group and copy the sampled action to peers."""

        generation_indices, source_by_sample = _select_group_generation_sources(
            sample_indices,
            group_ids,
            coupled_group_ids,
        )
        if not generation_indices:
            return {}

        generation_chats = [chat_by_idx[idx] for idx in generation_indices]
        outputs, tokens, stops, _, output_token_ids = self._generate_from_chat_list(
            role,
            generation_chats,
            tokenizers,
            meta_info,
            response_length,
            max_new_tokens=max_new_tokens,
        )
        generated = {
            idx: (
                outputs[local_idx],
                tokens[local_idx],
                stops[local_idx],
                list(output_token_ids[local_idx]),
            )
            for local_idx, idx in enumerate(generation_indices)
        }
        records = {}
        for raw_idx in sample_indices:
            idx = int(raw_idx)
            output, num_tokens, stop_reason, token_ids = generated[
                source_by_sample[idx]
            ]
            records[idx] = (
                output,
                num_tokens,
                stop_reason,
                list(token_ids),
            )
        return records

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
        response_ids = unpad_output.batch["input_ids"][:, -response_length:]
        response_mask = unpad_output.batch["attention_mask"][:, -response_length:].bool()
        response_token_ids = [
            token_row[mask_row].tolist()
            for token_row, mask_row in zip(response_ids, response_mask)
        ]
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
        return vllm_output_text, num_gen_tokens, stop_reasons, resp_lens, response_token_ids

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
        response_token_ids: List[List[int]],
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
                "token_ids": response_token_ids[i],
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
                            "token_ids":
                            [],
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
                current_outputs, num_gen_tokens, stop_reasons, resp_lens, response_token_ids = (
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
                    response_token_ids,
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
    def _extract_subtasks(
        plan_text: str,
        max_subtasks: Optional[int] = None,
    ) -> List[Tuple[str, str]]:
        subtasks = []
        seen_subtasks = set()
        for line in plan_text.splitlines():
            match = re.match(
                r"\s*(?:[-*]\s*)?(?:\d+\.\s*)?(?:\*\*)?\s*(S\d+)\s*(?:\*\*)?\s*[:.)-]\s*(?:\*\*)?\s*(.+?)\s*$",
                line,
                re.IGNORECASE,
            )
            if match:
                subtask_id = match.group(1).upper()
                if subtask_id in seen_subtasks:
                    continue
                subtasks.append((subtask_id, match.group(2).strip()))
                seen_subtasks.add(subtask_id)
                if max_subtasks is not None and len(subtasks) >= max_subtasks:
                    break
        if subtasks:
            return subtasks

        # The fixed derive/verify protocol asks the planner for natural
        # strategy and checking guidance. Normalize those sections into the
        # same internal S1/S2 representation used by routing and replay.
        section_matches = list(re.finditer(
            r"(?im)^\s*(STRATEGY|CHECKS)\s*:\s*",
            plan_text,
        ))
        section_to_subtask = {
            "STRATEGY": "S1",
            "CHECKS": "S2",
        }
        for section_idx, match in enumerate(section_matches):
            section_name = match.group(1).upper()
            content_start = match.end()
            content_end = (
                section_matches[section_idx + 1].start()
                if section_idx + 1 < len(section_matches)
                else len(plan_text)
            )
            description = plan_text[content_start:content_end].strip()
            if not description:
                continue
            subtasks.append((section_to_subtask[section_name], description))
            if max_subtasks is not None and len(subtasks) >= max_subtasks:
                break
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
    def _build_derive_verify_stages(
        subtasks: List[Tuple[str, str]],
        stage_roles: List[str],
        default_worker: str,
    ) -> Tuple[
        List[Tuple[str, str]],
        List[Tuple[str, str, List[Tuple[str, str]]]],
    ]:
        """Normalize a plan into fixed derive, verify, and final stages."""
        if len(stage_roles) != 3:
            raise ValueError(
                "routing_mode=derive_verify requires exactly three worker stages "
                "(derive, verify/repair, final)"
            )

        descriptions = [
            description.strip()
            for _, description in subtasks[:2]
            if description.strip()
        ]
        derive_description = (
            descriptions[0]
            if descriptions
            else (
                "Derive the main calculation or a candidate solution from the "
                "reference problem, preserving the intermediate facts needed for verification."
            )
        )
        verify_description = (
            descriptions[1]
            if len(descriptions) > 1
            else (
                "Use the S1 result to verify the candidate against every stated "
                "constraint, repair any error or omitted case, and state the corrected result."
            )
        )
        normalized_subtasks = [
            ("S1", derive_description),
            ("S2", verify_description),
        ]
        ordered_stages = [
            (stage_roles[0], default_worker, [normalized_subtasks[0]]),
            (stage_roles[1], default_worker, [normalized_subtasks[1]]),
            (stage_roles[2], default_worker, []),
        ]
        return normalized_subtasks, ordered_stages

    @staticmethod
    def _build_sequential_plan_stages(
        subtasks: List[Tuple[str, str]],
        stage_roles: List[str],
        default_worker: str,
    ) -> Tuple[
        List[Tuple[str, str]],
        List[Tuple[str, str, List[Tuple[str, str]]]],
    ]:
        """Map each planned subtask to one worker and keep the final stage fixed."""
        if len(stage_roles) < 2:
            raise ValueError(
                "routing_mode=sequential_plan requires at least one worker "
                "stage followed by one final stage"
            )

        worker_stage_capacity = len(stage_roles) - 1
        descriptions = [
            description.strip()
            for _, description in subtasks[:worker_stage_capacity]
            if description.strip()
        ]
        if not descriptions:
            descriptions = [
                "Solve the reference problem and return the self-contained result "
                "needed to synthesize the final answer."
            ]

        normalized_subtasks = [
            (f"S{subtask_idx + 1}", description)
            for subtask_idx, description in enumerate(descriptions)
        ]
        ordered_stages = [
            (stage_roles[subtask_idx], default_worker, [subtask])
            for subtask_idx, subtask in enumerate(normalized_subtasks)
        ]
        ordered_stages.append((stage_roles[-1], default_worker, []))
        return normalized_subtasks, ordered_stages

    @staticmethod
    def _format_deterministic_assignments(
        default_worker: str,
        subtasks: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        assignment_lines = [
            f"- {subtask_id} -> {default_worker}"
            for subtask_id, _ in (subtasks or [("S1", ""), ("S2", "")])
        ]
        return "\n".join([
            "ASSIGNMENTS:",
            *assignment_lines,
            f"- FINAL -> {default_worker}",
        ])

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

    @classmethod
    def _format_previous_local_results(
        cls,
        completed_results: List[Tuple[str, str, str, str]],
    ) -> str:
        sections = []
        for _, _, subtask_ids, output in completed_results:
            local_result = cls._extract_local_result(output)
            if not local_result:
                continue
            label = subtask_ids or "previous subtask"
            sections.append(f"{label} LOCAL_RESULT: {local_result}")
        if not sections:
            return ""
        return "PREVIOUS LOCAL RESULTS:\n" + "\n".join(sections)

    @staticmethod
    def _format_worker_results_for_final(completed_results: List[Tuple[str, str, str, str]]) -> str:
        sections = [
            f"{stage_role} as {worker_type} ({subtask_ids}):\n{output.strip()}"
            for stage_role, worker_type, subtask_ids, output in completed_results
            if output and output.strip()
        ]
        if not sections:
            return ""
        return "WORKER RESULTS:\n" + "\n\n".join(sections)

    @staticmethod
    def _format_final_question_block(question: str, final_context_mode: str) -> str:
        modes_without_question = {
            "notes_only",
            "notes",
            "no_question",
            "worker_results_only",
            "workers_only",
            "local_results_only",
            "plan_and_worker_results",
            "parsed_plan_and_worker_results",
        }
        if str(final_context_mode).lower() in modes_without_question:
            return ""
        return f"Question:\n{question}\n\n"

    def _format_plan_and_worker_results_for_final(
        self,
        subtasks: List[Tuple[str, str]],
        completed_results: List[Tuple[str, str, str, str]],
    ) -> str:
        sections = []
        parsed_plan = self._format_subtasks(subtasks)
        if parsed_plan:
            sections.append(f"PARSED PLAN:\n{parsed_plan}")
        worker_results = self._format_worker_results_for_final(completed_results)
        if worker_results:
            sections.append(worker_results)
        return "\n\n".join(sections)

    @staticmethod
    def _format_final_notes(decomposer_output: str, worker_outputs: str) -> str:
        note_parts = [
            part.strip()
            for part in (decomposer_output, worker_outputs)
            if isinstance(part, str) and part.strip()
        ]
        if not note_parts:
            return ""
        return (
            "Here are notes from earlier attempts. They may contain useful strategy, "
            "partial calculations, or mistakes. Use them critically.\n\n"
            + "\n\n".join(note_parts)
        )

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
            result = match.group(1).strip()
        else:
            result = ""
            boxed_starts = list(re.finditer(r"\\boxed\s*\{", output))
            for boxed_start in reversed(boxed_starts):
                brace_start = output.find("{", boxed_start.start())
                depth = 0
                for char_idx in range(brace_start, len(output)):
                    if output[char_idx] == "{":
                        depth += 1
                    elif output[char_idx] == "}":
                        depth -= 1
                        if depth == 0:
                            result = output[boxed_start.start():char_idx + 1]
                            break
                if result:
                    break
            if not result:
                lines = [line.strip() for line in output.splitlines() if line.strip()]
                result = lines[-1] if lines else ""
        result = " ".join(result.split())
        return result[:600]

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

    @staticmethod
    def _compact_feedback_text(text: str, max_chars: int) -> str:
        if not text or max_chars <= 0:
            return ""
        text = text.strip()
        if len(text) <= max_chars:
            return text
        head_chars = max(int(max_chars * 0.65), 1)
        tail_chars = max(max_chars - head_chars, 1)
        return (
            text[:head_chars].rstrip()
            + "\n...[middle omitted]...\n"
            + text[-tail_chars:].lstrip()
        )

    def _format_hierarchical_feedback(
        self,
        plan: str,
        assignments: str,
        worker_results: Dict[str, str],
        last_worker_output: str,
        worker_roles: List[str],
        worker_reasoning_max_chars: int = 2200,
        final_reasoning_max_chars: int = 1400,
    ) -> str:
        sections = []
        if plan and plan.strip():
            sections.append(f"PREVIOUS PLAN:\n{plan.strip()}")
        if assignments and assignments.strip():
            sections.append(f"PREVIOUS ASSIGNMENTS:\n{assignments.strip()}")

        worker_sections = []
        nonfinal_worker_roles = worker_roles[:-1] if len(worker_roles) > 1 else worker_roles
        for worker_role in nonfinal_worker_roles:
            output = worker_results.get(worker_role, "")
            if output and output.strip():
                reasoning = self._compact_feedback_text(
                    self._extract_reasoning(output),
                    worker_reasoning_max_chars,
                )
                local_result = self._extract_local_result(output)
                worker_parts = [f"{worker_role}:"]
                if reasoning:
                    worker_parts.append(f"REASONING:\n{reasoning}")
                if local_result:
                    worker_parts.append(f"LOCAL_RESULT:\n{local_result}")
                worker_sections.append("\n".join(worker_parts))
        if worker_sections:
            sections.append("PREVIOUS WORKER RESULTS:\n" + "\n\n".join(worker_sections))

        if last_worker_output and last_worker_output.strip():
            final_output = last_worker_output.strip()
            boxed_matches = re.findall(
                r"\\boxed\s*\{(?:[^{}]|\{[^{}]*\})*\}",
                final_output,
                re.DOTALL,
            )
            if boxed_matches:
                final_answer = boxed_matches[-1]
                answer_start = final_output.rfind(final_answer)
                final_diagnosis = (
                    final_output[:answer_start] + final_output[answer_start + len(final_answer):]
                ).strip()
            else:
                final_answer = ""
                final_diagnosis = final_output
            final_diagnosis = self._compact_feedback_text(
                final_diagnosis,
                final_reasoning_max_chars,
            )
            if final_diagnosis:
                sections.append(
                    f"PREVIOUS FINAL DIAGNOSIS:\n{final_diagnosis}"
                )
            if final_answer:
                sections.append(f"PREVIOUS FINAL ANSWER:\n{final_answer}")

        if not sections:
            return ""
        return (
            "FEEDBACK FROM THE PREVIOUS ROUND:\n"
            "Use this to decide what to keep, repair, or replace in the next plan.\n\n"
            + "\n\n".join(sections)
        )

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
        max_planned_subtasks = int(
            hierarchy_config.get(
                "max_planned_subtasks",
                max(len(stage_roles) - 1, 1),
            )
        )
        default_worker = hierarchy_config.get("default_worker", worker_types[-1] if worker_types else selector_role)
        worker_specs = hierarchy_config.get("worker_specs", {})
        pass_question_to_workers = hierarchy_config.get("pass_question_to_workers", False)
        worker_context_mode = hierarchy_config.get(
            "worker_context_mode",
            "full_question" if pass_question_to_workers else "subtask_context",
        )
        pass_question_to_workers = worker_context_mode in {
            "full_question",
            "question",
            "full",
        }
        final_context_mode = hierarchy_config.get("final_context_mode", "full_question")
        routing_mode = str(hierarchy_config.get("routing_mode", "selector")).lower()
        deterministic_routing = routing_mode in {
            "derive_verify",
            "sequential_plan",
        }
        decomposer_max_new_tokens = hierarchy_config.get("decomposer_max_new_tokens")
        selector_max_new_tokens = hierarchy_config.get("selector_max_new_tokens")
        worker_max_new_tokens = hierarchy_config.get("worker_max_new_tokens")
        final_max_new_tokens = hierarchy_config.get(
            "final_max_new_tokens",
            worker_max_new_tokens,
        )
        feedback_worker_reasoning_max_chars = int(
            hierarchy_config.get("feedback_worker_reasoning_max_chars", 2200)
        )
        feedback_final_reasoning_max_chars = int(
            hierarchy_config.get("feedback_final_reasoning_max_chars", 1400)
        )
        accept_revise_config = hierarchy_config.get("accept_revise", {})
        accept_revise_enabled = bool(accept_revise_config.get("enable", False))
        scoped_c3_config = hierarchy_config.get("scoped_c3_grpo", {})
        c3_focal_role = prompts.meta_info.get("c3_focal_role")
        c3_group_ids = list(
            prompts.non_tensor_batch.get(
                "uid",
                np.arange(batch_size, dtype=object),
            )
        )
        curriculum_step = int(prompts.meta_info.get("curriculum_step", 0))
        is_validation = bool(prompts.meta_info.get("validate", False))
        agent12_curriculum = hierarchy_config.get("agent12_curriculum", {}) or {}
        teacher_solution_key = str(
            prompts.meta_info.get("teacher_solution_key", "teacher_solution")
        )
        teacher_solution_probability = (
            0.0
            if is_validation
            else float(
                prompts.meta_info.get("teacher_solution_probability", 0.0)
            )
        )
        teacher_solutions = prompts.non_tensor_batch.get(
            teacher_solution_key,
            np.asarray([""] * batch_size, dtype=object),
        )
        teacher_solution_max_chars = max(
            int(agent12_curriculum.get("teacher_solution_max_chars", 8000)),
            1,
        )
        teacher_solution_visible = np.asarray(
            [
                bool(str(teacher_solutions[idx]).strip())
                and curriculum_context_is_visible(
                    c3_group_ids[idx],
                    curriculum_step,
                    teacher_solution_probability,
                    salt="teacher_solution",
                )
                for idx in range(batch_size)
            ],
            dtype=bool,
        )
        worker_question_probability = (
            float(
                agent12_curriculum.get(
                    "worker_question_eval_probability",
                    1.0,
                )
            )
            if is_validation and agent12_curriculum.get("enable", False)
            else float(
                prompts.meta_info.get("worker_question_probability", 1.0)
            )
        )
        worker_question_visible = np.asarray(
            [
                pass_question_to_workers
                and curriculum_context_is_visible(
                    c3_group_ids[idx],
                    curriculum_step,
                    worker_question_probability,
                    salt="worker_question",
                )
                for idx in range(batch_size)
            ],
            dtype=bool,
        )
        prompts.non_tensor_batch["teacher_solution_visible"] = (
            teacher_solution_visible
        )
        prompts.non_tensor_batch["worker_question_visible"] = (
            worker_question_visible
        )
        c3_active = bool(
            scoped_c3_config.get("enable", False)
            and not prompts.meta_info.get("validate", False)
            and c3_focal_role in agent_roles
            and "uid" in prompts.non_tensor_batch
        )
        configured_branch_turn = scoped_c3_config.get("branch_turn", "latest")
        if configured_branch_turn == "latest":
            c3_branch_turn = max(max_num_turns - 1, 0)
        else:
            c3_branch_turn = int(configured_branch_turn)
            if not 0 <= c3_branch_turn < max_num_turns:
                raise ValueError(
                    f"scoped C3 branch_turn={c3_branch_turn} must be in "
                    f"[0, {max_num_turns})"
                )
        if c3_active:
            print(
                "scoped C3 rollout: "
                f"focal_role={c3_focal_role}, "
                f"branch_turn={c3_branch_turn + 1}/{max_num_turns}"
            )
        worker_spec_text = self._format_worker_specs(worker_specs, worker_types)

        conversation_history = {
            role: [None for _ in range(batch_size)]
            for role in agent_roles
        }
        previous_feedback = [None for _ in range(batch_size)]
        latest_outputs = ["" for _ in range(batch_size)]
        c3_action_records = [None for _ in range(batch_size)]
        candidate_outputs = ["" for _ in range(batch_size)]
        candidate_source_turn = [-1 for _ in range(batch_size)]
        accepted = [False for _ in range(batch_size)]
        decision_valid = [True for _ in range(batch_size)]
        round_attempted = [
            [False for _ in range(max_num_turns)]
            for _ in range(batch_size)
        ]

        def append_history(
            idx,
            role,
            content,
            num_gen_tokens,
            stop_reason,
            token_ids,
            **metadata,
        ):
            message = {
                "role": role,
                "content": content,
                "num_gen_tokens": num_gen_tokens,
                "stop_reason": stop_reason,
                "token_ids": token_ids,
                "executed": metadata.pop("executed", True),
            }
            message.update(metadata)
            history[idx].append(message)

        def build_prompt(role, idx, content):
            return [
                {"role": "system", "content": system_prompts[role]},
                {"role": "user", "content": content},
            ]

        def build_selected_worker_prompt(stage_role, worker_type, idx, content, system_prompt_override=None):
            system_prompt = system_prompt_override or system_prompts.get(worker_type, system_prompts[stage_role])
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]

        def record_prompt_and_output(
            idx,
            role,
            chat,
            output,
            num_gen_tokens,
            stop_reason,
            token_ids,
            **metadata,
        ):
            conversation_history[role][idx] = chat
            append_history(
                idx,
                role,
                output,
                num_gen_tokens,
                stop_reason,
                token_ids,
                **metadata,
            )

        def record_c3_branch_action(
            idx,
            role,
            chat,
            output,
            num_gen_tokens,
            stop_reason,
            token_ids,
            *,
            assigned_subtasks=None,
            plan_subtasks=None,
        ):
            if not (
                c3_active
                and i_turn == c3_branch_turn
                and role == c3_focal_role
            ):
                return
            c3_action_records[idx] = {
                "role": role,
                "turn_idx": i_turn,
                "chat": [dict(message) for message in chat],
                "output": output,
                "num_gen_tokens": num_gen_tokens,
                "stop_reason": stop_reason,
                "token_ids": list(token_ids),
                "assigned_subtasks": list(assigned_subtasks or []),
                "plan_subtasks": list(plan_subtasks or []),
            }

        for i_turn in range(max_num_turns):
            unfinished_indices = np.where(~finish_flags)[0]
            print(f"hierarchical turn {i_turn+1} of {max_num_turns}, "
                  f"{len(unfinished_indices)}/{batch_size} unfinished")
            if len(unfinished_indices) == 0:
                break

            branch_open_group_ids = set()

            def groups_for_indices(indices):
                return {
                    c3_group_ids[int(idx)]
                    for idx in indices
                }

            def coupled_groups_for(role, indices):
                if not c3_active:
                    return set()
                present_groups = groups_for_indices(indices)
                if i_turn < c3_branch_turn:
                    return present_groups
                if i_turn > c3_branch_turn:
                    return set()
                if role == c3_focal_role:
                    return set()
                return present_groups - branch_open_group_ids

            def mark_branch_open(role, indices):
                if (
                    c3_active
                    and i_turn == c3_branch_turn
                    and role == c3_focal_role
                ):
                    branch_open_group_ids.update(groups_for_indices(indices))

            # 1. Decompose or revise the plan using previous round feedback.
            decomposer_chats_by_idx = {}
            for idx in unfinished_indices:
                content = f"Question:\n{questions[idx]}"
                if teacher_solution_visible[idx]:
                    teacher_scaffold = str(teacher_solutions[idx])[
                        :teacher_solution_max_chars
                    ]
                    content += (
                        "\n\nCorrect teacher solution (planning scaffold):\n"
                        f"{teacher_scaffold}\n\n"
                        "Use this solution to identify a useful sequence of "
                        "dependent subtasks. Produce a plan that remains usable "
                        "when the solution is absent; do not copy its final answer "
                        "into the plan."
                    )
                if previous_feedback[idx]:
                    content += f"\n\n{previous_feedback[idx]}"
                if accept_revise_enabled:
                    if candidate_source_turn[idx] >= 0:
                        content += (
                            "\n\nReview the previous final answer. Begin with "
                            "DECISION: ACCEPT to keep it, or DECISION: REVISE "
                            "to run a repaired plan."
                        )
                    else:
                        content += (
                            "\n\nNo previous final answer exists. Begin with "
                            "DECISION: REVISE."
                        )
                decomposer_chats_by_idx[idx] = build_prompt(
                    decomposer_role,
                    idx,
                    content,
                )
            decomposer_records = self._generate_from_hierarchical_chat_map(
                decomposer_role,
                unfinished_indices,
                decomposer_chats_by_idx,
                tokenizers,
                prompts.meta_info,
                response_length,
                max_new_tokens=decomposer_max_new_tokens,
                group_ids=c3_group_ids,
                coupled_group_ids=coupled_groups_for(
                    decomposer_role,
                    unfinished_indices,
                ),
            )
            mark_branch_open(decomposer_role, unfinished_indices)
            current_plan = {}
            for idx in unfinished_indices:
                output, num_tokens, stop_reason, token_ids = decomposer_records[idx]
                current_plan[idx] = output
                record_prompt_and_output(
                    idx,
                    decomposer_role,
                    decomposer_chats_by_idx[idx],
                    output,
                    num_tokens,
                    stop_reason,
                    token_ids,
                )
                record_c3_branch_action(
                    idx,
                    decomposer_role,
                    decomposer_chats_by_idx[idx],
                    output,
                    num_tokens,
                    stop_reason,
                    token_ids,
                )

            revise_indices = list(unfinished_indices)
            if accept_revise_enabled:
                revise_indices = []
                for idx in unfinished_indices:
                    _requested, effective, valid, forced = _parse_decomposer_decision(
                        current_plan[idx],
                        has_candidate=candidate_source_turn[idx] >= 0,
                    )
                    decision_valid[idx] = valid and not forced
                    if effective == "REVISE":
                        revise_indices.append(idx)
                        continue

                    accepted[idx] = True
                    latest_outputs[idx] = candidate_outputs[idx]
                    finish_flags[idx] = True
                    finish_reason[idx] = "decomposer_accept"

                    selector_chat = build_prompt(
                        selector_role,
                        idx,
                        "The decomposer accepted the previous final answer.",
                    )
                    record_prompt_and_output(
                        idx,
                        selector_role,
                        selector_chat,
                        "",
                        0,
                        "stop",
                        [],
                        executed=False,
                        skipped_due_to_accept=True,
                    )
                    for stage_role in stage_roles:
                        is_final_role = stage_role == stage_roles[-1]
                        carried_output = candidate_outputs[idx] if is_final_role else ""
                        stage_chat = build_prompt(
                            stage_role,
                            idx,
                            "The decomposer accepted the previous final answer.",
                        )
                        record_prompt_and_output(
                            idx,
                            stage_role,
                            stage_chat,
                            carried_output,
                            0,
                            "stop",
                            [],
                            executed=False,
                            skipped_due_to_accept=True,
                            carried_forward=is_final_role,
                            candidate_source_turn=candidate_source_turn[idx],
                        )

                if not revise_indices:
                    continue

            # 2. Parse the plan and either route it deterministically or ask the selector.
            selector_chats_by_idx = {}
            parsed_subtasks = {}
            ordered_stages_by_idx = {}
            selector_output_by_idx = {}
            for idx in revise_indices:
                subtasks = self._extract_subtasks(
                    current_plan[idx],
                    max_subtasks=max_planned_subtasks,
                )
                if deterministic_routing:
                    if routing_mode == "derive_verify":
                        subtasks, ordered_stages = self._build_derive_verify_stages(
                            subtasks,
                            stage_roles,
                            default_worker,
                        )
                    else:
                        subtasks, ordered_stages = self._build_sequential_plan_stages(
                            subtasks,
                            stage_roles,
                            default_worker,
                        )
                    ordered_stages_by_idx[idx] = ordered_stages
                parsed_subtasks[idx] = subtasks
                selector_chats_by_idx[idx] = build_prompt(
                    selector_role,
                    idx,
                    (
                        f"Question:\n{questions[idx]}\n\n"
                        f"Plan:\n{current_plan[idx]}\n\n"
                        f"Available workers:\n{worker_spec_text}"
                    ),
                )
            if deterministic_routing:
                for idx in revise_indices:
                    deterministic_assignments = self._format_deterministic_assignments(
                        default_worker,
                        parsed_subtasks[idx],
                    )
                    selector_output_by_idx[idx] = deterministic_assignments
                    record_prompt_and_output(
                        idx,
                        selector_role,
                        selector_chats_by_idx[idx],
                        deterministic_assignments,
                        0,
                        "stop",
                        [],
                        executed=False,
                        deterministic_routing=True,
                    )
            else:
                selector_records = self._generate_from_hierarchical_chat_map(
                    selector_role,
                    revise_indices,
                    selector_chats_by_idx,
                    tokenizers,
                    prompts.meta_info,
                    response_length,
                    max_new_tokens=selector_max_new_tokens,
                    group_ids=c3_group_ids,
                    coupled_group_ids=coupled_groups_for(
                        selector_role,
                        revise_indices,
                    ),
                )
                mark_branch_open(selector_role, revise_indices)
                for idx in revise_indices:
                    output, num_tokens, stop_reason, token_ids = selector_records[idx]
                    selector_output_by_idx[idx] = output
                    record_prompt_and_output(
                        idx,
                        selector_role,
                        selector_chats_by_idx[idx],
                        output,
                        num_tokens,
                        stop_reason,
                        token_ids,
                    )
                    record_c3_branch_action(
                        idx,
                        selector_role,
                        selector_chats_by_idx[idx],
                        output,
                        num_tokens,
                        stop_reason,
                        token_ids,
                    )
                    ordered_stages_by_idx[idx] = self._parse_ordered_worker_stages(
                        output,
                        parsed_subtasks[idx],
                        stage_roles,
                        worker_types,
                        default_worker,
                    )
                    if stage_roles:
                        final_stage_role = stage_roles[-1]
                        if all(
                            stage_role != final_stage_role
                            for stage_role, _, _ in ordered_stages_by_idx[idx]
                        ):
                            ordered_stages_by_idx[idx].append(
                                (final_stage_role, default_worker, [])
                            )

            # 3. Execute selected worker stages sequentially. Stage roles encode
            # the order; each later worker sees previous results.
            worker_results = {idx: {role: "" for role in stage_roles} for idx in revise_indices}
            worker_records = {
                idx: {
                    role: None
                    for role in stage_roles
                }
                for idx in revise_indices
            }
            completed_results_by_idx = {idx: [] for idx in revise_indices}
            max_stage_count = max(
                [len(ordered_stages_by_idx[idx]) for idx in revise_indices],
                default=0,
            )
            for stage_idx in range(max_stage_count):
                for stage_role in stage_roles:
                    stage_indices = [
                        idx for idx in revise_indices
                        if (
                            stage_idx < len(ordered_stages_by_idx[idx])
                            and ordered_stages_by_idx[idx][stage_idx][0] == stage_role
                        )
                    ]
                    if not stage_indices:
                        continue

                    worker_chats_by_idx = {}
                    stage_subtasks_by_idx = {}
                    worker_type_by_idx = {}
                    for idx in stage_indices:
                        _, worker_type, assigned_subtasks = ordered_stages_by_idx[idx][stage_idx]
                        stage_subtasks_by_idx[idx] = assigned_subtasks
                        worker_type_by_idx[idx] = worker_type
                        is_final_stage = stage_idx == len(ordered_stages_by_idx[idx]) - 1
                        if is_final_stage:
                            question_block = self._format_final_question_block(
                                questions[idx],
                                final_context_mode,
                            )
                        elif worker_question_visible[idx]:
                            question_block = (
                                f"Reference problem:\n{questions[idx]}\n\n"
                                "Use the reference problem only to recover facts needed for the assigned subtask.\n\n"
                            )
                        else:
                            question_block = (
                                "The assigned subtask is your task context and should contain the needed facts. "
                                "Use previous LOCAL_RESULTs when they help.\n\n"
                            )
                        if is_final_stage:
                            if final_context_mode in {"worker_results_only", "workers_only", "local_results_only"}:
                                work_so_far = self._format_worker_results_for_final(
                                    completed_results_by_idx[idx]
                                )
                            elif final_context_mode in {
                                "plan_and_worker_results",
                                "parsed_plan_and_worker_results",
                            }:
                                work_so_far = self._format_plan_and_worker_results_for_final(
                                    parsed_subtasks[idx],
                                    completed_results_by_idx[idx],
                                )
                            else:
                                work_so_far = self._format_final_notes(
                                    current_plan.get(idx, ""),
                                    self._format_work_so_far(completed_results_by_idx[idx]),
                                )
                        else:
                            work_so_far = self._format_previous_local_results(
                                completed_results_by_idx[idx]
                            )
                        assigned_subtasks_text = self._format_subtasks(assigned_subtasks)
                        if is_final_stage:
                            stage_instruction = (
                                "Synthesize the final answer from the plan and worker results. "
                                "Reconcile their conclusions and repair only local inconsistencies needed for synthesis. "
                                "End with the final answer in \\boxed{}."
                            )
                        else:
                            role_instruction = "Work on the assigned subtask above. "
                            if routing_mode == "derive_verify" and stage_idx == 1:
                                role_instruction = (
                                    "Start from the previous S1 LOCAL_RESULT. Verify it against "
                                    "the reference problem, repair any error or omitted case, "
                                    "and state the corrected result. "
                                )
                            stage_instruction = (
                                f"{role_instruction}"
                                "Reason step by step with concrete calculations, transformations, or checks. "
                                "Keep the reasoning scoped to this subtask and check conditions that directly affect its result. "
                                "Finish with one concise LOCAL_RESULT for this subtask. "
                                "Output exactly:\n"
                                "REASONING:\n"
                                "<step-by-step reasoning for this subtask>\n\n"
                                "LOCAL_RESULT: \\boxed{<useful result of this subtask>}"
                            )
                        if is_final_stage and not assigned_subtasks_text:
                            assigned_subtasks_text = (
                                "- Use the work above to synthesize the final answer."
                            )
                        work_so_far_block = f"{work_so_far}\n\n" if work_so_far else ""
                        dependency_instruction = (
                            "Use the PREVIOUS LOCAL RESULTS when the current task depends on them.\n\n"
                            if work_so_far and not is_final_stage else ""
                        )
                        chat = build_selected_worker_prompt(
                            stage_role,
                            worker_type,
                            idx,
                            (
                                f"{question_block}"
                                f"{work_so_far_block}"
                                f"{dependency_instruction}"
                                f"CURRENT TASK:\n{assigned_subtasks_text}\n\n"
                                f"{stage_instruction}\n\n"
                            ),
                            system_prompts.get("finalizer") if is_final_stage else None,
                        )
                        worker_chats_by_idx[idx] = chat
                    stage_records = self._generate_from_hierarchical_chat_map(
                        stage_role,
                        stage_indices,
                        worker_chats_by_idx,
                        tokenizers,
                        prompts.meta_info,
                        response_length,
                        max_new_tokens=(
                            final_max_new_tokens
                            if all(
                                stage_idx == len(ordered_stages_by_idx[idx]) - 1
                                for idx in stage_indices
                            )
                            else worker_max_new_tokens
                        ),
                        group_ids=c3_group_ids,
                        coupled_group_ids=coupled_groups_for(
                            stage_role,
                            stage_indices,
                        ),
                    )
                    mark_branch_open(stage_role, stage_indices)
                    for idx in stage_indices:
                        output, num_tokens, stop_reason, token_ids = stage_records[idx]
                        record_c3_branch_action(
                            idx,
                            stage_role,
                            worker_chats_by_idx[idx],
                            output,
                            num_tokens,
                            stop_reason,
                            token_ids,
                            assigned_subtasks=stage_subtasks_by_idx[idx],
                            plan_subtasks=parsed_subtasks[idx],
                        )
                        worker_results[idx][stage_role] = output
                        worker_records[idx][stage_role] = (
                            worker_chats_by_idx[idx],
                            output,
                            num_tokens,
                            stop_reason,
                            token_ids,
                            [subtask_id for subtask_id, _ in stage_subtasks_by_idx[idx]],
                        )
                        subtask_ids = ", ".join([subtask_id for subtask_id, _ in stage_subtasks_by_idx[idx]])
                        previous_completed_results = list(completed_results_by_idx[idx])
                        completed_results_by_idx[idx].append((stage_role, worker_type_by_idx[idx], subtask_ids, output))

                        latest_outputs[idx] = output
                        is_final_stage = stage_idx == len(ordered_stages_by_idx[idx]) - 1
                        if is_final_stage:
                            candidate_outputs[idx] = output
                            candidate_source_turn[idx] = i_turn
                            round_attempted[idx][i_turn] = True
                            if self.config.stop_when_truncated and stop_reason == "length":
                                finish_flags[idx] = True
                                finish_reason[idx] = "stop_when_truncated"
                            elif (
                                not accept_revise_enabled
                                and
                                (not c3_active or i_turn >= c3_branch_turn)
                                and
                                _has_usable_final_boxed_answer(output)
                                and _final_uses_worker_local_result(output, previous_completed_results)
                            ):
                                finish_flags[idx] = True
                                finish_reason[idx] = "final_boxed_answer"

            # Keep exactly one history slot per role per hierarchical turn.
            for idx in revise_indices:
                for stage_role in stage_roles:
                    record = worker_records[idx].get(stage_role)
                    if record is None:
                        chat = build_prompt(stage_role, idx, "No subtasks were assigned to this worker stage.")
                        record_prompt_and_output(
                            idx,
                            stage_role,
                            chat,
                            "",
                            0,
                            "stop",
                            [],
                            executed=False,
                        )
                    else:
                        (
                            chat,
                            output,
                            num_gen_tokens,
                            stop_reason,
                            token_ids,
                            assigned_subtask_ids,
                        ) = record
                        record_prompt_and_output(
                            idx,
                            stage_role,
                            chat,
                            output,
                            num_gen_tokens,
                            stop_reason,
                            token_ids,
                        )
                        if history[idx] and history[idx][-1].get("role") == stage_role:
                            history[idx][-1]["assigned_subtasks"] = assigned_subtask_ids

            if i_turn + 1 < max_num_turns:
                for idx in revise_indices:
                    previous_feedback[idx] = self._format_hierarchical_feedback(
                        current_plan[idx], selector_output_by_idx[idx], worker_results[idx],
                        latest_outputs[idx], stage_roles,
                        worker_reasoning_max_chars=feedback_worker_reasoning_max_chars,
                        final_reasoning_max_chars=feedback_final_reasoning_max_chars,
                    )

        protocol_state = {
            "enabled": accept_revise_enabled,
            "accepted": accepted,
            "candidate_source_turn": candidate_source_turn,
            "decision_valid": decision_valid,
            "round_attempted": round_attempted,
        }
        return latest_outputs, conversation_history, c3_action_records, protocol_state

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
        latest_round_only: bool = False,
        c3_focal_role: Optional[str] = None,
        c3_action_records: Optional[List[Optional[Dict]]] = None,
        last_round_executed: Optional[List[Dict[str, bool]]] = None,
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
        encoded_num_gen_token_lst = {
            role: [] for role in conversation_history.keys()
        }
        encoded_stop_reason_lst = {
            role: [] for role in conversation_history.keys()
        }
        prompt_truncated_lst = {
            role: [] for role in conversation_history.keys()
        }
        response_truncated_lst = {
            role: [] for role in conversation_history.keys()
        }

        # build tensors for training
        for i_batch in range(len(last_round_responses)):
            for role in conversation_history.keys():
                if latest_round_only:
                    selected_record = None
                    if (
                        role == c3_focal_role
                        and c3_action_records is not None
                    ):
                        selected_record = c3_action_records[i_batch]
                    if selected_record is not None:
                        selected_turn_idx = int(selected_record["turn_idx"])
                        selected_stop_reason = selected_record["stop_reason"]
                        selected_conversation = [
                            dict(message)
                            for message in selected_record["chat"]
                        ] + [{
                            "role": "assistant",
                            "content": selected_record["output"],
                        }]
                    else:
                        role_num_gen_tokens = num_gen_token_lst[role][i_batch]
                        role_stop_reasons = stop_reason_lst[role][i_batch]
                        selected_turn_idx = max(len(role_num_gen_tokens) - 1, 0)
                        selected_stop_reason = (
                            role_stop_reasons[-1] if role_stop_reasons else "stop"
                        )
                        selected_conversation = conversation_history[role][i_batch]
                    (
                        input_ids,
                        labels,
                        step_ids,
                        encoded_num_gen_tokens,
                        encoded_stop_reason,
                        prompt_was_truncated,
                        response_was_truncated,
                    ) = _encode_latest_conversation(
                        selected_conversation,
                        tokenizers[role],
                        selected_stop_reason,
                        selected_turn_idx,
                        self.config.prompt_length,
                        self.config.response_length + self.config.prompt_length,
                    )
                    selected_was_executed = (
                        selected_record is not None
                        or last_round_executed is None
                        or bool(last_round_executed[i_batch].get(role, True))
                    )
                    if not selected_was_executed:
                        labels = [-100] * len(labels)
                        step_ids = [-100] * len(step_ids)
                        encoded_num_gen_tokens = 0
                    if selected_turn_idx >= max_num_turns:
                        raise ValueError(
                            f"Selected turn index {selected_turn_idx} exceeds "
                            f"configured max_num_turns={max_num_turns}"
                        )
                    sparse_num_gen_tokens = [0] * max_num_turns
                    sparse_stop_reasons = ["not_encoded"] * max_num_turns
                    sparse_num_gen_tokens[selected_turn_idx] = encoded_num_gen_tokens
                    sparse_stop_reasons[selected_turn_idx] = encoded_stop_reason
                else:
                    input_ids, labels, step_ids = _encode_conversation(
                        conversation_history[role][i_batch],
                        tokenizers[role],
                        num_gen_token_lst[role][i_batch],
                        stop_reason_lst[role][i_batch],
                    )
                    sparse_num_gen_tokens = list(
                        num_gen_token_lst[role][i_batch]
                    )
                    sparse_stop_reasons = list(
                        stop_reason_lst[role][i_batch]
                    )
                    prompt_was_truncated = False
                    response_was_truncated = False
                input_ids_lst[role].append(input_ids)
                labels_lst[role].append(labels)
                step_ids_lst[role].append(step_ids)
                encoded_num_gen_token_lst[role].append(sparse_num_gen_tokens)
                encoded_stop_reason_lst[role].append(sparse_stop_reasons)
                prompt_truncated_lst[role].append(prompt_was_truncated)
                response_truncated_lst[role].append(response_was_truncated)

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
            elif fr == "final_boxed_answer":
                finish_reason_array.append(4)
            elif fr == "decomposer_accept":
                finish_reason_array.append(5)
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
            for i, num_gen_tokens in enumerate(encoded_num_gen_token_lst[role]):
                padded_num_gen_tokens[i, :len(num_gen_tokens)] = torch.tensor(
                    num_gen_tokens, dtype=torch.long)
            padded_stop_reasons = torch.full((batch_size, max_num_turns),
                                             0,
                                             dtype=torch.bool)

            for i, stop_reasons in enumerate(encoded_stop_reason_lst[role]):
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
                    "prompt_truncated": torch.tensor(
                        prompt_truncated_lst[role],
                        dtype=torch.bool,
                    ),
                    "response_retokenized_truncated": torch.tensor(
                        response_truncated_lst[role],
                        dtype=torch.bool,
                    ),
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
        c3_focal_role: Optional[str] = None,
        c3_action_records: Optional[List[Optional[Dict]]] = None,
        protocol_state: Optional[Dict] = None,
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
        if protocol_state is not None:
            non_tensor_batch["accept_revise_enabled"] = np.array(
                [bool(protocol_state.get("enabled", False))] * len(history),
                dtype=bool,
            )
            non_tensor_batch["accepted"] = np.asarray(
                protocol_state.get("accepted", [False] * len(history)),
                dtype=bool,
            )
            non_tensor_batch["candidate_source_turn"] = np.asarray(
                protocol_state.get("candidate_source_turn", [-1] * len(history)),
                dtype=np.int64,
            )
            non_tensor_batch["accept_revise_decision_valid"] = np.asarray(
                protocol_state.get("decision_valid", [True] * len(history)),
                dtype=bool,
            )
            round_attempted = np.empty(len(history), dtype=object)
            round_attempted[:] = [
                list(values)
                for values in protocol_state.get("round_attempted", [])
            ]
            non_tensor_batch["round_attempted"] = round_attempted

        for role in agent_roles:
            role_action_token_ids = np.empty(len(history), dtype=object)
            selected_role_records = (
                c3_action_records
                if role == c3_focal_role and c3_action_records is not None
                else None
            )
            if selected_role_records is not None:
                role_action_token_ids[:] = [
                    list(record.get("token_ids", [])) if record else []
                    for record in selected_role_records
                ]
            else:
                role_action_token_ids[:] = [
                    next(
                        (
                            list(message.get("token_ids", []))
                            for message in reversed(sample_history)
                            if isinstance(message, dict) and message.get("role") == role
                        ),
                        [],
                    )
                    for sample_history in history
                ]
            non_tensor_batch[f"{role}_action_token_ids"] = role_action_token_ids

        if c3_focal_role and c3_action_records is not None:
            non_tensor_batch["c3_action_turn"] = np.array([
                int(record["turn_idx"]) if record else -1
                for record in c3_action_records
            ], dtype=np.int64)
            c3_assigned_subtasks = np.empty(len(c3_action_records), dtype=object)
            c3_assigned_subtasks[:] = [
                list(record.get("assigned_subtasks", [])) if record else []
                for record in c3_action_records
            ]
            non_tensor_batch["c3_action_assigned_subtasks"] = c3_assigned_subtasks
            c3_plan_subtasks = np.empty(len(c3_action_records), dtype=object)
            c3_plan_subtasks[:] = [
                list(record.get("plan_subtasks", [])) if record else []
                for record in c3_action_records
            ]
            non_tensor_batch["c3_action_plan_subtasks"] = c3_plan_subtasks
            non_tensor_batch["c3_action_stop_reason"] = np.array([
                record.get("stop_reason", "stop") if record else "stop"
                for record in c3_action_records
            ], dtype=object)

        # Keep raw sampled token ids out of the verbose public history. CPCR
        # receives them through the dedicated per-role arrays above.
        clean_history = [
            [
                {key: value for key, value in message.items() if key != "token_ids"}
                for message in sample_history
            ]
            for sample_history in history
        ]

        max_history_length = max(2 * self.config.max_num_turns,
                                 len(agent_roles) * self.config.max_num_turns)
        padded_history = _pad_history(clean_history, max_history_length)
        padded_conversation_history = {}
        for role in agent_roles:
            role_conversations = conversation_history[role]
            if (
                role == c3_focal_role
                and c3_action_records is not None
            ):
                role_conversations = [
                    [dict(message) for message in record["chat"]]
                    if record else conversation_history[role][sample_idx]
                    for sample_idx, record in enumerate(c3_action_records)
                ]
            padded_conversation_history[role] = _pad_history(
                role_conversations,
                max_history_length,
            )

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
        c3_focal_role = None
        c3_action_records = None
        protocol_state = None
        if hierarchy_config.get("enable", False):
            (
                latest_outputs,
                conversation_history,
                c3_action_records,
                protocol_state,
            ) = self._run_hierarchical_conversation(
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
            c3_focal_role = prompts.meta_info.get("c3_focal_role")
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
        last_round_executed = [{
            m["role"]: bool(m.get("executed", True))
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
            latest_round_only=hierarchy_config.get("enable", False),
            c3_focal_role=c3_focal_role,
            c3_action_records=c3_action_records,
            last_round_executed=last_round_executed,
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
            c3_focal_role=c3_focal_role,
            c3_action_records=c3_action_records,
            protocol_state=protocol_state,
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

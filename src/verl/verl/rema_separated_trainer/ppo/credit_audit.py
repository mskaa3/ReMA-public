"""Observational, action-level C3 trace. Never changes rewards or loss masks."""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class CreditAudit:
    def __init__(self, directory, console_examples=2):
        self.directory = Path(directory)
        self.console_examples = console_examples
        self.pending = {}

    def _write(self, step, events):
        self.directory.mkdir(parents=True, exist_ok=True)
        # Append: resumed sessions must not overwrite an earlier attempt.
        with (self.directory / f"train_step_{step}.jsonl").open("a") as stream:
            for event in events:
                stream.write(json.dumps(_json_value(event), allow_nan=False) + "\n")

    def capture(self, batch, estimate, *, step, role):
        groups = defaultdict(list)
        ids = np.empty(len(batch), dtype=object)
        fallback_actions = Counter()
        action_numbers = {}
        for i, uid in enumerate(batch.non_tensor_batch['uid']):
            actions = batch.non_tensor_batch.get('c3_action_index')
            action = int(actions[i]) if actions is not None else fallback_actions[str(uid)]
            fallback_actions[str(uid)] += 1
            audit_id = f"{uid}:{role}:{action}"
            ids[i] = audit_id
            groups[audit_id].append(i)
            action_numbers[audit_id] = action
        batch.non_tensor_batch['credit_audit_id'] = ids
        events = []
        for audit_id, indices in groups.items():
            suffixes = batch.non_tensor_batch.get('c3_suffix_index')
            indices.sort(key=lambda i: int(suffixes[i]) if suffixes is not None else 0)
            i = indices[0]
            def values(key, default):
                tensor = batch.batch.get(key)
                return [tensor[j].item() if tensor is not None else default for j in indices]

            valid = values('prefix_probe_gate_valid', True)
            gates = values('prefix_probe_collaboration_eligible', True)
            gates = [bool(v and g) for v, g in zip(valid, gates)]
            record = {
                'audit_id': audit_id, 'rollout_step': int(step), 'role': role,
                'uid': str(batch.non_tensor_batch['uid'][i]),
                'action_index': action_numbers[audit_id],
                'suffix_outcomes': values('scoped_c3_raw_outcome_score', None),
                'suffix_gate_valid': [bool(v) for v in valid],
                'suffix_gates': gates,
                'all_suffix_gates_pass': all(gates),
                'action_mean_score': float(batch.batch['scoped_c3_outcome_score'][i]),
                'advantage': float(estimate.advantage[i]),
                'c3_effective': bool(estimate.effective_mask[i]),
                'c3_positive_only': bool(batch.batch['scoped_c3_positive_only_mask'][i]),
                'c3_group_has_eligible_success': bool(estimate.eligible_success_group_mask[i]),
            }
            if audit_id in self.pending:
                raise ValueError(f"Duplicate credit audit action: {audit_id}")
            self.pending[audit_id] = record
            messages = batch.non_tensor_batch.get(f'{role}_conversation_history')
            histories = batch.non_tensor_batch.get('history')
            events.append({
                **record, 'event': 'generated',
                'question': batch.non_tensor_batch.get('question', [''] * len(batch))[i],
                'reward_model': batch.non_tensor_batch.get('reward_model', [None] * len(batch))[i],
                'focal_messages': messages[i] if messages is not None else None,
                'history': histories[i] if histories is not None else None,
                'suffix_indices': [int(suffixes[j]) if suffixes is not None else 0 for j in indices],
                'suffix_terminal_responses': [
                    batch.non_tensor_batch.get('response', [''] * len(batch))[j] for j in indices
                ],
            })
        self._write(step, events)

    def finish(self, batch, *, actor_updated, actor_update_step, skip_reason=None):
        final_rows = {}
        ids = batch.non_tensor_batch.get('credit_audit_id', [])
        for i, audit_id in enumerate(ids):
            if 'actor_padding_mask' in batch.batch and bool(batch.batch['actor_padding_mask'][i]):
                continue
            if 'scoped_c3_action_representative' in batch.batch and not bool(batch.batch['scoped_c3_action_representative'][i]):
                continue
            mask = batch.batch['labels'][i].ne(-100) & batch.batch['step_ids'][i].ne(-100)
            tokens = int(mask.sum())
            advantage = float(batch.batch['advantages'][i][mask][0]) if tokens else 0.0
            final_rows[audit_id] = (tokens, advantage)

        counts = Counter({f'status/{reason}': 0 for reason in (
            'included_positive', 'included_negative', 'actor_update_failed_or_partial',
            'suffix_gate_rejected_or_invalid', 'no_effective_c3_contrast',
            'nonpositive_advantage_policy', 'no_effective_c3_signal',
            'not_selected_for_optimizer_batch', 'no_trainable_tokens',
            'zero_trainable_batch', 'sparse_batch', 'critic_warmup',
            'actor_update_not_completed',
        )})
        events = []
        for audit_id, record in self.pending.items():
            selected = audit_id in final_rows
            tokens, advantage = final_rows.get(audit_id, (0, 0.0))
            included = bool(actor_updated and selected and tokens and advantage != 0)
            unknown = bool(skip_reason == 'actor_update_failed_or_partial' and selected and tokens)
            if included:
                reason = 'included_positive' if advantage > 0 else 'included_negative'
            elif unknown:
                reason = 'actor_update_failed_or_partial'
            elif not record['all_suffix_gates_pass']:
                reason = 'suffix_gate_rejected_or_invalid'
            elif record['advantage'] == 0:
                reason = 'no_effective_c3_contrast'
            elif record['c3_positive_only'] and record['advantage'] < 0:
                reason = 'nonpositive_advantage_policy'
            elif not record['c3_effective']:
                reason = 'no_effective_c3_signal'
            elif not selected:
                reason = 'not_selected_for_optimizer_batch'
            elif not tokens:
                reason = 'no_trainable_tokens'
            else:
                reason = skip_reason or 'actor_update_not_completed'
            event = {
                **record, 'event': 'disposition',
                'actor_update_step': int(actor_update_step),
                'selected_for_optimizer_batch': selected,
                'final_trainable_tokens': tokens, 'final_advantage': advantage,
                'actor_update_completed': bool(actor_updated),
                'update_outcome_unknown': unknown,
                'included_in_actor_update': None if unknown else included,
                'positive_training_signal': None if unknown else included and advantage > 0,
                'negative_training_signal': None if unknown else included and advantage < 0,
                'status': reason,
            }
            events.append(event)
            counts['generated_actions'] += 1
            counts['gate_passing_actions'] += int(record['all_suffix_gates_pass'])
            counts['positive_advantage_actions'] += int(record['advantage'] > 0)
            counts['used_positive_actions'] += int(bool(event['positive_training_signal']))
            counts['used_negative_actions'] += int(bool(event['negative_training_signal']))
            counts['excluded_actions'] += int(not included and not unknown)
            counts['unknown_update_actions'] += int(unknown)
            counts[f'status/{reason}'] += 1
        if events:
            self._write(events[0]['rollout_step'], events)
            # Include a used action when available, not only early rejected actions.
            ordered = sorted(events, key=lambda e: not e['included_in_actor_update'])
            for event in ordered[:self.console_examples]:
                print(
                    f"[credit/example] id={event['audit_id']} "
                    f"outcomes={event['suffix_outcomes']} gates={event['suffix_gates']} "
                    f"mean={event['action_mean_score']:.3f} advantage={event['advantage']:+.3f} "
                    f"trainable_tokens={event['final_trainable_tokens']} "
                    f"actor_updated={int(actor_updated)} status={event['status']}", flush=True,
                )
            print(f"[credit/summary] actions={counts['generated_actions']} "
                  f"used_positive={counts['used_positive_actions']} "
                  f"used_negative={counts['used_negative_actions']} "
                  f"excluded={counts['excluded_actions']} "
                  f"unknown={counts['unknown_update_actions']}", flush=True)
        self.pending.clear()
        return {f'train/credit/{key}': float(value) for key, value in counts.items()}

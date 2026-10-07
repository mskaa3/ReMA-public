"""Shared Graph+PRM input contract, vendored unchanged into ReMA's runtime.

Keep this file byte-identical to hierarchical_rema/graphprm_inputs.py. It has no
training dependencies and never treats raw audit output as a delivered artifact.
"""
from __future__ import annotations

import copy
import hashlib
import re

INPUT_VERSION = "graphprm_inputs_v2"
STEP_SEPARATOR = "<extra_0>"
_SCRATCH_NAMES = r"(?:worker_scratchpad|decomposer_scratchpad|selector_scratchpad|scratchpad)"
_SCRATCH_BLOCK = re.compile(rf"<(?P<tag>{_SCRATCH_NAMES})\s*>.*?</(?P=tag)\s*>", re.I | re.S)
_UNCLOSED_SCRATCH = re.compile(rf"<{_SCRATCH_NAMES}\s*>.*\Z", re.I | re.S)
_RESULT_BLOCK = re.compile(r"<(?P<tag>worker_result|result|answer)\s*>(.*?)</(?P=tag)\s*>", re.I | re.S)
_PROTOCOL_TAG = re.compile(rf"</?(?:{_SCRATCH_NAMES}|worker_result|result|answer)\s*>", re.I)


def clean_result_text(value):
    """Preserve math and newlines; strip only recognized protocol markup."""
    text = "" if value is None else str(value)
    text = _SCRATCH_BLOCK.sub("", text)
    text = _UNCLOSED_SCRATCH.sub("", text)
    match = _RESULT_BLOCK.search(text)
    if match:
        text = match.group(2)
    return _PROTOCOL_TAG.sub("", text).strip()


def delivered_result(worker):
    for key in ("output_text", "worker_output_text", "result"):
        if key in worker:
            return clean_result_text(worker[key])
    return ""


def normalize_text_record(record):
    """Preserve raw output for audit and normalize only accepted text fields."""
    result = copy.deepcopy(record)
    for worker in result.get("workers", []):
        worker["output_text"] = delivered_result(worker)
        for context in worker.get("upstream_context", []):
            for key in ("used_value", "output_text"):
                if key in context:
                    context[key] = clean_result_text(context[key])
        if "dependency_outputs" in worker:
            worker["dependency_outputs"] = {
                key: clean_result_text(value) for key, value in worker["dependency_outputs"].items()
            }
    trajectory = result.get("trajectory", {})
    if "final_answer" in trajectory:
        trajectory["final_answer"] = clean_result_text(trajectory["final_answer"])
    return result


def feature_only_record(record):
    """Prevent older graph builders' raw-output fallback, without losing audit data."""
    result = normalize_text_record(record)
    for worker in result.get("workers", []):
        worker["raw_output_text"] = worker["output_text"]
        for context in worker.get("upstream_context", []):
            if "used_value" in context:
                context["output_text"] = context["used_value"]
    return result


def escape_prm_field(value, tokenizer=None):
    text = "" if value is None else str(value)
    reserved = {STEP_SEPARATOR, "<|im_start|>", "<|im_end|>", "<|endoftext|>"}
    reserved.update(str(token) for token in (getattr(tokenizer, "all_special_tokens", []) or []))
    for token in sorted(reserved, key=len, reverse=True):
        if token:
            replacement = token.replace("<", "&lt;").replace(">", "&gt;")
            if replacement == token:
                replacement = "[reserved token]"
            text = text.replace(token, replacement)
    return text


def check_rendered_markers(prompt, targets):
    if prompt.count(STEP_SEPARATOR) != len(targets):
        raise ValueError("PRM marker count does not match rendered node targets")
    keys = [(target["scope"], str(target.get("node_id", ""))) for target in targets]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate PRM node targets")


def marker_alignment(tokenizer, prompt, targets, max_length):
    """Verify exact prefix truncation so surviving marker i belongs to target i."""
    check_rendered_markers(prompt, targets)
    if getattr(tokenizer, "truncation_side", "right") != "right":
        raise ValueError("Graph+PRM requires right truncation for prefix marker alignment")
    separator_ids = tokenizer.encode(STEP_SEPARATOR, add_special_tokens=False)
    if len(separator_ids) != 1:
        raise ValueError("PRM step separator must be exactly one token")
    separator_id = separator_ids[0]
    full = tokenizer(prompt, truncation=False)["input_ids"]
    kept = tokenizer(prompt, truncation=True, max_length=int(max_length))["input_ids"]
    if full.count(separator_id) != len(targets):
        raise ValueError("Unexpected reserved PRM markers in tokenized input")
    if kept != full[:len(kept)]:
        raise ValueError("PRM tokenization did not preserve the sequence prefix")
    positions = [i for i, token in enumerate(kept) if token == separator_id]
    return {
        "version": INPUT_VERSION,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "target_keys": [[t["scope"], str(t.get("node_id", ""))] for t in targets],
        "retained_target_indices": list(range(len(positions))),
        "marker_positions": positions,
        "total_tokens": len(full),
        "retained_tokens": len(kept),
    }


def validate_marker_payload(payload, alignment, *, include_hidden=True, include_score=True):
    """Validate new extraction and cached features before positional assignment."""
    expected = len(alignment["marker_positions"])
    if int(payload.get("marker_count", -1)) != expected:
        raise ValueError("PRM feature count does not match verified marker positions")
    if "marker_positions" in payload and payload["marker_positions"] != alignment["marker_positions"]:
        raise ValueError("PRM extracted marker positions do not match rendered targets")
    if "marker_alignment" in payload and payload["marker_alignment"] != alignment:
        raise ValueError("Cached PRM marker alignment does not match this input")
    if include_score and len(payload.get("scores", [])) != expected:
        raise ValueError("PRM score count does not match marker targets")
    if include_hidden:
        hidden = payload.get("hidden_states")
        if expected and (hidden is None or len(hidden.shape) != 2 or hidden.shape[0] != expected):
            raise ValueError("PRM hidden-state rows do not match marker targets")
        if not expected and hidden is not None and hidden.shape[0] != 0:
            raise ValueError("PRM hidden states exist without matching markers")
    payload["marker_alignment"] = copy.deepcopy(alignment)


def apply_protocol_reward_guards(node_rewards, record, penalty=0.10):
    """Keep invalid delivered outputs from receiving positive compiled rewards."""
    invalid = {}
    for worker in record.get("workers", []):
        reason = str(worker.get("invalid_reason") or "")
        if not delivered_result(worker):
            reason = reason or "empty_delivered_result"
        if reason:
            invalid[str(worker["node_id"])] = reason

    def guard(payload, reason):
        before = float(payload["reward"])
        payload.update(reward_before_protocol_guard=before, protocol_invalid_reason=reason,
                       protocol_penalty=float(penalty), protocol_positive_cap_applied=before > 0)
        payload["reward"] = payload["reward_raw"] = min(before, 0.0) - penalty

    for node, reason in invalid.items():
        if node in node_rewards.get("workers", {}):
            guard(node_rewards["workers"][node], reason)
    final_id = str(record.get("decomposition", {}).get("final_node_id")
                   or record.get("trajectory", {}).get("final_node_id") or "")
    final_reason = invalid.get(final_id)
    if not clean_result_text(record.get("trajectory", {}).get("final_answer")):
        final_reason = final_reason or "empty_final_answer"
    if final_reason and "final" in node_rewards:
        guard(node_rewards["final"], final_reason)
    return invalid

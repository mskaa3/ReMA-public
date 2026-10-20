"""Opt-in, complete evaluation records and fixed-plan evaluation support."""

import hashlib
import json
import os
from pathlib import Path


def json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def question_id(question):
    return hashlib.sha256(" ".join(str(question).split()).encode()).hexdigest()


def plan_id(plan):
    return hashlib.sha256(str(plan).encode()).hexdigest()


def validation_plans(non_tensors, *, enabled, validation, max_turns):
    if not enabled:
        return None
    if not validation or max_turns != 1:
        raise ValueError("Fixed-plan replay is evaluation-only and requires exactly one round")
    values = non_tensors.get("fixed_decomposer_output")
    if values is None or len(values) != len(non_tensors["question"]):
        raise ValueError("Replay requires fixed_decomposer_output for every evaluation row")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Empty fixed plans cannot silently fall back to a generated plan")
    return list(values)


class ValidationRecorder:
    """Publish only complete evaluations; retain partial evidence after a failure."""

    def __init__(self, directory, context, step, update_step, expected_rows, config):
        self.path = Path(directory) / f"{context}-step{step}-update{update_step}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.path.with_suffix(".jsonl.partial")
        if self.path.exists() or self.partial.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation evidence: {self.path}")
        self.partial.touch(exist_ok=False)
        self.expected_rows = expected_rows
        self.count = 0
        self.config = config
        self.context, self.step, self.update_step = context, step, update_step

    def add_batch(self, batch, histories, responses, scores, terminal_roles):
        size = len(histories)
        if any(len(values) != size for values in (responses, scores, terminal_roles, batch['question'])):
            raise ValueError("Evaluation fields do not have the same number of rows")
        with self.partial.open("a") as stream:
            for index in range(size):
                history = histories[index]
                plans = [item['content'] for item in history if item['role'] == 'decomposer']
                record = {
                    "schema_version": 1, "context": self.context,
                    "step": self.step, "actor_update_step": self.update_step,
                    "record_index": self.count,
                    "question_id": question_id(batch['question'][index]),
                    "question": str(batch['question'][index]),
                    "data_source": str(batch['data_source'][index]),
                    "subset": str(batch.get('subset', batch['data_source'])[index]),
                    "ground_truth": str(batch['reward_model'][index]['ground_truth']),
                    "response": str(responses[index]), "raw_score": float(scores[index]),
                    "terminal_role": str(terminal_roles[index]),
                    "decomposer_output": plans[-1] if plans else "",
                    "plan_id": plan_id(plans[-1]) if plans else "",
                    "history": history,
                    "completion_tokens": sum(int(item.get('num_gen_tokens', 0)) for item in history),
                }
                for key in ('eval_item_id', 'plan_source', 'answer_free_reviewed'):
                    if key in batch:
                        value = batch[key][index]
                        record[key] = value.item() if hasattr(value, 'item') else value
                stream.write(json.dumps(record, ensure_ascii=True, default=json_default) + "\n")
                self.count += 1

    def finish(self, metrics, *, probe_records=None):
        if self.count != self.expected_rows:
            raise ValueError(f"Incomplete evaluation: {self.count}/{self.expected_rows} rows")
        metadata = {
            "schema_version": 1, "rows": self.count, "metrics": metrics,
            "config": self.config, "records_sha256": hashlib.sha256(self.partial.read_bytes()).hexdigest(),
        }
        if probe_records is not None:
            probe_path = self.path.with_suffix('.probes.jsonl')
            probe_partial = probe_path.with_suffix('.jsonl.partial')
            with probe_partial.open('x') as stream:
                for record in probe_records:
                    stream.write(json.dumps(record, ensure_ascii=True, default=json_default) + '\n')
            metadata['probe_records_sha256'] = hashlib.sha256(probe_partial.read_bytes()).hexdigest()
            os.replace(probe_partial, probe_path)
        meta_partial = self.path.with_suffix('.metadata.json.partial')
        meta_partial.write_text(json.dumps(metadata, indent=2, ensure_ascii=True, default=json_default) + '\n')
        os.replace(self.partial, self.path)
        os.replace(meta_partial, self.path.with_suffix('.metadata.json'))
        return self.path

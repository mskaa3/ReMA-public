from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .schema import RolloutLoggingConfig, TaskRollout


@dataclass
class _RankedRecord:
    score: float
    payload: Dict


class RolloutRecorder:
    def __init__(self, config: RolloutLoggingConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._best_selection_records: List[_RankedRecord] = []
        self._best_decomposition_records: List[_RankedRecord] = []

    def record_task_rollout(self, rollout: TaskRollout) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        task_payload = {
            "timestamp": timestamp,
            "task_id": rollout.task.task_id,
            "rollout": rollout.to_dict(),
        }
        if self.config.save_all_rollouts:
            self._append_jsonl(self.output_dir / "all_rollouts.jsonl", task_payload)

        if not self.config.save_best_rollouts:
            return

        for decomposition in rollout.decompositions:
            decomposition_payload = {
                "timestamp": timestamp,
                "task_id": rollout.task.task_id,
                "decomposition_id": decomposition.decomposition.decomposition_id,
                "score": decomposition.decomposition_reward,
                "rollout": decomposition.to_dict(),
            }
            self._push_best(
                self._best_decomposition_records,
                decomposition.decomposition_reward,
                decomposition_payload,
            )

            for selection in decomposition.selections:
                selection_payload = {
                    "timestamp": timestamp,
                    "task_id": rollout.task.task_id,
                    "decomposition_id": decomposition.decomposition.decomposition_id,
                    "selection_id": selection.selection.selection_id,
                    "score": selection.reward.total_reward,
                    "rollout": selection.to_dict(),
                }
                self._push_best(
                    self._best_selection_records,
                    selection.reward.total_reward,
                    selection_payload,
                )

        self._rewrite_jsonl(
            self.output_dir / "best_decompositions.jsonl",
            [record.payload for record in self._best_decomposition_records],
        )
        self._rewrite_jsonl(
            self.output_dir / "best_selections.jsonl",
            [record.payload for record in self._best_selection_records],
        )

    def _push_best(self, bucket: List[_RankedRecord], score: float, payload: Dict) -> None:
        bucket.append(_RankedRecord(score=score, payload=payload))
        bucket.sort(key=lambda record: record.score, reverse=True)
        del bucket[self.config.best_k :]

    @staticmethod
    def _append_jsonl(path: Path, payload: Dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    @staticmethod
    def _rewrite_jsonl(path: Path, payloads: List[Dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

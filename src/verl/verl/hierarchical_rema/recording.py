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
        task_payload = self._task_payload(rollout=rollout, timestamp=timestamp)
        if self.config.save_all_rollouts:
            self._append_jsonl(self.output_dir / "all_rollouts.jsonl", task_payload)

        if not self.config.save_best_rollouts:
            return

        best_decomposition = max(
            rollout.decompositions,
            key=lambda item: item.decomposition_reward,
        )
        self._append_jsonl(
            self.output_dir / "best_decompositions.jsonl",
            self._decomposition_payload(
                task_rollout=rollout,
                decomposition_rollout=best_decomposition,
                timestamp=timestamp,
            ),
        )
        best_selection_decomposition, best_selection = max(
            (
                (decomposition, selection)
                for decomposition in rollout.decompositions
                for selection in decomposition.selections
            ),
            key=lambda item: item[1].reward.total_reward,
        )
        self._append_jsonl(
            self.output_dir / "best_selections.jsonl",
            self._selection_payload(
                task_rollout=rollout,
                decomposition_rollout=best_selection_decomposition,
                selection_rollout=best_selection,
                timestamp=timestamp,
            ),
        )

        for decomposition in rollout.decompositions:
            decomposition_payload = self._decomposition_payload(
                task_rollout=rollout,
                decomposition_rollout=decomposition,
                timestamp=timestamp,
            )
            self._push_best(
                self._best_decomposition_records,
                decomposition.decomposition_reward,
                decomposition_payload,
            )

            for selection in decomposition.selections:
                selection_payload = self._selection_payload(
                    task_rollout=rollout,
                    decomposition_rollout=decomposition,
                    selection_rollout=selection,
                    timestamp=timestamp,
                )
                self._push_best(
                    self._best_selection_records,
                    selection.reward.total_reward,
                    selection_payload,
                )

        self._rewrite_jsonl(
            self.output_dir / "topk_best_decompositions.jsonl",
            [record.payload for record in self._best_decomposition_records],
        )
        self._rewrite_jsonl(
            self.output_dir / "topk_best_selections.jsonl",
            [record.payload for record in self._best_selection_records],
        )

    def _push_best(self, bucket: List[_RankedRecord], score: float, payload: Dict) -> None:
        bucket.append(_RankedRecord(score=score, payload=payload))
        bucket.sort(key=lambda record: record.score, reverse=True)
        del bucket[self.config.best_k :]

    def _task_payload(self, rollout: TaskRollout, timestamp: str) -> Dict:
        if not self.config.compact_mode:
            return {
                "timestamp": timestamp,
                "task_id": rollout.task.task_id,
                "rollout": rollout.to_dict(),
            }
        best_decomposition = max(rollout.decompositions, key=lambda item: item.decomposition_reward)
        best_selection = max(
            (
                selection
                for decomposition in rollout.decompositions
                for selection in decomposition.selections
            ),
            key=lambda item: item.reward.total_reward,
        )
        return {
            "timestamp": timestamp,
            "task_id": rollout.task.task_id,
            "task_prompt": rollout.task.prompt,
            "ground_truth": rollout.task.ground_truth,
            "schedule": {
                "mode": rollout.schedule.mode.value,
                "phase": rollout.schedule.alternating_phase.value,
            },
            "best_decomposition_id": best_decomposition.decomposition.decomposition_id,
            "best_decomposition_reward": best_decomposition.decomposition_reward,
            "best_selection_id": best_selection.selection.selection_id,
            "best_selection_reward": best_selection.reward.total_reward,
            "best_final_correctness": best_selection.reward.final_answer_correctness,
        }

    def _decomposition_payload(
        self,
        task_rollout: TaskRollout,
        decomposition_rollout,
        timestamp: str,
    ) -> Dict:
        if not self.config.compact_mode:
            return {
                "timestamp": timestamp,
                "task_id": task_rollout.task.task_id,
                "task_prompt": task_rollout.task.prompt,
                "ground_truth": task_rollout.task.ground_truth,
                "schedule": {
                    "mode": task_rollout.schedule.mode.value,
                    "phase": task_rollout.schedule.alternating_phase.value,
                },
                "decomposition_id": decomposition_rollout.decomposition.decomposition_id,
                "score": decomposition_rollout.decomposition_reward,
                "rollout": decomposition_rollout.to_dict(),
            }
        decomposition = decomposition_rollout.decomposition
        return {
            "timestamp": timestamp,
            "task_id": task_rollout.task.task_id,
            "task_prompt": task_rollout.task.prompt,
            "ground_truth": task_rollout.task.ground_truth,
            "schedule": {
                "mode": task_rollout.schedule.mode.value,
                "phase": task_rollout.schedule.alternating_phase.value,
            },
            "decomposition_id": decomposition.decomposition_id,
            "final_node_id": decomposition.final_node_id,
            "score": decomposition_rollout.decomposition_reward,
            "base_reward": decomposition_rollout.base_decomposition_reward,
            "summary": decomposition.summary,
            "num_hops": decomposition.num_hops,
            "effective_num_hops": decomposition.effective_num_hops,
            "soft_penalty": decomposition.soft_penalty,
            "was_hard_truncated": decomposition.was_hard_truncated,
            "decomposition_raw_text": decomposition.raw_text,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "instruction": node.instruction,
                    "dependencies": list(node.dependencies),
                    "required_skills": list(node.required_skills),
                    "output_key": node.output_key,
                }
                for node in decomposition.nodes
            ],
        }

    def _selection_payload(
        self,
        task_rollout: TaskRollout,
        decomposition_rollout,
        selection_rollout,
        timestamp: str,
    ) -> Dict:
        if not self.config.compact_mode:
            return {
                "timestamp": timestamp,
                "task_id": task_rollout.task.task_id,
                "task_prompt": task_rollout.task.prompt,
                "ground_truth": task_rollout.task.ground_truth,
                "schedule": {
                    "mode": task_rollout.schedule.mode.value,
                    "phase": task_rollout.schedule.alternating_phase.value,
                },
                "decomposition_id": decomposition_rollout.decomposition.decomposition_id,
                "selection_id": selection_rollout.selection.selection_id,
                "score": selection_rollout.reward.total_reward,
                "rollout": selection_rollout.to_dict(),
            }
        return {
            "timestamp": timestamp,
            "task_id": task_rollout.task.task_id,
            "task_prompt": task_rollout.task.prompt,
            "ground_truth": task_rollout.task.ground_truth,
            "schedule": {
                "mode": task_rollout.schedule.mode.value,
                "phase": task_rollout.schedule.alternating_phase.value,
            },
            "decomposition_id": decomposition_rollout.decomposition.decomposition_id,
            "final_node_id": decomposition_rollout.decomposition.final_node_id,
            "decomposition_summary": decomposition_rollout.decomposition.summary,
            "decomposition_raw_text": decomposition_rollout.decomposition.raw_text,
            "selection_id": selection_rollout.selection.selection_id,
            "score": selection_rollout.reward.total_reward,
            "selection_raw_text": selection_rollout.selection.raw_text,
            "final_answer": selection_rollout.final_answer,
            "reward": selection_rollout.reward.to_dict(),
            "assignments": [
                {
                    "node_id": assignment.node_id,
                    "worker_id": assignment.worker_id,
                    "compatibility": assignment.compatibility,
                }
                for assignment in selection_rollout.selection.assignments
            ],
            "executions": [
                {
                    "node_id": execution.node_id,
                    "worker_id": execution.worker_id,
                    "output_text": execution.output_text,
                    "dependency_outputs": execution.dependency_outputs,
                    "entropy": execution.entropy,
                    "confidence_reward": execution.confidence_reward,
                    "compatibility": execution.compatibility,
                    "completed": execution.completed,
                    "success": execution.success,
                }
                for execution in selection_rollout.executions
            ],
        }

    @staticmethod
    def _append_jsonl(path: Path, payload: Dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    @staticmethod
    def _rewrite_jsonl(path: Path, payloads: List[Dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

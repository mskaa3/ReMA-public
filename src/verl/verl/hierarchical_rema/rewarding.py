from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

from .schema import (
    RewardWeights,
    SelectionRewardBreakdown,
    WorkerExecution,
    WorkerPerformanceSnapshot,
    WorkerRewardMode,
    WorkerPoolConfig,
    WorkerSpec,
)


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def entropy_to_confidence_reward(entropy: float, entropy_cap: float) -> float:
    if entropy_cap <= 0:
        return 0.0
    clipped = min(max(entropy, 0.0), entropy_cap)
    return 1.0 - (clipped / entropy_cap)


def group_relative_advantages(values: Sequence[float]) -> List[float]:
    if len(values) <= 1:
        return [0.0 for _ in values]
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std = variance ** 0.5
    if std < 1e-8:
        return [0.0 for _ in values]
    return [(value - mean_value) / std for value in values]


@dataclass
class WorkerPerformanceMemory:
    smoothing: float = 0.2
    initial_prior: float = 0.5
    max_recent_history: int = 5
    _worker_outcome_ema: Dict[str, float] = field(default_factory=dict)
    _worker_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def prior_for(self, worker_id: str) -> float:
        return self._worker_outcome_ema.get(worker_id, self.initial_prior)

    def _stats_for(self, worker_id: str) -> Dict[str, Any]:
        if worker_id not in self._worker_stats:
            self._worker_stats[worker_id] = {
                "num_assignments": 0,
                "total_reward": 0.0,
                "total_confidence_reward": 0.0,
                "total_compatibility": 0.0,
                "recent_history": [],
            }
        return self._worker_stats[worker_id]

    def update(self, worker_id: str, outcome: float) -> None:
        previous = self.prior_for(worker_id)
        self._worker_outcome_ema[worker_id] = (1.0 - self.smoothing) * previous + self.smoothing * outcome

    def update_many(self, worker_ids: Iterable[str], outcome: float) -> None:
        for worker_id in worker_ids:
            self.update(worker_id, outcome)

    def record_execution(
        self,
        task_id: str,
        execution: WorkerExecution,
        selection_reward: SelectionRewardBreakdown,
        reward_weights: RewardWeights,
    ) -> None:
        stats = self._stats_for(execution.worker_id)
        stats["num_assignments"] += 1
        stats["total_reward"] += selection_reward.total_reward
        stats["total_confidence_reward"] += execution.confidence_reward
        stats["total_compatibility"] += execution.compatibility
        stats["recent_history"].append(
            {
                "task_id": task_id,
                "node_id": execution.node_id,
                "selection_reward": selection_reward.total_reward,
                "final_answer_correctness": selection_reward.final_answer_correctness,
                "confidence_reward": execution.confidence_reward,
                "compatibility": execution.compatibility,
                "entropy": execution.entropy,
            }
        )
        if len(stats["recent_history"]) > self.max_recent_history:
            stats["recent_history"] = stats["recent_history"][-self.max_recent_history:]

        outcome = selection_reward.final_answer_correctness
        self.update(execution.worker_id, outcome)

    def snapshot_for(self, worker_id: str) -> WorkerPerformanceSnapshot:
        stats = self._stats_for(worker_id)
        num_assignments = int(stats["num_assignments"])
        denom = max(num_assignments, 1)
        return WorkerPerformanceSnapshot(
            worker_id=worker_id,
            ema_outcome=self.prior_for(worker_id),
            num_assignments=num_assignments,
            average_reward=stats["total_reward"] / denom if num_assignments else 0.0,
            average_confidence_reward=(
                stats["total_confidence_reward"] / denom if num_assignments else 0.0
            ),
            average_compatibility=(
                stats["total_compatibility"] / denom if num_assignments else 0.0
            ),
            recent_history=list(stats["recent_history"]),
        )

    def snapshot(self, worker_pool: WorkerPoolConfig) -> Dict[str, WorkerPerformanceSnapshot]:
        return {
            worker.worker_id: self.snapshot_for(worker.worker_id)
            for worker in worker_pool.workers
        }

    def to_dict(self) -> Dict[str, float]:
        return dict(self._worker_outcome_ema)


def skill_match_score(required_skills: Sequence[str], worker: WorkerSpec) -> float:
    if not required_skills:
        return 0.5
    required = {skill.lower() for skill in required_skills}
    available = {skill.lower() for skill in worker.skills}
    overlap = len(required & available)
    return overlap / max(len(required), 1)


def compatibility_score(
    required_skills: Sequence[str],
    worker: WorkerSpec,
    worker_performance: Dict[str, WorkerPerformanceSnapshot],
) -> float:
    return skill_match_score(required_skills, worker)


def build_selection_reward(
    final_answer: str,
    ground_truth: str,
    executions: Sequence[WorkerExecution],
    weights: RewardWeights,
) -> SelectionRewardBreakdown:
    final_correct = exact_match(final_answer, ground_truth)
    confidence_reward = (
        sum(
            entropy_to_confidence_reward(execution.entropy, weights.entropy_cap)
            for execution in executions
        )
        / len(executions)
        if executions
        else 0.0
    )
    compatibility_reward = (
        sum(execution.compatibility for execution in executions) / len(executions)
        if executions
        else 0.0
    )
    if weights.worker_reward_mode == WorkerRewardMode.FINAL_ANSWER_CORRECTNESS_ONLY:
        total_reward = final_correct
    else:
        total_reward = (
            weights.final_answer * final_correct
            + weights.confidence * confidence_reward
            + weights.compatibility * compatibility_reward
        )
    return SelectionRewardBreakdown(
        final_answer_correctness=final_correct,
        confidence_reward=confidence_reward,
        compatibility_reward=compatibility_reward,
        total_reward=total_reward,
    )

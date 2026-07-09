from __future__ import annotations

from dataclasses import dataclass, field
import re
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

WORKER_INVALID_RESULT_PENALTY = 0.10
INTERMEDIATE_FINAL_ANSWER_PENALTY = 0.25
NON_FINAL_ANSWER_CONTAINMENT_PENALTY = 0.125
_DEFAULT_COMPUTE_SCORE = None
_DEFAULT_COMPUTE_SCORE_LOADED = False
_BOXED_ANSWER_PATTERN = re.compile(r"\\boxed\s*\{([^{}]+)\}")
_FINAL_CLAUSE_PATTERN = re.compile(
    r"(?:the\s+answer\s+is|the\s+value\s+is|therefore|thus|so|hence|must\s+be|equals?)\s*[:=]?\s*(.+)",
    flags=re.IGNORECASE,
)
_NAMED_RESULT_CLAUSE_PATTERN = re.compile(
    r"^(?:therefore|thus|hence|so)?\s*,?\s*the\s+[a-z0-9_\\{}^$().\-\s]{1,80}?\s+is\s*[:=]?\s*(.*)$",
    flags=re.IGNORECASE,
)
_BOUNDARY_SAFE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def _strip_outer_math_delimiters(candidate: str) -> str:
    cleaned = str(candidate or "").strip()
    if len(cleaned) >= 2 and cleaned.startswith("$") and cleaned.endswith("$"):
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("\\(") and cleaned.endswith("\\)"):
        cleaned = cleaned[2:-2].strip()
    if cleaned.startswith("\\[") and cleaned.endswith("\\]"):
        cleaned = cleaned[2:-2].strip()
    if cleaned.startswith("$"):
        cleaned = cleaned[1:].strip()
    if cleaned.endswith("$"):
        cleaned = cleaned[:-1].strip()
    return cleaned.rstrip(".,;:")


def _contains_normalized_reference(text: str, reference: str) -> bool:
    normalized_text = normalize_answer(text)
    normalized_reference = normalize_answer(reference)
    if not normalized_text or not normalized_reference:
        return False
    if normalized_reference not in normalized_text:
        return False
    if all(ch in _BOUNDARY_SAFE_CHARS for ch in normalized_reference):
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_reference)}(?![a-z0-9])"
        return re.search(pattern, normalized_text) is not None
    return True


def _extract_answer_like_candidates(text: str) -> List[str]:
    stripped = str(text or "").strip()
    if not stripped:
        return []

    candidates: List[str] = [stripped]
    seen = {stripped}

    def _add(candidate: str) -> None:
        cleaned = candidate.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            candidates.append(cleaned)

    def _strip_outer_math_delimiters(candidate: str) -> str:
        cleaned = candidate.strip()
        if len(cleaned) >= 2 and cleaned.startswith("$") and cleaned.endswith("$"):
            cleaned = cleaned[1:-1].strip()
        if cleaned.startswith("\\(") and cleaned.endswith("\\)"):
            cleaned = cleaned[2:-2].strip()
        if cleaned.startswith("\\[") and cleaned.endswith("\\]"):
            cleaned = cleaned[2:-2].strip()
        if cleaned.startswith("$"):
            cleaned = cleaned[1:].strip()
        if cleaned.endswith("$"):
            cleaned = cleaned[:-1].strip()
        cleaned = cleaned.rstrip(".,;:")
        return cleaned

    def _add_with_variants(candidate: str) -> None:
        _add(candidate)
        stripped_candidate = _strip_outer_math_delimiters(candidate)
        if stripped_candidate != candidate.strip():
            _add(stripped_candidate)

    for match in _BOXED_ANSWER_PATTERN.finditer(stripped):
        _add_with_variants(match.group(1))

    lines = [line.strip(" -*\t") for line in stripped.splitlines() if line.strip()]
    if lines:
        _add_with_variants(lines[-1])
    for line in lines[-3:]:
        match = _FINAL_CLAUSE_PATTERN.search(line)
        if match:
            _add_with_variants(match.group(1))
        if ":" in line:
            _add_with_variants(line.rsplit(":", 1)[-1])
        if "=" in line:
            _add_with_variants(line.rsplit("=", 1)[-1])

    for segment in re.split(r"[;\n]", stripped):
        candidate = segment.strip()
        if 0 < len(candidate) <= 64:
            _add_with_variants(candidate)

    return candidates


def _extract_non_final_leak_candidates(text: str) -> List[str]:
    stripped = str(text or "").strip()
    if not stripped:
        return []

    candidates: List[str] = []
    seen = set()

    def _add(candidate: str) -> None:
        cleaned = candidate.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            candidates.append(cleaned)

    def _strip_outer_math_delimiters(candidate: str) -> str:
        cleaned = candidate.strip()
        if len(cleaned) >= 2 and cleaned.startswith("$") and cleaned.endswith("$"):
            cleaned = cleaned[1:-1].strip()
        if cleaned.startswith("\\(") and cleaned.endswith("\\)"):
            cleaned = cleaned[2:-2].strip()
        if cleaned.startswith("\\[") and cleaned.endswith("\\]"):
            cleaned = cleaned[2:-2].strip()
        if cleaned.startswith("$"):
            cleaned = cleaned[1:].strip()
        if cleaned.endswith("$"):
            cleaned = cleaned[:-1].strip()
        cleaned = cleaned.rstrip(".,;:")
        return cleaned

    def _add_with_variants(candidate: str) -> None:
        _add(candidate)
        stripped_candidate = _strip_outer_math_delimiters(candidate)
        if stripped_candidate != candidate.strip():
            _add(stripped_candidate)

    _add_with_variants(stripped)

    lines = [line.strip(" -*\t") for line in stripped.splitlines() if line.strip()]
    if lines:
        _add_with_variants(lines[-1])
    start_index = max(len(lines) - 3, 0)
    for offset, line in enumerate(lines[start_index:], start=start_index):
        match = _FINAL_CLAUSE_PATTERN.search(line)
        if match:
            _add_with_variants(match.group(1))
        named_match = _NAMED_RESULT_CLAUSE_PATTERN.search(line)
        if named_match:
            remainder = named_match.group(1).strip()
            if remainder:
                _add_with_variants(remainder)
            elif offset + 1 < len(lines):
                trailing_block = "\n".join(lines[offset + 1 :]).strip()
                if trailing_block:
                    _add_with_variants(trailing_block)

    for match in _BOXED_ANSWER_PATTERN.finditer(stripped):
        _add_with_variants(match.group(1))

    return candidates


def is_non_final_answer_leak(
    text: str,
    reference: str,
    task_metadata: Dict[str, Any] | None = None,
) -> bool:
    for candidate in _extract_non_final_leak_candidates(text):
        if compute_final_answer_correctness(
            candidate,
            reference,
            task_metadata=task_metadata,
        ) > 0.0:
            return True
    return False


def contains_answer_like_content(
    text: str,
    reference: str,
    task_metadata: Dict[str, Any] | None = None,
) -> bool:
    if _contains_normalized_reference(text, reference):
        return True
    for candidate in _extract_answer_like_candidates(text):
        if compute_final_answer_correctness(
            candidate,
            reference,
            task_metadata=task_metadata,
        ) > 0.0:
            return True
    return False


def _load_default_compute_score():
    global _DEFAULT_COMPUTE_SCORE, _DEFAULT_COMPUTE_SCORE_LOADED
    if _DEFAULT_COMPUTE_SCORE_LOADED:
        return _DEFAULT_COMPUTE_SCORE
    try:
        from ..utils.reward_score import _default_compute_score

        _DEFAULT_COMPUTE_SCORE = _default_compute_score
    except Exception:
        _DEFAULT_COMPUTE_SCORE = None
    _DEFAULT_COMPUTE_SCORE_LOADED = True
    return _DEFAULT_COMPUTE_SCORE


def _score_single_prediction_candidate(
    prediction: str,
    reference: str,
    task_metadata: Dict[str, Any] | None = None,
) -> float:
    if not str(prediction or "").strip() or not str(reference or "").strip():
        return 0.0

    compute_score = _load_default_compute_score()
    metadata = task_metadata if isinstance(task_metadata, dict) else {}
    data_source = str(metadata.get("data_source") or "ReMA-math")
    extra_info = metadata.get("extra_info")
    if compute_score is not None:
        try:
            score = float(
                compute_score(
                    data_source=data_source,
                    solution_str=str(prediction),
                    ground_truth=str(reference),
                    extra_info=extra_info,
                )
            )
            return min(max(score, 0.0), 1.0)
        except Exception:
            pass

    return exact_match(prediction, reference)


def compute_final_answer_correctness(
    prediction: str,
    reference: str,
    task_metadata: Dict[str, Any] | None = None,
) -> float:
    prediction_text = str(prediction or "").strip()
    reference_text = str(reference or "").strip()
    if not prediction_text or not reference_text:
        return 0.0

    best_score = 0.0
    for candidate in _extract_answer_like_candidates(prediction_text):
        best_score = max(
            best_score,
            _score_single_prediction_candidate(
                candidate,
                reference_text,
                task_metadata=task_metadata,
            ),
        )
        if best_score >= 1.0:
            return 1.0
    return best_score


def entropy_to_confidence_reward(entropy: float, entropy_cap: float) -> float:
    if entropy_cap <= 0:
        return 0.0
    clipped = min(max(entropy, 0.0), entropy_cap)
    return 1.0 - (clipped / entropy_cap)


def _execution_signature(text: str) -> str:
    return normalize_answer(_strip_outer_math_delimiters(text))


def _is_valid_execution_output(execution: WorkerExecution) -> bool:
    return bool(str(execution.output_text or "").strip()) and not execution.invalid_reason


def _usage_rate(text: str, candidate_values: Iterable[str]) -> float:
    unique_signatures: List[str] = []
    seen = set()
    for candidate_value in candidate_values:
        signature = _execution_signature(candidate_value)
        if not signature or signature in seen:
            continue
        seen.add(signature)
        unique_signatures.append(signature)
    if not unique_signatures:
        return 0.0
    used_count = sum(
        1
        for signature in unique_signatures
        if _contains_normalized_reference(text, signature)
    )
    return used_count / len(unique_signatures)


def _compute_hierarchy_reward_stats(
    executions: Sequence[WorkerExecution],
    final_node_id: str | None,
) -> Dict[str, float | bool]:
    for execution in executions:
        execution.unique_result = False
        execution.downstream_used = False
        execution.dependency_used = False

    signature_by_node: Dict[str, str] = {}
    signature_counts: Dict[str, int] = {}
    valid_executions: List[WorkerExecution] = []
    for execution in executions:
        if not _is_valid_execution_output(execution):
            continue
        signature = _execution_signature(execution.output_text)
        if not signature:
            continue
        valid_executions.append(execution)
        signature_by_node[execution.node_id] = signature
        signature_counts[signature] = signature_counts.get(signature, 0) + 1

    valid_non_final_executions = [
        execution
        for execution in valid_executions
        if execution.node_id != final_node_id
    ]

    for index, execution in enumerate(executions):
        signature = signature_by_node.get(execution.node_id)
        if not signature:
            continue
        execution.unique_result = signature_counts.get(signature, 0) == 1
        later_outputs = [
            later_execution.output_text
            for later_execution in executions[index + 1 :]
            if _is_valid_execution_output(later_execution)
        ]
        execution.downstream_used = any(
            _contains_normalized_reference(later_output, signature)
            for later_output in later_outputs
        )
        if execution.node_id == final_node_id:
            continue
        if execution.dependency_outputs:
            execution.dependency_used = (
                _usage_rate(execution.output_text, execution.dependency_outputs.values()) > 0.0
            )

    dependency_usage_candidates = [
        execution
        for execution in valid_non_final_executions
        if execution.dependency_outputs
    ]
    dependency_usage_rate = (
        sum(1 for execution in dependency_usage_candidates if execution.dependency_used)
        / len(dependency_usage_candidates)
        if dependency_usage_candidates
        else 0.0
    )

    final_execution = next(
        (execution for execution in executions if execution.node_id == final_node_id),
        None,
    )
    final_has_dependencies = bool(
        final_execution
        and any(
            _execution_signature(dependency_output)
            for dependency_output in final_execution.dependency_outputs.values()
        )
    )
    final_dependency_usage_rate = (
        _usage_rate(final_execution.output_text, final_execution.dependency_outputs.values())
        if final_execution is not None and _is_valid_execution_output(final_execution) and final_has_dependencies
        else 0.0
    )
    final_output_uses_dependencies = final_dependency_usage_rate > 0.0
    multiple_nodes_executed = len(valid_executions) >= 2 and len(valid_non_final_executions) >= 1
    hierarchy_utilization_gate = 1.0 if (
        multiple_nodes_executed
        and final_has_dependencies
        and final_output_uses_dependencies
    ) else 0.0

    unique_result_rate = (
        sum(1 for execution in valid_executions if execution.unique_result)
        / len(valid_executions)
        if valid_executions
        else 0.0
    )
    duplicate_result_rate = (
        sum(
            1
            for execution in valid_executions
            if signature_counts.get(signature_by_node.get(execution.node_id, ""), 0) > 1
        )
        / len(valid_executions)
        if valid_executions
        else 0.0
    )
    downstream_used_rate = (
        sum(1 for execution in valid_non_final_executions if execution.downstream_used)
        / len(valid_non_final_executions)
        if valid_non_final_executions
        else 0.0
    )

    return {
        "unique_result_rate": unique_result_rate,
        "duplicate_result_rate": duplicate_result_rate,
        "downstream_used_rate": downstream_used_rate,
        "dependency_usage_rate": dependency_usage_rate,
        "final_dependency_usage_rate": final_dependency_usage_rate,
        "multiple_nodes_executed": multiple_nodes_executed,
        "final_has_dependencies": final_has_dependencies,
        "final_output_uses_dependencies": final_output_uses_dependencies,
        "hierarchy_utilization_gate": hierarchy_utilization_gate,
    }


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
        uses_reward_model = (
            selection_reward.reward_model_source == "gfam_v1"
            and execution.reward_model_reward is not None
        )
        observed_reward = (
            float(execution.reward_model_reward)
            if execution.reward_model_reward is not None
            else float(selection_reward.total_reward)
        )
        stats["num_assignments"] += 1
        stats["total_reward"] += observed_reward
        if uses_reward_model:
            stats["recent_history"].append(
                {
                    "task_id": task_id,
                    "node_id": execution.node_id,
                    "reward_source": selection_reward.reward_model_source,
                    "observed_reward": observed_reward,
                    "entropy": execution.entropy,
                }
            )
        else:
            stats["total_confidence_reward"] += execution.confidence_reward
            stats["total_compatibility"] += execution.compatibility
            stats["recent_history"].append(
                {
                    "task_id": task_id,
                    "node_id": execution.node_id,
                    "selection_reward": selection_reward.total_reward,
                    "observed_reward": observed_reward,
                    "final_answer_correctness": selection_reward.final_answer_correctness,
                    "confidence_reward": execution.confidence_reward,
                    "compatibility": execution.compatibility,
                    "entropy": execution.entropy,
                }
            )
        if len(stats["recent_history"]) > self.max_recent_history:
            stats["recent_history"] = stats["recent_history"][-self.max_recent_history:]

        outcome = observed_reward if uses_reward_model else selection_reward.final_answer_correctness
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
    task_metadata: Dict[str, Any] | None = None,
    final_node_id: str | None = None,
) -> SelectionRewardBreakdown:
    final_correct = compute_final_answer_correctness(
        final_answer,
        ground_truth,
        task_metadata=task_metadata,
    )
    hierarchy_stats = _compute_hierarchy_reward_stats(
        executions=executions,
        final_node_id=final_node_id,
    )
    positive_bonus_gate = 1.0 if final_correct > 0.0 else 0.0
    hierarchy_bonus_gate = positive_bonus_gate * float(
        hierarchy_stats["hierarchy_utilization_gate"]
    )
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
    worker_format_penalty = (
        WORKER_INVALID_RESULT_PENALTY
        * (
            sum(1 for execution in executions if execution.invalid_reason)
            / len(executions)
        )
        if executions
        else 0.0
    )
    intermediate_final_answer_penalty = (
        INTERMEDIATE_FINAL_ANSWER_PENALTY
        * (
            sum(1 for execution in executions if execution.final_answer_leak)
            / len(executions)
        )
        if executions
        else 0.0
    )
    non_final_answer_containment_count = 0
    if executions:
        for execution in executions:
            if execution.node_id == final_node_id or execution.final_answer_leak or not execution.output_text.strip():
                execution.answer_containment = False
                continue
            execution.answer_containment = (
                compute_final_answer_correctness(
                    execution.output_text,
                    final_answer,
                    task_metadata=task_metadata,
                )
                > 0.0
                or compute_final_answer_correctness(
                    execution.output_text,
                    ground_truth,
                    task_metadata=task_metadata,
                )
                > 0.0
            )
            if execution.answer_containment:
                non_final_answer_containment_count += 1
    non_final_answer_containment_penalty = (
        NON_FINAL_ANSWER_CONTAINMENT_PENALTY * (non_final_answer_containment_count / len(executions))
        if executions
        else 0.0
    )
    final_ignores_hierarchy_penalty = 0.0
    final_raw_score_usage_multiplier = 1.0
    if weights.worker_reward_mode == WorkerRewardMode.FINAL_ANSWER_CORRECTNESS_ONLY:
        total_reward = (
            final_correct
            - worker_format_penalty
            - intermediate_final_answer_penalty
            - non_final_answer_containment_penalty
        )
        worker_unique_result_bonus = 0.0
        worker_duplicate_result_penalty = 0.0
        worker_downstream_used_bonus = 0.0
        final_stage_usage_bonus = 0.0
    else:
        worker_unique_result_bonus = (
            weights.worker_unique_result_bonus
            * float(hierarchy_stats["unique_result_rate"])
            * hierarchy_bonus_gate
        )
        worker_duplicate_result_penalty = (
            weights.worker_duplicate_result_penalty
            * float(hierarchy_stats["duplicate_result_rate"])
        )
        worker_downstream_used_bonus = (
            weights.worker_downstream_used_bonus
            * float(hierarchy_stats["downstream_used_rate"])
            * hierarchy_bonus_gate
        )
        final_stage_usage_bonus = (
            weights.final_stage_usage_bonus
            * float(hierarchy_stats["final_dependency_usage_rate"])
            * hierarchy_bonus_gate
        )
        if (
            hierarchy_stats["multiple_nodes_executed"]
            and hierarchy_stats["final_has_dependencies"]
        ):
            final_raw_score_usage_multiplier = (
                1.0
                if hierarchy_stats["final_output_uses_dependencies"]
                else min(max(weights.final_raw_score_usage_floor, 0.0), 1.0)
            )
            if not hierarchy_stats["final_output_uses_dependencies"]:
                final_ignores_hierarchy_penalty = weights.final_ignores_hierarchy_penalty
        total_reward = (
            weights.final_answer * final_correct * final_raw_score_usage_multiplier
            + weights.confidence * confidence_reward
            + weights.compatibility * compatibility_reward
            + worker_unique_result_bonus
            + worker_downstream_used_bonus
            + final_stage_usage_bonus
            - worker_format_penalty
            - intermediate_final_answer_penalty
            - non_final_answer_containment_penalty
            - worker_duplicate_result_penalty
            - final_ignores_hierarchy_penalty
        )
    return SelectionRewardBreakdown(
        final_answer_correctness=final_correct,
        confidence_reward=confidence_reward,
        compatibility_reward=compatibility_reward,
        total_reward=total_reward,
        worker_format_penalty=worker_format_penalty,
        intermediate_final_answer_penalty=intermediate_final_answer_penalty,
        non_final_answer_containment_penalty=non_final_answer_containment_penalty,
        worker_unique_result_bonus=worker_unique_result_bonus,
        worker_duplicate_result_penalty=worker_duplicate_result_penalty,
        worker_downstream_used_bonus=worker_downstream_used_bonus,
        final_stage_usage_bonus=final_stage_usage_bonus,
        final_ignores_hierarchy_penalty=final_ignores_hierarchy_penalty,
        positive_bonus_gate=positive_bonus_gate,
        hierarchy_utilization_gate=float(hierarchy_stats["hierarchy_utilization_gate"]),
        dependency_usage_rate=float(hierarchy_stats["dependency_usage_rate"]),
        final_dependency_usage_rate=float(hierarchy_stats["final_dependency_usage_rate"]),
        final_raw_score_usage_multiplier=final_raw_score_usage_multiplier,
    )

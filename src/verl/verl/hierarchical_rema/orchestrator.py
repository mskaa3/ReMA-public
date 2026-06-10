from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Dict, List, Optional, Sequence

from .backends import (
    DecompositionRequest,
    HierarchicalBackend,
    MockHierarchicalBackend,
    RayVLLMHierarchicalBackend,
    SelectionRequest,
    TransformersHierarchicalBackend,
    WorkerExecutionRequest,
)
from .recording import RolloutRecorder
from .rewarding import (
    INTERMEDIATE_FINAL_ANSWER_PENALTY,
    NON_FINAL_ANSWER_CONTAINMENT_PENALTY,
    WORKER_INVALID_RESULT_PENALTY,
    WorkerPerformanceMemory,
    build_selection_reward,
    group_relative_advantages,
)
from .schema import (
    AlternatingPhase,
    ControllerPolicyConfig,
    ControllerTrainingSample,
    DecompositionCandidate,
    DecompositionRollout,
    HFBackendConfig,
    HierarchicalTrainingBatch,
    RewardWeights,
    RolloutLoggingConfig,
    RolloutConfig,
    SelectionCandidate,
    SelectionRollout,
    TaskExample,
    TaskRollout,
    TrainingMode,
    TrainingScheduleConfig,
    VLLMBackendConfig,
    WorkerExecution,
    WorkerPoolConfig,
)
from .structured import format_decomposition_plan, format_selection_plan


@dataclass
class _SelectionExecutionState:
    task_index: int
    decomposition_index: int
    selection_index: int
    task: TaskExample
    decomposition: DecompositionCandidate
    selection: SelectionCandidate
    topo_order: List[str]
    next_node_index: int = 0
    outputs: Dict[str, str] = field(default_factory=dict)
    executions: List[WorkerExecution] = field(default_factory=list)


class HierarchicalReMAOrchestrator:
    def __init__(
        self,
        backend: HierarchicalBackend,
        reward_weights: RewardWeights,
        worker_memory: WorkerPerformanceMemory,
        controller_format_retry_penalty: float = 0.0,
        controller_format_fallback_penalty: float = 0.0,
        selector_partial_completion_penalty: float = 0.0,
        decomposer_reward_aggregation: str = "best",
        decomposer_no_correct_selection_scale: float = 0.25,
        track_workers_history: bool = True,
        train_worker_model: bool = False,
        min_worker_grpo_group_size: int = 3,
    ) -> None:
        self.backend = backend
        self.reward_weights = reward_weights
        self.worker_memory = worker_memory
        self.controller_format_retry_penalty = max(float(controller_format_retry_penalty), 0.0)
        self.controller_format_fallback_penalty = max(float(controller_format_fallback_penalty), 0.0)
        self.selector_partial_completion_penalty = max(
            float(selector_partial_completion_penalty),
            0.0,
        )
        normalized_aggregation = str(decomposer_reward_aggregation).strip().lower()
        if normalized_aggregation not in {"mean", "best"}:
            raise ValueError(
                "decomposer_reward_aggregation must be one of: mean, best"
            )
        self.decomposer_reward_aggregation = normalized_aggregation
        self.decomposer_no_correct_selection_scale = min(
            max(float(decomposer_no_correct_selection_scale), 0.0),
            1.0,
        )
        self.track_workers_history = bool(track_workers_history)
        self.train_worker_model = bool(train_worker_model)
        self.min_worker_grpo_group_size = max(int(min_worker_grpo_group_size), 1)

    @staticmethod
    def _canonical_decomposition_completion(candidate: DecompositionCandidate) -> str:
        return format_decomposition_plan(candidate)

    @staticmethod
    def _canonical_selection_completion(
        candidate: SelectionCandidate,
        decomposition: DecompositionCandidate,
    ) -> str:
        return format_selection_plan(
            candidate,
            node_order=[node.node_id for node in decomposition.nodes],
        )

    @staticmethod
    def _normalized_instruction_key(instruction: str) -> str:
        normalized = re.sub(r"\s+", " ", str(instruction or "").strip().lower())
        return normalized or "unknown_instruction"

    @staticmethod
    def _worker_training_reward(selection: SelectionRollout, execution: WorkerExecution) -> float:
        reward = float(selection.reward.total_reward)
        if execution.invalid_reason:
            reward -= WORKER_INVALID_RESULT_PENALTY
        if execution.final_answer_leak:
            reward -= INTERMEDIATE_FINAL_ANSWER_PENALTY
        elif execution.answer_containment:
            reward -= NON_FINAL_ANSWER_CONTAINMENT_PENALTY
        return reward

    def _aggregate_decomposition_selection_reward(
        self,
        selection_rollouts: Sequence[SelectionRollout],
        selection_training_rewards: Sequence[float],
    ) -> float:
        if not selection_training_rewards:
            return 0.0
        if self.decomposer_reward_aggregation == "best":
            aggregated_reward = max(selection_training_rewards)
        else:
            aggregated_reward = sum(selection_training_rewards) / max(
                len(selection_training_rewards), 1
            )

        has_correct_selection = any(
            selection.reward.final_answer_correctness > 0.0
            for selection in selection_rollouts
        )
        if not has_correct_selection:
            positive_reward = max(aggregated_reward, 0.0)
            negative_reward = min(aggregated_reward, 0.0)
            aggregated_reward = (
                negative_reward
                + positive_reward * self.decomposer_no_correct_selection_scale
            )
        return aggregated_reward

    @staticmethod
    def _raw_controller_validation(payload: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(payload, dict):
            return {}
        validation = payload.get("validation")
        if isinstance(validation, dict):
            return validation
        return {}

    @staticmethod
    def _controller_validation_info(payload: Dict[str, object]) -> Dict[str, object]:
        validation = HierarchicalReMAOrchestrator._raw_controller_validation(payload)
        if not validation:
            return {}
        errors_before_success = validation.get("errors_before_success")
        errors = validation.get("errors")
        attempt_raw = validation.get("attempt", 0)
        try:
            attempt = max(int(attempt_raw), 0)
        except (TypeError, ValueError):
            attempt = 0
        return {
            "backend": validation.get("backend"),
            "attempt": attempt,
            "fallback_used": bool(validation.get("fallback_used")),
            "batch_repair_fallback": bool(validation.get("batch_repair_fallback")),
            "partial_completion_used": bool(validation.get("partial_completion_used")),
            "unparseable_batch_output": bool(validation.get("unparseable_batch_output")),
            "num_errors_before_success": len(errors_before_success) if isinstance(errors_before_success, list) else 0,
            "num_errors": len(errors) if isinstance(errors, list) else 0,
        }

    def _controller_format_penalty(self, payload: Dict[str, object], *, role: str) -> float:
        validation = self._raw_controller_validation(payload)
        if not validation:
            return 0.0

        fallback_used = bool(validation.get("fallback_used"))
        if fallback_used:
            return self.controller_format_fallback_penalty

        attempt_raw = validation.get("attempt", 0)
        try:
            attempt_count = max(int(attempt_raw), 0)
        except (TypeError, ValueError):
            attempt_count = 0

        errors_before_success = validation.get("errors_before_success")
        if isinstance(errors_before_success, list):
            attempt_count = max(attempt_count, len(errors_before_success))
        if validation.get("batch_repair_fallback"):
            attempt_count = max(attempt_count, 1)

        penalty = 0.0
        if attempt_count > 0:
            penalty += attempt_count * self.controller_format_retry_penalty
        if role == "selector" and validation.get("partial_completion_used"):
            penalty += self.selector_partial_completion_penalty
        return penalty

    def _format_adjusted_reward(
        self,
        raw_reward: float,
        payload: Dict[str, object],
        *,
        role: str,
    ) -> float:
        return raw_reward - self._controller_format_penalty(payload, role=role)

    def run_task(
        self,
        task: TaskExample,
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        rollout_config: RolloutConfig,
        schedule: TrainingScheduleConfig,
        progress_label: str | None = None,
        update_worker_memory: bool = True,
    ) -> TaskRollout:
        return self.run_tasks(
            tasks=[task],
            worker_pool=worker_pool,
            policy_config=policy_config,
            rollout_config=rollout_config,
            schedule=schedule,
            progress_label=progress_label,
            update_worker_memory=update_worker_memory,
        )[0]

    def run_tasks(
        self,
        tasks: Sequence[TaskExample],
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        rollout_config: RolloutConfig,
        schedule: TrainingScheduleConfig,
        progress_label: str | None = None,
        update_worker_memory: bool = True,
    ) -> List[TaskRollout]:
        if not tasks:
            return []

        frozen_worker_performance = (
            self.worker_memory.snapshot(worker_pool)
            if self.track_workers_history
            else {}
        )
        num_decompositions, num_selections = self._effective_rollout_counts(
            rollout_config=rollout_config,
            schedule=schedule,
        )
        progress_suffix = f" epoch_tasks={progress_label}" if progress_label else ""
        print(
            f"[hierarchical-rema][rollout] stage=decomposer "
            f"tasks={len(tasks)}{progress_suffix} decompositions_per_task={num_decompositions} "
            f"requests={len(tasks) * num_decompositions}"
        )

        decomposition_requests: List[DecompositionRequest] = []
        decomposition_metadata: List[tuple[int, int]] = []
        for task_index, task in enumerate(tasks):
            for decomposition_index in range(num_decompositions):
                decomposition_requests.append(
                    DecompositionRequest(
                        task=task,
                        worker_pool=worker_pool,
                        policy_config=policy_config,
                        rollout_config=rollout_config,
                        decomposition_index=decomposition_index,
                        worker_performance=frozen_worker_performance,
                    )
                )
                decomposition_metadata.append((task_index, decomposition_index))
        decomposition_candidates = self.backend.sample_decompositions_batch(decomposition_requests)
        task_decompositions: List[List[DecompositionCandidate]] = [[] for _ in tasks]
        for (task_index, _), decomposition in zip(decomposition_metadata, decomposition_candidates):
            decomposition.topological_order()
            task_decompositions[task_index].append(decomposition)

        print(
            f"[hierarchical-rema][rollout] stage=selector "
            f"tasks={len(tasks)}{progress_suffix} selections_per_decomposition={num_selections} "
            f"requests={len(tasks) * num_decompositions * num_selections}"
        )
        selection_requests: List[SelectionRequest] = []
        request_metadata: List[tuple[int, int, int]] = []
        for task_index, task in enumerate(tasks):
            for decomposition_index, decomposition in enumerate(task_decompositions[task_index]):
                for selection_index in range(num_selections):
                    selection_requests.append(
                        SelectionRequest(
                            task=task,
                            decomposition=decomposition,
                            worker_pool=worker_pool,
                            policy_config=policy_config,
                            selection_index=selection_index,
                            worker_performance=frozen_worker_performance,
                        )
                    )
                    request_metadata.append((task_index, decomposition_index, selection_index))

        selection_candidates = self.backend.sample_selections_batch(selection_requests)
        selection_states: List[_SelectionExecutionState] = []
        for (task_index, decomposition_index, selection_index), selection in zip(
            request_metadata,
            selection_candidates,
        ):
            decomposition = task_decompositions[task_index][decomposition_index]
            selection_states.append(
                _SelectionExecutionState(
                    task_index=task_index,
                    decomposition_index=decomposition_index,
                    selection_index=selection_index,
                    task=tasks[task_index],
                    decomposition=decomposition,
                    selection=selection,
                    topo_order=decomposition.topological_order(),
                )
            )

        selection_rollout_map = self._execute_selections_batch(
            selection_states=selection_states,
            worker_pool=worker_pool,
            progress_label=progress_label,
        )

        task_rollouts: List[TaskRollout] = []
        for task_index, task in enumerate(tasks):
            decomposition_rollouts: List[DecompositionRollout] = []
            for decomposition_index, decomposition in enumerate(task_decompositions[task_index]):
                selection_rollouts = [
                    selection_rollout_map[(task_index, decomposition_index, selection_index)]
                    for selection_index in range(num_selections)
                ]
                selection_training_rewards = [
                    self._format_adjusted_reward(
                        selection.reward.total_reward,
                        selection.selection.raw_payload,
                        role="selector",
                    )
                    for selection in selection_rollouts
                ]
                selection_advantages = group_relative_advantages(
                    selection_training_rewards
                )
                for selection_rollout, advantage in zip(selection_rollouts, selection_advantages):
                    selection_rollout.selector_advantage = advantage

                base_decomposition_reward = self._aggregate_decomposition_selection_reward(
                    selection_rollouts=selection_rollouts,
                    selection_training_rewards=selection_training_rewards,
                )
                decomposition_reward = base_decomposition_reward - decomposition.soft_penalty
                decomposition_rollouts.append(
                    DecompositionRollout(
                        decomposition=decomposition,
                        selections=selection_rollouts,
                        base_decomposition_reward=base_decomposition_reward,
                        decomposition_reward=decomposition_reward,
                    )
                )

            decomposition_training_rewards = [
                self._format_adjusted_reward(
                    decomposition_rollout.decomposition_reward,
                    decomposition_rollout.decomposition.raw_payload,
                    role="decomposer",
                )
                for decomposition_rollout in decomposition_rollouts
            ]
            decomposition_advantages = group_relative_advantages(decomposition_training_rewards)
            for decomposition_rollout, advantage in zip(
                decomposition_rollouts,
                decomposition_advantages,
            ):
                decomposition_rollout.decomposer_advantage = advantage

            if update_worker_memory and self.track_workers_history:
                self._update_worker_memory(task, decomposition_rollouts)
            task_rollouts.append(
                self._finalize_task_rollout(
                    task=task,
                    worker_pool=worker_pool,
                    policy_config=policy_config,
                    rollout_config=rollout_config,
                    schedule=schedule,
                    decomposition_rollouts=decomposition_rollouts,
                )
            )

        return task_rollouts

    def _effective_rollout_counts(
        self,
        rollout_config: RolloutConfig,
        schedule: TrainingScheduleConfig,
    ) -> tuple[int, int]:
        num_decompositions = rollout_config.num_decompositions
        num_selections = rollout_config.num_selections_per_decomposition
        if schedule.mode == TrainingMode.ALTERNATING:
            if schedule.alternating_phase == AlternatingPhase.SELECTOR:
                num_decompositions = 1
            else:
                num_selections = 1
        return num_decompositions, num_selections

    def _finalize_task_rollout(
        self,
        task: TaskExample,
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        rollout_config: RolloutConfig,
        schedule: TrainingScheduleConfig,
        decomposition_rollouts: List[DecompositionRollout],
    ) -> TaskRollout:
        training_batch = self._build_training_batch(
            task=task,
            worker_pool=worker_pool,
            policy_config=policy_config,
            schedule=schedule,
            decompositions=decomposition_rollouts,
        )
        return TaskRollout(
            task=task,
            policy_config=policy_config,
            rollout_config=rollout_config,
            schedule=schedule,
            decompositions=decomposition_rollouts,
            training_batch=training_batch,
        )

    def _execute_selections_batch(
        self,
        selection_states: Sequence[_SelectionExecutionState],
        worker_pool: WorkerPoolConfig,
        progress_label: str | None = None,
    ) -> Dict[tuple[int, int, int], SelectionRollout]:
        worker_map = worker_pool.workers_by_id()
        node_maps = {
            (state.task_index, state.decomposition_index): state.decomposition.nodes_by_id()
            for state in selection_states
        }
        active_states = list(selection_states)
        frontier_step = 0
        total_states = len(active_states)
        total_tasks = len({state.task_index for state in active_states})
        max_frontier_steps = max((len(state.topo_order) for state in active_states), default=0)
        progress_suffix = f" epoch_tasks={progress_label}" if progress_label else ""
        while True:
            worker_requests: List[WorkerExecutionRequest] = []
            request_states: List[_SelectionExecutionState] = []
            for state in active_states:
                if state.next_node_index >= len(state.topo_order):
                    continue
                node_id = state.topo_order[state.next_node_index]
                node = node_maps[(state.task_index, state.decomposition_index)][node_id]
                assignment = state.selection.assignment_for(node_id)
                worker = worker_map[assignment.worker_id]
                dependency_outputs = {
                    dependency: state.outputs[dependency]
                    for dependency in node.dependencies
                }
                worker_requests.append(
                    WorkerExecutionRequest(
                        task=state.task,
                        decomposition=state.decomposition,
                        node=node,
                        worker=worker,
                        dependency_outputs=dependency_outputs,
                        compatibility=assignment.compatibility,
                    )
                )
                request_states.append(state)

            if not worker_requests:
                break

            frontier_step += 1
            print(
                f"[hierarchical-rema][rollout] stage=workers_dispatch "
                f"{progress_suffix.lstrip()} frontier={frontier_step}/{max_frontier_steps} "
                f"requests={len(worker_requests)}"
            )
            worker_executions = self.backend.execute_workers_batch(worker_requests)
            for state, execution in zip(request_states, worker_executions):
                state.executions.append(execution)
                state.outputs[execution.node_id] = execution.output_text
                state.next_node_index += 1

            completed_states = sum(
                1 for state in active_states if state.next_node_index >= len(state.topo_order)
            )
            per_task_counts: Dict[int, List[int]] = {}
            for state in active_states:
                bucket = per_task_counts.setdefault(state.task_index, [0, 0])
                bucket[1] += 1
                if state.next_node_index >= len(state.topo_order):
                    bucket[0] += 1
            completed_tasks = sum(1 for done, total in per_task_counts.values() if done == total)
            print(
                f"[hierarchical-rema][rollout] stage=workers "
                f"{progress_suffix.lstrip()} frontier={frontier_step}/{max_frontier_steps} "
                f"batch_tasks_done={completed_tasks}/{total_tasks} "
                f"states_done={completed_states}/{total_states} "
                f"requests={len(worker_requests)}"
            )

        selection_rollout_map: Dict[tuple[int, int, int], SelectionRollout] = {}
        for state in active_states:
            final_answer = state.outputs[state.decomposition.final_node_id]
            reward = build_selection_reward(
                final_answer=final_answer,
                ground_truth=state.task.ground_truth,
                executions=state.executions,
                weights=self.reward_weights,
                task_metadata=state.task.metadata,
                final_node_id=state.decomposition.final_node_id,
            )
            selection_rollout_map[
                (state.task_index, state.decomposition_index, state.selection_index)
            ] = SelectionRollout(
                selection=state.selection,
                executions=state.executions,
                final_answer=final_answer,
                reward=reward,
            )
        return selection_rollout_map

    def _execute_selection(
        self,
        task: TaskExample,
        decomposition: DecompositionCandidate,
        selection: SelectionCandidate,
        worker_pool: WorkerPoolConfig,
    ) -> SelectionRollout:
        worker_map = worker_pool.workers_by_id()
        node_map = decomposition.nodes_by_id()
        topo_order = decomposition.topological_order()
        outputs: Dict[str, str] = {}
        executions = []
        for node_id in topo_order:
            node = node_map[node_id]
            assignment = selection.assignment_for(node_id)
            worker = worker_map[assignment.worker_id]
            dependency_outputs = {dep: outputs[dep] for dep in node.dependencies}
            execution = self.backend.execute_worker(
                task=task,
                decomposition=decomposition,
                node=node,
                worker=worker,
                dependency_outputs=dependency_outputs,
                compatibility=assignment.compatibility,
            )
            executions.append(execution)
            outputs[node_id] = execution.output_text

        final_answer = outputs[decomposition.final_node_id]
        reward = build_selection_reward(
            final_answer=final_answer,
            ground_truth=task.ground_truth,
            executions=executions,
            weights=self.reward_weights,
            task_metadata=task.metadata,
            final_node_id=decomposition.final_node_id,
        )
        return SelectionRollout(
            selection=selection,
            executions=executions,
            final_answer=final_answer,
            reward=reward,
        )

    def _update_worker_memory(
        self,
        task: TaskExample,
        decompositions: List[DecompositionRollout],
    ) -> None:
        for decomposition_rollout in decompositions:
            for selection_rollout in decomposition_rollout.selections:
                for execution in selection_rollout.executions:
                    self.worker_memory.record_execution(
                        task_id=task.task_id,
                        execution=execution,
                        selection_reward=selection_rollout.reward,
                        reward_weights=self.reward_weights,
                    )

    def _build_training_batch(
        self,
        task: TaskExample,
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        schedule: TrainingScheduleConfig,
        decompositions: List[DecompositionRollout],
    ) -> HierarchicalTrainingBatch:
        decomposer_samples: List[ControllerTrainingSample] = []
        selector_samples: List[ControllerTrainingSample] = []
        worker_samples: List[ControllerTrainingSample] = []
        worker_grpo_stats: Dict[str, float | int] = {
            "min_group_size": self.min_worker_grpo_group_size,
            "num_groups_total": 0,
            "num_groups_used": 0,
            "num_groups_skipped": 0,
            "num_samples_total": 0,
            "num_samples_used": 0,
            "num_samples_skipped": 0,
            "mean_group_size_used": 0.0,
        }

        include_decomposer = schedule.mode == TrainingMode.JOINT or (
            schedule.mode == TrainingMode.ALTERNATING
            and schedule.alternating_phase == AlternatingPhase.DECOMPOSER
        )
        include_selector = schedule.mode == TrainingMode.JOINT or (
            schedule.mode == TrainingMode.ALTERNATING
            and schedule.alternating_phase == AlternatingPhase.SELECTOR
        )

        if include_decomposer:
            for decomposition_rollout in decompositions:
                adjusted_reward = self._format_adjusted_reward(
                    decomposition_rollout.decomposition_reward,
                    decomposition_rollout.decomposition.raw_payload,
                    role="decomposer",
                )
                format_penalty = decomposition_rollout.decomposition_reward - adjusted_reward
                adjusted_advantage = decomposition_rollout.decomposer_advantage
                decomposer_samples.append(
                    ControllerTrainingSample(
                        role="decomposer",
                        policy_id=policy_config.policy_id("decomposer"),
                        group_id=task.task_id,
                        prompt_text=decomposition_rollout.decomposition.raw_payload["controller_prompt"],
                        completion_text=self._canonical_decomposition_completion(
                            decomposition_rollout.decomposition
                        ),
                        reward=adjusted_reward,
                        advantage=adjusted_advantage,
                        metadata={
                            "model_path": policy_config.model_for_role("decomposer"),
                            "parameter_sharing": policy_config.parameter_sharing,
                            "reward_before_format_penalty": decomposition_rollout.decomposition_reward,
                            "advantage_used_for_training": adjusted_advantage,
                            "format_penalty": format_penalty,
                            "format_validation": self._controller_validation_info(
                                decomposition_rollout.decomposition.raw_payload
                            ),
                            "training_target_source": "canonical_decomposition_plan",
                        },
                    )
                )

        if include_selector:
            for decomposition_rollout in decompositions:
                for selection_rollout in decomposition_rollout.selections:
                    adjusted_reward = self._format_adjusted_reward(
                        selection_rollout.reward.total_reward,
                        selection_rollout.selection.raw_payload,
                        role="selector",
                    )
                    format_penalty = selection_rollout.reward.total_reward - adjusted_reward
                    adjusted_advantage = selection_rollout.selector_advantage
                    selector_samples.append(
                        ControllerTrainingSample(
                            role="selector",
                            policy_id=policy_config.policy_id("selector"),
                            group_id=decomposition_rollout.decomposition.decomposition_id,
                            prompt_text=selection_rollout.selection.raw_payload["controller_prompt"],
                            completion_text=self._canonical_selection_completion(
                                selection_rollout.selection,
                                decomposition_rollout.decomposition,
                            ),
                            reward=adjusted_reward,
                            advantage=adjusted_advantage,
                            metadata={
                                "model_path": policy_config.model_for_role("selector"),
                                "parameter_sharing": policy_config.parameter_sharing,
                                "reward_before_format_penalty": selection_rollout.reward.total_reward,
                                "advantage_used_for_training": adjusted_advantage,
                                "format_penalty": format_penalty,
                                "format_validation": self._controller_validation_info(
                                    selection_rollout.selection.raw_payload
                                ),
                                "training_target_source": "canonical_selection_plan",
                            },
                        )
                    )

        if self.train_worker_model:
            worker_lookup = worker_pool.workers_by_id()
            worker_groups: Dict[str, List[tuple[SelectionRollout, WorkerExecution, str, float, str]]] = {}
            for decomposition_rollout in decompositions:
                node_map = decomposition_rollout.decomposition.nodes_by_id()
                for selection_rollout in decomposition_rollout.selections:
                    for execution in selection_rollout.executions:
                        prompt_text = str(execution.worker_prompt or "").strip()
                        completion_text = str(execution.raw_output_text or "").strip()
                        if not prompt_text or not completion_text:
                            continue
                        worker_spec = worker_lookup.get(execution.worker_id)
                        model_path = (
                            worker_spec.base_model_path
                            if worker_spec is not None and worker_spec.base_model_path
                            else worker_pool.base_model_path
                        )
                        reward = self._worker_training_reward(selection_rollout, execution)
                        node = node_map.get(execution.node_id)
                        normalized_instruction = self._normalized_instruction_key(
                            node.instruction if node is not None else execution.node_id
                        )
                        worker_group_key = f"{execution.worker_id}::{normalized_instruction}"
                        worker_groups.setdefault(worker_group_key, []).append(
                            (
                                selection_rollout,
                                execution,
                                model_path or "",
                                reward,
                                normalized_instruction,
                            )
                        )

            used_group_sizes: List[int] = []
            for worker_group_key, grouped_payloads in worker_groups.items():
                group_size = len(grouped_payloads)
                worker_grpo_stats["num_groups_total"] += 1
                worker_grpo_stats["num_samples_total"] += group_size
                if group_size < self.min_worker_grpo_group_size:
                    worker_grpo_stats["num_groups_skipped"] += 1
                    worker_grpo_stats["num_samples_skipped"] += group_size
                    continue
                worker_grpo_stats["num_groups_used"] += 1
                worker_grpo_stats["num_samples_used"] += group_size
                used_group_sizes.append(group_size)
                grouped_rewards = [payload[3] for payload in grouped_payloads]
                grouped_advantages = group_relative_advantages(grouped_rewards)
                worker_id = grouped_payloads[0][1].worker_id
                normalized_instruction = grouped_payloads[0][4]
                instruction_hash = hashlib.sha1(
                    normalized_instruction.encode("utf-8")
                ).hexdigest()[:12]
                group_id = (
                    f"task:{task.task_id}:worker:{worker_id}:"
                    f"instr:{instruction_hash}"
                )
                for advantage, payload in zip(grouped_advantages, grouped_payloads):
                    selection_rollout, execution, model_path, reward, normalized_instruction = payload
                    worker_samples.append(
                        ControllerTrainingSample(
                            role="worker",
                            policy_id="shared_worker",
                            group_id=group_id,
                            prompt_text=execution.worker_prompt,
                            completion_text=execution.raw_output_text,
                            reward=reward,
                            advantage=advantage,
                            metadata={
                                "model_path": model_path,
                                "worker_id": execution.worker_id,
                                "node_id": execution.node_id,
                                "selection_id": selection_rollout.selection.selection_id,
                                "advantage_group_size": group_size,
                                "advantage_group_kind": "worker_id_and_normalized_instruction_within_task",
                                "normalized_node_instruction": normalized_instruction,
                                "invalid_reason": execution.invalid_reason,
                                "final_answer_leak": execution.final_answer_leak,
                                "answer_containment": execution.answer_containment,
                                "reward_before_local_penalties": selection_rollout.reward.total_reward,
                                "final_answer_correctness": selection_rollout.reward.final_answer_correctness,
                                "worker_format_penalty": WORKER_INVALID_RESULT_PENALTY if execution.invalid_reason else 0.0,
                                "intermediate_final_answer_penalty": (
                                    INTERMEDIATE_FINAL_ANSWER_PENALTY if execution.final_answer_leak else 0.0
                                ),
                                "non_final_answer_containment_penalty": (
                                    NON_FINAL_ANSWER_CONTAINMENT_PENALTY
                                    if execution.answer_containment and not execution.final_answer_leak
                                    else 0.0
                                ),
                            },
                        )
                    )
            if used_group_sizes:
                worker_grpo_stats["mean_group_size_used"] = (
                    sum(used_group_sizes) / len(used_group_sizes)
                )

        frozen_roles: List[str] = []
        if schedule.mode == TrainingMode.ALTERNATING:
            if schedule.alternating_phase == AlternatingPhase.SELECTOR:
                frozen_roles.append("decomposer")
            else:
                frozen_roles.append("selector")

        return HierarchicalTrainingBatch(
            decomposer_samples=decomposer_samples,
            selector_samples=selector_samples,
            worker_samples=worker_samples,
            frozen_roles=frozen_roles,
            worker_grpo_stats=worker_grpo_stats,
        )


@dataclass
class HierarchicalGRPOTrainer:
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    worker_memory: WorkerPerformanceMemory = field(default_factory=WorkerPerformanceMemory)
    backend_type: str = "mock"
    hf_backend_config: HFBackendConfig = field(default_factory=HFBackendConfig)
    vllm_backend_config: VLLMBackendConfig = field(default_factory=VLLMBackendConfig)
    rollout_recorder: Optional[RolloutRecorder] = None
    rollout_logging_config: Optional[RolloutLoggingConfig] = None
    backend: Optional[HierarchicalBackend] = None
    controller_format_retry_penalty: float = 0.0
    controller_format_fallback_penalty: float = 0.0
    selector_partial_completion_penalty: float = 0.0
    decomposer_reward_aggregation: str = "best"
    decomposer_no_correct_selection_scale: float = 0.25
    track_workers_history: bool = True
    train_worker_model: bool = False
    min_worker_grpo_group_size: int = 3

    def __post_init__(self) -> None:
        if self.backend is None:
            if self.backend_type == "mock":
                self.backend = MockHierarchicalBackend(worker_memory=self.worker_memory)
            elif self.backend_type == "hf":
                self.backend = TransformersHierarchicalBackend(config=self.hf_backend_config)
            elif self.backend_type == "vllm":
                self.backend = RayVLLMHierarchicalBackend(config=self.vllm_backend_config)
            else:
                raise ValueError(f"Unknown backend_type: {self.backend_type}")

        if self.rollout_recorder is None and self.rollout_logging_config is not None:
            self.rollout_recorder = RolloutRecorder(self.rollout_logging_config)

        self.orchestrator = HierarchicalReMAOrchestrator(
            backend=self.backend,
            reward_weights=self.reward_weights,
            worker_memory=self.worker_memory,
            controller_format_retry_penalty=self.controller_format_retry_penalty,
            controller_format_fallback_penalty=self.controller_format_fallback_penalty,
            selector_partial_completion_penalty=self.selector_partial_completion_penalty,
            decomposer_reward_aggregation=self.decomposer_reward_aggregation,
            decomposer_no_correct_selection_scale=self.decomposer_no_correct_selection_scale,
            track_workers_history=self.track_workers_history,
            train_worker_model=self.train_worker_model,
            min_worker_grpo_group_size=self.min_worker_grpo_group_size,
        )
        self._current_phase = AlternatingPhase.SELECTOR

    def run(
        self,
        task: TaskExample,
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        rollout_config: RolloutConfig,
        schedule: TrainingScheduleConfig,
        progress_label: str | None = None,
        update_worker_memory: bool = True,
    ) -> TaskRollout:
        return self.run_many(
            tasks=[task],
            worker_pool=worker_pool,
            policy_config=policy_config,
            rollout_config=rollout_config,
            schedule=schedule,
            progress_label=progress_label,
            update_worker_memory=update_worker_memory,
        )[0]

    def run_many(
        self,
        tasks: Sequence[TaskExample],
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        rollout_config: RolloutConfig,
        schedule: TrainingScheduleConfig,
        progress_label: str | None = None,
        update_worker_memory: bool = True,
    ) -> List[TaskRollout]:
        rollouts = self.orchestrator.run_tasks(
            tasks=tasks,
            worker_pool=worker_pool,
            policy_config=policy_config,
            rollout_config=rollout_config,
            schedule=schedule,
            progress_label=progress_label,
            update_worker_memory=update_worker_memory,
        )
        if self.rollout_recorder is not None:
            for rollout in rollouts:
                self.rollout_recorder.record_task_rollout(rollout)
        return rollouts

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()

    def next_alternating_schedule(self) -> TrainingScheduleConfig:
        schedule = TrainingScheduleConfig(
            mode=TrainingMode.ALTERNATING,
            alternating_phase=self._current_phase,
        )
        self._current_phase = (
            AlternatingPhase.DECOMPOSER
            if self._current_phase == AlternatingPhase.SELECTOR
            else AlternatingPhase.SELECTOR
        )
        return schedule

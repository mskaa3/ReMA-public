from __future__ import annotations

from dataclasses import dataclass, field
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
from .rewarding import WorkerPerformanceMemory, build_selection_reward, group_relative_advantages
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
    ) -> None:
        self.backend = backend
        self.reward_weights = reward_weights
        self.worker_memory = worker_memory

    def run_task(
        self,
        task: TaskExample,
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        rollout_config: RolloutConfig,
        schedule: TrainingScheduleConfig,
    ) -> TaskRollout:
        return self.run_tasks(
            tasks=[task],
            worker_pool=worker_pool,
            policy_config=policy_config,
            rollout_config=rollout_config,
            schedule=schedule,
        )[0]

    def run_tasks(
        self,
        tasks: Sequence[TaskExample],
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        rollout_config: RolloutConfig,
        schedule: TrainingScheduleConfig,
    ) -> List[TaskRollout]:
        if not tasks:
            return []

        frozen_worker_performance = self.worker_memory.snapshot(worker_pool)
        num_decompositions, num_selections = self._effective_rollout_counts(
            rollout_config=rollout_config,
            schedule=schedule,
        )
        print(
            f"[hierarchical-rema][rollout] stage=decomposer "
            f"tasks={len(tasks)} decompositions_per_task={num_decompositions} "
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
            f"tasks={len(tasks)} selections_per_decomposition={num_selections} "
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
        )

        task_rollouts: List[TaskRollout] = []
        for task_index, task in enumerate(tasks):
            decomposition_rollouts: List[DecompositionRollout] = []
            for decomposition_index, decomposition in enumerate(task_decompositions[task_index]):
                selection_rollouts = [
                    selection_rollout_map[(task_index, decomposition_index, selection_index)]
                    for selection_index in range(num_selections)
                ]
                selection_advantages = group_relative_advantages(
                    [selection.reward.total_reward for selection in selection_rollouts]
                )
                for selection_rollout, advantage in zip(selection_rollouts, selection_advantages):
                    selection_rollout.selector_advantage = advantage

                base_decomposition_reward = sum(
                    selection.reward.total_reward for selection in selection_rollouts
                ) / max(len(selection_rollouts), 1)
                decomposition_reward = base_decomposition_reward - decomposition.soft_penalty
                decomposition_rollouts.append(
                    DecompositionRollout(
                        decomposition=decomposition,
                        selections=selection_rollouts,
                        base_decomposition_reward=base_decomposition_reward,
                        decomposition_reward=decomposition_reward,
                    )
                )

            decomposition_advantages = group_relative_advantages(
                [decomposition.decomposition_reward for decomposition in decomposition_rollouts]
            )
            for decomposition_rollout, advantage in zip(
                decomposition_rollouts,
                decomposition_advantages,
            ):
                decomposition_rollout.decomposer_advantage = advantage

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
    ) -> Dict[tuple[int, int, int], SelectionRollout]:
        worker_map = worker_pool.workers_by_id()
        node_maps = {
            (state.task_index, state.decomposition_index): state.decomposition.nodes_by_id()
            for state in selection_states
        }
        active_states = list(selection_states)
        frontier_step = 0
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
                f"[hierarchical-rema][rollout] stage=workers "
                f"frontier_step={frontier_step} active_states={len(active_states)} "
                f"requests={len(worker_requests)}"
            )

            worker_executions = self.backend.execute_workers_batch(worker_requests)
            for state, execution in zip(request_states, worker_executions):
                state.executions.append(execution)
                state.outputs[execution.node_id] = execution.output_text
                state.next_node_index += 1

        selection_rollout_map: Dict[tuple[int, int, int], SelectionRollout] = {}
        for state in active_states:
            final_answer = state.outputs[state.decomposition.final_node_id]
            reward = build_selection_reward(
                final_answer=final_answer,
                ground_truth=state.task.ground_truth,
                executions=state.executions,
                weights=self.reward_weights,
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
                decomposer_samples.append(
                    ControllerTrainingSample(
                        role="decomposer",
                        policy_id=policy_config.policy_id("decomposer"),
                        group_id=task.task_id,
                        prompt_text=decomposition_rollout.decomposition.raw_payload["controller_prompt"],
                        completion_text=decomposition_rollout.decomposition.raw_text,
                        reward=decomposition_rollout.decomposition_reward,
                        advantage=decomposition_rollout.decomposer_advantage,
                        metadata={
                            "model_path": policy_config.model_for_role("decomposer"),
                            "parameter_sharing": policy_config.parameter_sharing,
                        },
                    )
                )

        if include_selector:
            for decomposition_rollout in decompositions:
                for selection_rollout in decomposition_rollout.selections:
                    selector_samples.append(
                        ControllerTrainingSample(
                            role="selector",
                            policy_id=policy_config.policy_id("selector"),
                            group_id=decomposition_rollout.decomposition.decomposition_id,
                            prompt_text=selection_rollout.selection.raw_payload["controller_prompt"],
                            completion_text=selection_rollout.selection.raw_text,
                            reward=selection_rollout.reward.total_reward,
                            advantage=selection_rollout.selector_advantage,
                            metadata={
                                "model_path": policy_config.model_for_role("selector"),
                                "parameter_sharing": policy_config.parameter_sharing,
                            },
                        )
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
            frozen_roles=frozen_roles,
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
        )
        self._current_phase = AlternatingPhase.SELECTOR

    def run(
        self,
        task: TaskExample,
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        rollout_config: RolloutConfig,
        schedule: TrainingScheduleConfig,
    ) -> TaskRollout:
        return self.run_many(
            tasks=[task],
            worker_pool=worker_pool,
            policy_config=policy_config,
            rollout_config=rollout_config,
            schedule=schedule,
        )[0]

    def run_many(
        self,
        tasks: Sequence[TaskExample],
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        rollout_config: RolloutConfig,
        schedule: TrainingScheduleConfig,
    ) -> List[TaskRollout]:
        rollouts = self.orchestrator.run_tasks(
            tasks=tasks,
            worker_pool=worker_pool,
            policy_config=policy_config,
            rollout_config=rollout_config,
            schedule=schedule,
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

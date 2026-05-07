from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from graphlib import CycleError, TopologicalSorter
from typing import Any, Dict, List, Optional, Sequence


class TrainingMode(str, Enum):
    JOINT = "joint"
    ALTERNATING = "alternating"


class AlternatingPhase(str, Enum):
    SELECTOR = "selector"
    DECOMPOSER = "decomposer"


@dataclass
class ControllerPolicyConfig:
    parameter_sharing: bool = False
    shared_model_path: Optional[str] = None
    decomposer_model_path: Optional[str] = None
    selector_model_path: Optional[str] = None

    def model_for_role(self, role: str) -> Optional[str]:
        if self.parameter_sharing:
            return self.shared_model_path or self.decomposer_model_path or self.selector_model_path
        if role == "decomposer":
            return self.decomposer_model_path
        if role == "selector":
            return self.selector_model_path
        raise ValueError(f"Unknown controller role: {role}")

    def policy_id(self, role: str) -> str:
        if self.parameter_sharing:
            return "shared_controller"
        return f"{role}_controller"


@dataclass
class RewardWeights:
    final_answer: float = 1.0
    confidence: float = 0.2
    compatibility: float = 0.2
    entropy_cap: float = 2.0


@dataclass
class RolloutConfig:
    num_decompositions: int = 3
    num_selections_per_decomposition: int = 2
    max_nodes_per_decomposition: int = 4
    soft_max_hops: Optional[int] = None
    hard_max_hops: Optional[int] = None
    soft_hop_penalty: float = 0.1
    soft_hop_penalty_power: float = 1.0


@dataclass
class TrainingScheduleConfig:
    mode: TrainingMode = TrainingMode.JOINT
    alternating_phase: AlternatingPhase = AlternatingPhase.SELECTOR


@dataclass
class HFBackendConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    controller_max_new_tokens: int = 768
    worker_max_new_tokens: int = 256
    controller_batch_size: int = 8
    worker_batch_size: int = 16
    max_format_retries: int = 2
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True


@dataclass
class VLLMBackendConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    prompt_length: int = 2048
    controller_max_new_tokens: int = 768
    worker_max_new_tokens: int = 256
    controller_batch_size: int = 8
    worker_batch_size: int = 16
    max_format_retries: int = 2
    nnodes: int = 1
    n_gpus_per_node: int = 1
    tensor_model_parallel_size: int = 1
    gpu_memory_utilization: float = 0.5
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 1024
    max_model_len: Optional[int] = None
    dtype: str = "bfloat16"
    enforce_eager: bool = True
    free_cache_engine: bool = True
    enable_chunked_prefill: bool = True
    load_format: str = "dummy_dtensor"
    disable_log_stats: bool = True
    detokenize: bool = True
    trust_remote_code: bool = True


@dataclass
class RolloutLoggingConfig:
    output_dir: str = "outputs/hierarchical_rema"
    save_all_rollouts: bool = True
    save_best_rollouts: bool = True
    best_k: int = 10


@dataclass
class TaskExample:
    task_id: str
    prompt: str
    ground_truth: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkerSpec:
    worker_id: str
    description: str
    skills: List[str]
    system_prompt: str
    base_model_path: Optional[str] = None
    lora_adapter_path: Optional[str] = None
    trainable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkerPoolConfig:
    base_model_path: Optional[str]
    workers: List[WorkerSpec]
    enable_role_lora: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def workers_by_id(self) -> Dict[str, WorkerSpec]:
        return {worker.worker_id: worker for worker in self.workers}


@dataclass
class WorkerPerformanceSnapshot:
    worker_id: str
    ema_outcome: float
    num_assignments: int
    num_completed: int
    completion_rate: float
    num_successes: int
    success_rate: float
    average_reward: float
    average_confidence_reward: float
    average_compatibility: float
    recent_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubtaskNode:
    node_id: str
    instruction: str
    dependencies: List[str] = field(default_factory=list)
    required_skills: List[str] = field(default_factory=list)
    output_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecompositionCandidate:
    decomposition_id: str
    summary: str
    nodes: List[SubtaskNode]
    final_node_id: str
    num_hops: int = 0
    effective_num_hops: int = 0
    soft_penalty: float = 0.0
    was_hard_truncated: bool = False
    raw_text: str = ""
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    def nodes_by_id(self) -> Dict[str, SubtaskNode]:
        return {node.node_id: node for node in self.nodes}

    def topological_order(self) -> List[str]:
        node_map = self.nodes_by_id()
        graph = {node.node_id: tuple(node.dependencies) for node in self.nodes}
        if self.final_node_id not in node_map:
            raise ValueError(f"Final node {self.final_node_id} is not part of the decomposition")
        for node in self.nodes:
            for dependency in node.dependencies:
                if dependency not in node_map:
                    raise ValueError(
                        f"Node {node.node_id} depends on unknown node {dependency}"
                    )
        try:
            order = list(TopologicalSorter(graph).static_order())
        except CycleError as exc:
            raise ValueError(f"Decomposition {self.decomposition_id} is not a DAG") from exc
        return order

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decomposition_id": self.decomposition_id,
            "summary": self.summary,
            "final_node_id": self.final_node_id,
            "num_hops": self.num_hops,
            "effective_num_hops": self.effective_num_hops,
            "soft_penalty": self.soft_penalty,
            "was_hard_truncated": self.was_hard_truncated,
            "nodes": [node.to_dict() for node in self.nodes],
            "raw_text": self.raw_text,
            "raw_payload": self.raw_payload,
        }


@dataclass
class WorkerAssignment:
    node_id: str
    worker_id: str
    rationale: str
    compatibility: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectionCandidate:
    selection_id: str
    assignments: List[WorkerAssignment]
    raw_text: str = ""
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    def assignment_for(self, node_id: str) -> WorkerAssignment:
        for assignment in self.assignments:
            if assignment.node_id == node_id:
                return assignment
        raise KeyError(f"No worker assignment found for node {node_id}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection_id": self.selection_id,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "raw_text": self.raw_text,
            "raw_payload": self.raw_payload,
        }


@dataclass
class WorkerExecution:
    node_id: str
    worker_id: str
    output_text: str
    entropy: float
    confidence_reward: float
    compatibility: float
    dependency_outputs: Dict[str, str] = field(default_factory=dict)
    completed: bool = True
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectionRewardBreakdown:
    final_answer_correctness: float
    confidence_reward: float
    compatibility_reward: float
    total_reward: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectionRollout:
    selection: SelectionCandidate
    executions: List[WorkerExecution]
    final_answer: str
    reward: SelectionRewardBreakdown
    selector_advantage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection": self.selection.to_dict(),
            "executions": [execution.to_dict() for execution in self.executions],
            "final_answer": self.final_answer,
            "reward": self.reward.to_dict(),
            "selector_advantage": self.selector_advantage,
        }


@dataclass
class DecompositionRollout:
    decomposition: DecompositionCandidate
    selections: List[SelectionRollout]
    base_decomposition_reward: float
    decomposition_reward: float
    decomposer_advantage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decomposition": self.decomposition.to_dict(),
            "selections": [selection.to_dict() for selection in self.selections],
            "base_decomposition_reward": self.base_decomposition_reward,
            "decomposition_reward": self.decomposition_reward,
            "decomposer_advantage": self.decomposer_advantage,
        }


@dataclass
class ControllerTrainingSample:
    role: str
    policy_id: str
    group_id: str
    prompt_text: str
    completion_text: str
    reward: float
    advantage: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HierarchicalTrainingBatch:
    decomposer_samples: List[ControllerTrainingSample] = field(default_factory=list)
    selector_samples: List[ControllerTrainingSample] = field(default_factory=list)
    frozen_roles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decomposer_samples": [sample.to_dict() for sample in self.decomposer_samples],
            "selector_samples": [sample.to_dict() for sample in self.selector_samples],
            "frozen_roles": list(self.frozen_roles),
        }


@dataclass
class TaskRollout:
    task: TaskExample
    policy_config: ControllerPolicyConfig
    rollout_config: RolloutConfig
    schedule: TrainingScheduleConfig
    decompositions: List[DecompositionRollout]
    training_batch: HierarchicalTrainingBatch

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "policy_config": asdict(self.policy_config),
            "rollout_config": asdict(self.rollout_config),
            "schedule": {
                "mode": self.schedule.mode.value,
                "alternating_phase": self.schedule.alternating_phase.value,
            },
            "decompositions": [decomposition.to_dict() for decomposition in self.decompositions],
            "training_batch": self.training_batch.to_dict(),
        }


def average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))

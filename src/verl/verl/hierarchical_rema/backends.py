from __future__ import annotations

import importlib.util
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .prompts import (
    render_decomposer_prompt,
    render_selector_output_skeleton,
    render_selector_prompt,
    render_worker_prompt,
)
from .rewarding import compatibility_score, entropy_to_confidence_reward, skill_match_score
from .schema import (
    ControllerPolicyConfig,
    DecompositionCandidate,
    HFBackendConfig,
    RolloutConfig,
    SelectionCandidate,
    SubtaskNode,
    TaskExample,
    WorkerAssignment,
    WorkerExecution,
    WorkerPerformanceSnapshot,
    WorkerPoolConfig,
    WorkerSpec,
    VLLMBackendConfig,
)
from .ray_generation import RayVLLMGenerationManager
from .structured import (
    build_fallback_decomposition,
    build_fallback_selection,
    extract_worker_result_text,
    extract_decomposition_payload,
    extract_selection_payload,
    format_decomposition_plan,
    format_selection_plan,
    validate_decomposition_payload,
    validate_selection_payload,
)


@dataclass
class DecompositionRequest:
    task: TaskExample
    worker_pool: WorkerPoolConfig
    policy_config: ControllerPolicyConfig
    rollout_config: RolloutConfig
    decomposition_index: int
    worker_performance: Dict[str, WorkerPerformanceSnapshot]


@dataclass
class SelectionRequest:
    task: TaskExample
    decomposition: DecompositionCandidate
    worker_pool: WorkerPoolConfig
    policy_config: ControllerPolicyConfig
    selection_index: int
    worker_performance: Dict[str, WorkerPerformanceSnapshot]


@dataclass
class WorkerExecutionRequest:
    task: TaskExample
    decomposition: DecompositionCandidate
    node: SubtaskNode
    worker: WorkerSpec
    dependency_outputs: Dict[str, str]
    compatibility: float


def _canonicalize_selection_candidate(
    candidate: SelectionCandidate,
    decomposition: DecompositionCandidate,
    worker_pool: WorkerPoolConfig,
    worker_performance: Dict[str, WorkerPerformanceSnapshot],
) -> SelectionCandidate:
    worker_map = worker_pool.workers_by_id()
    node_map = decomposition.nodes_by_id()
    normalized_assignments: List[WorkerAssignment] = []
    for assignment in candidate.assignments:
        worker = worker_map[assignment.worker_id]
        node = node_map[assignment.node_id]
        normalized_assignments.append(
            WorkerAssignment(
                node_id=assignment.node_id,
                worker_id=assignment.worker_id,
                rationale=assignment.rationale or f"Selected for node {assignment.node_id}.",
                compatibility=round(
                    compatibility_score(node.required_skills, worker, worker_performance),
                    4,
                ),
            )
        )
    candidate.assignments = normalized_assignments
    return candidate


class HierarchicalBackend(ABC):
    @abstractmethod
    def sample_decomposition(
        self,
        task: TaskExample,
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        rollout_config: RolloutConfig,
        decomposition_index: int,
        worker_performance: Dict[str, WorkerPerformanceSnapshot],
    ) -> DecompositionCandidate:
        raise NotImplementedError

    @abstractmethod
    def sample_selection(
        self,
        task: TaskExample,
        decomposition: DecompositionCandidate,
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        selection_index: int,
        worker_performance: Dict[str, WorkerPerformanceSnapshot],
    ) -> SelectionCandidate:
        raise NotImplementedError

    @abstractmethod
    def execute_worker(
        self,
        task: TaskExample,
        decomposition: DecompositionCandidate,
        node: SubtaskNode,
        worker: WorkerSpec,
        dependency_outputs: Dict[str, str],
        compatibility: float,
    ) -> WorkerExecution:
        raise NotImplementedError

    def sample_decompositions_batch(
        self,
        requests: Sequence[DecompositionRequest],
    ) -> List[DecompositionCandidate]:
        return [
            self.sample_decomposition(
                task=request.task,
                worker_pool=request.worker_pool,
                policy_config=request.policy_config,
                rollout_config=request.rollout_config,
                decomposition_index=request.decomposition_index,
                worker_performance=request.worker_performance,
            )
            for request in requests
        ]

    def sample_selections_batch(
        self,
        requests: Sequence[SelectionRequest],
    ) -> List[SelectionCandidate]:
        return [
            self.sample_selection(
                task=request.task,
                decomposition=request.decomposition,
                worker_pool=request.worker_pool,
                policy_config=request.policy_config,
                selection_index=request.selection_index,
                worker_performance=request.worker_performance,
            )
            for request in requests
        ]

    def execute_workers_batch(
        self,
        requests: Sequence[WorkerExecutionRequest],
    ) -> List[WorkerExecution]:
        return [
            self.execute_worker(
                task=request.task,
                decomposition=request.decomposition,
                node=request.node,
                worker=request.worker,
                dependency_outputs=request.dependency_outputs,
                compatibility=request.compatibility,
            )
            for request in requests
        ]

    def close(self) -> None:
        return None


class MockHierarchicalBackend(HierarchicalBackend):
    """Small deterministic backend for validating orchestration and reward flow.

    This does not call a real LLM. It emits structured controller outputs and
    worker responses with confidence proxies so the full DAG rollout can be
    tested end-to-end before wiring it into vLLM or HF inference.
    """

    def __init__(self, worker_memory) -> None:
        self.worker_memory = worker_memory

    def sample_decomposition(
        self,
        task: TaskExample,
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        rollout_config: RolloutConfig,
        decomposition_index: int,
        worker_performance: Dict[str, WorkerPerformanceSnapshot],
    ) -> DecompositionCandidate:
        del policy_config
        skill_focus = task.metadata.get("skill_focus", "algebra")
        secondary_skill = "analysis" if skill_focus == "algebra" else "algebra"
        prompt_text = render_decomposer_prompt(
            task,
            max_nodes_hint=rollout_config.max_nodes_per_decomposition,
            soft_max_hops_hint=rollout_config.soft_max_hops,
            hard_max_hops_hint=rollout_config.hard_max_hops,
        )

        template_index = decomposition_index % 3
        if template_index == 0:
            nodes = [
                SubtaskNode(
                    node_id="1",
                    instruction="Identify the core mathematical structure.",
                    dependencies=[],
                    required_skills=[skill_focus],
                    output_key="core_structure",
                ),
                SubtaskNode(
                    node_id="2",
                    instruction="Use the core structure to compute the final answer.",
                    dependencies=["1"],
                    required_skills=[skill_focus],
                    output_key="final_answer",
                ),
            ]
            summary = "Compact decomposition aligned to the dominant skill."
        elif template_index == 1:
            nodes = [
                SubtaskNode(
                    node_id="1",
                    instruction="List the known quantities and target expression.",
                    dependencies=[],
                    required_skills=[skill_focus],
                    output_key="knowns",
                ),
                SubtaskNode(
                    node_id="2",
                    instruction="Choose the most relevant theorem or manipulation.",
                    dependencies=["1"],
                    required_skills=[skill_focus],
                    output_key="method",
                ),
                SubtaskNode(
                    node_id="3",
                    instruction="Apply the chosen method to produce the final answer.",
                    dependencies=["2"],
                    required_skills=[skill_focus],
                    output_key="final_answer",
                ),
            ]
            summary = "Longer but still coherent decomposition."
        else:
            nodes = [
                SubtaskNode(
                    node_id="1",
                    instruction="Take an unnecessary detour through a less relevant subdomain.",
                    dependencies=[],
                    required_skills=[secondary_skill],
                    output_key="detour",
                ),
                SubtaskNode(
                    node_id="2",
                    instruction="Recover from the detour and attempt the final answer.",
                    dependencies=["1"],
                    required_skills=[skill_focus],
                    output_key="final_answer",
                ),
            ]
            summary = "Noisier decomposition with a skill mismatch in the first node."

        raw_payload = {
            "decomposition_id": f"{task.task_id}-decomp-{decomposition_index}",
            "summary": summary,
            "final_node_id": nodes[-1].node_id,
            "nodes": [node.to_dict() for node in nodes],
            "controller_prompt": prompt_text,
        }
        candidate = validate_decomposition_payload(
            payload=raw_payload,
            rollout_config=rollout_config,
            fallback_id=raw_payload["decomposition_id"],
        )
        raw_payload["validation"] = {
            "backend": "mock",
            "fallback_used": False,
        }
        candidate.raw_payload = raw_payload
        candidate.raw_text = format_decomposition_plan(candidate)
        return candidate

    def sample_selection(
        self,
        task: TaskExample,
        decomposition: DecompositionCandidate,
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        selection_index: int,
        worker_performance: Dict[str, WorkerPerformanceSnapshot],
    ) -> SelectionCandidate:
        del policy_config
        worker_map = worker_pool.workers_by_id()
        prompt_text = render_selector_prompt(task, decomposition, worker_pool, worker_performance)
        assignments: List[WorkerAssignment] = []
        all_workers = list(worker_map.values())
        for node_position, node in enumerate(decomposition.nodes):
            scored_workers = sorted(
                (
                    (
                        compatibility_score(node.required_skills, worker, worker_performance),
                        worker,
                    )
                    for worker in all_workers
                ),
                key=lambda item: (-item[0], item[1].worker_id),
            )

            if selection_index % 2 == 0 or len(scored_workers) == 1:
                chosen_score, chosen_worker = scored_workers[0]
            else:
                # On alternate rollouts, flip one assignment to create a weaker selector branch.
                pick_index = 1 if node_position == 0 and len(scored_workers) > 1 else 0
                chosen_score, chosen_worker = scored_workers[pick_index]

            rationale = (
                f"Selected {chosen_worker.worker_id} for node {node.node_id} "
                f"because its skills best cover {node.required_skills}."
            )
            assignments.append(
                WorkerAssignment(
                    node_id=node.node_id,
                    worker_id=chosen_worker.worker_id,
                    rationale=rationale,
                    compatibility=round(chosen_score, 4),
                )
            )

        raw_payload = {
            "selection_id": f"{decomposition.decomposition_id}-sel-{selection_index}",
            "assignments": [assignment.to_dict() for assignment in assignments],
            "controller_prompt": prompt_text,
        }
        candidate = validate_selection_payload(
            payload=raw_payload,
            decomposition=decomposition,
            worker_pool=worker_pool,
        )
        raw_payload["validation"] = {
            "backend": "mock",
            "fallback_used": False,
        }
        candidate.raw_payload = raw_payload
        node_order = [node.node_id for node in decomposition.nodes]
        worker_index_by_id = {
            worker.worker_id: idx
            for idx, worker in enumerate(worker_pool.workers, start=1)
        }
        candidate.raw_text = format_selection_plan(
            candidate,
            node_order=node_order,
            worker_index_by_id=worker_index_by_id,
        )
        return candidate

    def execute_worker(
        self,
        task: TaskExample,
        decomposition: DecompositionCandidate,
        node: SubtaskNode,
        worker: WorkerSpec,
        dependency_outputs: Dict[str, str],
        compatibility: float,
    ) -> WorkerExecution:
        prompt_text = render_worker_prompt(task, decomposition, node, worker, dependency_outputs)
        required_match = skill_match_score(node.required_skills, worker)
        upstream_failed = any("[incorrect]" in output for output in dependency_outputs.values())
        success = required_match >= 0.99 and not upstream_failed

        if node.node_id == decomposition.final_node_id:
            if success:
                output_text = task.ground_truth
            else:
                output_text = task.metadata.get("distractor_answer", "incorrect")
        else:
            if success:
                output_text = (
                    f"[correct] {worker.worker_id} completed {node.node_id}: {node.output_key}"
                )
            else:
                output_text = (
                    f"[incorrect] {worker.worker_id} drifted on {node.node_id}: {node.output_key}"
                )

        entropy = 0.15 if success else 1.35
        confidence_reward = entropy_to_confidence_reward(entropy, entropy_cap=2.0)
        execution_payload = {
            "worker_prompt": prompt_text,
            "required_skills": node.required_skills,
            "worker_skills": worker.skills,
            "dependency_outputs": dependency_outputs,
        }
        del execution_payload
        return WorkerExecution(
            node_id=node.node_id,
            worker_id=worker.worker_id,
            output_text=output_text,
            entropy=entropy,
            confidence_reward=confidence_reward,
            compatibility=compatibility,
            dependency_outputs=dict(dependency_outputs),
            completed=bool(output_text.strip()),
            success=success,
        )


class TransformersHierarchicalBackend(HierarchicalBackend):
    backend_name = "transformers"

    def __init__(self, config: HFBackendConfig | None = None) -> None:
        self.config = config or HFBackendConfig()
        self._bundle_cache: Dict[Tuple[str, str | None], Tuple[object, object]] = {}

    @staticmethod
    def _require_runtime() -> None:
        missing = [
            package
            for package in ("transformers", "torch")
            if importlib.util.find_spec(package) is None
        ]
        if missing:
            raise ImportError(
                "TransformersHierarchicalBackend requires missing packages: "
                + ", ".join(missing)
            )

    def _load_bundle(
        self,
        base_model_path: str,
        lora_adapter_path: str | None = None,
    ) -> Tuple[object, object]:
        self._require_runtime()
        cache_key = (base_model_path, lora_adapter_path)
        if cache_key in self._bundle_cache:
            return self._bundle_cache[cache_key]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        torch_dtype = self.config.torch_dtype
        if torch_dtype != "auto":
            torch_dtype = getattr(torch, torch_dtype)

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=self.config.trust_remote_code,
            device_map=self.config.device_map,
            torch_dtype=torch_dtype,
        )
        if lora_adapter_path is not None:
            if importlib.util.find_spec("peft") is None:
                raise ImportError("peft is required to load worker LoRA adapters")
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, lora_adapter_path)

        model.eval()
        self._bundle_cache[cache_key] = (tokenizer, model)
        return tokenizer, model

    @staticmethod
    def _safe_model_device(model: object) -> object:
        if hasattr(model, "device"):
            return model.device
        return next(model.parameters()).device

    def _build_prompt(
        self,
        tokenizer: object,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                pass
        if system_prompt:
            return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"
        return user_prompt

    @staticmethod
    def _chunk_range(total_size: int, chunk_size: int) -> Sequence[Tuple[int, int]]:
        safe_chunk_size = max(int(chunk_size), 1)
        return [
            (start, min(start + safe_chunk_size, total_size))
            for start in range(0, total_size, safe_chunk_size)
        ]

    def _controller_temperature(self) -> float:
        if self.config.controller_temperature is not None:
            return float(self.config.controller_temperature)
        return float(self.config.temperature)

    def _worker_temperature(self) -> float:
        if self.config.worker_temperature is not None:
            return float(self.config.worker_temperature)
        return float(self.config.temperature)

    def _controller_max_new_tokens(self, role: str) -> int:
        if role == "decomposer" and self.config.decomposer_max_new_tokens is not None:
            return int(self.config.decomposer_max_new_tokens)
        if role == "selector" and self.config.selector_max_new_tokens is not None:
            return int(self.config.selector_max_new_tokens)
        return int(self.config.controller_max_new_tokens)

    @staticmethod
    def _selection_index_context(
        decomposition: DecompositionCandidate,
        worker_pool: WorkerPoolConfig,
    ) -> tuple[List[str], Dict[str, int]]:
        node_order = [node.node_id for node in decomposition.nodes]
        worker_index_by_id = {
            worker.worker_id: index
            for index, worker in enumerate(worker_pool.workers, start=1)
        }
        return node_order, worker_index_by_id

    def _format_selection_completion_text(
        self,
        *,
        candidate: SelectionCandidate,
        decomposition: DecompositionCandidate,
        worker_pool: WorkerPoolConfig,
    ) -> str:
        node_order, worker_index_by_id = self._selection_index_context(
            decomposition=decomposition,
            worker_pool=worker_pool,
        )
        return format_selection_plan(
            candidate,
            node_order=node_order,
            worker_index_by_id=worker_index_by_id,
        )

    def _controller_sampling_overrides(
        self,
        role: str,
        *,
        node_order: Sequence[str] | None = None,
        worker_count: int | None = None,
    ) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}
        stop_tag = "</decomposition_plan>" if role == "decomposer" else "</selection_plan>"
        overrides["stop"] = [stop_tag]
        overrides["include_stop_str_in_output"] = True

        if getattr(self.config, "controller_constrained_decoding", False):
            if role == "decomposer":
                regex = r"(?s)<decomposition_plan>.*?</decomposition_plan>"
            else:
                regex = r"(?s)<selection_plan>\s*(?:\d+\s*:\s*\d+\s*)+</selection_plan>"
                if node_order and worker_count and worker_count > 0:
                    worker_index_pattern = "|".join(str(index) for index in range(1, worker_count + 1))
                    assignment_lines = [
                        rf"{re.escape(str(node_id))}\s*:\s*(?:{worker_index_pattern})\s*"
                        for node_id in node_order
                    ]
                    regex = r"(?s)<selection_plan>\s*" + "".join(assignment_lines) + r"</selection_plan>"
            # Keep alias variants for vLLM versions that differ in field names.
            overrides["structured_outputs"] = {"regex": regex}
            overrides["guided_decoding"] = {"regex": regex}
            overrides["guided_regex"] = regex
        return overrides

    def _estimate_entropy(self, scores: List[object]) -> float:
        entropies = self._estimate_batch_entropy(scores)
        return entropies[0] if entropies else 0.0

    def _estimate_batch_entropy(self, scores: List[object]) -> List[float]:
        if not scores:
            return []
        import torch

        entropies = []
        for step_scores in scores:
            probs = torch.softmax(step_scores.float(), dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
            entropies.append(entropy)
        stacked = torch.stack(entropies, dim=0)
        return stacked.mean(dim=0).detach().cpu().tolist()

    def _generate_text(
        self,
        base_model_path: str,
        prompt_text: str,
        max_new_tokens: int,
        lora_adapter_path: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float]:
        del sampling_overrides
        tokenizer, model = self._load_bundle(base_model_path, lora_adapter_path=lora_adapter_path)
        full_prompt = self._build_prompt(tokenizer, prompt_text, system_prompt=system_prompt)
        resolved_temperature = self.config.temperature if temperature is None else temperature

        import torch

        inputs = tokenizer(full_prompt, return_tensors="pt")
        model_device = self._safe_model_device(model)
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=resolved_temperature > 0.0,
                temperature=resolved_temperature,
                top_p=self.config.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences[0, input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        entropy = self._estimate_entropy(list(outputs.scores))
        return generated_text, entropy

    def _generate_text_batch(
        self,
        base_model_path: str,
        prompt_texts: Sequence[str],
        max_new_tokens: int,
        batch_size: int,
        lora_adapter_path: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        sampling_overrides: Optional[Dict[str, Any]] = None,
        log_label: str | None = None,
    ) -> List[Tuple[str, float]]:
        del sampling_overrides
        del log_label
        if not prompt_texts:
            return []

        tokenizer, model = self._load_bundle(base_model_path, lora_adapter_path=lora_adapter_path)
        resolved_temperature = self.config.temperature if temperature is None else temperature

        import torch

        results: List[Tuple[str, float]] = []
        for start, end in self._chunk_range(len(prompt_texts), batch_size):
            prompt_chunk = [
                self._build_prompt(tokenizer, prompt_text, system_prompt=system_prompt)
                for prompt_text in prompt_texts[start:end]
            ]
            inputs = tokenizer(prompt_chunk, return_tensors="pt", padding=True)
            model_device = self._safe_model_device(model)
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
            input_lengths = inputs["attention_mask"].sum(dim=1)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=resolved_temperature > 0.0,
                    temperature=resolved_temperature,
                    top_p=self.config.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            entropies = self._estimate_batch_entropy(list(outputs.scores))
            for row_idx in range(outputs.sequences.shape[0]):
                generated_ids = outputs.sequences[row_idx, int(input_lengths[row_idx].item()):]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                entropy = entropies[row_idx] if row_idx < len(entropies) else 0.0
                results.append((generated_text, float(entropy)))

        return results

    @staticmethod
    def _set_decomposition_artifacts(
        candidate: DecompositionCandidate,
        prompt_text: str,
        validation: Dict[str, object],
    ) -> DecompositionCandidate:
        candidate.raw_payload = {
            "decomposition": {
                "decomposition_id": candidate.decomposition_id,
                "summary": candidate.summary,
                "final_node_id": candidate.final_node_id,
                "num_hops": candidate.num_hops,
                "effective_num_hops": candidate.effective_num_hops,
                "soft_penalty": candidate.soft_penalty,
                "was_hard_truncated": candidate.was_hard_truncated,
                "nodes": [node.to_dict() for node in candidate.nodes],
            },
            "controller_prompt": prompt_text,
            "validation": dict(validation),
        }
        candidate.raw_text = format_decomposition_plan(candidate)
        return candidate

    @staticmethod
    def _set_selection_artifacts(
        candidate: SelectionCandidate,
        prompt_text: str,
        validation: Dict[str, object],
        decomposition: DecompositionCandidate | None = None,
        worker_pool: WorkerPoolConfig | None = None,
    ) -> SelectionCandidate:
        candidate.raw_payload = {
            "selection": {
                "selection_id": candidate.selection_id,
                "assignments": [assignment.to_dict() for assignment in candidate.assignments],
            },
            "controller_prompt": prompt_text,
            "validation": dict(validation),
        }
        if decomposition is not None and worker_pool is not None:
            node_order = [node.node_id for node in decomposition.nodes]
            worker_index_by_id = {
                worker.worker_id: idx
                for idx, worker in enumerate(worker_pool.workers, start=1)
            }
            candidate.raw_text = format_selection_plan(
                candidate,
                node_order=node_order,
                worker_index_by_id=worker_index_by_id,
            )
        else:
            candidate.raw_text = format_selection_plan(candidate)
        return candidate

    def _generate_validated_decomposition(
        self,
        prompt_text: str,
        task: TaskExample,
        policy_config: ControllerPolicyConfig,
        rollout_config: RolloutConfig,
        fallback_id: str,
    ) -> DecompositionCandidate:
        model_path = policy_config.model_for_role("decomposer")
        if not model_path:
            raise ValueError(f"No decomposer model path configured for {type(self).__name__}")

        repair_prompt = prompt_text
        errors: List[str] = []
        last_raw_text = ""
        for attempt in range(self.config.max_format_retries + 1):
            last_raw_text, _ = self._generate_text(
                base_model_path=model_path,
                prompt_text=repair_prompt,
                max_new_tokens=self._controller_max_new_tokens("decomposer"),
                temperature=self._controller_temperature(),
                sampling_overrides=self._controller_sampling_overrides("decomposer"),
            )
            try:
                payload = extract_decomposition_payload(last_raw_text)
                candidate = validate_decomposition_payload(
                    payload=payload,
                    rollout_config=rollout_config,
                    fallback_id=fallback_id,
                )
                return self._set_decomposition_artifacts(
                    candidate=candidate,
                    prompt_text=prompt_text,
                    validation={
                        "backend": self.backend_name,
                        "attempt": attempt,
                        "errors_before_success": list(errors),
                        "fallback_used": False,
                        "raw_model_text": last_raw_text,
                    },
                )
            except Exception as exc:
                errors.append(str(exc))
                repair_prompt = (
                    f"{prompt_text}\n\nYour previous answer did not match the required decomposition format. "
                    f"Error: {exc}\nReturn ONLY the corrected <decomposition_plan> block. "
                    "Do not add commentary, bullets, or repeated task text. "
                    "Every node must include NODE_ID, INSTRUCTION, DEPENDENCIES, REQUIRED_SKILLS, and OUTPUT_KEY. "
                    "Use only allowed numeric node IDs (1, 2, ...). "
                    "Use `none` for REQUIRED_SKILLS only on pure routing or final-answer wrapper nodes. "
                    "FINAL_NODE_ID must match the last declared node and that node must be the terminal final-answer node."
                )

        candidate = build_fallback_decomposition(
            task_id=task.task_id,
            raw_text=last_raw_text,
            error_message=" | ".join(errors) if errors else "Unknown decomposition format error",
            rollout_config=rollout_config,
        )
        candidate.raw_payload["controller_prompt"] = prompt_text
        candidate.raw_payload.setdefault("validation", {})
        candidate.raw_payload["validation"].update(
            {
                "backend": self.backend_name,
                "errors": list(errors),
            }
        )
        candidate.raw_text = format_decomposition_plan(candidate)
        return candidate

    def _generate_validated_selection(
        self,
        prompt_text: str,
        task: TaskExample,
        decomposition: DecompositionCandidate,
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        worker_performance: Dict[str, WorkerPerformanceSnapshot],
        fallback_id: str,
    ) -> SelectionCandidate:
        del task
        model_path = policy_config.model_for_role("selector")
        if not model_path:
            raise ValueError(f"No selector model path configured for {type(self).__name__}")

        repair_prompt = prompt_text
        errors: List[str] = []
        last_raw_text = ""
        selector_skeleton = render_selector_output_skeleton(
            [node.node_id for node in decomposition.nodes],
            len(worker_pool.workers),
        )
        for attempt in range(self.config.max_format_retries + 1):
            last_raw_text, _ = self._generate_text(
                base_model_path=model_path,
                prompt_text=repair_prompt,
                max_new_tokens=self._controller_max_new_tokens("selector"),
                temperature=self._controller_temperature(),
                sampling_overrides=self._controller_sampling_overrides(
                    "selector",
                    node_order=[node.node_id for node in decomposition.nodes],
                    worker_count=len(worker_pool.workers),
                ),
            )
            try:
                payload = extract_selection_payload(last_raw_text)
                payload.setdefault("selection_id", fallback_id)
                candidate = validate_selection_payload(
                    payload=payload,
                    decomposition=decomposition,
                    worker_pool=worker_pool,
                )
                candidate = _canonicalize_selection_candidate(
                    candidate=candidate,
                    decomposition=decomposition,
                    worker_pool=worker_pool,
                    worker_performance=worker_performance,
                )
                return self._set_selection_artifacts(
                    candidate=candidate,
                    prompt_text=prompt_text,
                    validation={
                        "backend": self.backend_name,
                        "attempt": attempt,
                        "errors_before_success": list(errors),
                        "fallback_used": False,
                        "raw_model_text": last_raw_text,
                    },
                    decomposition=decomposition,
                    worker_pool=worker_pool,
                )
            except Exception as exc:
                errors.append(str(exc))
                repair_prompt = (
                    f"{prompt_text}\n\nYour previous answer did not match the required selection format. "
                    f"Error: {exc}\nReturn ONLY the corrected <selection_plan> block. "
                    "Do not add commentary, bullets, repeated task text, or extra sections. "
                    "Use numeric node IDs only with one `node_id: worker_index` line per node. "
                    "The simplest valid form is:\n"
                    f"{selector_skeleton}"
                    "\nTreat the worker indices shown above as placeholders for the required output shape; "
                    "choose the actual best valid worker index for each node."
                )

        candidate = build_fallback_selection(
            decomposition=decomposition,
            worker_pool=worker_pool,
            raw_text=last_raw_text,
            error_message=" | ".join(errors) if errors else "Unknown selection format error",
        )
        candidate = _canonicalize_selection_candidate(
            candidate=candidate,
            decomposition=decomposition,
            worker_pool=worker_pool,
            worker_performance=worker_performance,
        )
        candidate.raw_payload["controller_prompt"] = prompt_text
        candidate.raw_payload.setdefault("validation", {})
        candidate.raw_payload["validation"].update(
            {
                "backend": self.backend_name,
                "errors": list(errors),
            }
        )
        candidate.raw_text = self._format_selection_completion_text(
            candidate=candidate,
            decomposition=decomposition,
            worker_pool=worker_pool,
        )
        return candidate

    def sample_decomposition(
        self,
        task: TaskExample,
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        rollout_config: RolloutConfig,
        decomposition_index: int,
        worker_performance: Dict[str, WorkerPerformanceSnapshot],
    ) -> DecompositionCandidate:
        prompt_text = render_decomposer_prompt(
            task,
            max_nodes_hint=rollout_config.max_nodes_per_decomposition,
            soft_max_hops_hint=rollout_config.soft_max_hops,
            hard_max_hops_hint=rollout_config.hard_max_hops,
        )
        return self._generate_validated_decomposition(
            prompt_text=prompt_text,
            task=task,
            policy_config=policy_config,
            rollout_config=rollout_config,
            fallback_id=f"{task.task_id}-decomp-{decomposition_index}",
        )

    def sample_selection(
        self,
        task: TaskExample,
        decomposition: DecompositionCandidate,
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
        selection_index: int,
        worker_performance: Dict[str, WorkerPerformanceSnapshot],
    ) -> SelectionCandidate:
        prompt_text = render_selector_prompt(task, decomposition, worker_pool, worker_performance)
        return self._generate_validated_selection(
            prompt_text=prompt_text,
            task=task,
            decomposition=decomposition,
            worker_pool=worker_pool,
            policy_config=policy_config,
            worker_performance=worker_performance,
            fallback_id=f"{decomposition.decomposition_id}-sel-{selection_index}",
        )

    def execute_worker(
        self,
        task: TaskExample,
        decomposition: DecompositionCandidate,
        node: SubtaskNode,
        worker: WorkerSpec,
        dependency_outputs: Dict[str, str],
        compatibility: float,
    ) -> WorkerExecution:
        base_model_path = worker.base_model_path
        if not base_model_path:
            raise ValueError(f"Worker {worker.worker_id} is missing base_model_path")

        prompt_text = render_worker_prompt(task, decomposition, node, worker, dependency_outputs)
        output_text, entropy = self._generate_text(
            base_model_path=base_model_path,
            lora_adapter_path=worker.lora_adapter_path,
            prompt_text=prompt_text,
            system_prompt=worker.system_prompt,
            max_new_tokens=self.config.worker_max_new_tokens,
            temperature=self._worker_temperature(),
        )
        normalized_output = extract_worker_result_text(output_text).strip()
        completed = bool(normalized_output)
        success = completed and "i don't know" not in output_text.lower()
        return WorkerExecution(
            node_id=node.node_id,
            worker_id=worker.worker_id,
            output_text=normalized_output,
            entropy=entropy,
            confidence_reward=entropy_to_confidence_reward(entropy, entropy_cap=2.0),
            compatibility=compatibility,
            dependency_outputs=dict(dependency_outputs),
            completed=completed,
            success=success,
        )

    def sample_decompositions_batch(
        self,
        requests: Sequence[DecompositionRequest],
    ) -> List[DecompositionCandidate]:
        if not requests:
            return []

        grouped: Dict[str, List[Tuple[int, DecompositionRequest, str]]] = {}
        for index, request in enumerate(requests):
            model_path = request.policy_config.model_for_role("decomposer")
            if not model_path:
                raise ValueError(f"No decomposer model path configured for {type(self).__name__}")
            prompt_text = render_decomposer_prompt(
                request.task,
                max_nodes_hint=request.rollout_config.max_nodes_per_decomposition,
                soft_max_hops_hint=request.rollout_config.soft_max_hops,
                hard_max_hops_hint=request.rollout_config.hard_max_hops,
            )
            grouped.setdefault(model_path, []).append((index, request, prompt_text))

        results: List[DecompositionCandidate | None] = [None] * len(requests)
        for model_path, grouped_requests in grouped.items():
            prompt_texts = [prompt_text for _, _, prompt_text in grouped_requests]
            generated = self._generate_text_batch(
                base_model_path=model_path,
                prompt_texts=prompt_texts,
                max_new_tokens=self._controller_max_new_tokens("decomposer"),
                batch_size=self.config.controller_batch_size,
                temperature=self._controller_temperature(),
                sampling_overrides=self._controller_sampling_overrides("decomposer"),
                log_label="decomposer",
            )
            repair_count = 0
            for (result_index, request, prompt_text), (raw_text, entropy) in zip(grouped_requests, generated):
                fallback_id = f"{request.task.task_id}-decomp-{request.decomposition_index}"
                try:
                    payload = extract_decomposition_payload(raw_text)
                    candidate = validate_decomposition_payload(
                        payload=payload,
                        rollout_config=request.rollout_config,
                        fallback_id=fallback_id,
                    )
                    candidate = self._set_decomposition_artifacts(
                        candidate=candidate,
                        prompt_text=prompt_text,
                        validation={
                            "backend": self.backend_name,
                            "attempt": 0,
                            "errors_before_success": [],
                            "fallback_used": False,
                            "raw_model_text": raw_text,
                            "batch_generated": True,
                            "entropy": entropy,
                        },
                    )
                except Exception:
                    repair_count += 1
                    candidate = self._generate_validated_decomposition(
                        prompt_text=prompt_text,
                        task=request.task,
                        policy_config=request.policy_config,
                        rollout_config=request.rollout_config,
                        fallback_id=fallback_id,
                    )
                    candidate.raw_payload.setdefault("validation", {})
                    candidate.raw_payload["validation"].update(
                        {
                            "batch_generated": True,
                            "batch_repair_fallback": True,
                        }
                    )
                    candidate.raw_text = format_decomposition_plan(candidate)
                results[result_index] = candidate
            if repair_count:
                print(
                    f"[hierarchical-rema][generation] role=decomposer "
                    f"model={model_path} repair_fallbacks={repair_count}/{len(grouped_requests)}"
                )

        if any(candidate is None for candidate in results):
            raise RuntimeError("Batched decomposition generation did not produce a result for every request")
        return [candidate for candidate in results if candidate is not None]

    def sample_selections_batch(
        self,
        requests: Sequence[SelectionRequest],
    ) -> List[SelectionCandidate]:
        if not requests:
            return []

        grouped: Dict[Tuple[str, Tuple[str, ...], int], List[Tuple[int, SelectionRequest, str]]] = {}
        for index, request in enumerate(requests):
            model_path = request.policy_config.model_for_role("selector")
            if not model_path:
                raise ValueError(f"No selector model path configured for {type(self).__name__}")
            prompt_text = render_selector_prompt(
                request.task,
                request.decomposition,
                request.worker_pool,
                request.worker_performance,
            )
            node_order = tuple(node.node_id for node in request.decomposition.nodes)
            grouped.setdefault((model_path, node_order, len(request.worker_pool.workers)), []).append((index, request, prompt_text))

        results: List[SelectionCandidate | None] = [None] * len(requests)
        for (model_path, node_order, worker_count), grouped_requests in grouped.items():
            prompt_texts = [prompt_text for _, _, prompt_text in grouped_requests]
            generated = self._generate_text_batch(
                base_model_path=model_path,
                prompt_texts=prompt_texts,
                max_new_tokens=self._controller_max_new_tokens("selector"),
                batch_size=self.config.controller_batch_size,
                temperature=self._controller_temperature(),
                sampling_overrides=self._controller_sampling_overrides(
                    "selector",
                    node_order=node_order,
                    worker_count=worker_count,
                ),
                log_label="selector",
            )
            repair_count = 0
            for (result_index, request, prompt_text), (raw_text, entropy) in zip(grouped_requests, generated):
                fallback_id = f"{request.decomposition.decomposition_id}-sel-{request.selection_index}"
                try:
                    payload = extract_selection_payload(raw_text)
                    payload.setdefault("selection_id", fallback_id)
                    candidate = validate_selection_payload(
                        payload=payload,
                        decomposition=request.decomposition,
                        worker_pool=request.worker_pool,
                    )
                    candidate = _canonicalize_selection_candidate(
                        candidate=candidate,
                        decomposition=request.decomposition,
                        worker_pool=request.worker_pool,
                        worker_performance=request.worker_performance,
                    )
                    candidate = self._set_selection_artifacts(
                        candidate=candidate,
                        prompt_text=prompt_text,
                        validation={
                            "backend": self.backend_name,
                            "attempt": 0,
                            "errors_before_success": [],
                            "fallback_used": False,
                            "raw_model_text": raw_text,
                            "batch_generated": True,
                            "entropy": entropy,
                        },
                        decomposition=request.decomposition,
                        worker_pool=request.worker_pool,
                    )
                except Exception:
                    repair_count += 1
                    candidate = self._generate_validated_selection(
                        prompt_text=prompt_text,
                        task=request.task,
                        decomposition=request.decomposition,
                        worker_pool=request.worker_pool,
                        policy_config=request.policy_config,
                        worker_performance=request.worker_performance,
                        fallback_id=fallback_id,
                    )
                    candidate.raw_payload.setdefault("validation", {})
                    candidate.raw_payload["validation"].update(
                        {
                            "batch_generated": True,
                            "batch_repair_fallback": True,
                        }
                    )
                    candidate.raw_text = self._format_selection_completion_text(
                        candidate=candidate,
                        decomposition=request.decomposition,
                        worker_pool=request.worker_pool,
                    )
                results[result_index] = candidate
            if repair_count:
                print(
                    f"[hierarchical-rema][generation] role=selector "
                    f"model={model_path} repair_fallbacks={repair_count}/{len(grouped_requests)}"
                )

        if any(candidate is None for candidate in results):
            raise RuntimeError("Batched selection generation did not produce a result for every request")
        return [candidate for candidate in results if candidate is not None]

    def execute_workers_batch(
        self,
        requests: Sequence[WorkerExecutionRequest],
    ) -> List[WorkerExecution]:
        if not requests:
            return []

        grouped: Dict[Tuple[str, str | None, str], List[Tuple[int, WorkerExecutionRequest, str]]] = {}
        for index, request in enumerate(requests):
            base_model_path = request.worker.base_model_path
            if not base_model_path:
                raise ValueError(f"Worker {request.worker.worker_id} is missing base_model_path")
            prompt_text = render_worker_prompt(
                request.task,
                request.decomposition,
                request.node,
                request.worker,
                request.dependency_outputs,
            )
            cache_key = (
                base_model_path,
                request.worker.lora_adapter_path,
                request.worker.system_prompt,
            )
            grouped.setdefault(cache_key, []).append((index, request, prompt_text))

        results: List[WorkerExecution | None] = [None] * len(requests)
        for (base_model_path, lora_adapter_path, system_prompt), grouped_requests in grouped.items():
            prompt_texts = [prompt_text for _, _, prompt_text in grouped_requests]
            generated = self._generate_text_batch(
                base_model_path=base_model_path,
                prompt_texts=prompt_texts,
                max_new_tokens=self.config.worker_max_new_tokens,
                batch_size=self.config.worker_batch_size,
                lora_adapter_path=lora_adapter_path,
                system_prompt=system_prompt,
                temperature=self._worker_temperature(),
                log_label="worker",
            )
            for (result_index, request, _), (output_text, entropy) in zip(grouped_requests, generated):
                normalized_output = extract_worker_result_text(output_text).strip()
                completed = bool(normalized_output)
                success = completed and "i don't know" not in output_text.lower()
                results[result_index] = WorkerExecution(
                    node_id=request.node.node_id,
                    worker_id=request.worker.worker_id,
                    output_text=normalized_output,
                    entropy=float(entropy),
                    confidence_reward=entropy_to_confidence_reward(entropy, entropy_cap=2.0),
                    compatibility=request.compatibility,
                    dependency_outputs=dict(request.dependency_outputs),
                    completed=completed,
                    success=success,
                )

        if any(execution is None for execution in results):
            raise RuntimeError("Batched worker execution did not produce a result for every request")
        return [execution for execution in results if execution is not None]


class RayVLLMHierarchicalBackend(TransformersHierarchicalBackend):
    backend_name = "ray_vllm"

    def __init__(self, config: VLLMBackendConfig | None = None) -> None:
        self.config = config or VLLMBackendConfig()
        self._manager = RayVLLMGenerationManager(self.config)

    def _generate_text(
        self,
        base_model_path: str,
        prompt_text: str,
        max_new_tokens: int,
        lora_adapter_path: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float]:
        if lora_adapter_path is not None:
            raise NotImplementedError("Worker LoRA adapters are not implemented for the Ray/vLLM backend yet")
        result = self._manager.generate_one(
            model_path=base_model_path,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
            temperature=self.config.temperature if temperature is None else temperature,
            sampling_overrides=sampling_overrides,
        )
        return result.text, result.entropy

    def _generate_text_batch(
        self,
        base_model_path: str,
        prompt_texts: Sequence[str],
        max_new_tokens: int,
        batch_size: int,
        lora_adapter_path: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        sampling_overrides: Optional[Dict[str, Any]] = None,
        log_label: str | None = None,
    ) -> List[Tuple[str, float]]:
        if lora_adapter_path is not None:
            raise NotImplementedError("Worker LoRA adapters are not implemented for the Ray/vLLM backend yet")
        generated = self._manager.generate_batch(
            model_path=base_model_path,
            prompt_texts=prompt_texts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            system_prompt=system_prompt,
            temperature=self.config.temperature if temperature is None else temperature,
            sampling_overrides=sampling_overrides,
            log_label=log_label,
        )
        return [(item.text, item.entropy) for item in generated]

    def close(self) -> None:
        self._manager.close()

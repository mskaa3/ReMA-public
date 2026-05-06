from __future__ import annotations

import importlib.util
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from .prompts import render_decomposer_prompt, render_selector_prompt, render_worker_prompt
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
)
from .structured import (
    StructuredOutputError,
    build_fallback_decomposition,
    build_fallback_selection,
    extract_json_dict,
    validate_decomposition_payload,
    validate_selection_payload,
)


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
        prompt_text = render_decomposer_prompt(task, worker_pool, worker_performance)

        template_index = decomposition_index % 3
        if template_index == 0:
            nodes = [
                SubtaskNode(
                    node_id="n1",
                    instruction="Identify the core mathematical structure.",
                    dependencies=[],
                    required_skills=[skill_focus],
                    output_key="core_structure",
                ),
                SubtaskNode(
                    node_id="n2",
                    instruction="Use the core structure to compute the final answer.",
                    dependencies=["n1"],
                    required_skills=[skill_focus],
                    output_key="final_answer",
                ),
            ]
            summary = "Compact decomposition aligned to the dominant skill."
        elif template_index == 1:
            nodes = [
                SubtaskNode(
                    node_id="n1",
                    instruction="List the known quantities and target expression.",
                    dependencies=[],
                    required_skills=[skill_focus],
                    output_key="knowns",
                ),
                SubtaskNode(
                    node_id="n2",
                    instruction="Choose the most relevant theorem or manipulation.",
                    dependencies=["n1"],
                    required_skills=[skill_focus],
                    output_key="method",
                ),
                SubtaskNode(
                    node_id="n3",
                    instruction="Apply the chosen method to produce the final answer.",
                    dependencies=["n2"],
                    required_skills=[skill_focus],
                    output_key="final_answer",
                ),
            ]
            summary = "Longer but still coherent decomposition."
        else:
            nodes = [
                SubtaskNode(
                    node_id="n1",
                    instruction="Take an unnecessary detour through a less relevant subdomain.",
                    dependencies=[],
                    required_skills=[secondary_skill],
                    output_key="detour",
                ),
                SubtaskNode(
                    node_id="n2",
                    instruction="Recover from the detour and attempt the final answer.",
                    dependencies=["n1"],
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
        candidate.raw_text = json.dumps(raw_payload, indent=2, sort_keys=True)
        candidate.raw_payload = raw_payload
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
        candidate.raw_text = json.dumps(raw_payload, indent=2, sort_keys=True)
        candidate.raw_payload = raw_payload
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
        del decomposition
        prompt_text = render_worker_prompt(task, node, worker, dependency_outputs)
        required_match = skill_match_score(node.required_skills, worker)
        upstream_failed = any("[incorrect]" in output for output in dependency_outputs.values())
        success = required_match >= 0.99 and not upstream_failed

        if node.output_key == "final_answer":
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

    def _estimate_entropy(self, scores: List[object]) -> float:
        if not scores:
            return 0.0
        import torch

        entropies = []
        for step_scores in scores:
            probs = torch.softmax(step_scores.float(), dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
            entropies.append(entropy.item())
        return float(sum(entropies) / len(entropies))

    def _generate_text(
        self,
        base_model_path: str,
        prompt_text: str,
        max_new_tokens: int,
        lora_adapter_path: str | None = None,
        system_prompt: str | None = None,
    ) -> Tuple[str, float]:
        tokenizer, model = self._load_bundle(base_model_path, lora_adapter_path=lora_adapter_path)
        full_prompt = self._build_prompt(tokenizer, prompt_text, system_prompt=system_prompt)

        import torch

        inputs = tokenizer(full_prompt, return_tensors="pt")
        model_device = self._safe_model_device(model)
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
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

    def _generate_validated_decomposition(
        self,
        prompt_text: str,
        task: TaskExample,
        policy_config: ControllerPolicyConfig,
        rollout_config: RolloutConfig,
    ) -> DecompositionCandidate:
        model_path = policy_config.model_for_role("decomposer")
        if not model_path:
            raise ValueError("No decomposer model path configured for TransformersHierarchicalBackend")

        repair_prompt = prompt_text
        errors: List[str] = []
        last_raw_text = ""
        for attempt in range(self.config.max_format_retries + 1):
            last_raw_text, _ = self._generate_text(
                base_model_path=model_path,
                prompt_text=repair_prompt,
                max_new_tokens=self.config.controller_max_new_tokens,
            )
            try:
                payload = extract_json_dict(last_raw_text)
                candidate = validate_decomposition_payload(
                    payload=payload,
                    rollout_config=rollout_config,
                    fallback_id=f"{task.task_id}-decomp",
                )
                payload["controller_prompt"] = prompt_text
                payload["validation"] = {
                    "backend": "transformers",
                    "attempt": attempt,
                    "errors_before_success": list(errors),
                    "fallback_used": False,
                    "raw_model_text": last_raw_text,
                }
                candidate.raw_payload = payload
                candidate.raw_text = json.dumps(payload, indent=2, sort_keys=True)
                return candidate
            except Exception as exc:
                errors.append(str(exc))
                repair_prompt = (
                    f"{prompt_text}\n\nYour previous answer was invalid JSON for the schema. "
                    f"Error: {exc}\nReturn ONLY corrected JSON."
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
                "backend": "transformers",
                "errors": list(errors),
            }
        )
        candidate.raw_text = json.dumps(candidate.raw_payload, indent=2, sort_keys=True)
        return candidate

    def _generate_validated_selection(
        self,
        prompt_text: str,
        task: TaskExample,
        decomposition: DecompositionCandidate,
        worker_pool: WorkerPoolConfig,
        policy_config: ControllerPolicyConfig,
    ) -> SelectionCandidate:
        model_path = policy_config.model_for_role("selector")
        if not model_path:
            raise ValueError("No selector model path configured for TransformersHierarchicalBackend")

        repair_prompt = prompt_text
        errors: List[str] = []
        last_raw_text = ""
        for attempt in range(self.config.max_format_retries + 1):
            last_raw_text, _ = self._generate_text(
                base_model_path=model_path,
                prompt_text=repair_prompt,
                max_new_tokens=self.config.controller_max_new_tokens,
            )
            try:
                payload = extract_json_dict(last_raw_text)
                candidate = validate_selection_payload(
                    payload=payload,
                    decomposition=decomposition,
                    worker_pool=worker_pool,
                )
                payload["controller_prompt"] = prompt_text
                payload["validation"] = {
                    "backend": "transformers",
                    "attempt": attempt,
                    "errors_before_success": list(errors),
                    "fallback_used": False,
                    "raw_model_text": last_raw_text,
                }
                candidate.raw_payload = payload
                candidate.raw_text = json.dumps(payload, indent=2, sort_keys=True)
                return candidate
            except Exception as exc:
                errors.append(str(exc))
                repair_prompt = (
                    f"{prompt_text}\n\nYour previous answer was invalid JSON for the schema. "
                    f"Error: {exc}\nReturn ONLY corrected JSON."
                )

        candidate = build_fallback_selection(
            decomposition=decomposition,
            worker_pool=worker_pool,
            raw_text=last_raw_text,
            error_message=" | ".join(errors) if errors else "Unknown selection format error",
        )
        candidate.raw_payload["controller_prompt"] = prompt_text
        candidate.raw_payload.setdefault("validation", {})
        candidate.raw_payload["validation"].update(
            {
                "backend": "transformers",
                "errors": list(errors),
            }
        )
        candidate.raw_text = json.dumps(candidate.raw_payload, indent=2, sort_keys=True)
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
        del decomposition_index
        prompt_text = render_decomposer_prompt(task, worker_pool, worker_performance)
        return self._generate_validated_decomposition(
            prompt_text=prompt_text,
            task=task,
            policy_config=policy_config,
            rollout_config=rollout_config,
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
        del selection_index
        prompt_text = render_selector_prompt(task, decomposition, worker_pool, worker_performance)
        return self._generate_validated_selection(
            prompt_text=prompt_text,
            task=task,
            decomposition=decomposition,
            worker_pool=worker_pool,
            policy_config=policy_config,
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
        del decomposition
        base_model_path = worker.base_model_path
        if not base_model_path:
            raise ValueError(f"Worker {worker.worker_id} is missing base_model_path")

        prompt_text = render_worker_prompt(task, node, worker, dependency_outputs)
        output_text, entropy = self._generate_text(
            base_model_path=base_model_path,
            lora_adapter_path=worker.lora_adapter_path,
            prompt_text=prompt_text,
            system_prompt=worker.system_prompt,
            max_new_tokens=self.config.worker_max_new_tokens,
        )
        normalized_output = output_text.strip()
        completed = bool(normalized_output)
        success = completed and "i don't know" not in normalized_output.lower()
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

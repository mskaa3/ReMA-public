from __future__ import annotations

import json
from typing import Dict

from .schema import (
    DecompositionCandidate,
    SubtaskNode,
    TaskExample,
    WorkerPerformanceSnapshot,
    WorkerPoolConfig,
    WorkerSpec,
)


DECOMPOSER_SYSTEM_PROMPT = """You are the Decomposer controller.
Produce a compact DAG decomposition for the task.

Return exactly one block in this format:
<decomposition_plan>
DECOMPOSITION_ID: short_id
SUMMARY: short summary
FINAL_NODE_ID: n_last
NODE: n1
INSTRUCTION: short instruction
DEPENDENCIES: none
REQUIRED_SKILLS: algebra
OUTPUT_KEY: partial_result
NODE: n_last
INSTRUCTION: produce the final answer
DEPENDENCIES: n1
REQUIRED_SKILLS: analysis
OUTPUT_KEY: final_answer
</decomposition_plan>

Rules:
1. Do not use markdown fences.
2. Keep the block compact and easy to parse.
3. Use a DAG, not a linear chain unless the task truly requires one.
4. Use short node instructions.
5. The final node must produce the final answer.
6. Prefer decompositions that fit the available worker pool and prior worker performance.
"""


SELECTOR_SYSTEM_PROMPT = """You are the Selector controller.
Assign workers to the DAG nodes.

Return exactly one block in this format:
<selection_plan>
SELECTION_ID: short_id
n1 -> worker_a
n2 -> worker_b
</selection_plan>

Rules:
1. Do not use markdown fences.
2. Keep the block compact and easy to parse.
3. Assign exactly one worker to each node.
4. Use worker skills, prior compatibility, and performance history.
5. The simplest valid output is one assignment line per node in the form `node_id -> worker_id`.
6. Compatibility and rationale are optional; if you include them, keep them short.
"""


DEFAULT_ALGEBRA_WORKER_PROMPT = """You are an algebra-focused worker.
You are strongest at symbolic manipulation, equation solving, simplification, and exact arithmetic.
Be concise and return the subtask result directly.
"""


DEFAULT_ANALYSIS_WORKER_PROMPT = """You are an analysis-focused worker.
You are strongest at calculus, limits, continuity, derivatives, and theorem-driven reasoning.
Be concise and return the subtask result directly.
"""


def _compact_performance_summary(snapshot: WorkerPerformanceSnapshot) -> dict:
    return {
        "ema_outcome": snapshot.ema_outcome,
        "num_assignments": snapshot.num_assignments,
        "completion_rate": snapshot.completion_rate,
        "success_rate": snapshot.success_rate,
        "average_reward": snapshot.average_reward,
        "average_confidence_reward": snapshot.average_confidence_reward,
        "average_compatibility": snapshot.average_compatibility,
        "recent_history": snapshot.recent_history[-3:],
    }


def _worker_context(
    worker_pool: WorkerPoolConfig,
    worker_performance: Dict[str, WorkerPerformanceSnapshot],
) -> list[dict]:
    worker_context = []
    for worker in worker_pool.workers:
        performance_snapshot = worker_performance.get(worker.worker_id)
        worker_context.append(
            {
                "worker_id": worker.worker_id,
                "skills": list(worker.skills),
                "description": worker.description,
                "trainable": worker.trainable,
                "lora_adapter_path": worker.lora_adapter_path,
                "performance_summary": _compact_performance_summary(
                    performance_snapshot
                    if performance_snapshot is not None
                    else WorkerPerformanceSnapshot(
                        worker_id=worker.worker_id,
                        ema_outcome=0.5,
                        num_assignments=0,
                        num_completed=0,
                        completion_rate=0.0,
                        num_successes=0,
                        success_rate=0.0,
                        average_reward=0.0,
                        average_confidence_reward=0.0,
                        average_compatibility=0.0,
                    )
                ),
            }
        )
    return worker_context


def _decomposition_context(decomposition: DecompositionCandidate) -> dict:
    return {
        "decomposition_id": decomposition.decomposition_id,
        "summary": decomposition.summary,
        "final_node_id": decomposition.final_node_id,
        "num_hops": decomposition.num_hops,
        "effective_num_hops": decomposition.effective_num_hops,
        "soft_penalty": decomposition.soft_penalty,
        "was_hard_truncated": decomposition.was_hard_truncated,
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


def render_decomposer_prompt(
    task: TaskExample,
    worker_pool: WorkerPoolConfig,
    worker_performance: Dict[str, WorkerPerformanceSnapshot],
) -> str:
    payload = {
        "task_id": task.task_id,
        "task": task.prompt,
        "available_workers": _worker_context(worker_pool, worker_performance),
        "preferred_output_format": [
            "<decomposition_plan>",
            "DECOMPOSITION_ID: short_id",
            "SUMMARY: short summary",
            "FINAL_NODE_ID: n_last",
            "NODE: n1",
            "INSTRUCTION: short instruction",
            "DEPENDENCIES: none",
            "REQUIRED_SKILLS: algebra",
            "OUTPUT_KEY: partial_result",
            "NODE: n_last",
            "INSTRUCTION: produce the final answer",
            "DEPENDENCIES: n1",
            "REQUIRED_SKILLS: analysis",
            "OUTPUT_KEY: final_answer",
            "</decomposition_plan>",
        ],
        "required_fields": [
            "decomposition_id",
            "summary",
            "final_node_id",
            "nodes[].node_id",
            "nodes[].instruction",
            "nodes[].dependencies",
            "nodes[].required_skills",
            "nodes[].output_key",
        ],
    }
    return f"{DECOMPOSER_SYSTEM_PROMPT}\n\n{json.dumps(payload, indent=2, sort_keys=True)}"


def render_selector_prompt(
    task: TaskExample,
    decomposition: DecompositionCandidate,
    worker_pool: WorkerPoolConfig,
    worker_performance: Dict[str, WorkerPerformanceSnapshot],
) -> str:
    node_lines = []
    for node in decomposition.nodes:
        dependencies = ",".join(node.dependencies) if node.dependencies else "none"
        skills = ",".join(node.required_skills) if node.required_skills else "none"
        node_lines.append(
            f"- {node.node_id} | deps={dependencies} | skills={skills} | output={node.output_key} | instruction={node.instruction}"
        )

    worker_lines = []
    for worker in worker_pool.workers:
        snapshot = worker_performance.get(worker.worker_id)
        success_rate = snapshot.success_rate if snapshot is not None else 0.0
        avg_reward = snapshot.average_reward if snapshot is not None else 0.0
        skills = ",".join(worker.skills) if worker.skills else "none"
        worker_lines.append(
            f"- {worker.worker_id} | skills={skills} | success={success_rate:.2f} | avg_reward={avg_reward:.2f} | desc={worker.description}"
        )

    return (
        f"{SELECTOR_SYSTEM_PROMPT}\n\n"
        f"TASK_ID: {task.task_id}\n"
        f"TASK: {task.prompt}\n"
        f"DECOMPOSITION_ID: {decomposition.decomposition_id}\n"
        f"FINAL_NODE_ID: {decomposition.final_node_id}\n"
        "NODES:\n"
        f"{chr(10).join(node_lines)}\n"
        "AVAILABLE_WORKERS:\n"
        f"{chr(10).join(worker_lines)}\n\n"
        "Return ONLY the <selection_plan> block."
    )


def render_worker_prompt(
    task: TaskExample,
    node: SubtaskNode,
    worker: WorkerSpec,
    dependency_outputs: dict[str, str],
) -> str:
    payload = {
        "task_id": task.task_id,
        "task": task.prompt,
        "worker": {
            "worker_id": worker.worker_id,
            "skills": worker.skills,
            "description": worker.description,
        },
        "node": node.to_dict(),
        "dependency_outputs": dependency_outputs,
        "instructions": "Return the subtask result only.",
    }
    return json.dumps(payload, indent=2, sort_keys=True)

from __future__ import annotations

import json
from typing import Dict, Iterable

from .schema import (
    DecompositionCandidate,
    SubtaskNode,
    TaskExample,
    WorkerPerformanceSnapshot,
    WorkerPoolConfig,
    WorkerSpec,
)


DECOMPOSER_SYSTEM_PROMPT = """You are the Decomposer controller.
Return a DAG decomposition for the user task as strict JSON.

Rules:
1. Output only JSON.
2. Use a DAG, not a linear chain unless the task truly requires one.
3. Each node must contain: node_id, instruction, dependencies, required_skills, output_key.
4. The final node must produce the final answer.
5. Prefer decompositions that fit the available worker pool.
6. Use worker characteristics and historical performance when deciding the DAG shape.
"""


SELECTOR_SYSTEM_PROMPT = """You are the Selector controller.
Return a worker assignment for each DAG node as strict JSON.

Rules:
1. Output only JSON.
2. Assign exactly one worker to each node.
3. Use worker skills, prior compatibility, and decomposition structure.
4. Each assignment must contain: node_id, worker_id, rationale, compatibility.
5. Use each worker's performance history, including reward and completion behavior.
"""


DEFAULT_ALGEBRA_WORKER_PROMPT = """You are an algebra-focused worker.
You are strongest at symbolic manipulation, equation solving, simplification, and exact arithmetic.
Be concise and return the subtask result directly.
"""


DEFAULT_ANALYSIS_WORKER_PROMPT = """You are an analysis-focused worker.
You are strongest at calculus, limits, continuity, derivatives, and theorem-driven reasoning.
Be concise and return the subtask result directly.
"""


def _worker_catalog(workers: Iterable[WorkerSpec]) -> str:
    return json.dumps(
        [
            {
                "worker_id": worker.worker_id,
                "skills": worker.skills,
                "description": worker.description,
                "lora_adapter_path": worker.lora_adapter_path,
                "trainable": worker.trainable,
            }
            for worker in workers
        ],
        indent=2,
        sort_keys=True,
    )


def _worker_context(
    worker_pool: WorkerPoolConfig,
    worker_performance: Dict[str, WorkerPerformanceSnapshot],
) -> list[dict]:
    worker_context = []
    for worker in worker_pool.workers:
        performance_snapshot = worker_performance.get(worker.worker_id)
        worker_context.append(
            {
                **worker.to_dict(),
                "performance_summary": (
                    performance_snapshot.to_dict()
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
                    ).to_dict()
                ),
            }
        )
    return worker_context


def render_decomposer_prompt(
    task: TaskExample,
    worker_pool: WorkerPoolConfig,
    worker_performance: Dict[str, WorkerPerformanceSnapshot],
) -> str:
    payload = {
        "task_id": task.task_id,
        "task": task.prompt,
        "available_workers": _worker_context(worker_pool, worker_performance),
        "target_schema": {
            "decomposition_id": "string",
            "summary": "string",
            "final_node_id": "string",
            "nodes": [
                {
                    "node_id": "string",
                    "instruction": "string",
                    "dependencies": ["node_id"],
                    "required_skills": ["skill"],
                    "output_key": "string",
                }
            ],
        },
    }
    return f"{DECOMPOSER_SYSTEM_PROMPT}\n\n{json.dumps(payload, indent=2, sort_keys=True)}"


def render_selector_prompt(
    task: TaskExample,
    decomposition: DecompositionCandidate,
    worker_pool: WorkerPoolConfig,
    worker_performance: Dict[str, WorkerPerformanceSnapshot],
) -> str:
    payload = {
        "task_id": task.task_id,
        "task": task.prompt,
        "decomposition": decomposition.to_dict(),
        "available_workers": _worker_context(worker_pool, worker_performance),
        "target_schema": {
            "selection_id": "string",
            "assignments": [
                {
                    "node_id": "string",
                    "worker_id": "string",
                    "rationale": "string",
                    "compatibility": "float in [0, 1]",
                }
            ],
        },
    }
    return f"{SELECTOR_SYSTEM_PROMPT}\n\n{json.dumps(payload, indent=2, sort_keys=True)}"


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

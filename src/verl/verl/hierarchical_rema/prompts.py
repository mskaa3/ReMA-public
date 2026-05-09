from __future__ import annotations

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

Your entire response must be exactly one XML-like block and nothing else.

Return exactly one block in this format:
<decomposition_plan>
SUMMARY: short summary
FINAL_NODE_ID: 2
NODE_ID: 1
INSTRUCTION: short instruction
DEPENDENCIES: none
REQUIRED_SKILLS: algebra
OUTPUT_KEY: partial_result
NODE_ID: 2
INSTRUCTION: produce the final answer
DEPENDENCIES: 1
REQUIRED_SKILLS: analysis
OUTPUT_KEY: final_answer
</decomposition_plan>

Rules:
1. Do not use markdown fences.
2. Do not output prose before the opening tag or after the closing tag.
3. Use the field names exactly as shown: SUMMARY, FINAL_NODE_ID, NODE_ID, INSTRUCTION, DEPENDENCIES, REQUIRED_SKILLS, OUTPUT_KEY.
4. Every NODE_ID must be followed by exactly one INSTRUCTION, one DEPENDENCIES, one REQUIRED_SKILLS, and one OUTPUT_KEY line.
5. Use a DAG, not a linear chain unless the task truly requires one.
6. Use short node instructions and short summaries.
7. Prefer 2 to 4 nodes unless the task truly needs more or fewer.
8. The final node must produce the final answer, and FINAL_NODE_ID must match one declared node ID.
9. Dependencies must be `none` or a comma-separated list of previously declared node IDs.
10. REQUIRED_SKILLS should match the available worker pool whenever possible.
11. OUTPUT_KEY values should be short snake_case names.
12. Do not invent extra sections, commentary, explanations, bullets, or JSON.
13. Only use plain numeric node IDs like 1, 2, 3, ... and keep them consistent.

If you are unsure, output the simplest valid decomposition_plan block that satisfies the format.
"""


SELECTOR_SYSTEM_PROMPT = """You are the Selector controller.
Assign workers to the DAG nodes.

Your entire response must be exactly one XML-like block and nothing else.

Return exactly one block in this format:
<selection_plan>
1: 2
2: 1
</selection_plan>

Rules:
1. Do not use markdown fences.
2. Do not output prose before the opening tag or after the closing tag.
3. Assign exactly one worker to each node ID from the decomposition.
4. Use only worker indices listed in WORKERS_BY_INDEX from the prompt.
5. Preferred output is one mapping line per node in the form `node_id: worker_index`.
6. Do not skip nodes, do not add extra assignments, and do not assign multiple workers to one node.
7. Prefer the worker whose skills and past performance best match each node.
8. Keep the output minimal. Do not include explanations, commentary, bullets, JSON, or repeated task text.
9. Compatibility and rationale are optional; omit them unless explicitly requested.
10. Legacy `node_id -> worker_id` lines are accepted, but numeric node-ID mapping is preferred.

If you are unsure, output the simplest valid selection_plan block with one assignment per node.
"""


DEFAULT_ARITHMETIC_PREALGEBRA_WORKER_PROMPT = """You are an arithmetic and prealgebra worker.
You are strongest at exact numeric computation, fractions, ratios, percentages, signs, simplification, and straightforward expression cleanup.
Prefer exact forms over decimals unless the task explicitly asks for approximation.
Be concise and return the subtask result directly.
"""


DEFAULT_ALGEBRA_SYMBOLIC_WORKER_PROMPT = """You are an algebra and symbolic manipulation worker.
You are strongest at solving equations, substitutions, polynomial manipulation, factoring, expanding, and symbolic simplification.
Keep expressions exact and transform them carefully step by step when needed.
Be concise and return the subtask result directly.
"""


DEFAULT_GEOMETRY_TRIGONOMETRY_WORKER_PROMPT = """You are a geometry and trigonometry worker.
You are strongest at Euclidean geometry, coordinate geometry, angle and length relations, standard formulas, and trigonometric identities.
Use the relevant geometric constraints precisely and keep notation clean.
Be concise and return the subtask result directly.
"""


DEFAULT_CALCULUS_ANALYSIS_WORKER_PROMPT = """You are a calculus and analysis worker.
You are strongest at limits, derivatives, integrals, continuity, monotonicity, extrema, and function behavior.
Apply standard theorems and derivative or integral rules carefully, keeping the result mathematically exact.
Be concise and return the subtask result directly.
"""


DEFAULT_DISCRETE_NUMBER_THEORY_WORKER_PROMPT = """You are a discrete mathematics and number theory worker.
You are strongest at divisibility, modular arithmetic, parity, counting, combinatorics, invariants, and elementary probability.
Break the problem into precise cases or arithmetic constraints when helpful.
Be concise and return the subtask result directly.
"""


# Backward-compatible aliases for any older imports that still expect the two-worker setup.
DEFAULT_ALGEBRA_WORKER_PROMPT = DEFAULT_ALGEBRA_SYMBOLIC_WORKER_PROMPT
DEFAULT_ANALYSIS_WORKER_PROMPT = DEFAULT_CALCULUS_ANALYSIS_WORKER_PROMPT


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
    max_nodes_hint: int | None = None,
) -> str:
    worker_lines = []
    for worker in worker_pool.workers:
        snapshot = worker_performance.get(worker.worker_id)
        success_rate = snapshot.success_rate if snapshot is not None else 0.0
        avg_reward = snapshot.average_reward if snapshot is not None else 0.0
        completion_rate = snapshot.completion_rate if snapshot is not None else 0.0
        skills = ",".join(worker.skills) if worker.skills else "none"
        worker_lines.append(
            f"- {worker.worker_id} | skills={skills} | success={success_rate:.2f} | "
            f"complete={completion_rate:.2f} | avg_reward={avg_reward:.2f} | desc={worker.description}"
        )

    max_nodes = max(1, int(max_nodes_hint or 4))
    allowed_node_ids = ", ".join(str(i) for i in range(1, max_nodes + 1))
    return (
        f"{DECOMPOSER_SYSTEM_PROMPT}\n\n"
        "OUTPUT CONTRACT:\n"
        "- Response must start with <decomposition_plan> and end with </decomposition_plan>.\n"
        "- Use plain numeric node IDs like 1, 2, 3 in topological order.\n"
        "- Every node block must include NODE_ID, INSTRUCTION, DEPENDENCIES, REQUIRED_SKILLS, OUTPUT_KEY.\n"
        "- Do not repeat the task outside the block.\n"
        "- Allowed field keys: SUMMARY, FINAL_NODE_ID, NODE_ID, INSTRUCTION, DEPENDENCIES, REQUIRED_SKILLS, OUTPUT_KEY.\n"
        f"- Allowed node IDs: {allowed_node_ids}.\n"
        "- Allowed dependency tokens: `none` or comma-separated node IDs from the allowed set.\n"
        "- Forbidden output patterns: markdown fences, JSON, bullets, prose outside tags.\n\n"
        f"TASK_ID: {task.task_id}\n"
        f"TASK: {task.prompt}\n"
        "AVAILABLE_WORKERS:\n"
        f"{chr(10).join(worker_lines)}\n\n"
        "Return ONLY the <decomposition_plan> block."
    )


def render_selector_prompt(
    task: TaskExample,
    decomposition: DecompositionCandidate,
    worker_pool: WorkerPoolConfig,
    worker_performance: Dict[str, WorkerPerformanceSnapshot],
) -> str:
    node_lines = []
    ordered_node_ids = [node.node_id for node in decomposition.nodes]
    for node in decomposition.nodes:
        dependencies = ",".join(node.dependencies) if node.dependencies else "none"
        skills = ",".join(node.required_skills) if node.required_skills else "none"
        node_lines.append(
            f"{node.node_id}: deps={dependencies} | skills={skills} | output={node.output_key} | instruction={node.instruction}"
        )

    worker_lines = []
    for worker_index, worker in enumerate(worker_pool.workers, start=1):
        snapshot = worker_performance.get(worker.worker_id)
        success_rate = snapshot.success_rate if snapshot is not None else 0.0
        avg_reward = snapshot.average_reward if snapshot is not None else 0.0
        skills = ",".join(worker.skills) if worker.skills else "none"
        worker_lines.append(
            f"{worker_index}: {worker.worker_id} | skills={skills} | success={success_rate:.2f} | avg_reward={avg_reward:.2f} | desc={worker.description}"
        )

    allowed_worker_indices = ", ".join(str(index) for index in range(1, len(worker_pool.workers) + 1))
    return (
        f"{SELECTOR_SYSTEM_PROMPT}\n\n"
        "OUTPUT CONTRACT:\n"
        "- Response must start with <selection_plan> and end with </selection_plan>.\n"
        "- Preferred answer is one line per node: `node_id: worker_index`.\n"
        "- The left side is the numeric node ID from NODES_BY_ID.\n"
        "- The right side is the worker index from WORKERS_BY_INDEX.\n"
        f"- Number of mapping lines must equal number of nodes ({len(ordered_node_ids)}).\n"
        f"- Allowed node IDs: {', '.join(ordered_node_ids)}.\n"
        f"- Allowed worker indices: {allowed_worker_indices}.\n"
        "- Do not repeat the task, decomposition, or worker descriptions in the output.\n"
        "- Forbidden output patterns: markdown fences, JSON, bullets, prose outside tags.\n\n"
        f"TASK_ID: {task.task_id}\n"
        f"TASK: {task.prompt}\n"
        f"FINAL_NODE_ID: {decomposition.final_node_id}\n"
        "NODES_BY_ID:\n"
        f"{chr(10).join(node_lines)}\n"
        "WORKERS_BY_INDEX:\n"
        f"{chr(10).join(worker_lines)}\n\n"
        "Return ONLY the <selection_plan> block. Preferred minimal form:\n"
        "<selection_plan>\n"
        "1: 2\n"
        "2: 1\n"
        "3: 1\n"
        "</selection_plan>"
    )


def render_worker_prompt(
    task: TaskExample,
    node: SubtaskNode,
    worker: WorkerSpec,
    dependency_outputs: dict[str, str],
) -> str:
    dependency_lines = []
    for dependency_id, dependency_output in dependency_outputs.items():
        dependency_lines.append(f"- {dependency_id}: {dependency_output}")
    if not dependency_lines:
        dependency_lines.append("- none")

    skills = ",".join(worker.skills) if worker.skills else "none"
    node_dependencies = ",".join(node.dependencies) if node.dependencies else "none"
    node_required_skills = ",".join(node.required_skills) if node.required_skills else "none"

    return (
        f"TASK_ID: {task.task_id}\n"
        f"TASK: {task.prompt}\n"
        f"WORKER_ID: {worker.worker_id}\n"
        f"WORKER_SKILLS: {skills}\n"
        f"WORKER_DESCRIPTION: {worker.description}\n"
        f"NODE_ID: {node.node_id}\n"
        f"NODE_INSTRUCTION: {node.instruction}\n"
        f"NODE_DEPENDENCIES: {node_dependencies}\n"
        f"NODE_REQUIRED_SKILLS: {node_required_skills}\n"
        f"NODE_OUTPUT_KEY: {node.output_key}\n"
        "DEPENDENCY_OUTPUTS:\n"
        f"{chr(10).join(dependency_lines)}\n\n"
        "Return the subtask result only."
    )

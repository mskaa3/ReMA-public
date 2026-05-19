from __future__ import annotations

from typing import Dict, Sequence

from .schema import (
    CANONICAL_SKILL_TAGS,
    DecompositionCandidate,
    SubtaskNode,
    TaskExample,
    WorkerPerformanceSnapshot,
    WorkerPoolConfig,
    WorkerSpec,
)


DECOMPOSER_SYSTEM_PROMPT = """You are the Decomposer controller.
Break the task into a compact DAG of atomic reasoning steps.
"""


SELECTOR_SYSTEM_PROMPT = """You are the Selector controller.
Assign workers to the DAG nodes.
"""


DECOMPOSER_ONE_SHOT_EXAMPLE = """ONE-SHOT EXAMPLE:
EXAMPLE_TASK: Solve for x: 2x + 3 = 11
EXAMPLE_OUTPUT:
<decomposition_plan>
SUMMARY: isolate x and compute the final value
FINAL_NODE_ID: 2
NODE_ID: 1
INSTRUCTION: rearrange the equation to isolate the variable term
DEPENDENCIES: none
REQUIRED_SKILLS: algebra
OUTPUT_KEY: isolated_equation
NODE_ID: 2
INSTRUCTION: compute the value of x and return the final answer
DEPENDENCIES: 1
REQUIRED_SKILLS: arithmetic
OUTPUT_KEY: final_answer
</decomposition_plan>"""


SELECTOR_ONE_SHOT_EXAMPLE = """ONE-SHOT EXAMPLE:
EXAMPLE_NODES_BY_ID:
1: deps=none | skills=algebra | output=isolated_equation | instruction=rearrange the equation to isolate the variable term
2: deps=1 | skills=arithmetic | output=final_answer | instruction=compute the value of x and return the final answer
EXAMPLE_WORKERS_BY_INDEX:
1: arithmetic_prealgebra_worker | skills=arithmetic,prealgebra,fractions,simplification | success=0.72 | avg_reward=0.44 | desc=Exact arithmetic, fractions, ratios, and simplification specialist.
2: algebra_symbolic_worker | skills=algebra,symbolic_manipulation,equations,polynomials | success=0.81 | avg_reward=0.57 | desc=Equation solving and symbolic algebra specialist.
EXAMPLE_OUTPUT:
<selection_plan>
1: 2
2: 1
</selection_plan>"""


WORKER_ONE_SHOT_EXAMPLE = """ONE-SHOT EXAMPLE:
NODE_INSTRUCTION: compute the value of x and return the final answer
FINAL_NODE: yes
DEPENDENCY_RESULTS:
- NODE 1 (rearrange the equation to isolate the variable term): 2x = 8
EXAMPLE_OUTPUT:
<worker_result>
4
</worker_result>"""


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


def render_selector_output_skeleton(node_ids: Sequence[str], worker_count: int) -> str:
    safe_worker_count = max(1, int(worker_count))
    lines = ["<selection_plan>"]
    for line_index, node_id in enumerate(node_ids, start=1):
        worker_index = ((line_index - 1) % safe_worker_count) + 1
        lines.append(f"{node_id}: {worker_index}")
    lines.append("</selection_plan>")
    return "\n".join(lines)


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
    max_nodes_hint: int | None = None,
    soft_max_hops_hint: int | None = None,
    hard_max_hops_hint: int | None = None,
) -> str:
    max_nodes = max(1, int(max_nodes_hint or 4))
    allowed_node_ids = ", ".join(str(i) for i in range(1, max_nodes + 1))
    skill_tags = ", ".join(CANONICAL_SKILL_TAGS)
    node_budget_lines = []
    if soft_max_hops_hint is not None:
        node_budget_lines.append(
            f"- Prefer at most {int(soft_max_hops_hint)} nodes; decompositions with more nodes are penalized."
        )
    node_budget_lines.append(f"- Hard node cap: at most {max_nodes} nodes.")
    node_budget_contract = "\n".join(node_budget_lines)
    if node_budget_contract:
        node_budget_contract += "\n"
    return (
        f"{DECOMPOSER_SYSTEM_PROMPT}\n\n"
        "OUTPUT CONTRACT:\n"
        "- Return exactly one <decomposition_plan> block and nothing else.\n"
        "- Allowed field keys: SUMMARY, FINAL_NODE_ID, NODE_ID, INSTRUCTION, DEPENDENCIES, REQUIRED_SKILLS, OUTPUT_KEY.\n"
        "- Use this exact skeleton:\n"
        "<decomposition_plan>\n"
        "SUMMARY: short summary\n"
        "FINAL_NODE_ID: 2\n"
        "NODE_ID: 1\n"
        "INSTRUCTION: short instruction\n"
        "DEPENDENCIES: none\n"
        "REQUIRED_SKILLS: algebra\n"
        "OUTPUT_KEY: partial_result\n"
        "NODE_ID: 2\n"
        "INSTRUCTION: produce the final answer\n"
        "DEPENDENCIES: 1\n"
        "REQUIRED_SKILLS: analysis\n"
        "OUTPUT_KEY: final_answer\n"
        "</decomposition_plan>\n"
        "- Every NODE_ID must be followed by exactly one INSTRUCTION, one DEPENDENCIES, one REQUIRED_SKILLS, and one OUTPUT_KEY line.\n"
        f"- Allowed node IDs: {allowed_node_ids}.\n"
        f"{node_budget_contract}"
        "- Allowed dependency tokens: `none` or comma-separated node IDs from the allowed set.\n"
        f"- REQUIRED_SKILLS must use only these abstract tags: {skill_tags}.\n"
        "- Use `none` only for pure routing or final-answer wrapper nodes; for substantive math steps choose at least one concrete REQUIRED_SKILLS tag.\n"
        "- Do not tailor the decomposition to a particular worker roster.\n"
        "- Use a DAG, not a linear chain unless the task truly requires one.\n"
        "- Use short node instructions, short summaries, and short snake_case OUTPUT_KEY values.\n"
        "- The final node must produce the final answer, be the last declared NODE_ID, and have no downstream dependents.\n"
        "- Forbidden output patterns: markdown fences, JSON, bullets, prose outside tags.\n\n"
        f"{DECOMPOSER_ONE_SHOT_EXAMPLE}\n\n"
        f"TASK_ID: {task.task_id}\n"
        f"TASK: {task.prompt}\n"
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

    selector_skeleton = render_selector_output_skeleton(ordered_node_ids, len(worker_pool.workers))
    allowed_worker_indices = ", ".join(str(index) for index in range(1, len(worker_pool.workers) + 1))
    return (
        f"{SELECTOR_SYSTEM_PROMPT}\n\n"
        "OUTPUT CONTRACT:\n"
        "- Return exactly one <selection_plan> block and nothing else.\n"
        "- Use this exact skeleton:\n"
        f"{selector_skeleton}\n"
        "- Treat the worker indices shown in the skeleton as format placeholders only; choose the actual best worker index for each node.\n"
        "- Use exactly one line per node in the form: `node_id: worker_index`.\n"
        "- The left side is the numeric node ID from NODES_BY_ID.\n"
        "- The right side is the worker index from WORKERS_BY_INDEX.\n"
        "- Assign exactly one worker to each node ID from the decomposition.\n"
        f"- Number of mapping lines must equal number of nodes ({len(ordered_node_ids)}).\n"
        f"- Allowed node IDs: {', '.join(ordered_node_ids)}.\n"
        f"- Allowed worker indices: {allowed_worker_indices}.\n"
        "- Do not skip nodes, do not add extra assignments, and do not assign multiple workers to one node.\n"
        "- Prefer the worker whose skills and past performance best match each node.\n"
        "- Do not repeat the task, decomposition, or worker descriptions in the output.\n"
        "- Forbidden output patterns: markdown fences, JSON, bullets, prose outside tags.\n\n"
        f"{SELECTOR_ONE_SHOT_EXAMPLE}\n\n"
        f"TASK_ID: {task.task_id}\n"
        f"TASK: {task.prompt}\n"
        f"FINAL_NODE_ID: {decomposition.final_node_id}\n"
        "NODES_BY_ID:\n"
        f"{chr(10).join(node_lines)}\n"
        "WORKERS_BY_INDEX:\n"
        f"{chr(10).join(worker_lines)}\n\n"
        "Return ONLY the <selection_plan> block."
    )


def render_worker_prompt(
    task: TaskExample,
    decomposition: DecompositionCandidate,
    node: SubtaskNode,
    worker: WorkerSpec,
    dependency_outputs: dict[str, str],
) -> str:
    node_map = decomposition.nodes_by_id()
    dependency_lines = []
    for dependency_id in node.dependencies:
        dependency_output = dependency_outputs.get(dependency_id, "").strip()
        dependency_node = node_map.get(dependency_id)
        dependency_instruction = dependency_node.instruction if dependency_node else ""
        rendered_output = dependency_output.replace("\n", "\n  ") if dependency_output else "[missing]"
        if dependency_instruction:
            dependency_lines.append(
                f"- NODE {dependency_id} ({dependency_instruction}): {rendered_output}"
            )
        else:
            dependency_lines.append(f"- NODE {dependency_id}: {rendered_output}")
    if not dependency_lines:
        dependency_lines.append("- none")

    skills = ",".join(worker.skills) if worker.skills else "none"
    node_dependencies = ",".join(node.dependencies) if node.dependencies else "none"
    node_required_skills = ",".join(node.required_skills) if node.required_skills else "none"
    is_final_node = decomposition.final_node_id == node.node_id
    final_node_contract = (
        "- This is the FINAL_NODE. Put only the final answer inside <worker_result>.\n"
        "- Do not include explanations, labels, sentences, or variable assignments such as `x = 4`; write only `4`.\n"
    )
    if not is_final_node:
        final_node_contract = (
            "- This is an intermediate node. Put only the minimal downstream-usable result inside <worker_result>.\n"
        )

    return (
        f"TASK_ID: {task.task_id}\n"
        f"TASK: {task.prompt}\n"
        f"WORKER_ID: {worker.worker_id}\n"
        f"WORKER_SKILLS: {skills}\n"
        f"WORKER_DESCRIPTION: {worker.description}\n"
        f"NODE_ID: {node.node_id}\n"
        f"NODE_INSTRUCTION: {node.instruction}\n"
        f"FINAL_NODE: {'yes' if is_final_node else 'no'}\n"
        f"NODE_DEPENDENCIES: {node_dependencies}\n"
        f"NODE_REQUIRED_SKILLS: {node_required_skills}\n"
        "OUTPUT CONTRACT:\n"
        "- Do only the current NODE_INSTRUCTION.\n"
        "- Use DEPENDENCY_RESULTS as the current working context when they are provided.\n"
        "- Treat dependency results as factual inputs from earlier nodes. If a dependency says `sin(alpha) = 1/2`, use that value directly.\n"
        "- Do not solve future nodes, repeat the full task, or add explanations unless the instruction explicitly asks for them.\n"
        "- Return exactly one <worker_result> block and nothing else.\n"
        "- Use this exact skeleton:\n"
        "<worker_result>\n"
        "concise result\n"
        "</worker_result>\n"
        "- Put only the node result inside the tags. Do not include field labels like `RESULT:` or `OUTPUT_KEY:`.\n"
        "- For expressions, equations, values, or short case splits, return just that content inside the tags.\n"
        "- If there are multiple items, keep them compact and separate them with `;` when possible.\n"
        f"{final_node_contract}"
        "- Forbidden output patterns: prose outside tags, markdown fences, JSON, bullets, or solving nodes that were not assigned.\n\n"
        f"{WORKER_ONE_SHOT_EXAMPLE}\n\n"
        "DEPENDENCY_RESULTS:\n"
        f"{chr(10).join(dependency_lines)}\n\n"
        "Return ONLY the <worker_result> block."
    )

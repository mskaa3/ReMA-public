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
Break each task into a compact DAG of concrete mathematical steps whose outputs are reusable by downstream nodes and end at the final answer.
"""


SELECTOR_SYSTEM_PROMPT = """You are the Selector controller.
Assign the most suitable worker to each decomposition node using the node requirements and the workers' capabilities and track record.
"""


SELECTOR_SYSTEM_PROMPT_NO_HISTORY = """You are the Selector controller.
Assign the most suitable worker to each decomposition node using the node requirements and the workers' capabilities.
"""


# WORKER_ONE_SHOT_EXAMPLE = """ONE-SHOT EXAMPLE:
# NODE_INSTRUCTION: compute the value of x and return the final answer
# FINAL_NODE: yes
# DEPENDENCY_RESULTS:
# - NODE 1 (rearrange the equation to isolate the variable term): 2x = 8
# EXAMPLE_OUTPUT:
# <worker_scratchpad>
# From 2x = 8, divide both sides by 2.
# </worker_scratchpad>
# <worker_result>
# 4
# </worker_result>"""


DEFAULT_ARITHMETIC_PREALGEBRA_WORKER_PROMPT = """You are an arithmetic and prealgebra worker.
Be exact with fractions, ratios, signs, and straightforward simplifications, and prefer exact forms over decimals unless the task asks for approximation.

"""


DEFAULT_ALGEBRA_SYMBOLIC_WORKER_PROMPT = """You are an algebra and symbolic manipulation worker.
Solve equations and carry out substitutions, factoring, expanding, and symbolic simplification carefully while keeping expressions exact.

"""


DEFAULT_GEOMETRY_TRIGONOMETRY_WORKER_PROMPT = """You are a geometry and trigonometry worker.
Use Euclidean or coordinate geometry, angle and length relations, and trigonometric identities precisely, keeping notation clean and exact.

"""


DEFAULT_CALCULUS_ANALYSIS_WORKER_PROMPT = """You are a calculus and analysis worker.
Reason carefully about limits, derivatives, integrals, continuity, extrema, and function behavior, and keep the result mathematically exact.

"""


DEFAULT_DISCRETE_NUMBER_THEORY_WORKER_PROMPT = """You are a discrete mathematics and number theory worker.
Use divisibility, modular arithmetic, parity, counting, combinatorics, invariants, and elementary probability with precise case splits or arithmetic constraints when helpful.

"""


# Backward-compatible aliases for any older imports that still expect the two-worker setup.
DEFAULT_ALGEBRA_WORKER_PROMPT = DEFAULT_ALGEBRA_SYMBOLIC_WORKER_PROMPT
DEFAULT_ANALYSIS_WORKER_PROMPT = DEFAULT_CALCULUS_ANALYSIS_WORKER_PROMPT


def _compact_performance_summary(snapshot: WorkerPerformanceSnapshot) -> dict:
    return {
        "num_assignments": snapshot.num_assignments,
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
                        average_reward=0.0,
                        average_confidence_reward=0.0,
                        average_compatibility=0.0,
                    )
                ),
            }
        )
    return worker_context


def render_selector_output_skeleton(node_ids: Sequence[str]) -> str:
    lines = ["<selection_plan>"]
    for node_id in node_ids:
        lines.append(f"{node_id}: best_worker_id_here")
    lines.append("</selection_plan>")
    return "\n".join(lines)


def _expected_worker_output_hint(
    decomposition: DecompositionCandidate,
    node: SubtaskNode,
) -> str:
    if decomposition.final_node_id == node.node_id:
        return "final_scalar_answer"

    instruction = node.instruction.lower()
    output_key = (node.output_key or "").lower()

    if "equation" in instruction or "equation" in output_key:
        return "transformed_equation"
    if any(token in instruction for token in ("expression", "simplify", "expand", "factor")):
        return "simplified_expression"
    if any(token in instruction for token in ("theorem", "method", "strategy", "approach", "choose")):
        return "chosen_method"
    if any(token in instruction for token in ("value", "count", "probability", "ratio")):
        return "intermediate_numeric_result"
    return "intermediate_mathematical_state"


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
        "OUTPUT CONTRACT:\n"
        "- Return exactly one <decomposition_plan> block and nothing else.\n"
        "- Use only these keys: SUMMARY, FINAL_NODE_ID, NODE_ID, INSTRUCTION, DEPENDENCIES, REQUIRED_SKILLS, OUTPUT_KEY.\n"
        "- Use this skeleton:\n"
        "<decomposition_plan>\n"
        "SUMMARY: short summary\n"
        "FINAL_NODE_ID: 2\n"
        "NODE_ID: 1\n"
        "INSTRUCTION: short instruction\n"
        "DEPENDENCIES: none\n"
        "REQUIRED_SKILLS: algebra\n"
        "NODE_ID: 2\n"
        "INSTRUCTION: produce the final answer\n"
        "DEPENDENCIES: 1\n"
        "REQUIRED_SKILLS: analysis\n"
        "</decomposition_plan>\n"
        f"- Allowed node IDs: {allowed_node_ids}. Use contiguous numeric NODE_ID values in declaration order: 1, 2, ..., N.\n"
        "- Each node must contain exactly one INSTRUCTION line and one DEPENDENCIES line.\n"
        f"{node_budget_contract}"
        "- DEPENDENCIES must be `none` or comma-separated earlier node IDs from the allowed set.\n"
        f"- REQUIRED_SKILLS is preferred for substantive nodes; when present, use only these tags: {skill_tags}.\n"
        "- Use at most 3 REQUIRED_SKILLS tags per node, and prefer 1-2 when possible.\n"
        "- Do not list every possible skill or repeat broad generic tags across all nodes; include only the tags truly needed for that node.\n"
        "- OUTPUT_KEY is optional and only for readability.\n"
        "- Do not tailor the decomposition to a particular worker roster.\n"
        "- Return a DAG, not a chain unless the task truly needs one.\n"
        "- Prefer a few smaller meaningful steps over a single node; use one node only when the task is genuinely atomic or cannot be usefully divided.\n"
        "- Keep the summary and node instructions short.\n"
        "- Every node instruction must name the concrete mathematical artifact it should output.\n"
        "- Prefer grounded instructions like `rewrite ... as ...`, `return the simplified expression ...`, `name the chosen method ...`, or `return the final scalar answer ...`.\n"
        "- For root nodes, reference the actual equation, expression, case split, or target quantity from TASK instead of vague text like `simplify both sides`.\n"
        "- For intermediate nodes, preserve reusable symbolic state; avoid bare numbers unless the node explicitly asks for a numeric sub-result.\n"
        "- The final node must be a terminal sink node that produces the final answer.\n"
        "- Forbidden output patterns: markdown fences, JSON, bullets, prose outside tags.\n\n"
        f"TASK_ID: {task.task_id}\n"
        f"TASK: {task.prompt}\n"
        "Return ONLY the <decomposition_plan> block."
    )


def render_selector_prompt(
    task: TaskExample,
    decomposition: DecompositionCandidate,
    worker_pool: WorkerPoolConfig,
    worker_performance: Dict[str, WorkerPerformanceSnapshot],
    track_workers_history: bool = True,
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
    for worker in worker_pool.workers:
        snapshot = worker_performance.get(worker.worker_id)
        skills = ",".join(worker.skills) if worker.skills else "none"
        if track_workers_history:
            avg_reward = snapshot.average_reward if snapshot is not None else 0.0
            worker_lines.append(
                f"- {worker.worker_id} | skills={skills} | avg_reward={avg_reward:.2f} | desc={worker.description}"
            )
        else:
            worker_lines.append(
                f"- {worker.worker_id} | skills={skills} | desc={worker.description}"
            )

    selector_skeleton = render_selector_output_skeleton(ordered_node_ids)
    allowed_worker_ids = ", ".join(worker.worker_id for worker in worker_pool.workers)
    worker_preference_line = (
        "- Prefer the worker whose skills and past performance best match each node.\n"
        if track_workers_history
        else "- Prefer the worker whose skills best match each node.\n"
    )
    return (
        "OUTPUT CONTRACT:\n"
        "- Return exactly one <selection_plan> block and nothing else.\n"
        "- Use this skeleton:\n"
        f"{selector_skeleton}\n"
        "- Use exactly one line per node in the form: `node_id: worker_id`.\n"
        "- Replace the placeholder worker ID with the actual best worker_id from WORKERS_BY_ID.\n"
        "- Assign exactly one worker to every node from NODES_BY_ID.\n"
        f"- The number of mapping lines must equal the number of nodes ({len(ordered_node_ids)}).\n"
        f"- Allowed node IDs: {', '.join(ordered_node_ids)}.\n"
        f"- Allowed worker IDs: {allowed_worker_ids}.\n"
        "- Do not skip nodes, add extra mappings, repeat a node, or assign multiple workers to one node.\n"
        f"{worker_preference_line}"
        "- Do not assign workers by list position or by a repeated numeric pattern like `1->1, 2->2, 3->3`; choose based on node requirements and worker fit.\n"
        "- Forbidden output patterns: markdown fences, JSON, bullets, prose outside tags.\n\n"
        f"TASK_ID: {task.task_id}\n"
        f"TASK: {task.prompt}\n"
        f"FINAL_NODE_ID: {decomposition.final_node_id}\n"
        "NODES_BY_ID:\n"
        f"{chr(10).join(node_lines)}\n"
        "WORKERS_BY_ID:\n"
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
    is_root_node = not node.dependencies
    expected_output_hint = _expected_worker_output_hint(decomposition, node)
    node_context_contract = ""
    if is_root_node:
        node_context_contract = (
            "- This node has no dependencies. The full TASK is provided below. Ground your result directly in TASK and explicitly carry forward the relevant equation, expression, or target quantity from the problem.\n"
            "- For root algebra or manipulation nodes, do not return a bare scalar unless NODE_INSTRUCTION explicitly asks for one.\n"
        )
    else:
        node_context_contract = (
            "- This node has dependencies. The full TASK is intentionally omitted.\n"
            "- Solve only the current NODE_INSTRUCTION using DEPENDENCY_RESULTS. Do not reconstruct or re-solve the entire task on your own.\n"
        )
    final_node_contract = (
        "- This is the FINAL_NODE. Put only the final answer inside <worker_result>.\n"
        "- Do not include explanations, labels, sentences, or variable assignments such as `x = 4`; write only `4`.\n"
    )
    if not is_final_node:
        final_node_contract = (
            "- This is an intermediate node. Put only the downstream-usable result inside <worker_result>.\n"
            f"- Suggested output shape for this node: {expected_output_hint}.\n"
        )

    return (
        "- Do only the current NODE_INSTRUCTION. Do not solve future nodes, repeat the full task, or add explanations unless the instruction explicitly asks for them.\n"
        "- Use DEPENDENCY_RESULTS as the current working context when they are provided.\n"
        "- Treat dependency results as factual inputs from earlier nodes. If a dependency says `sin(alpha) = 1/2`, use that value directly.\n"
        "- You may include at most one <worker_scratchpad> block before the final <worker_result> block.\n"
        "- If you use <worker_scratchpad>, make it a short but logically connected piece of reasoning that helps solve the current node.\n"
        "- The scratchpad should show useful mathematical thinking such as a transformation, substitution, case split, geometric fact, or intermediate deduction that moves from the available context toward the required node artifact.\n"
        "- Do not use the scratchpad as filler, a paraphrase of NODE_INSTRUCTION, or a copied template; it should contain reasoning that is genuinely helpful for solving this task.\n"
        "- Return exactly one <worker_result> block. Only the content inside <worker_result> is passed to downstream nodes, so put the final node artifact there.\n"
        "- Use one of these exact skeletons:\n"
        "<worker_scratchpad>\n"
        "Logical derivation of the solution\n"
        "</worker_scratchpad>\n"
        "<worker_result>\n"
        "concise result\n"
        "</worker_result>\n"
        "- Put only the node result inside the tags <worker_result>. Do not include field labels like `RESULT:` or `OUTPUT_KEY:`.\n"
        "- For expressions, equations, values, or short case splits, return just that content inside the tags.\n"
        "- Do not jump to a scalar too early; preserve an equation, expression, or other reusable symbolic state unless NODE_INSTRUCTION explicitly asks for a numeric result.\n"
        "- If there are multiple items, keep them compact and separate them with `;` when possible.\n"
        f"{node_context_contract}"
        f"{final_node_contract}"
        "- Forbidden output patterns: prose outside the allowed tags, markdown fences, JSON, bullets, or solving nodes that were not assigned.\n\n"
        # f"{WORKER_ONE_SHOT_EXAMPLE}\n\n"
        "Task details:\n"
        f"TASK_ID: {task.task_id}\n"
        f"{f'TASK: {task.prompt}\\n' if is_root_node else ''}"
        f"WORKER_ID: {worker.worker_id}\n"
        f"WORKER_SKILLS: {skills}\n"
        f"WORKER_DESCRIPTION: {worker.description}\n"
        f"NODE_ID: {node.node_id}\n"
        f"NODE_INSTRUCTION: {node.instruction}\n"
        f"NODE_REQUIRED_SKILLS: {node_required_skills}\n"
        f"FINAL_NODE: {'yes' if is_final_node else 'no'}\n"
        f"NODE_DEPENDENCIES: {node_dependencies}\n"
        f"DEPENDENCY_RESULTS:\n"
        f"{chr(10).join(dependency_lines)}\n\n"
        f"EXPECTED_OUTPUT_HINT: {expected_output_hint}\n"
        "Return ONLY an optional <worker_scratchpad> block followed by the required <worker_result> block.\n"
        "OUTPUT:\n"
    )

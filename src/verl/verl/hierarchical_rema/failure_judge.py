"""Optional validation-only audit; never contributes rewards or policy samples."""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys

from .failure_judge_contract import CATEGORIES, PROMPT, VERSION, artifact_text, digest, validate_annotation


@dataclass(frozen=True)
class FailureJudgeConfig:
    enabled: bool = False
    model: str = "google/gemma-4-31B-it"
    revision: str = "main"
    batch_size: int = 4
    max_model_len: int = 16384
    max_tokens: int = 3072
    gpu_memory_utilization: float = .85
    retries: int = 1
    timeout: float = 7200.
    python: str = sys.executable
    seed: int = 42

    @classmethod
    def from_env(cls):
        # Disabled means no dependency checks, model access, or GPU work.
        enabled = os.environ.get("REMA_MAST_ENABLED", "false").lower() in ("true", "1")
        if not enabled:
            return cls()
        config = cls(enabled=True,
            model=os.environ.get("REMA_MAST_MODEL", cls.model),
            revision=os.environ.get("REMA_MAST_REVISION", cls.revision),
            batch_size=int(os.environ.get("REMA_MAST_BATCH_SIZE", cls.batch_size)),
            max_model_len=int(os.environ.get("REMA_MAST_MAX_MODEL_LEN", cls.max_model_len)),
            max_tokens=int(os.environ.get("REMA_MAST_MAX_TOKENS", cls.max_tokens)),
            gpu_memory_utilization=float(os.environ.get("REMA_MAST_GPU_MEMORY_UTILIZATION", .85)),
            retries=int(os.environ.get("REMA_MAST_RETRIES", cls.retries)),
            timeout=float(os.environ.get("REMA_MAST_TIMEOUT", cls.timeout)),
            python=os.environ.get("REMA_MAST_PYTHON") or sys.executable,
            seed=int(os.environ.get("REMA_MAST_SEED", cls.seed)))
        if not config.model.strip() or not config.revision.strip():
            raise ValueError("Failure judge needs a model and revision")
        if not 1 <= config.batch_size <= 64 or not 0 <= config.retries <= 3:
            raise ValueError("Failure judge batch size must be 1..64 and retries 0..3")
        if not 1 <= config.max_tokens < config.max_model_len:
            raise ValueError("Failure judge output budget must be smaller than context limit")
        if not math.isfinite(config.timeout) or config.timeout <= 0:
            raise ValueError("Failure judge timeout must be positive and finite")
        if not 0 < config.gpu_memory_utilization < 1:
            raise ValueError("Failure judge GPU utilization must be in (0, 1)")
        return config


def project_trajectory(rollout, plan, execution, worker_pool):
    """Whitelist delivered fields; never serialize reward objects or raw outputs."""
    from .prompts import DECOMPOSER_SYSTEM_PROMPT, render_decomposer_prompt
    decomposition = plan.decomposition
    settings = rollout.rollout_config
    contract = render_decomposer_prompt(rollout.task,
        max_nodes_hint=settings.max_nodes_per_decomposition,
        soft_max_hops_hint=settings.soft_max_hops,
        hard_max_hops_hint=settings.hard_max_hops).split("ILLUSTRATIVE EXAMPLES", 1)[0]
    contract = "\n".join(line for line in contract.splitlines() if "scratchpad" not in line)
    workers = {worker.worker_id: worker for worker in worker_pool.workers}
    nodes = [{"node_id": str(node.node_id), "instruction": artifact_text(node.instruction),
              "dependencies": [str(dep) for dep in node.dependencies]}
             for node in decomposition.nodes]
    node_map = {node["node_id"]: node for node in nodes}
    results = []
    for result in execution.executions:
        node = node_map.get(str(result.node_id))
        dependencies = getattr(result, "dependency_outputs", None)
        # Empty delivered output remains empty, even when raw_output_text is populated.
        results.append({"node_id": str(result.node_id), "worker_id": str(result.worker_id),
            "task_context_expected": (not node["dependencies"]) if node else None,
            "dependency_inputs": ({str(key): artifact_text(value) for key, value in dependencies.items()}
                                  if isinstance(dependencies, dict) else None),
            "output": artifact_text(result.output_text)})
    used_workers = {item["worker_id"] for item in results}
    return {
        "task": artifact_text(rollout.task.prompt),
        "roles": {"planner": DECOMPOSER_SYSTEM_PROMPT + "\n" + contract,
                  "workers": {key: artifact_text(workers[key].system_prompt)
                              for key in sorted(used_workers) if key in workers}},
        "plan": {"target_quantity": artifact_text(decomposition.target_quantity),
                 "final_answer_format": artifact_text(decomposition.final_answer_format_hint),
                 "final_node_id": str(decomposition.final_node_id), "nodes": nodes},
        "executions": results, "final_answer": artifact_text(execution.final_answer),
        "observation_limits": [
            "Structured executed plan, not raw planner generation; repairs are not assessed.",
            "Only root workers are intended to receive the whole task.",
            "Dependency snapshots record assembled inputs, not post-tokenization visibility.",
            "Null inputs mean missing records, not proof that no inputs were provided.",
            "Private reasoning, rejected raw output, reference and correctness verdict are excluded.",
        ],
    }


def aggregate(rows):
    categories = {}
    total = len(rows)
    for code, name in CATEGORIES.items():
        task_values = defaultdict(list)
        statuses = Counter()
        positives = assessed = 0
        for row in rows:
            item = row.get("annotation", {}).get("categories", {}).get(code)
            status = item["status"] if item else row["status"]
            statuses[status] += 1
            if item and item["value"] is not None:
                assessed += 1
                positives += item["value"]
                task_values[(row["dataset"], row["task_id"])].append(item["value"])
        categories[code] = {"name": name,
            "rate": (sum(sum(values) / len(values) for values in task_values.values()) / len(task_values)
                     if task_values else None),
            "positive_rollouts": positives, "assessed_rollouts": assessed,
            "assessed_tasks": len(task_values), "total_rollouts": total,
            "coverage": assessed / total if total else None, "status_counts": dict(statuses)}
    return {"total_rollouts": total, "status_counts": dict(Counter(row["status"] for row in rows)),
            "categories": categories}


class FailureJudgeAudit:
    def __init__(self, worker_pool, config=None):
        self.config = config or FailureJudgeConfig.from_env()
        self.worker_pool, self.rows = worker_pool, []

    def add(self, rollout, dataset):
        if not self.config.enabled:
            return
        for plan in rollout.decompositions:
            for execution in plan.selections:
                identity = {"dataset": dataset, "task_id": str(rollout.task.task_id),
                            "decomposition_id": str(plan.decomposition.decomposition_id),
                            "execution_id": str(execution.selection.selection_id)}
                try:
                    document = project_trajectory(rollout, plan, execution, self.worker_pool)
                    self.rows.append({**identity, "input_id": digest(document),
                                      "input": document, "status": "pending"})
                except (AttributeError, TypeError, ValueError) as exc:
                    self.rows.append({**identity, "status": "projection_error", "error": type(exc).__name__})

    def finish(self, output_dir, runner=None):
        metadata = {"enabled": self.config.enabled, "version": VERSION,
                    "model": self.config.model, "requested_revision": self.config.revision,
                    "prompt_sha256": digest(PROMPT), "prompt": PROMPT,
                    "taxonomy_source": "https://arxiv.org/abs/2503.13657",
                    "adaptation": "Eight result-only categories; not full-taxonomy replication.",
                    "rate_definition": "Mean within task of assessed rollout labels, then equal mean across tasks.",
                    "coverage_definition": "Assessed rollouts / all observed rollouts, separately per category.",
                    "configuration": {k: v for k, v in asdict(self.config).items() if k != "python"}}
        pending = {row["input_id"]: row["input"] for row in self.rows if row["status"] == "pending"}
        if self.config.enabled and pending:
            from .failure_judge_runtime import run_local_judge, ResourceRestoreError
            try:
                report = (runner or run_local_judge)(self.config, pending, Path(output_dir))
                metadata.update(report.get("metadata", {}))
                results = report.get("results", {})
                for row in self.rows:
                    if row["status"] != "pending":
                        continue
                    result = results.get(row["input_id"], {"status": "unavailable"})
                    row.update({key: result[key] for key in ("status", "error", "input_tokens", "attempts", "cache_hit")
                                if key in result})
                    if row["status"] == "ok":
                        try:
                            row["annotation"] = validate_annotation(result.get("annotation"), row["input"])
                        except (ValueError, TypeError, KeyError):
                            row.update(status="invalid_annotation", error="Evidence/schema validation failed")
                    if row["status"] == "pending":
                        row["status"] = "unavailable"
            except ResourceRestoreError:
                # Continuing with a sleeping reward provider would corrupt/hang the next update.
                raise
            except Exception as exc:
                metadata["runtime_error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
                for row in self.rows:
                    if row["status"] == "pending":
                        row.update(status="unavailable", error=metadata["runtime_error"])
        overall = aggregate(self.rows)
        if self.config.enabled:
            print(f"[failure-judge] status={overall['status_counts']} "
                  f"details={Path(output_dir) / 'validation_failures.jsonl'}", flush=True)
            if any(row["status"] != "ok" for row in self.rows):
                print("[failure-judge] WARNING: unavailable judgments are excluded, not counted as clean. "
                      "Inspect failure_judge.log and failure_judge_definition.json.", flush=True)
        return {"metadata": metadata, "overall": overall, "rows": self.rows,
                "subsets": {name: aggregate([row for row in self.rows if row["dataset"] == name])
                            for name in sorted({row["dataset"] for row in self.rows})}}

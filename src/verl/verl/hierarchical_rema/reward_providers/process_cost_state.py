"""Driver-owned dual state. Scorers receive immutable snapshots, never update it."""
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .process_costs import COSTS, SETTING, VERSION, ProcessCostConfig, validate_lambdas


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=8)
def _checkpoint_digest(path, size, mtime_ns):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".process-cost-", delete=False) as handle:
            temporary = handle.name
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def segment_statistics(rollouts):
    """Equal task/plan weighting; ordinary repeats OR complete branch groups.

    Never use training eligibility, advantage signs or best-selection filters.
    In joint mode ordinary executions estimate plan-level costs, without also
    counting their correlated fixed-prefix branches. In executor-only prefix
    mode each action/continuation/branch point receives equal nested weight.
    """
    tasks, identities, snapshots = [], [], set()
    plans_total = plans_complete = graphs_seen = graphs_scored = 0
    for task in rollouts:
        plans = []
        for plan in task.decompositions:
            plans_total += 1
            ordinary = [s for s in plan.selections if not s.branch_metadata]
            selected = ordinary or list(plan.selections)
            entries, groups = [], {}
            valid = bool(selected)
            for selection in selected:
                graphs_seen += 1
                summary = selection.reward_model_outputs.get("graph_summary", {})
                costs = summary.get("process_costs")
                y = selection.reward.final_answer_correctness
                available = (selection.reward_model_outputs.get("status") == "scored"
                             and selection.reward_model_outputs.get("source") == SETTING
                             and type(y) in (int, float) and y in (0, 1)
                             and isinstance(costs, dict) and set(costs) == set(COSTS))
                if available:
                    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                           for v in costs.values()):
                        raise ValueError("Invalid process-cost statistics")
                    graphs_scored += 1
                    snapshots.add(json.dumps([summary["process_cost_update_count"],
                                               summary["process_cost_lambdas"]], sort_keys=True))
                else:
                    valid = False
                identities.append([task.task.task_id, plan.decomposition.decomposition_id,
                                   selection.selection.selection_id, costs if available else None])
                if ordinary:
                    if available:
                        entries.append({**costs, "correctness": y})
                else:
                    meta = selection.branch_metadata
                    group = groups.setdefault(meta["group_id"], {})
                    key = (meta["action_index"], meta["continuation_index"])
                    if key in group:
                        raise ValueError("Duplicate continuation in process-cost statistics")
                    group[key] = ({**costs, "correctness": y} if available else None, meta)
            if not ordinary:
                for comparison in plan.worker_comparisons:
                    if comparison.get("status") == "skipped" and not comparison.get("group_id"):
                        valid = False
                    elif comparison.get("group_id") not in groups:
                        valid = False
                for group in groups.values():
                    meta = next(iter(group.values()))[1]
                    count, repeats = meta["expected_actions"], meta["expected_continuations"]
                    expected = {(a, c) for a in range(count) for c in range(repeats)}
                    if set(group) != expected or any(value is None for value, _ in group.values()):
                        valid = False
                        continue
                    # Equal repeats per action; this mean is the nested action mean.
                    entries.append({k: _mean([group[key][0][k] for key in sorted(expected)])
                                    for k in (*COSTS, "correctness")})
            if valid and entries:
                plans_complete += 1
                plans.append({k: _mean([entry[k] for entry in entries]) for k in (*COSTS, "correctness")})
        if plans:
            tasks.append({k: _mean([plan[k] for plan in plans]) for k in (*COSTS, "correctness")})
    if len(snapshots) > 1:
        raise ValueError("Penalty weights changed within a rollout/update segment")
    return {
        "mean_costs": {k: _mean([task[k] for task in tasks]) for k in COSTS},
        "mean_correctness": _mean([task["correctness"] for task in tasks]),
        "plans_total": plans_total, "plans_complete": plans_complete,
        "coverage": plans_complete / plans_total if plans_total else 0.,
        "graphs_seen": graphs_seen, "graphs_scored": graphs_scored,
        "graph_coverage": graphs_scored / graphs_seen if graphs_seen else 0.,
        "snapshot": json.loads(next(iter(snapshots))) if snapshots else None,
        "fingerprint": _digest(identities),
        "aggregation": "equal_tasks_plans_complete_repeats_or_branch_groups_v1",
    }


class ProcessCostState:
    def __init__(self, path, config, checkpoint_sha256, initial_state=None):
        self.path, self.config = Path(path), config
        self.identity = {"version": VERSION, "config": asdict(config), "checkpoint_sha256": checkpoint_sha256}
        if self.path.exists():
            self.read()
        else:
            if initial_state:
                state = json.loads(Path(initial_state).expanduser().read_text())
                self.validate(state)
                state = {**state, "events": {}, "restored_from": str(initial_state)}
            else:
                state = {**self.identity, "lambdas": {k: getattr(config, "initial_" + k) for k in COSTS},
                         "update_count": 0, "events": {}}
            atomic_json(self.path, state)

    def validate(self, state):
        if any(state.get(k) != v for k, v in self.identity.items()):
            raise ValueError("Process-cost state belongs to a different configuration/checkpoint")
        validate_lambdas(state["lambdas"])
        if type(state.get("update_count")) is not int or state["update_count"] < 0 or not isinstance(state.get("events"), dict):
            raise ValueError("Malformed process-cost state")

    def read(self):
        state = json.loads(self.path.read_text())
        self.validate(state)
        return state

    def finish(self, segment_id, statistics):
        state = self.read()
        if segment_id in state["events"]:
            event = state["events"][segment_id]
            if event["statistics"] != statistics:
                raise ValueError("Segment already updated with different rollouts; do not reuse an output directory")
            return event
        expected = [state["update_count"], state["lambdas"]]
        if statistics["snapshot"] is not None and statistics["snapshot"] != expected:
            raise ValueError("Stale process-cost snapshot; refusing a duplicate/out-of-order dual update")
        before = dict(state["lambdas"])
        sufficient = (statistics["coverage"] >= self.config.min_coverage
                      and all(statistics["mean_costs"][k] is not None for k in COSTS))
        if sufficient:
            for cost in COSTS:
                state["lambdas"][cost] = max(0., before[cost] + self.config.learning_rate * (
                    statistics["mean_costs"][cost] - getattr(self.config, "target_" + cost)))
            state["update_count"] += 1
        event = {"segment": segment_id, "statistics": statistics, "before": before,
                 "after": dict(state["lambdas"]), "updated": sufficient,
                 "reason": "dual_update" if sufficient else "insufficient_complete_plan_coverage",
                 "update_count": state["update_count"], "identity": self.identity}
        state["events"][segment_id] = event
        atomic_json(self.path, state)
        return event


def state_for_args(args):
    checkpoint = Path(args.gfam_reward_model_pkl).expanduser().resolve()
    stat = checkpoint.stat()
    digest = _checkpoint_digest(str(checkpoint), stat.st_size, stat.st_mtime_ns)
    return ProcessCostState(Path(args.output_dir) / "process_costs" / "state.json",
                            ProcessCostConfig.from_environment(), digest,
                            os.environ.get("REMA_PROCESS_COST_INITIAL_STATE") or None)


def finish_process_cost_segment(args, rollouts, segment_dir, tracking=None, step=0):
    if getattr(args, "reward_setting", None) != SETTING:
        return None
    segment_dir = Path(segment_dir)
    state = state_for_args(args)
    event = state.finish(str(segment_dir.resolve()), segment_statistics(rollouts))
    atomic_json(segment_dir / "process_cost_update.json", event)
    # Portable sidecar for the policy checkpoints produced by this segment.
    # The root state retains the full ledger; a restored sidecar starts a new ledger.
    atomic_json(segment_dir / "process_cost_state.json", {
        **state.identity, "lambdas": event["after"], "update_count": event["update_count"], "events": {}})
    metrics = {"process_costs/coverage": event["statistics"]["coverage"],
               "process_costs/graph_coverage": event["statistics"]["graph_coverage"],
               "process_costs/dual_updates": event["update_count"],
               "process_costs/update_applied": int(event["updated"])}
    for cost in COSTS:
        metrics[f"process_costs/lambda/{cost}"] = event["after"][cost]
        metrics[f"process_costs/target/{cost}"] = event["identity"]["config"]["target_" + cost]
        mean = event["statistics"]["mean_costs"][cost]
        if mean is not None:
            metrics[f"process_costs/mean/{cost}"] = mean
    if event["statistics"]["mean_correctness"] is not None:
        metrics["process_costs/verified_correctness"] = event["statistics"]["mean_correctness"]
    if tracking is not None:
        tracking.log(metrics, step=step)
    print(f"[process-costs] segment={segment_dir.name} updated={event['updated']} "
          f"coverage={event['statistics']['coverage']:.3f} lambdas={event['after']}", flush=True)
    return event

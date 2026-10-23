"""Role-aware verified-outcome rewards, independent of the legacy reward gates."""
from dataclasses import asdict, dataclass
import math
import os

from .base import validate_result

SETTING = "graph_prm_process_costs"
VERSION = "role_aware_process_costs_v3"
COSTS = ("collapse", "discipline", "task_fulfillment", "planner_discipline", "planner_task_fulfillment")


@dataclass(frozen=True)
class ProcessCostConfig:
    target_collapse: float = .10
    target_discipline: float = .10
    target_task_fulfillment: float = .10
    target_planner_discipline: float = .10
    target_planner_task_fulfillment: float = .10
    initial_collapse: float = .10
    initial_discipline: float = .10
    initial_task_fulfillment: float = .10
    initial_planner_discipline: float = .10
    initial_planner_task_fulfillment: float = .10
    learning_rate: float = .01
    min_coverage: float = .95

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Process-cost {name} must be finite and nonnegative")
            if (name.startswith("target_") or name == "min_coverage") and value > 1:
                raise ValueError(f"Process-cost {name} must be in [0, 1]")
        if self.min_coverage == 0:
            raise ValueError("Process-cost min_coverage must be positive")

    @classmethod
    def from_environment(cls):
        defaults = cls()
        return cls(**{name: float(os.environ.get("REMA_PROCESS_COST_" + name.upper(), value))
                      for name, value in asdict(defaults).items()})


def validate_lambdas(values):
    if set(values) != set(COSTS) or any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values.values()
    ):
        raise ValueError("Process-cost multipliers must contain finite nonnegative " + ", ".join(COSTS))
    return dict(values)


def failure_probability(item):
    p = item["probabilities"]
    if set(p) != {"0", "1"} or any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()
    ) or not math.isclose(p["0"] + p["1"], 1., abs_tol=1e-5):
        raise ValueError("Process costs require normalized binary probabilities")
    return 1. - p["1"]


def compile_process_costs(predictions, request, lambdas, update_count):
    outcome = request.get("verified_final_correctness")
    if type(outcome) not in (int, float) or outcome not in (0, 1):
        raise ValueError("Process costs require verified binary final correctness")
    lambdas = validate_lambdas(lambdas)
    final_id = str(request["decomposition"].final_node_id)
    executions = request["executions"]
    ids = [str(ex.node_id) for ex in executions]
    if not ids or len(set(ids)) != len(ids) or ids.count(final_id) != 1:
        raise ValueError("Process costs require unique executed nodes including exactly one final node")
    if set(predictions["workers"]) != set(ids):
        raise ValueError("Process-cost prediction/execution node IDs differ")
    collapse = failure_probability(predictions["graph"]["anti_collapse"])
    planner_discipline = failure_probability(predictions["decomposer"]["execution_discipline"])
    planner_task_fulfillment = failure_probability(predictions["decomposer"]["task_fulfillment"])
    workers, costs, predicted_costs, invalid = {}, {}, {}, []
    task_costs = {}
    for ex in executions:
        node = str(ex.node_id)
        # The separate final head is the authoritative head for the final execution.
        labels = predictions["final"] if node == final_id else predictions["workers"][node]
        predicted = failure_probability(labels["execution_discipline"])
        task_cost = failure_probability(labels["task_fulfillment"])
        protocol_invalid = bool(ex.invalid_reason) or not str(ex.output_text or "").strip()
        cost = 1. if protocol_invalid else predicted
        if protocol_invalid:
            invalid.append(node)
        costs[node], predicted_costs[node] = cost, predicted
        task_costs[node] = task_cost
        value = (float(outcome) - lambdas["discipline"] * cost
                 - lambdas["task_fulfillment"] * task_cost)
        workers[node] = {"reward": value, "reward_raw": value, "verified_outcome": float(outcome),
                         "discipline_cost": cost, "predicted_discipline_cost": predicted,
                         "task_fulfillment_cost": task_cost,
                         "protocol_cost_override": protocol_invalid}
    mean_discipline = math.fsum(costs.values()) / len(costs)
    mean_task_fulfillment = math.fsum(task_costs.values()) / len(task_costs)
    planner = (float(outcome) - lambdas["collapse"] * collapse
               - lambdas["planner_discipline"] * planner_discipline
               - lambdas["planner_task_fulfillment"] * planner_task_fulfillment
               - lambdas["discipline"] * mean_discipline
               - lambdas["task_fulfillment"] * mean_task_fulfillment)
    return {
        "compiled_rewards": {"workers": workers, "final": dict(workers[final_id]),
                             "decomposer": {"reward": planner, "reward_raw": planner,
                                            "verified_outcome": float(outcome), "collapse_cost": collapse,
                                            "discipline_cost": planner_discipline,
                                            "task_fulfillment_cost": planner_task_fulfillment,
                                            "mean_worker_discipline_cost": mean_discipline,
                                            "mean_worker_task_fulfillment_cost": mean_task_fulfillment}},
        "graph_summary": {
            "reward_compiler": VERSION, "verified_final_correctness": float(outcome),
            "execution_outcome": float(outcome), "execution_outcome_source": "verified",
            "process_costs": {"collapse": collapse, "discipline": mean_discipline,
                              "task_fulfillment": mean_task_fulfillment,
                              "planner_discipline": planner_discipline,
                              "planner_task_fulfillment": planner_task_fulfillment},
            "node_discipline_costs": costs, "node_predicted_discipline_costs": predicted_costs,
            "node_task_fulfillment_costs": task_costs,
            "process_cost_lambdas": dict(lambdas), "process_cost_update_count": update_count,
            "protocol_invalid_workers": invalid, "final_head_used_for_final_execution": True,
            "legacy_reward_gates_applied": False,
        },
    }


class ProcessCostProvider:
    source = SETTING
    supports_verified_final_correctness = True
    required_auxiliary_checks = ()
    uses_process_costs = True

    def __init__(self, scorer, lambdas, update_count=0):
        if not callable(getattr(scorer, "predict_rollouts", None)):
            raise ValueError("graph_prm_process_costs requires a v2 GraphPRM checkpoint with raw predictions")
        self.scorer = scorer
        self.lambdas = validate_lambdas(lambdas)
        self.update_count = update_count
        self.rollout_batch_size = scorer.rollout_batch_size

    def score_rollouts(self, requests):
        results = [None] * len(requests)
        valid = []
        for index, request in enumerate(requests):
            y = request.get("verified_final_correctness")
            if type(y) not in (int, float) or y not in (0, 1):
                results[index] = {"source": self.source, "status": "unscored", "compiled_rewards": {},
                                  "error": {"code": "verified_outcome_required",
                                            "message": "No predicted-outcome fallback in process-cost mode"}}
            else:
                valid.append(index)
        # Verifier evidence is only for compilation, never part of neural inputs.
        inputs = [{k: v for k, v in requests[i].items() if k != "verified_final_correctness"} for i in valid]
        raw = self.scorer.predict_rollouts(inputs) if inputs else []
        if len(raw) != len(valid):
            raise ValueError("GraphPRM returned misaligned process-cost results")
        for index, result in zip(valid, raw):
            if result.get("status") == "unscored":
                results[index] = {**result, "source": self.source}
                continue
            compiled = compile_process_costs(result["model_predictions"], requests[index],
                                             self.lambdas, self.update_count)
            results[index] = validate_result({
                **result, "source": self.source,
                "compiled_rewards": compiled["compiled_rewards"],
                "graph_summary": {**result.get("graph_summary", {}), **compiled["graph_summary"]},
            }, requests[index])
        return results

    def score_rollout(self, **request):
        return self.score_rollouts([request])[0]

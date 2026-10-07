"""Inference-time reward policy, independent of learned weights and labels.

Provided edges only encode available information. They NEVER create a downstream
blame/reach penalty. Plan execution feedback is gated by predicted plan quality.
"""
import math
from .graphprm_inputs import apply_protocol_reward_guards


# Provisional symmetric baseline, not learned or empirically calibrated weights.
DECOMPOSER_LABEL_WEIGHTS = dict(task_fulfillment=.5, execution_discipline=.5)
EXECUTION_LABEL_WEIGHTS = dict(task_fulfillment=1/3, execution_discipline=1/3, causal_utility=1/3)


def compile_rewards(predictions, record, bad_class_penalty=1.0, verified_final_correctness=None):
    if not math.isfinite(bad_class_penalty) or bad_class_penalty < 0:
        raise ValueError('bad_class_penalty must be finite and nonnegative')
    def utility(item):
        p = item['probabilities']
        return p['1'] - bad_class_penalty * p['0']
    def quality(labels, weights):
        return sum(weight * utility(labels[name]) for name, weight in weights.items())
    graph, plan = predictions['graph'], predictions['decomposer']
    p_safe = graph['anti_collapse']['probabilities']['1']
    p_success = graph['global_outcome']['probabilities']['1']
    if verified_final_correctness is not None:
        verified_final_correctness = float(verified_final_correctness)
        if not math.isfinite(verified_final_correctness) or verified_final_correctness not in (0., 1.):
            raise ValueError('verified_final_correctness must be binary or None')
    # Evaluation evidence is supplied separately, never through model features.
    outcome = p_success if verified_final_correctness is None else verified_final_correctness
    plan_good = min(plan[n]['probabilities']['1'] for n in plan)
    weights = EXECUTION_LABEL_WEIGHTS
    def local(labels):
        q = quality(labels, weights)
        # Positive local credit is attenuated, not made negative solely by failure elsewhere.
        r = min(q, 0.) + max(q, 0.) * (.5 + .5 * p_safe)
        return {'reward': r, 'reward_raw': r, 'local_quality': q}
    workers = {node: local(labels) for node, labels in predictions['workers'].items()}
    final = local(predictions['final'])
    plan_quality = quality(plan, DECOMPOSER_LABEL_WEIGHTS)
    plan_bonus = .15 * plan_good * p_safe * outcome
    failure_penalty = .15 * (1 - plan_good) * (1 - outcome)
    collapse_penalty = .20 * (1 - p_safe)
    plan_penalty = failure_penalty + collapse_penalty
    plan_reward = min(plan_quality, 0.) + max(plan_quality, 0.) * p_safe + plan_bonus - plan_penalty
    node_rewards = {'decomposer': {'reward': plan_reward, 'reward_raw': plan_reward,
                    'local_quality': plan_quality, 'execution_bonus': plan_bonus,
                    'plan_consistency_penalty': plan_penalty,
                    'failure_penalty': failure_penalty, 'collapse_penalty': collapse_penalty,
                    'plan_quality_gate': plan_good, 'anti_collapse_gate': p_safe},
                    'workers': workers, 'final': final}
    invalid = apply_protocol_reward_guards(node_rewards, record)
    return {'node_rewards': node_rewards, 'graph_summary': {
        'reward_compiler': 'provided_dependencies_v2_verified_outcome', 'bad_class_penalty': bad_class_penalty,
        'reward_weight_scheme': 'equal_labels_v1',
        'label_weights': {'decomposer': dict(DECOMPOSER_LABEL_WEIGHTS),
                          'worker': dict(EXECUTION_LABEL_WEIGHTS),
                          'final': dict(EXECUTION_LABEL_WEIGHTS)},
        'predicted_success': p_success, 'predicted_anti_collapse': p_safe,
        'verified_final_correctness': verified_final_correctness,
        'execution_outcome': outcome,
        'execution_outcome_source': 'verified' if verified_final_correctness is not None else 'predicted',
        'downstream_causal_penalties_enabled': False, 'protocol_invalid_workers': invalid}}

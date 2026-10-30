#!/usr/bin/env python3
"""CPU reference checks. Requires math-verify 0.7.0, not model weights or a GPU."""
import argparse
import importlib.metadata
import importlib.util
import sys
from pathlib import Path

import build_review as draft
from check_math import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New JSON path; never overwritten')
    args = parser.parse_args()
    assert importlib.metadata.version('math-verify') == '0.7.0', 'Use the training verifier version'
    root = Path(__file__).resolve().parent
    evaluation = draft.evaluation
    data = evaluation.load_dataset(root / 'frozen/dataset.json')
    review = evaluation.read_json(root / 'review.json')
    assert evaluation.digest(review) == data['study']['review_sha256']
    # Load the actual production comparator without importing the GPU/Ray package.
    import local_results
    sys.modules['verl.rema_separated_trainer.ppo.local_results'] = local_results
    spec = importlib.util.spec_from_file_location('review_production_probe', evaluation.PPO / 'prefix_probe.py')
    probe = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = probe
    spec.loader.exec_module(probe)

    def equal(a, b):
        return probe.compare_worker_final_answers('LOCAL_RESULT: \\boxed{' + a + '}', '\\boxed{' + b + '}')

    refs = sorted({ref for row in data['examples'] for ref in [*row['reference_answers'], row['final_answer']]})
    for ref in refs:
        assert equal(ref, ref) is True, ('unparsable reference', ref)
    included = [r for r in review['examples'] if r['decision'] == 'include']
    for row in included:
        assert row['tasks'] == draft.PLANS[row['question_number']], 'Draft and frozen review differ'
        final = row.get('canonical_final_answer', row['final_answer'])
        assert equal(row['tasks'][-1]['reference_answers'][0], final) is True, row['question_number']
    controls = [('7', '8'), ('(1,2)', '(2,1)'),
                (r'5.3\times10^{-9}', r'9.9\times10^{-9}'),
                (r'3.61\times10^{-10}', r'4.61\times10^{-10}')]
    for left, right in controls:
        assert equal(left, right) is False, ('negative control accepted', left, right)
    exact = run(draft.PLANS)

    import sympy as s
    x, m, a, b, c = s.symbols('x m a b c')
    assert s.expand((m*m-m+s.Rational(1,2))*(m*m+m+s.Rational(1,2)) - m**4-s.Rational(1,4)) == 0
    assert s.expand((x-1)*(x+1)*(x*x+1)*(x*x-2*x+2)*(x*x+2*x+2)-(x**8+3*x**4-4)) == 0
    assert s.expand(a*b*(a*a-b*b)+b*c*(b*b-c*c)+c*a*(c*c-a*a)
                    +(a-b)*(b-c)*(c-a)*(a+b+c)) == 0
    assert s.limit(x/(x*x/3+1)**2/s.sqrt(2), x, 1) == 9*s.sqrt(2)/32
    report = dict(dataset_sha256=evaluation.digest(data), review_sha256=evaluation.digest(review),
        verifier_version=importlib.metadata.version('math-verify'), sympy_version=s.__version__,
        unique_reference_self_checks=len(refs), terminal_reference_checks=len(included),
        negative_controls=len(controls), exact_arithmetic_and_enumeration_checks=len(exact),
        questions_with_exact_checks=len({c['question_number'] for c in exact}),
        additional_symbolic_checks=4, source_hashes=evaluation.source_hashes(),
        model_inference_run=False, independent_review=False,
        caveat='Self-parsing is not proof of mathematical correctness. Exact computations cover a subset; '
               'the remaining derivations are model self-review, not independently certified proofs.')
    evaluation.write_json(args.output, report)
    print(f"Verified {len(refs)} reference strings, {len(included)} terminal matches, and {len(exact)} exact checks.")


if __name__ == '__main__':
    main()

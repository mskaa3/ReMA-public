"""Real parser regression controls, including misleading and adversarial answers."""

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src/verl/verl/rema_separated_trainer/ppo'))
import eac3_math_answers as answers
from local_results import extract_local_result


class BoundaryTests(unittest.TestCase):
    def test_attributed_quote_and_own_result(self):
        for quote in ('Given S1 LOCAL_RESULT: \\boxed{8}',
                      '1. S2 LOCAL_RESULT: 8', 'S3 LOCAL_RESULT: \\boxed{8}'):
            self.assertEqual(extract_local_result(quote + '\nLOCAL_RESULT: \\boxed{24}'), r'\boxed{24}')

    def test_do_not_select_among_answers_or_fall_back_to_quote(self):
        for text in ('S1 LOCAL_RESULT: \\boxed{8}',
                     'S1 LOCAL_RESULT: \\boxed{8}\nLOCAL_RESULT: \\boxed{',
                     'LOCAL_RESULT: \\boxed{8}\nLOCAL_RESULT: \\boxed{24}',
                     'Given LOCAL_RESULT: \\boxed{8}\nLOCAL_RESULT: \\boxed{24}',
                     r'LOCAL_RESULT: \boxed{8} or \boxed{24}'):
            self.assertEqual(extract_local_result(text), '')


@unittest.skipUnless(importlib.util.find_spec('math_verify'), 'Requires math-verify integration environment')
class MathTests(unittest.TestCase):
    def check(self, answer, reference, expected, task=''):
        self.assertIs(answers.equivalent((answer, [reference], task)), expected)

    def test_numeric_equations_compare_values_not_truth(self):
        self.check(r'7\times289=17\times119', '2123=11*193', False)
        self.check('2123=11*193', '2123', True)
        self.check('2+2=4', '2123=11*193', False)
        self.check('2+2=5', '5', None)
        self.check('x=2123', '2123', True)
        self.check('y_1=-0.5', '-0.5', True)
        self.check('False', 'True', None)

    def test_labels_and_units_preserve_values_and_order(self):
        self.check(r'150 \text{ Trekking bikes, 45 BMX bikes, 105 MTB bikes}', '(150,45,105)', True,
                   'Return Trekking, BMX, MTB bike counts.')
        self.check(r'140 \text{ rubies},280 \text{ emeralds}', '(140,280)', True,
                   'Return (rubies, emeralds).')
        self.check('280 rubies,140 emeralds', '(140,280)', False, 'Return rubies, emeralds.')
        self.check('Carlos=11,Carrey=22', '(22,11)', False, 'Find Carlos,Carrey.')
        self.check('Carrey=22,Carlos=11', '(22,11)', False, 'Find (Carlos, Carrey).')
        self.check('300 balloons', '300', True, 'Find initial balloons.')
        self.check('300 not balloons', '300', None, 'Find balloons.')
        self.check('300 meters', '300', None, 'Find weight in kilograms.')
        self.check('5 x', '5', None, 'Find x.')
        self.check('5 e', '5', None, 'Find e.')
        self.check('$90/hour$', '90', True, 'Find hourly revenue difference.')
        self.check(r'66\%', '66', True, 'Find percentage points lost.')
        self.check(r'66\%', '66', None, 'Find a fraction.')

    def test_reference_tuple_cannot_be_compared_as_unordered_set(self):
        self.check('45,150,105', '(150,45,105)', False)
        self.check('150,45,105', '(150,45,105)', True)
        self.check('150,45', '(150,45,105)', False)
        self.check('150,45,105,0', '(150,45,105)', False)

    def test_vector_and_interval_notation(self):
        self.check(r'31\mathbf{i}-6\mathbf{j}+58\mathbf{k}', '(31,-6,58)', True)
        self.check(r'31\mathbf{i}+6\mathbf{j}+58\mathbf{k}', '(31,-6,58)', False)
        self.check('-1<x<5', '(-1,5)', True, 'Return an open interval.')
        self.check('-1<=x<=5', '(-1,5)', False, 'Return an open interval.')

    def test_complete_expression_not_partial_parse(self):
        self.check(r'\frac{e \cdot f+d \cdot f+d \cdot e}{d \cdot e \cdot f}',
                   r'\frac{de+df+ef}{def}', True, 'For positive d, e, f combine fractions.')
        self.check('31 apples and 58 pears', '31', None, 'Find apples and pears.')
        self.check('x+1', '1', None)

    def test_equivalence_is_separate_from_requested_form(self):
        expr = 'ab(a-b)(a+b)+bc(b-c)(b+c)+ca(c-a)(c+a)'
        ref = '-(a-b)(b-c)(c-a)(a+b+c)'
        task = 'Factor ab(a^2-b^2)+bc(b^2-c^2)+ca(c^2-a^2) into linear factors.'
        self.check(expr, ref, True, task)
        self.assertIs(answers.requested_form((expr, task)), False)
        self.assertIs(answers.requested_form((ref, task)), True)
        self.assertIsNone(answers.requested_form((ref, 'Simplify the expression.')))
        task = 'Factor as a product of two monic quadratic polynomials.'
        self.assertIs(answers.requested_form((r'(m^2-m+\frac12)(m^2+m+\frac12)', task)), True)
        self.assertIs(answers.requested_form((r'm^4+\frac14', task)), False)

    def test_scoring_keeps_value_and_form_separate(self):
        import eac3_fixed_prefix_eval as evaluation
        task = 'Factor ab(a^2-b^2)+bc(b^2-c^2)+ca(c^2-a^2) into linear factors.'
        row = dict(id='test', question=task, subtasks=[task], target=1, family='fixture', kind='compute',
                   reference_answers=['-(a-b)(b-c)(c-a)(a+b+c)'], final_answer='0')
        response = dict(id='test', response=r'LOCAL_RESULT: \boxed{ab(a-b)(a+b)+bc(b-c)(b+c)+ca(c-a)(c+a)}')
        bounded = lambda fn, jobs, **kw: ([fn(j) for j in jobs], {'checks': len(jobs)})
        result = evaluation.score_records({'examples': [row]}, {'test': response}, bounded)['records'][0]
        self.assertEqual(result['value_status'], 'correct')
        self.assertEqual(result['reported_answer_status'], 'correct')
        self.assertEqual(result['task_form_status'], 'failed')
        self.assertEqual(result['status'], 'wrong')


if __name__ == '__main__':
    unittest.main()

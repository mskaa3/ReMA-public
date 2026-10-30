"""Structural contracts for the model-reviewed exploratory study, not a math oracle."""
import copy
from contextlib import redirect_stdout
import io
import sys
import unittest
from unittest.mock import patch

from test_fixed_prefix_eval import ROOT, production_rollout_class
import eac3_fixed_prefix_eval as evaluation
import eac3_worker_study as study
import eac3_worker_study_stats as stats
import experiment_io


class ExploratoryArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / 'experiments/worker-specialization-300-exploratory-v1'
        cls.data = evaluation.load_dataset(cls.path / 'frozen/dataset.json')
        cls.review = evaluation.read_json(cls.path / 'review.json')

    def test_all_selected_questions_and_explicit_exclusions(self):
        selection = evaluation.read_json(self.path / 'frozen/selection.json')
        rows = study.validate_review(selection, self.review, exploratory_self_review=True)
        self.assertEqual(len(rows), 300)
        self.assertEqual(sum(r['decision'] == 'include' for r in rows), 276)
        self.assertEqual(evaluation.digest(self.review), self.data['study']['review_sha256'])
        self.assertFalse(self.data['study']['independently_reviewed'])
        with self.assertRaises(ValueError):
            study.validate_review(selection, self.review)
        stats.validate_pairs(self.data['examples'])
        self.assertEqual(len(self.data['examples']), 764)

    def test_all_prompt_boundaries_and_reference_isolation(self):
        rollout = production_rollout_class()
        for row in self.data['examples']:
            with self.subTest(case=row['id']), redirect_stdout(io.StringIO()), patch.dict(sys.modules, {
                    'verl.rema_separated_trainer.ppo.experiment_io': experiment_io}):
                messages = evaluation.capture_prompt(row, rollout)
                altered = copy.deepcopy(row)
                altered.update(reference_answers=['PRIVATE_REFERENCE'], reference_check='PRIVATE_REFERENCE',
                               final_answer='PRIVATE_REFERENCE', source_final_answer='PRIVATE_REFERENCE')
                self.assertEqual(messages, evaluation.capture_prompt(altered, rollout))
                user = messages[1]['content']
                self.assertEqual('Reference problem:' in user, row['question_mode'] == 'full_question')
                self.assertNotIn('PLAN:', user)
                self.assertIn(f"- S{row['target']}: {row['subtasks'][row['target'] - 1]}", user)
                for i, task in enumerate(row['subtasks'], 1):
                    if i != row['target']:
                        self.assertNotIn(f'- S{i}: {task}', user)
                for i, result in enumerate(row['previous_results'], 1):
                    self.assertIn(f'S{i} LOCAL_RESULT: {evaluation.extract_local_result(result)}', user)

    def test_cylinder_reference_and_fixed_prefix_use_explicit_fraction_braces(self):
        question_id = '76a1ec3d5b39fbdbcbc86230988e092afb0931eab017dbd0dd77226003a9784f'
        rows = [r for r in self.data['examples'] if r['question_id'] == question_id]
        self.assertEqual(len(rows), 3)
        answer = r'\frac{1}{r}+\frac{1}{h}'
        for row in rows:
            if row['target'] == 1:
                self.assertEqual(row['reference_answers'], [answer])
            else:
                self.assertEqual(row['previous_results'], [r'LOCAL_RESULT: \boxed{' + answer + '}'])
            self.assertEqual(row['final_answer'], '6')
        reviewed = next(r for r in self.review['examples'] if r['question_id'] == question_id)
        self.assertEqual(reviewed['tasks'][0]['reference_answers'], [answer])


if __name__ == '__main__':
    unittest.main()

"""Reserved sample packaging and secondary extraction must not loosen training gates."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import eac3_answer_diagnostic as diagnostic
import eac3_fixed_prefix_eval as evaluation
import eac3_worker_study_pack as pack
import eac3_worker_study_stats as stats
import test_worker_study as fixtures


class ScaleTests(unittest.TestCase):
    def test_inline_quote_is_not_a_second_standalone_answer(self):
        response = 'REASONING:\nGiven S2 LOCAL_RESULT: \\boxed{29}, multiply by seven.\n\nLOCAL_RESULT: \\boxed{203}'
        self.assertEqual(evaluation.extract_local_result(response), r'\boxed{203}')
        self.assertEqual(diagnostic.reported_answer(response)['answer'], '203')

    def test_ambiguous_or_truncated_outputs_do_not_fall_back(self):
        for text in (r'\boxed{3} or \boxed{4}',
                     'LOCAL_RESULT: \\boxed{3}\nLOCAL_RESULT: \\boxed{4}',
                     'Earlier \\boxed{3}\nLOCAL_RESULT: \\boxed{',
                     'Earlier \\boxed{3}\nLOCAL_RESULT: not known',
                     'Earlier \\boxed{3}\nLOCAL_RESULT: \\boxed{4} or \\boxed{5}'):
            with self.subTest(text=text):
                self.assertEqual(diagnostic.reported_answer(text)['answer'], '')
        self.assertEqual(diagnostic.reported_answer(r'Therefore \boxed{3\sqrt{13}}.')['answer'], r'3\sqrt{13}')
        self.assertEqual(diagnostic.reported_answer('**LOCAL_RESULT:** \\boxed{5}')['answer'], '5')
        self.assertEqual(diagnostic.reported_answer('')['answer'], '')
        self.assertEqual(diagnostic.reported_answer(r'Given S1 LOCAL_RESULT: \boxed{8}')['answer'], '')

    def test_package_reuses_reserved_sample_without_certifying(self):
        helper = fixtures.StudyTests()
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            selection, _ = helper.sample(tmp)
            with redirect_stdout(io.StringIO()):
                pack.prepare(tmp / 'sample/selection.json', tmp / 'package', 2)
            manifest = evaluation.read_json(tmp / 'package/manifest.json')
            self.assertEqual(manifest['sampled_questions'], 6)
            self.assertEqual(len(manifest['batches']), 3)
            self.assertEqual(evaluation.read_json(tmp / 'package/selection.json'), selection)
            first = evaluation.read_json(tmp / 'package/review-01.json')
            self.assertEqual(first['examples'][0]['decision'], 'pending')
            self.assertEqual(first['examples'][0]['checker'], '')
            with self.assertRaises(ValueError):
                pack.assemble(tmp / 'package', tmp / 'complete.json')
            self.assertFalse((tmp / 'complete.json').exists())
            for batch in manifest['batches']:
                path = tmp / 'package' / batch['file']
                path.write_text(json.dumps(helper.reviewed(evaluation.read_json(path))))
            with redirect_stdout(io.StringIO()):
                pack.assemble(tmp / 'package', tmp / 'complete.json')
            complete = evaluation.read_json(tmp / 'complete.json')
            self.assertEqual([r['question_id'] for r in complete['examples']],
                             [r['question_id'] for r in selection['splits']['reserved']])
            with self.assertRaises(FileExistsError):
                pack.prepare(tmp / 'sample/selection.json', tmp / 'package')
            path = tmp / 'package/review-01.json'
            bad = evaluation.read_json(path)
            bad['examples'].pop()
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                pack.assemble(tmp / 'package', tmp / 'bad.json')

    def test_diagnostic_does_not_replace_primary_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixtures.StudyTests().freeze(Path(directory))
            models = {name: [dict(id=r['id'], status='no_accepted_result',
                reported_answer_status='correct' if name == 'gated' else 'wrong')
                for r in data['examples']] for name in data['study']['model_ids']}
            result = stats.analyze(data, models)
            endpoint = 'compute_nonterminal_full_question'
            self.assertEqual(result['endpoints'][endpoint]['paired_differences']['gated - ungated']['mean'], 0)
            self.assertEqual(result['reported_answer_diagnostic'][endpoint]['paired_differences'][
                'gated - ungated']['mean'], 1)
            self.assertIn('Secondary Stated-Answer Diagnostic', '\n'.join(stats.markdown(result)))
            mixed = copy.deepcopy(models)
            del mixed['w0'][0]['reported_answer_status']
            with self.assertRaises(ValueError):
                stats.analyze(data, mixed)


if __name__ == '__main__':
    unittest.main()

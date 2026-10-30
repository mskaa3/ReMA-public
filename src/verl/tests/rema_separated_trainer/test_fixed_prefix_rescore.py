"""Offline rescoring must preserve generation provenance and never overwrite inputs."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import eac3_fixed_prefix_eval as evaluation
import eac3_rescore_fixed_prefix as rescore
import test_worker_study as fixtures


class RescoreTests(unittest.TestCase):
    def archive(self, directory, mutate=None):
        data = fixtures.StudyTests().freeze(directory)
        data['dataset_sha256'] = evaluation.digest(data)
        data['source_hashes'] = {'original_generation.py': 'original-sha'}
        for row in data['examples']:
            row['messages'] = [dict(role='user', content=row['id'])]
            row['prompt_sha256'] = evaluation.digest(row['messages'])
        encoded = lambda value: json.dumps(value).encode()
        files = {'prepared.json': encoded(data), 'config.json': encoded(dict(models=data['study']['models'])),
                 'comparison/report.json': b'{"original": true}'}
        for model in data['study']['models']:
            name = model['id']
            manifest = dict(model_id=name, model_source=model['source'],
                prepared_sha256=evaluation.digest(data), source_hashes=data['source_hashes'],
                settings={'temperature': 0}, versions={'vllm': 'test'}, max_model_len=8192, batch_size=8)
            records = [dict(id=r['id'], model_id=name, response=r'LOCAL_RESULT: \boxed{5}',
                prompt_sha256=r['prompt_sha256'], prompt_token_sha256='same-tokens',
                generated_tokens=8, finish_reason='stop') for r in data['examples']]
            old = [dict(**r, status='wrong', accepted_result=r'\boxed{5}', local_correct=False,
                full_question_answer_match=False, reported_answer_status='wrong') for r in records]
            files[name+'/manifest.json'] = encoded(manifest)
            files[name+'/responses.jsonl'] = b'\n'.join(encoded(r) for r in records)
            files[name+'/scored.json'] = encoded(dict(records=old))
        if mutate:
            mutate(files)
        path = directory / 'source.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            for name, content in files.items():
                archive.writestr('results/test/' + name, content)
        return path, files

    def test_rescore_preserves_inputs_and_reports_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path, files = self.archive(tmp)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            def bounded(function, jobs, **kwargs):
                return [True for _ in jobs], {'checks': len(jobs), 'unknown_count': 0}
            with patch.object(rescore, 'bounded_verification_map', bounded), redirect_stdout(io.StringIO()):
                rescore.rescore(path, tmp / 'rescored')
            out = tmp / 'rescored'
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertEqual((out / 'prepared.json').read_bytes(), files['prepared.json'])
            self.assertEqual((out / 'ungated/responses.jsonl').read_bytes(), files['ungated/responses.jsonl'])
            self.assertEqual((out / 'ungated/scored.original.json').read_bytes(), files['ungated/scored.json'])
            provenance = evaluation.read_json(out / 'rescore-provenance.json')
            self.assertTrue(provenance['post_hoc_scoring_revision'])
            self.assertFalse(provenance['model_inference_run'])
            self.assertGreater(provenance['changed_records'], 0)
            self.assertTrue((out / 'comparison/report.json').is_file())
            with self.assertRaises(FileExistsError):
                rescore.rescore(path, out)

    def test_reject_changed_prompt_tokens_raw_response_or_generation_settings(self):
        def change(files, path, function):
            obj = json.loads(files[path]); function(obj); files[path] = json.dumps(obj).encode()
        mutations = [
            lambda f: change(f, 'gated/manifest.json', lambda x: x['settings'].update(temperature=1)),
            lambda f: change(f, 'gated/scored.json', lambda x: x['records'][0].update(response='modified')),
            lambda f: f.update({'gated/responses.jsonl': b''}),
            lambda f: change(f, 'prepared.json', lambda x: x['examples'][0]['messages'][0].update(content='modified')),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as tmp:
                path, _ = self.archive(Path(tmp), mutate)
                with self.assertRaises(ValueError):
                    rescore.load_archive(path)

    def test_report_refuses_mixed_scorer_versions(self):
        # Covered at the report boundary, not by trusting directory names.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path, _ = self.archive(tmp)
            with patch.object(rescore, 'bounded_verification_map',
                    lambda fn, jobs, **kw: ([True]*len(jobs), {'checks': len(jobs)})), redirect_stdout(io.StringIO()):
                rescore.rescore(path, tmp / 'rescored')
            scored = tmp / 'rescored/gated/scored.json'
            data = evaluation.read_json(scored)
            data['scorer_version'] = 'different'
            scored.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'identical inputs'):
                evaluation.report(tmp / 'rescored/prepared.json',
                    [tmp / 'rescored' / m for m in ('w0', 'ungated', 'gated')], tmp / 'mixed')


if __name__ == '__main__':
    unittest.main()

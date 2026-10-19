import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import release_lab as lab


class ReleaseLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.output = Path(cls.directory.name)/'run'
        comparator = Path(os.environ.get('EVALDELTA_SCRIPT', str(lab.ROOT.parent/'evaldelta/evaldelta.py')))
        cls.result = lab.run(cls.output, comparator)
        cls.comparator = comparator.resolve()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_every_seed_is_reported_without_pooling_or_selection(self):
        self.assertEqual([r['seed'] for r in self.result['runs']], [42, 43, 44])
        self.assertEqual(self.result['primary_seed'], 42)
        for run in self.result['runs']:
            self.assertEqual(run['n'], 275)
            self.assertEqual(run['split_counts'], {'train':984, 'calibration':268, 'policy':270, 'test':275})

    def test_models_have_distinct_weights_and_exactly_identical_case_contracts(self):
        for run in self.result['runs']:
            folder = self.output/f"seed-{run['seed']}"
            baseline = [json.loads(s) for s in (folder/'baseline.jsonl').read_text().splitlines()]
            candidate = [json.loads(s) for s in (folder/'candidate.jsonl').read_text().splitlines()]
            self.assertNotEqual(run['models']['baseline']['sha256'], run['models']['candidate']['sha256'])
            for a, b, example in zip(baseline, candidate, run['examples'], strict=True):
                for key in ('id', 'case_hash', 'suite', 'scorer', 'tags'):
                    self.assertEqual(a[key], b[key])
                self.assertEqual(a['score'], int(example['baseline']['prediction'] == example['label']))
                self.assertEqual(b['score'], int(example['candidate']['prediction'] == example['label']))
                self.assertEqual(a['case_hash'], lab.case_hash(example['row'], example['pixels'], example['label']))

    def test_every_recorded_prediction_recomputes_from_its_exported_model(self):
        for run in self.result['runs']:
            folder = self.output/f"seed-{run['seed']}"
            for name in ('baseline', 'candidate'):
                artifact = json.loads((folder/f'{name}-model.json').read_text())
                params = {key: np.asarray(value) for key, value in artifact['weights'].items()}
                images = np.asarray([e['pixels'] for e in run['examples']])/16
                actual = lab.infer(params, artifact['temperature'], artifact['threshold'], images)
                for example, prediction in zip(run['examples'], actual, strict=True):
                    self.assertEqual(example[name]['prediction'], prediction['prediction'])
                    self.assertAlmostEqual(example[name]['confidence'], prediction['confidence'], places=12)
                    self.assertEqual(example[name]['decision'], prediction['decision'])

    def test_noisy_inputs_are_not_in_the_model_release_comparison(self):
        x, _ = lab.ml.load_data(lab.ROOT/'data/digits.csv.gz')
        for run in self.result['runs']:
            for example in run['examples']:
                np.testing.assert_array_equal(np.asarray(example['pixels'])/16, x[example['row']])
            self.assertIn('not a release comparison', run['noise_stress']['scope'])

    def test_inconclusive_is_preserved_instead_of_claiming_regression(self):
        run = self.result['runs'][1]
        self.assertGreater(run['candidate_accuracy'], run['baseline_accuracy'])
        self.assertEqual(run['gate']['decision'], 'block')
        self.assertEqual(run['gate']['blockers'], [{'slice':'tag:digit-5', 'quality':'inconclusive', 'latency':'pass'}])
        for run in self.result['runs']:
            self.assertEqual(run['control_decision'], 'pass')

    def test_policy_reports_both_success_and_residual_errors(self):
        policies = [run['policy']['candidate'] for run in self.result['runs']]
        self.assertEqual([p['deferred'] for p in policies], [0, 1, 7])
        self.assertEqual([p['accepted_errors'] for p in policies], [7, 6, 4])
        first = self.result['first_deferred_prediction']
        self.assertEqual(first['seed'], 43)
        self.assertEqual(first['candidate']['decision'], 'review')
        self.assertNotEqual(first['candidate']['prediction'], first['label'])

    def test_splits_are_disjoint_and_complete_within_each_seed(self):
        for run in self.result['runs']:
            splits = json.loads((self.output/f"seed-{run['seed']}"/'splits.json').read_text())
            combined = [i for values in splits.values() for i in values]
            self.assertEqual(len(combined), 1797)
            self.assertEqual(len(set(combined)), 1797)
            self.assertEqual(sorted(combined), list(range(1797)))

    def test_manifest_covers_all_files_and_detects_tampering(self):
        manifest = json.loads((self.output/'manifest.json').read_text())['sha256']
        actual = {str(p.relative_to(self.output)) for p in self.output.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        self.assertEqual(set(manifest), actual)
        for name, digest in manifest.items():
            self.assertEqual(digest, hashlib.sha256((self.output/name).read_bytes()).hexdigest())
        self.assertNotEqual(manifest['protocol.json'], hashlib.sha256(b'changed').hexdigest())

    def test_existing_output_and_unreviewed_comparator_are_rejected(self):
        with self.assertRaises(FileExistsError):
            lab.run(self.output, self.comparator)
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder)/'fake.py'
            fake.write_text('raise RuntimeError("must not execute")')
            with self.assertRaisesRegex(ValueError, 'pinned'):
                lab.run(Path(folder)/'output', fake)
            self.assertFalse((Path(folder)/'output').exists())

    def test_policy_empty_acceptance_has_no_fabricated_accuracy(self):
        summary = lab.policy_summary([{'decision':'review', 'prediction':1}], [0])
        self.assertIsNone(summary['accepted_accuracy'])
        self.assertEqual(summary['deferred_errors'], 1)

    def test_calibration_negative_results_are_retained(self):
        first = self.result['runs'][0]['models']['candidate']
        self.assertGreater(first['nll_after'], first['nll_before'])

    def test_cli_enforcement_returns_block_and_invalid_is_not_a_block(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'cli-run'
            command = [sys.executable, str(lab.ROOT/'release_lab.py'), '--evaldelta', str(self.comparator),
                       '--output', str(output), '--enforce']
            completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 1, completed.stderr)
            self.assertTrue((output/'summary.json').is_file())
            invalid = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertEqual(invalid.returncode, 2)


if __name__ == '__main__':
    unittest.main()

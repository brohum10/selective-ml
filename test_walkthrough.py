import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import walkthrough as demo


class WalkthroughTests(unittest.TestCase):
    def test_case_hash_covers_pixels_label_row(self):
        x = [0]*64
        baseline = demo.case_hash(1, x, 2)
        self.assertEqual(len(baseline), 64)
        self.assertNotEqual(baseline, demo.case_hash(2, x, 2))
        self.assertNotEqual(baseline, demo.case_hash(1, x, 3))
        self.assertNotEqual(baseline, demo.case_hash(1, [1]+x[1:], 2))

    def test_noise_is_seeded_bounded_and_zero_is_identity(self):
        x = np.full((3, 64), .5)
        np.testing.assert_array_equal(demo.make_noise(x, 0), x)
        np.testing.assert_array_equal(demo.make_noise(x, .2), demo.make_noise(x, .2))
        values = demo.make_noise(x, .5)
        self.assertTrue(((values >= 0) & (values <= 1)).all())
        for bad in (True, -.1, .6, float('nan'), float('inf'), '.2'):
            with self.assertRaises(ValueError):
                demo.make_noise(x, bad)

    def test_export_is_paired_matches_real_inference_and_held_out(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = demo.export(folder)
            baseline = [json.loads(line) for line in (Path(folder)/'baseline.jsonl').read_text().splitlines()]
            candidate = [json.loads(line) for line in (Path(folder)/'candidate.jsonl').read_text().splitlines()]
        self.assertEqual(trace['n'], 275)
        self.assertEqual(trace['clean_errors'], 7)
        self.assertEqual(trace['stress_errors'], 27)
        self.assertEqual(trace['learned_threshold'], 0)
        self.assertEqual(trace['model_sha256'], hashlib.sha256((demo.ROOT/'results/reference/model.json').read_bytes()).hexdigest())
        for a, b, e in zip(baseline, candidate, trace['examples'], strict=True):
            for key in ('id', 'case_hash', 'suite', 'scorer', 'tags'):
                self.assertEqual(a[key], b[key])
            self.assertEqual(a['score'], int(e['clean']['prediction'] == e['label']))
            self.assertEqual(b['score'], int(e['stress']['prediction'] == e['label']))
            self.assertGreaterEqual(a['latency_ms'], 0)
            self.assertGreaterEqual(b['latency_ms'], 0)
        chosen = next(e for e in trace['examples'] if e['row'] == trace['featured_row'])
        self.assertEqual(chosen['clean']['prediction'], chosen['label'])
        self.assertNotEqual(chosen['stress']['prediction'], chosen['label'])
        sweep = trace['noise_sweep']
        self.assertEqual(len(sweep), 41)
        self.assertEqual(sweep[0]['prediction'], chosen['clean']['prediction'])
        self.assertEqual(sweep[-1]['prediction'], chosen['stress']['prediction'])
        self.assertEqual(sweep[-1]['confidence'], chosen['stress']['confidence'])
        np.testing.assert_array_equal(sweep[-1]['pixels'], chosen['stress']['pixels'])
        artifact = json.loads((demo.ROOT/'results/reference/model.json').read_text())
        for point in sweep:
            prediction = demo.ml.predict(artifact, [point['pixels']])[0]
            self.assertEqual(prediction['confidence'], point['confidence'])
            self.assertEqual(prediction['prediction'], point['prediction'])

    def test_zero_noise_preserves_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = demo.export(folder, sigma=0)
        self.assertEqual(trace['clean_accuracy'], trace['stress_accuracy'])
        self.assertIsNone(trace['featured_row'])

    def test_changed_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder)
            (reference/'model.json').write_text((demo.ROOT/'results/reference/model.json').read_text())
            (reference/'splits.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'seed-42'):
                demo.export(reference/'out', reference=reference)


if __name__ == '__main__':
    unittest.main()

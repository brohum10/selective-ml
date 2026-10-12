import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import selective_ml as ml


class ModelTests(unittest.TestCase):
    def test_softmax_extreme_logits(self):
        p = ml.softmax(np.array([[10000., 10001., -10000.]]))
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(float(p.sum()), 1)

    def test_numerical_gradients_linear_and_neural(self):
        rng = np.random.default_rng(5)
        x, y = rng.random((5, 3)), np.array([0, 1, 2, 0, 1])
        for hidden in (0, 4):
            params = ml.initialize(3, hidden, 3, 5)
            _, gradients = ml.loss_grad(params, x, y)
            for key, value in params.items():
                for index in np.ndindex(value.shape):
                    original = value[index]
                    value[index] = original + 1e-6
                    plus = ml.loss_grad(params, x, y)[0]
                    value[index] = original - 1e-6
                    minus = ml.loss_grad(params, x, y)[0]
                    value[index] = original
                    self.assertAlmostEqual((plus-minus)/2e-6, gradients[key][index], places=6)

    def test_split_disjoint_complete_and_reproducible(self):
        y = np.repeat(np.arange(10), 100)
        split = ml.split_indices(y, 42)
        combined = np.concatenate(list(split.values()))
        self.assertEqual(len(np.unique(combined)), len(y))
        np.testing.assert_array_equal(np.sort(combined), np.arange(len(y)))
        for key in split:
            np.testing.assert_array_equal(split[key], ml.split_indices(y, 42)[key])
            self.assertEqual(len(np.unique(y[split[key]])), 10)

    def test_calibration_does_not_worsen_its_fitting_loss(self):
        z = np.array([[8., 0.], [8., 0.], [0., 8.]])
        y = np.array([0, 1, 1])
        self.assertLessEqual(ml.nll(z, y, ml.calibrate(z, y)), ml.nll(z, y))

    def test_policy_selects_maximum_feasible_coverage(self):
        p = np.array([[.9, .1], [.8, .2], [.6, .4], [.55, .45]])
        y = np.array([0, 0, 1, 1])
        self.assertEqual(ml.choose_threshold(p, y, target=1), .8)

    def test_infeasible_policy_abstains(self):
        self.assertGreater(ml.choose_threshold(np.array([[.9, .1]]), np.array([1])), 1)

    def test_metrics_perfect_predictions(self):
        y = np.arange(10)
        result = ml.metrics(np.eye(10), y)
        for key in ('accuracy', 'macro_f1', 'coverage', 'selective_accuracy'):
            self.assertEqual(result[key], 1)
        for key in ('nll', 'brier_multiclass', 'ece_10_bins'):
            self.assertEqual(result[key], 0)
        self.assertEqual(sum(b['count'] for b in result['reliability_bins']), 10)

    def test_empty_acceptance_is_not_perfect_accuracy(self):
        self.assertIsNone(ml.metrics(np.eye(10), np.arange(10), 1.000001)['selective_accuracy'])

    def test_training_repeatability_and_tiny_overfit(self):
        x = np.eye(10, 64)
        y = np.arange(10)
        a, b = ml.train(x, y), ml.train(x, y)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(ml.logits(a, x).argmax(axis=1), y)

    def test_json_roundtrip_prediction(self):
        artifact = ml.make_artifact(ml.initialize(), 1., .8)
        pixels = np.zeros((2, 64)).tolist()
        self.assertEqual(ml.predict(artifact, pixels), ml.predict(json.loads(json.dumps(artifact)), pixels))

    def test_reject_invalid_pixels(self):
        artifact = ml.make_artifact(ml.initialize(), 1., .8)
        for pixels in ([], [[0]*63], [[17]*64], [[-1]*64], [[float('nan')]*64]):
            with self.assertRaises(ValueError):
                ml.predict(artifact, pixels)

    def test_reject_malformed_artifact(self):
        artifact = ml.make_artifact(ml.initialize(), 1., .8)
        for key, value in (('version', 2), ('temperature', 0), ('threshold', -1), ('weights', {})):
            bad = copy.deepcopy(artifact)
            bad[key] = value
            with self.assertRaises(ValueError):
                ml.predict(bad, [[0]*64])

    def test_dataset_checksum_and_no_exact_duplicate_images(self):
        x, y = ml.load_data(Path('data/digits.csv.gz'))
        self.assertEqual(x.shape, (1797, 64))
        self.assertEqual(len(np.unique(x, axis=0)), len(x))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'bad.gz'
            path.write_bytes(b'wrong data')
            with self.assertRaises(ValueError):
                ml.load_data(path)

    def test_bootstrap_repeatability(self):
        p, y = np.eye(10), np.arange(10)
        self.assertEqual(ml.bootstrap_accuracy(p, y, 1), [1., 1.])


if __name__ == '__main__':
    unittest.main()

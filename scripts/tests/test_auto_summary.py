import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from auto_summary import (
    extract_features,
    feature_names,
    scaled_prediction_distance,
    train_auto_summary_model,
)
from sv_abc_core import simulate_sv


class AutoFeatureTests(unittest.TestCase):
    def setUp(self):
        self.returns = simulate_sv(
            np.array([-0.5, 0.95, 0.30]),
            100,
            np.random.default_rng(61),
        )

    def test_feature_groups_have_clear_fixed_sizes(self):
        marginal = extract_features(self.returns, "marginal")
        temporal = extract_features(self.returns, "temporal")
        full = extract_features(self.returns, "full")

        self.assertEqual(len(marginal), 11)
        self.assertEqual(len(temporal), 5)
        self.assertEqual(len(full), 16)
        np.testing.assert_allclose(full, np.concatenate([marginal, temporal]))
        self.assertEqual(len(feature_names("full")), 16)

    def test_temporal_features_are_squared_return_autocorrelations(self):
        returns = np.linspace(-1.0, 2.0, 30)
        squared = returns**2
        centred = squared - np.mean(squared)
        denominator = np.dot(centred, centred)
        expected = np.array(
            [
                np.dot(centred[lag:], centred[:-lag]) / denominator
                for lag in [1, 2, 3, 5, 10]
            ]
        )

        actual = extract_features(returns, "temporal")

        np.testing.assert_allclose(actual, expected)

    def test_invalid_series_returns_none(self):
        self.assertIsNone(extract_features(np.ones(100), "full"))

    def test_unknown_feature_group_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_features(self.returns, "unknown")

    def test_scaled_distance_gives_each_prediction_its_own_scale(self):
        distance = scaled_prediction_distance(
            np.array([2.0, 4.0, 8.0]),
            np.array([1.0, 2.0, 4.0]),
            np.array([2.0, 4.0, 8.0]),
        )

        self.assertAlmostEqual(distance, np.sqrt(0.75))


class AutoTrainingTests(unittest.TestCase):
    def test_training_returns_model_metrics_and_positive_prediction_scale(self):
        bundle, metrics, details = train_auto_summary_model(
            n_simulations=80,
            series_length=60,
            feature_group="full",
            random_seed=67,
            n_estimators=10,
        )

        self.assertEqual(bundle["feature_group"], "full")
        self.assertEqual(bundle["series_length"], 60)
        self.assertEqual(bundle["feature_names"], list(feature_names("full")))
        self.assertEqual(bundle["prediction_scale"].shape, (3,))
        self.assertTrue(np.all(bundle["prediction_scale"] > 0.0))
        self.assertEqual(
            bundle["prediction_scale_source"],
            "held-out validation predictions",
        )
        self.assertEqual(list(metrics["parameter"]), ["alpha", "beta", "sigma_eta"])
        self.assertEqual(details["valid_simulations"] + details["invalid_simulations"], 80)
        self.assertIn("validation_rows", details)
        self.assertIn("overall_validation_r2", details)

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.pkl"
            joblib.dump(bundle, model_path)
            loaded = joblib.load(model_path)

        np.testing.assert_allclose(
            loaded["prediction_scale"],
            bundle["prediction_scale"],
        )


if __name__ == "__main__":
    unittest.main()

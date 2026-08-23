import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from auto_abc_experiment import run_auto_rejection_abc, save_auto_abc_result
from auto_summary import train_auto_summary_model
from sv_abc_core import simulate_sv


class AutoABCExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle, _, _ = train_auto_summary_model(
            n_simulations=100,
            series_length=60,
            feature_group="full",
            random_seed=71,
            n_estimators=10,
        )
        cls.observed = simulate_sv(
            np.array([-0.5, 0.95, 0.30]),
            60,
            np.random.default_rng(73),
        )

    def test_small_run_is_reproducible_and_uses_scaled_distance(self):
        first = run_auto_rejection_abc(
            self.observed,
            self.bundle,
            n_simulations=20,
            acceptance_fraction=0.10,
            random_seed=79,
        )
        second = run_auto_rejection_abc(
            self.observed,
            self.bundle,
            n_simulations=20,
            acceptance_fraction=0.10,
            random_seed=79,
        )

        np.testing.assert_allclose(first.accepted, second.accepted)
        self.assertEqual(first.accepted.shape, (2, 4))
        self.assertEqual(first.valid_simulations + first.invalid_simulations, 20)
        np.testing.assert_allclose(first.prediction_scale, self.bundle["prediction_scale"])

    def test_saved_metadata_records_training_length_and_distance_scale(self):
        result = run_auto_rejection_abc(
            self.observed,
            self.bundle,
            n_simulations=20,
            acceptance_fraction=0.10,
            random_seed=83,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sample_path = root / "posterior.npy"
            metadata_path = root / "metadata.json"
            save_auto_abc_result(
                result,
                sample_path=sample_path,
                metadata_path=metadata_path,
                metadata={"dataset": "test data"},
            )
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(stored["training_series_length"], 60)
        self.assertEqual(stored["distance_scaling"], "prediction MAD")
        self.assertEqual(len(stored["prediction_scale"]), 3)


if __name__ == "__main__":
    unittest.main()

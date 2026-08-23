import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from manual_abc_experiment import (
    run_manual_rejection_abc,
    save_manual_abc_result,
)
from sv_abc_core import simulate_sv


class ManualRejectionExperimentTests(unittest.TestCase):
    def test_small_run_reports_valid_and_invalid_simulation_counts(self):
        observed = simulate_sv(
            np.array([-0.5, 0.95, 0.30]),
            80,
            np.random.default_rng(3),
        )

        result = run_manual_rejection_abc(
            observed,
            n_simulations=20,
            acceptance_fraction=0.10,
            random_seed=5,
            scale_simulations=12,
            scale_seed=7,
        )

        self.assertEqual(result.accepted.shape, (2, 4))
        self.assertEqual(result.valid_simulations + result.invalid_simulations, 20)
        self.assertEqual(
            result.scale_valid_simulations + result.scale_invalid_simulations,
            12,
        )
        self.assertEqual(result.summary_scale.shape, (4,))
        self.assertTrue(np.all(result.summary_scale > 0.0))

    def test_fixed_seeds_reproduce_the_complete_result(self):
        observed = simulate_sv(
            np.array([-0.5, 0.95, 0.30]),
            60,
            np.random.default_rng(11),
        )
        arguments = {
            "n_simulations": 12,
            "acceptance_fraction": 0.25,
            "random_seed": 13,
            "scale_simulations": 10,
            "scale_seed": 17,
        }

        first = run_manual_rejection_abc(observed, **arguments)
        second = run_manual_rejection_abc(observed, **arguments)

        np.testing.assert_allclose(first.accepted, second.accepted)
        np.testing.assert_allclose(first.summary_scale, second.summary_scale)
        self.assertEqual(first.valid_simulations, second.valid_simulations)
        self.assertEqual(first.invalid_simulations, second.invalid_simulations)

    def test_saved_outputs_include_scale_and_validity_metadata(self):
        observed = simulate_sv(
            np.array([-0.5, 0.95, 0.30]),
            60,
            np.random.default_rng(19),
        )
        result = run_manual_rejection_abc(
            observed,
            n_simulations=10,
            acceptance_fraction=0.20,
            random_seed=23,
            scale_simulations=8,
            scale_seed=29,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sample_path = root / "posterior.npy"
            metadata_path = root / "metadata.json"
            save_manual_abc_result(
                result,
                sample_path=sample_path,
                metadata_path=metadata_path,
                metadata={"dataset": "test data"},
            )

            stored_samples = np.load(sample_path)
            stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        np.testing.assert_allclose(stored_samples, result.accepted)
        self.assertEqual(stored_metadata["dataset"], "test data")
        self.assertEqual(stored_metadata["valid_simulations"], result.valid_simulations)
        self.assertEqual(stored_metadata["invalid_simulations"], result.invalid_simulations)
        self.assertEqual(stored_metadata["scale_seed"], 29)
        self.assertEqual(len(stored_metadata["summary_scale"]), 4)


if __name__ == "__main__":
    unittest.main()

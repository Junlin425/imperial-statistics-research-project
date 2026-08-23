import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sv_abc_core import simulate_sv
from manual_abc_experiment import run_manual_rejection_abc
from wasserstein_abc_experiment import (
    run_wasserstein_rejection_abc,
    save_wasserstein_abc_result,
)


class WassersteinRejectionExperimentTests(unittest.TestCase):
    def test_shared_simulation_bank_uses_the_same_validity_rule_as_manual(self):
        observed = simulate_sv(
            np.array([-0.5, 0.95, 0.30]),
            60,
            np.random.default_rng(47),
        )
        manual = run_manual_rejection_abc(
            observed,
            n_simulations=30,
            acceptance_fraction=0.10,
            random_seed=53,
            scale_simulations=10,
            scale_seed=59,
        )
        wasserstein = run_wasserstein_rejection_abc(
            observed,
            n_simulations=30,
            acceptance_fraction=0.10,
            random_seed=53,
        )

        self.assertEqual(
            wasserstein.valid_simulations,
            manual.valid_simulations,
        )
        self.assertEqual(
            wasserstein.invalid_simulations,
            manual.invalid_simulations,
        )

    def test_small_run_is_reproducible_and_reports_invalid_counts(self):
        observed = simulate_sv(
            np.array([-0.5, 0.95, 0.30]),
            60,
            np.random.default_rng(31),
        )

        first = run_wasserstein_rejection_abc(
            observed,
            n_simulations=12,
            acceptance_fraction=0.25,
            random_seed=37,
        )
        second = run_wasserstein_rejection_abc(
            observed,
            n_simulations=12,
            acceptance_fraction=0.25,
            random_seed=37,
        )

        np.testing.assert_allclose(first.accepted, second.accepted)
        self.assertEqual(first.accepted.shape, (3, 4))
        self.assertEqual(first.valid_simulations + first.invalid_simulations, 12)

    def test_saved_metadata_records_validity_and_output_path(self):
        observed = simulate_sv(
            np.array([-0.5, 0.95, 0.30]),
            60,
            np.random.default_rng(41),
        )
        result = run_wasserstein_rejection_abc(
            observed,
            n_simulations=10,
            acceptance_fraction=0.20,
            random_seed=43,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sample_path = root / "posterior.npy"
            metadata_path = root / "metadata.json"
            save_wasserstein_abc_result(
                result,
                sample_path=sample_path,
                metadata_path=metadata_path,
                metadata={"dataset": "test data"},
            )
            stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(stored_metadata["dataset"], "test data")
        self.assertEqual(stored_metadata["valid_simulations"], result.valid_simulations)
        self.assertEqual(stored_metadata["invalid_simulations"], result.invalid_simulations)
        self.assertEqual(stored_metadata["output_file"], str(sample_path))


if __name__ == "__main__":
    unittest.main()

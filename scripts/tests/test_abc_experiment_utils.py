import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from abc_experiment_utils import (
    evaluate_convergence_prefixes,
    save_run_metadata,
    select_top_fraction,
)


class SelectTopFractionTests(unittest.TestCase):

    def test_selects_exact_smallest_fraction(self):
        parameters = np.arange(30, dtype=float).reshape(10, 3)
        distances = np.array(
            [9, 1, 8, 2, 7, 3, 6, 4, 5, 0],
            dtype=float,
        )

        accepted, epsilon = select_top_fraction(
            parameters,
            distances,
            0.2,
        )

        self.assertEqual(accepted.shape, (2, 4))
        np.testing.assert_array_equal(
            accepted[:, 3],
            [0.0, 1.0],
        )
        self.assertEqual(epsilon, 1.0)

    def test_rounds_up_non_integer_acceptance_count(self):
        parameters = np.zeros((11, 3))
        distances = np.arange(11, dtype=float)

        accepted, _ = select_top_fraction(
            parameters,
            distances,
            0.05,
        )

        self.assertEqual(len(accepted), 1)

    def test_rejects_invalid_fraction(self):
        with self.assertRaises(ValueError):
            select_top_fraction(
                np.zeros((2, 3)),
                np.zeros(2),
                0,
            )

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            select_top_fraction(
                np.zeros((2, 3)),
                np.zeros(3),
                0.5,
            )


class ConvergenceGridTests(unittest.TestCase):

    def test_uses_prefix_and_reports_parameter_errors(self):
        parameters = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
            ],
            dtype=float,
        )

        rows = evaluate_convergence_prefixes(
            parameters=parameters,
            distances=np.array([0.1, 0.6, 0.2]),
            n_values=[2],
            epsilon_values=[0.5],
            truth=np.array([0.5, 0.5, 0.5]),
            seed=7,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seed"], 7)
        self.assertEqual(rows[0]["accepted"], 1)
        self.assertEqual(rows[0]["alpha_mean"], 0.0)
        self.assertEqual(rows[0]["alpha_abs_error"], 0.5)

    def test_empty_acceptance_produces_nan_estimates(self):
        rows = evaluate_convergence_prefixes(
            parameters=np.zeros((2, 3)),
            distances=np.ones(2),
            n_values=[2],
            epsilon_values=[0.1],
            truth=np.zeros(3),
            seed=0,
        )

        self.assertEqual(rows[0]["accepted"], 0)
        self.assertTrue(np.isnan(rows[0]["alpha_mean"]))
        self.assertTrue(np.isnan(rows[0]["alpha_abs_error"]))


class MetadataTests(unittest.TestCase):

    def test_writes_json_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metadata.json"

            save_run_metadata(
                path,
                {"method": "Manual ABC", "runtime_seconds": 1.25},
            )

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["method"], "Manual ABC")
            self.assertEqual(saved["runtime_seconds"], 1.25)

    def test_relative_path_is_resolved_inside_scripts_directory(self):
        original_directory = Path.cwd()

        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                written_path = save_run_metadata(
                    "relative_metadata_test.json",
                    {"seed": 7},
                )
            finally:
                os.chdir(original_directory)

        expected_path = (
            Path(__file__).resolve().parents[1]
            / "relative_metadata_test.json"
        )
        try:
            self.assertEqual(written_path, expected_path)
            saved = json.loads(expected_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["seed"], 7)
        finally:
            expected_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from supervisor_revision_diagnostics import (
    calculate_prior_scaled_errors,
    run_supervisor_diagnostics,
    simulate_prior_validity,
    summarize_induced_prior,
    summarize_wasserstein_ridge,
)


class ScaledErrorTests(unittest.TestCase):
    def test_errors_are_divided_by_the_matching_prior_width(self):
        errors = pd.DataFrame(
            {
                "method": ["Example ABC"],
                "alpha_error": [0.14],
                "beta_error": [0.0149],
                "sigma_eta_error": [0.045],
            }
        )
        bounds = np.array(
            [
                [-1.5, -0.1],
                [0.85, 0.999],
                [0.05, 0.50],
            ]
        )

        result = calculate_prior_scaled_errors(errors, bounds)

        self.assertAlmostEqual(result.loc[0, "alpha_scaled_error"], 0.10)
        self.assertAlmostEqual(result.loc[0, "beta_scaled_error"], 0.10)
        self.assertAlmostEqual(result.loc[0, "sigma_eta_scaled_error"], 0.10)
        self.assertAlmostEqual(result.loc[0, "mean_scaled_error"], 0.10)


class InducedPriorSummaryTests(unittest.TestCase):
    def test_summary_separates_nominal_and_valid_draws(self):
        parameters = np.array(
            [
                [-0.5, 0.90, 0.20],
                [-0.6, 0.95, 0.30],
                [-0.7, 0.98, 0.40],
            ]
        )
        valid = np.array([True, False, True])

        result = summarize_induced_prior(parameters, valid)

        alpha_nominal = result.query(
            "distribution == 'Nominal prior' and variable == 'alpha'"
        ).iloc[0]
        alpha_valid = result.query(
            "distribution == 'Valid-only prior' and variable == 'alpha'"
        ).iloc[0]
        self.assertEqual(alpha_nominal["n_draws"], 3)
        self.assertEqual(alpha_valid["n_draws"], 2)
        self.assertAlmostEqual(alpha_valid["mean"], -0.6)

    def test_validity_simulation_is_reproducible(self):
        first_parameters, first_valid = simulate_prior_validity(
            n_draws=12,
            series_length=20,
            random_seed=123,
        )
        second_parameters, second_valid = simulate_prior_validity(
            n_draws=12,
            series_length=20,
            random_seed=123,
        )

        np.testing.assert_allclose(first_parameters, second_parameters)
        np.testing.assert_array_equal(first_valid, second_valid)
        self.assertEqual(first_parameters.shape, (12, 3))
        self.assertEqual(first_valid.dtype, np.dtype(bool))


class WassersteinRidgeTests(unittest.TestCase):
    def test_summary_reports_parameter_ridge_and_long_run_level(self):
        beta = np.array([0.90, 0.92, 0.94, 0.96])
        alpha = -10.0 * (1.0 - beta)
        samples = np.column_stack(
            [alpha, beta, np.full(beta.shape, 0.30), np.zeros(beta.shape)]
        )

        result = summarize_wasserstein_ridge(samples, true_mu_h=-10.0)

        self.assertGreater(result["alpha_beta_correlation"], 0.99)
        self.assertAlmostEqual(result["mean_long_run_log_variance"], -10.0)
        self.assertAlmostEqual(result["absolute_mu_h_error"], 0.0)


class DiagnosticWorkflowTests(unittest.TestCase):
    def test_workflow_saves_tables_metadata_and_separate_figures(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            errors = root / "errors.csv"
            wasserstein = root / "wasserstein.npy"
            pd.DataFrame(
                {
                    "method": ["Example ABC"],
                    "alpha_error": [0.14],
                    "beta_error": [0.0149],
                    "sigma_eta_error": [0.045],
                }
            ).to_csv(errors, index=False)
            samples = np.array(
                [
                    [-1.0, 0.90, 0.30, 0.1],
                    [-0.8, 0.92, 0.30, 0.2],
                    [-0.6, 0.94, 0.30, 0.3],
                    [-0.4, 0.96, 0.30, 0.4],
                ]
            )
            np.save(wasserstein, samples)

            outputs = run_supervisor_diagnostics(
                error_table_path=errors,
                wasserstein_samples_path=wasserstein,
                output_directory=root / "tables",
                figure_directory=root / "figures",
                n_prior_draws=40,
                series_length=20,
                random_seed=123,
            )

            self.assertTrue(outputs["scaled_errors"].exists())
            self.assertTrue(outputs["induced_prior_summary"].exists())
            self.assertTrue(outputs["metadata"].exists())
            self.assertEqual(len(outputs["figures"]), 6)
            for figure in outputs["figures"]:
                self.assertTrue(figure.exists())
                self.assertGreater(figure.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

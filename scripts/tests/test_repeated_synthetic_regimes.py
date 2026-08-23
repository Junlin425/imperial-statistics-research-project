import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from abc_smc_utils import ABCSMCConfig
from repeated_synthetic_regimes import (
    aggregate_recovery_results,
    build_regime_table,
    calculate_block_paired_comparisons,
    calculate_method_win_rates,
    posterior_recovery_rows,
    run_repeated_regime_study,
    save_repeated_study_outputs,
    select_shared_bank_posteriors,
)


class FirstThreeFeaturesModel:
    def predict(self, feature_matrix):
        feature_matrix = np.asarray(feature_matrix, dtype=float)
        return feature_matrix[:, :3]


class RegimeDesignTests(unittest.TestCase):
    def test_long_run_level_is_fixed_in_every_regime(self):
        regimes = build_regime_table(
            long_run_log_variance=-10.0,
            beta_values=(0.90, 0.95),
            sigma_eta_values=(0.15,),
            data_seeds=(11, 12),
        )

        self.assertEqual(len(regimes), 4)
        self.assertEqual(regimes["dataset_id"].nunique(), 4)
        np.testing.assert_allclose(
            regimes["alpha"],
            [-1.0, -1.0, -0.5, -0.5],
        )
        calculated_level = regimes["alpha"] / (1.0 - regimes["beta"])
        np.testing.assert_allclose(calculated_level, -10.0)


class RecoveryAggregationTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame(
            [
                {
                    "dataset_id": "d1",
                    "beta_regime": 0.90,
                    "sigma_eta_regime": 0.15,
                    "method": "A",
                    "target": "alpha",
                    "target_type": "parameter",
                    "bias": 0.10,
                    "absolute_error": 0.10,
                    "scaled_absolute_error": 0.20,
                    "covered": True,
                    "interval_width": 0.40,
                    "simulator_calls": 100,
                },
                {
                    "dataset_id": "d2",
                    "beta_regime": 0.90,
                    "sigma_eta_regime": 0.15,
                    "method": "A",
                    "target": "alpha",
                    "target_type": "parameter",
                    "bias": -0.10,
                    "absolute_error": 0.10,
                    "scaled_absolute_error": 0.20,
                    "covered": False,
                    "interval_width": 0.60,
                    "simulator_calls": 100,
                },
                {
                    "dataset_id": "d1",
                    "beta_regime": 0.90,
                    "sigma_eta_regime": 0.15,
                    "method": "B",
                    "target": "alpha",
                    "target_type": "parameter",
                    "bias": 0.20,
                    "absolute_error": 0.20,
                    "scaled_absolute_error": 0.40,
                    "covered": True,
                    "interval_width": 0.80,
                    "simulator_calls": 120,
                },
                {
                    "dataset_id": "d2",
                    "beta_regime": 0.90,
                    "sigma_eta_regime": 0.15,
                    "method": "B",
                    "target": "alpha",
                    "target_type": "parameter",
                    "bias": 0.05,
                    "absolute_error": 0.05,
                    "scaled_absolute_error": 0.10,
                    "covered": True,
                    "interval_width": 0.70,
                    "simulator_calls": 120,
                },
            ]
        )
        self.raw["runtime_seconds"] = [1.0, 2.0, 3.0, 4.0]

    def test_aggregate_reports_bias_rmse_coverage_and_interval_width(self):
        aggregate = aggregate_recovery_results(
            self.raw,
            group_columns=(
                "beta_regime",
                "sigma_eta_regime",
                "method",
                "target",
                "target_type",
            ),
        )
        row = aggregate[aggregate["method"] == "A"].iloc[0]

        self.assertAlmostEqual(row["mean_bias"], 0.0)
        self.assertAlmostEqual(row["rmse"], 0.10)
        self.assertAlmostEqual(row["coverage_rate"], 0.50)
        self.assertAlmostEqual(row["mean_interval_width"], 0.50)
        self.assertAlmostEqual(row["mean_runtime_seconds"], 1.50)
        self.assertAlmostEqual(row["median_scaled_absolute_error"], 0.20)
        self.assertAlmostEqual(row["scaled_error_iqr"], 0.0)

    def test_win_rates_use_the_lowest_scaled_error_per_dataset(self):
        win_rates = calculate_method_win_rates(
            self.raw,
            method_order=("A", "B"),
        )

        self.assertEqual(win_rates["wins"].tolist(), [1, 1])
        self.assertEqual(win_rates["win_rate"].tolist(), [0.5, 0.5])

    def test_block_comparison_keeps_common_random_seeds_together(self):
        rows = []
        errors = {
            11: {"A": (0.10, 0.30), "B": (0.20, 0.40)},
            12: {"A": (0.30, 0.50), "B": (0.20, 0.40)},
        }
        for seed, methods in errors.items():
            for method, regime_errors in methods.items():
                for regime, error in enumerate(regime_errors):
                    rows.append(
                        {
                            "dataset_id": f"seed_{seed}_regime_{regime}",
                            "data_seed": seed,
                            "method": method,
                            "target": "alpha",
                            "scaled_absolute_error": error,
                        }
                    )

        comparisons = calculate_block_paired_comparisons(
            pd.DataFrame(rows),
            method_order=("A", "B"),
            n_bootstrap=200,
            random_seed=4,
        )
        row = comparisons.iloc[0]

        self.assertEqual(row["best_method"], "A")
        self.assertEqual(row["comparison_method"], "B")
        self.assertEqual(row["seed_blocks"], 2)
        self.assertAlmostEqual(row["mean_paired_difference"], 0.0)
        self.assertLessEqual(row["ci_lower"], row["mean_paired_difference"])
        self.assertGreaterEqual(row["ci_upper"], row["mean_paired_difference"])


class PosteriorRecoveryTests(unittest.TestCase):
    def test_each_method_selects_from_the_same_parameter_bank(self):
        parameters = np.array(
            [
                [-1.0, 0.90, 0.15],
                [-0.5, 0.95, 0.30],
                [-0.2, 0.98, 0.15],
                [-0.4, 0.96, 0.25],
            ]
        )
        distances = {
            "A": np.array([0.4, 0.1, 0.3, 0.2]),
            "B": np.array([0.1, 0.4, 0.2, 0.3]),
        }

        posteriors = select_shared_bank_posteriors(
            parameters,
            distances,
            acceptance_fraction=0.5,
        )

        np.testing.assert_allclose(
            posteriors["A"]["samples"],
            parameters[[1, 3]],
        )
        np.testing.assert_allclose(
            posteriors["B"]["samples"],
            parameters[[0, 2]],
        )
        np.testing.assert_allclose(posteriors["A"]["weights"], [0.5, 0.5])

    def test_true_posterior_draws_have_zero_raw_and_derived_errors(self):
        true_parameters = np.array([-0.5, 0.95, 0.30])
        samples = np.tile(true_parameters, (3, 1))
        rows = posterior_recovery_rows(
            samples,
            np.full(3, 1.0 / 3.0),
            true_parameters,
            {
                "dataset_id": "d1",
                "beta_regime": 0.95,
                "sigma_eta_regime": 0.30,
                "data_seed": 11,
            },
            method="A",
            simulator_calls=100,
            runtime_seconds=1.25,
        )

        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["covered"] for row in rows))
        self.assertTrue(
            all(abs(row["scaled_absolute_error"]) < 1e-12 for row in rows)
        )
        self.assertTrue(all(row["interval_width"] == 0.0 for row in rows))
        self.assertTrue(all(row["runtime_seconds"] == 1.25 for row in rows))


class OutputWorkflowTests(unittest.TestCase):
    def test_outputs_include_tables_posteriors_metadata_and_seven_figures(self):
        methods = ("A", "B")
        targets = (
            "alpha",
            "beta",
            "sigma_eta",
            "long_run_log_variance",
            "half_life_days",
            "stationary_log_volatility_variance",
        )
        raw_rows = []
        for dataset_index, beta in enumerate((0.90, 0.95)):
            for method_index, method in enumerate(methods):
                for target_index, target in enumerate(targets):
                    error = 0.05 + 0.01 * (
                        dataset_index + method_index + target_index
                    )
                    raw_rows.append(
                        {
                            "dataset_id": f"d{dataset_index}",
                            "data_seed": dataset_index,
                            "beta_regime": beta,
                            "sigma_eta_regime": 0.15,
                            "method": method,
                            "target": target,
                            "target_type": (
                                "parameter" if target_index < 3 else "derived"
                            ),
                            "bias": error,
                            "absolute_error": error,
                            "scaled_absolute_error": error,
                            "covered": True,
                            "interval_width": 0.5,
                            "simulator_calls": 100,
                        }
                    )
        raw = pd.DataFrame(raw_rows)
        regime_summary = aggregate_recovery_results(
            raw,
            group_columns=(
                "beta_regime",
                "sigma_eta_regime",
                "method",
                "target",
                "target_type",
            ),
        )
        overall_summary = aggregate_recovery_results(
            raw,
            group_columns=("method", "target", "target_type"),
        )
        win_rates = calculate_method_win_rates(raw, methods)
        posterior_samples = np.zeros((2, 2, 3, 3))
        posterior_weights = np.full((2, 2, 3), 1.0 / 3.0)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            outputs = save_repeated_study_outputs(
                raw,
                regime_summary,
                overall_summary,
                win_rates,
                posterior_samples,
                posterior_weights,
                {"dataset_count": 2},
                root / "tables",
                root / "figures",
                methods,
            )

            self.assertEqual(len(outputs["figures"]), 7)
            self.assertTrue(outputs["posterior_file"].exists())
            metadata = json.loads(
                outputs["metadata"].read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["dataset_count"], 2)
            for path in outputs["tables"] + outputs["figures"]:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


class SmallEndToEndStudyTests(unittest.TestCase):
    def test_one_dataset_runs_all_four_methods_and_six_targets(self):
        regimes = build_regime_table(
            beta_values=(0.95,),
            sigma_eta_values=(0.30,),
            data_seeds=(11,),
        )
        model_bundle = {
            "model": FirstThreeFeaturesModel(),
            "feature_group": "full",
            "prediction_scale": np.ones(3),
            "series_length": 50,
            "training_simulations": 20,
        }
        smc_config = ABCSMCConfig(
            n_particles=6,
            n_pilot=12,
            max_populations=1,
            epsilon_quantile=0.5,
            max_attempts_per_population=100,
            random_seed=7,
        )

        study = run_repeated_regime_study(
            regimes,
            model_bundle,
            series_length=50,
            rejection_simulations=30,
            acceptance_fraction=0.20,
            rejection_seed=7,
            scale_simulations=20,
            scale_seed=8,
            smc_config=smc_config,
            smc_workers=1,
        )

        self.assertEqual(len(study["raw_results"]), 24)
        self.assertEqual(study["posterior_samples"].shape, (1, 4, 6, 3))
        self.assertEqual(study["posterior_weights"].shape, (1, 4, 6))
        self.assertEqual(study["raw_results"]["method"].nunique(), 4)
        self.assertEqual(study["raw_results"]["target"].nunique(), 6)
        np.testing.assert_allclose(
            study["posterior_weights"].sum(axis=2),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()

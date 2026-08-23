import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from posterior_predictive_utils import (
    DEFAULT_REPLICATIONS,
    DEFAULT_SEED,
    METHOD_NAMES,
    STATISTIC_NAMES,
    PosteriorSource,
    build_pvalue_table,
    build_summary_table,
    build_temporal_acf_summary,
    build_temporal_rmse_table,
    compute_temporal_acf,
    compute_predictive_draws,
    load_posterior_sources,
    normalize_weights,
    prepare_plot_statistic,
    run_posterior_predictive_analysis,
    sample_parameter_vectors,
)


VALID_SAMPLES = np.array(
    [
        [-0.5, 0.95, 0.30],
        [-0.6, 0.94, 0.35],
    ],
    dtype=float,
)


def write_posterior_fixture(directory, smc_weights=(1.0, 3.0)):
    directory = Path(directory)
    np.save(directory / "abc_manual_sv.npy", VALID_SAMPLES)
    np.save(directory / "abc_auto_sv.npy", VALID_SAMPLES)
    np.save(directory / "abc_wasserstein_sv.npy", VALID_SAMPLES)
    pd.DataFrame(
        {
            "alpha": VALID_SAMPLES[:, 0],
            "beta": VALID_SAMPLES[:, 1],
            "sigma_eta": VALID_SAMPLES[:, 2],
            "distance": [0.2, 0.1],
            "weight": smc_weights,
        }
    ).to_csv(directory / "abc_smc_sv_final.csv", index=False)


class PosteriorLoadingTests(unittest.TestCase):
    def test_load_posterior_sources_includes_four_methods(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            write_posterior_fixture(temporary_directory)

            sources = load_posterior_sources(Path(temporary_directory))

        self.assertEqual(
            list(sources),
            ["Manual ABC", "Auto ABC", "Wasserstein ABC", "ABC-SMC"],
        )
        np.testing.assert_allclose(
            sources["ABC-SMC"].probabilities,
            [0.25, 0.75],
        )
        self.assertEqual(
            sources["ABC-SMC"].sampling_rule,
            "importance-weighted",
        )

    def test_rejection_posterior_distance_column_is_not_a_parameter(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            write_posterior_fixture(temporary_directory)
            four_column_samples = np.column_stack(
                (VALID_SAMPLES, np.array([0.10, 0.20]))
            )
            for filename in [
                "abc_manual_sv.npy",
                "abc_auto_sv.npy",
                "abc_wasserstein_sv.npy",
            ]:
                np.save(Path(temporary_directory) / filename, four_column_samples)

            sources = load_posterior_sources(Path(temporary_directory))

        for method_name in ["Manual ABC", "Auto ABC", "Wasserstein ABC"]:
            with self.subTest(method_name=method_name):
                self.assertEqual(sources[method_name].samples.shape, (2, 3))
                np.testing.assert_allclose(
                    sources[method_name].samples,
                    VALID_SAMPLES,
                )

    def test_malformed_rejection_posterior_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            write_posterior_fixture(temporary_directory)
            np.save(
                Path(temporary_directory) / "abc_manual_sv.npy",
                np.array([1.0, 2.0, 3.0]),
            )

            with self.assertRaisesRegex(ValueError, "Manual ABC posterior"):
                load_posterior_sources(Path(temporary_directory))


class ImportanceWeightTests(unittest.TestCase):
    def test_normalize_weights_scales_to_one(self):
        np.testing.assert_allclose(
            normalize_weights(np.array([1.0, 3.0])),
            [0.25, 0.75],
        )

    def test_invalid_weights_are_rejected(self):
        invalid_cases = [
            np.array([-1.0, 2.0]),
            np.array([np.inf, 1.0]),
            np.array([0.0, 0.0]),
        ]
        for weights in invalid_cases:
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(ValueError, "weights"):
                    normalize_weights(weights)

    def test_weighted_sampling_uses_abc_smc_probabilities(self):
        source = PosteriorSource(
            samples=np.array(
                [
                    [-0.5, 0.95, 0.30],
                    [-0.7, 0.92, 0.40],
                ]
            ),
            probabilities=np.array([0.0, 1.0]),
            sampling_rule="importance-weighted",
        )

        draws = sample_parameter_vectors(
            source,
            np.random.default_rng(42),
            20,
        )

        np.testing.assert_allclose(
            draws,
            np.repeat(source.samples[[1]], 20, axis=0),
        )

    def test_non_positive_replication_count_is_rejected(self):
        source = PosteriorSource(
            samples=VALID_SAMPLES,
            probabilities=None,
            sampling_rule="uniform",
        )
        for value in [0, -1]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "n_replications"):
                    sample_parameter_vectors(
                        source,
                        np.random.default_rng(42),
                        value,
                    )


class PredictiveCalculationTests(unittest.TestCase):
    def test_temporal_acf_returns_lags_one_to_twenty(self):
        returns = np.linspace(-0.2, 0.3, 30)

        result = compute_temporal_acf(returns)

        self.assertEqual(result.shape, (20,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_temporal_tables_have_one_row_per_method_and_lag(self):
        observed = np.linspace(0.20, 0.01, 20)
        draws = {
            method_name: np.vstack([observed, observed + 0.02])
            for method_name in METHOD_NAMES
        }

        summary = build_temporal_acf_summary(observed, draws)
        rmse = build_temporal_rmse_table(observed, draws)

        self.assertEqual(summary.shape, (80, 10))
        self.assertEqual(rmse.shape, (4, 7))
        self.assertEqual(summary["Lag"].min(), 1)
        self.assertEqual(summary["Lag"].max(), 20)
        self.assertTrue(np.allclose(rmse["Median-curve ACF RMSE"], 0.01))
        self.assertTrue(np.all(rmse["Replications"] == 2))

    def test_pvalue_counts_equality_as_exceedance(self):
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predictive_draws = {
            method_name: np.array([[1.0, 1.0, 4.0, 3.0]])
            for method_name in METHOD_NAMES
        }

        result = build_pvalue_table(observed, predictive_draws)

        manual = result[result["Method"] == "Manual ABC"]
        self.assertEqual(
            manual["Posterior Predictive p-value"].tolist(),
            [1.0, 0.0, 1.0, 0.0],
        )

    def test_predictive_draws_are_deterministic(self):
        observed = np.array([0.10, -0.20, 0.15, -0.10, 0.05, -0.04])
        samples = np.array([[-0.5, 0.95, 0.30]])
        sources = {
            method_name: PosteriorSource(samples, None, "uniform")
            for method_name in METHOD_NAMES
        }

        first = compute_predictive_draws(observed, sources, 3, 42)
        second = compute_predictive_draws(observed, sources, 3, 42)

        np.testing.assert_allclose(first[0], second[0])
        for method_name in METHOD_NAMES:
            np.testing.assert_allclose(
                first[1][method_name],
                second[1][method_name],
            )

    def test_summary_and_pvalue_tables_have_complete_shapes(self):
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predictive_draws = {
            method_name: np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, 3.0, 4.0, 5.0],
                ]
            )
            for method_name in METHOD_NAMES
        }

        summary = build_summary_table(observed, predictive_draws)
        pvalues = build_pvalue_table(observed, predictive_draws)

        self.assertEqual(summary.shape, (16, 9))
        self.assertEqual(pvalues.shape, (16, 3))
        self.assertEqual(summary["Method"].nunique(), 4)
        self.assertEqual(summary["Statistic"].nunique(), 4)
        self.assertTrue(np.all(summary["Replications"] == 2))
        self.assertEqual(
            tuple(summary["Statistic"].drop_duplicates()),
            STATISTIC_NAMES,
        )
        self.assertTrue(
            np.all(np.isfinite(summary.select_dtypes(include=[np.number])))
        )


class AnalysisOutputTests(unittest.TestCase):
    def test_formal_configuration_uses_one_thousand_replications(self):
        self.assertEqual(DEFAULT_REPLICATIONS, 1_000)
        self.assertEqual(DEFAULT_SEED, 42)

    def test_variance_plot_uses_disclosed_log_floor(self):
        values = [np.array([1e-80, 1e-6, 1e-4])]

        transformed, observed, axis_label, title = prepare_plot_statistic(
            "Variance",
            values,
            1e-4,
        )

        np.testing.assert_allclose(transformed[0], [-12.0, -6.0, -4.0])
        self.assertEqual(observed, -4.0)
        self.assertIn("clipped", axis_label)
        self.assertIn("log10", title)

    def test_runner_writes_complete_outputs_outside_current_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scripts_dir = root / "canonical_scripts"
            figures_dir = root / "canonical_figures"
            data_dir = root / "data"
            unrelated_directory = root / "unrelated_working_directory"
            scripts_dir.mkdir()
            figures_dir.mkdir()
            data_dir.mkdir()
            unrelated_directory.mkdir()
            write_posterior_fixture(scripts_dir)
            data_path = data_dir / "sp500_returns.csv"
            pd.DataFrame(
                {
                    "Return": np.sin(np.arange(30)) * 0.1
                }
            ).to_csv(data_path, index=False)

            original_directory = Path.cwd()
            try:
                os.chdir(unrelated_directory)
                result = run_posterior_predictive_analysis(
                    n_replications=2,
                    seed=42,
                    scripts_dir=scripts_dir,
                    figures_dir=figures_dir,
                    data_path=data_path,
                )
            finally:
                os.chdir(original_directory)

            expected_script_outputs = [
                scripts_dir / "posterior_predictive_check.csv",
                scripts_dir / "posterior_predictive_summary.csv",
                scripts_dir / "posterior_predictive_pvalues.csv",
                scripts_dir / "posterior_predictive_metadata.json",
                scripts_dir / "posterior_predictive_draws.npz",
                scripts_dir / "posterior_predictive_acf.csv",
                scripts_dir / "posterior_predictive_acf_rmse.csv",
            ]
            expected_figures = [
                figures_dir / "ppc_variance.png",
                figures_dir / "ppc_kurtosis.png",
                figures_dir / "ppc_acf1.png",
                figures_dir / "ppc_acf5.png",
                figures_dir / "ppc_temporal_manual.png",
                figures_dir / "ppc_temporal_auto.png",
                figures_dir / "ppc_temporal_wasserstein.png",
                figures_dir / "ppc_temporal_abc_smc.png",
            ]
            for output_path in [*expected_script_outputs, *expected_figures]:
                self.assertTrue(output_path.exists(), output_path)
            self.assertEqual(list(unrelated_directory.iterdir()), [])

            self.assertEqual(result["wide_table"].shape, (4, 6))
            self.assertEqual(result["summary_table"].shape, (16, 9))
            self.assertEqual(result["pvalue_table"].shape, (16, 3))
            self.assertEqual(result["temporal_acf_table"].shape, (80, 10))
            self.assertEqual(result["temporal_rmse_table"].shape, (4, 7))
            metadata = json.loads(
                (scripts_dir / "posterior_predictive_metadata.json").read_text()
            )
            self.assertEqual(metadata["seed"], 42)
            self.assertEqual(metadata["replications_per_method"], 2)
            self.assertEqual(
                metadata["sampling_rules"]["ABC-SMC"],
                "importance-weighted",
            )
            with np.load(
                scripts_dir / "posterior_predictive_draws.npz"
            ) as saved_draws:
                self.assertEqual(saved_draws["observed_summary"].shape, (4,))
                self.assertEqual(saved_draws["manual"].shape, (2, 4))
                self.assertEqual(saved_draws["abc_smc"].shape, (2, 4))
                self.assertEqual(saved_draws["observed_acf"].shape, (20,))
                self.assertEqual(saved_draws["manual_acf"].shape, (2, 20))
            self.assertEqual(
                set(result["paths"]["figures"].values())
                | set(result["paths"]["temporal_figures"].values()),
                {path.resolve() for path in expected_figures},
            )


if __name__ == "__main__":
    unittest.main()

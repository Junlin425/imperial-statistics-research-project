import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from abc_smc_comparison import (
    METHOD_ORDER,
    _runtime_and_calls,
    run_comparison,
    weighted_posterior_summary,
)


SCRIPTS_DIR = Path(__file__).resolve().parents[1]


class PlotBackendTests(unittest.TestCase):
    def test_comparison_module_uses_a_headless_plot_backend(self):
        command = (
            "import abc_smc_comparison, matplotlib; "
            "print(matplotlib.get_backend())"
        )
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=SCRIPTS_DIR,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertEqual(result.stdout.strip().lower(), "agg")


class WeightedPosteriorSummaryTests(unittest.TestCase):
    def test_weighted_statistics_match_hand_calculation(self):
        summary = weighted_posterior_summary(
            np.array([0.0, 10.0, 20.0]),
            np.array([0.2, 0.6, 0.2]),
        )

        self.assertAlmostEqual(summary["mean"], 10.0)
        self.assertAlmostEqual(summary["sd"], np.sqrt(40.0))
        self.assertAlmostEqual(summary["median"], 10.0)
        self.assertAlmostEqual(summary["q2.5"], 0.0)
        self.assertAlmostEqual(summary["q97.5"], 20.0)

    def test_equal_weights_reproduce_unweighted_mean(self):
        values = np.array([-2.0, 1.0, 4.0, 9.0])
        summary = weighted_posterior_summary(
            values,
            np.full(4, 0.25),
        )

        self.assertAlmostEqual(summary["mean"], float(values.mean()))


class MetadataComparisonTests(unittest.TestCase):
    def test_main_inference_budget_excludes_separate_scaling_bank(self):
        runtime, calls = _runtime_and_calls(
            {
                "inference_seconds": 12.5,
                "scale_seconds": 3.0,
                "simulation_budget": 10_000,
                "total_simulator_calls": 12_000,
            }
        )

        self.assertEqual(runtime, 12.5)
        self.assertEqual(calls, 10_000)


class ComparisonWorkflowTests(unittest.TestCase):
    @staticmethod
    def _write_metadata(path, runtime, simulator_calls):
        path.write_text(
            json.dumps(
                {
                    "runtime_seconds": runtime,
                    "simulation_budget": simulator_calls,
                    "total_simulator_calls": simulator_calls,
                }
            ),
            encoding="utf-8",
        )

    def test_fixture_workflow_writes_tables_and_separate_figures(self):
        rng = np.random.default_rng(8)
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scripts_dir = root / "scripts"
            figures_dir = root / "figures"
            scripts_dir.mkdir()

            rejection_files = {
                "abc_manual_sv.npy": "abc_manual_sv_metadata.json",
                "abc_auto_sv.npy": "abc_auto_sv_metadata.json",
                "abc_wasserstein_sv.npy": "abc_wasserstein_sv_metadata.json",
                "abc_manual_synthetic_sv.npy": "abc_manual_synthetic_sv_metadata.json",
                "abc_auto_synthetic_sv.npy": "abc_auto_synthetic_sv_metadata.json",
                "abc_wasserstein_synthetic_sv.npy": "abc_wasserstein_synthetic_sv_metadata.json",
            }
            for npy_name, metadata_name in rejection_files.items():
                parameters = np.column_stack(
                    (
                        rng.normal(-0.5, 0.15, 40),
                        rng.normal(0.95, 0.015, 40),
                        rng.normal(0.30, 0.04, 40),
                        rng.uniform(0.0, 1.0, 40),
                    )
                )
                np.save(scripts_dir / npy_name, parameters)
                self._write_metadata(
                    scripts_dir / metadata_name,
                    runtime=1.0,
                    simulator_calls=10_000,
                )

            for prefix in ["abc_smc_sv", "abc_smc_synthetic_sv"]:
                particles = np.stack(
                    [
                        np.column_stack(
                            (
                                rng.normal(-0.6, 0.15, 40),
                                rng.normal(0.94, 0.015, 40),
                                rng.normal(0.31, 0.04, 40),
                            )
                        ),
                        np.column_stack(
                            (
                                rng.normal(-0.5, 0.10, 40),
                                rng.normal(0.95, 0.010, 40),
                                rng.normal(0.30, 0.03, 40),
                            )
                        ),
                    ]
                )
                weights = np.full((2, 40), 1.0 / 40)
                np.savez_compressed(
                    scripts_dir / f"{prefix}_history.npz",
                    particles=particles,
                    weights=weights,
                    distances=rng.uniform(0.0, 1.0, (2, 40)),
                    epsilons=np.array([1.0, 0.5]),
                    candidate_simulations=np.array([80, 100]),
                    eligible_counts=np.array([40, 40]),
                    acceptance_rates=np.array([0.5, 0.4]),
                    cumulative_simulator_calls=np.array([80, 180]),
                    effective_sample_sizes=np.array([40.0, 40.0]),
                )
                self._write_metadata(
                    scripts_dir / f"{prefix}_metadata.json",
                    runtime=2.0,
                    simulator_calls=180,
                )

            outputs = run_comparison(scripts_dir, figures_dir)

            comparison = pd.read_csv(outputs["comparison_table"])
            errors = pd.read_csv(outputs["synthetic_error_table"])
            legacy_summary = pd.read_csv(outputs["legacy_summary_table"])
            legacy_real = pd.read_csv(outputs["legacy_real_table"])
            legacy_errors = pd.read_csv(outputs["legacy_error_table"])
            self.assertEqual(len(comparison), 24)
            self.assertEqual(len(errors), 4)
            self.assertTrue(comparison.equals(legacy_summary))
            self.assertTrue(errors.equals(legacy_errors))
            self.assertEqual(
                list(legacy_real.columns),
                ["method", "alpha", "beta", "sigma_eta"],
            )
            self.assertEqual(list(legacy_real["method"]), list(METHOD_ORDER))
            self.assertEqual(len(outputs["figures"]), 9)
            for figure in outputs["figures"]:
                self.assertTrue(figure.exists())
                self.assertGreater(figure.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from abc_smc_utils import ABCSMCConfig
from fixed_budget_study import (
    aggregate_fixed_budget_results,
    run_fixed_budget_study,
    save_fixed_budget_outputs,
    select_rejection_budget_posteriors,
)
from sv_abc_core import simulate_sv


class FirstThreeFeaturesModel:
    def predict(self, feature_matrix):
        feature_matrix = np.asarray(feature_matrix, dtype=float)
        return feature_matrix[:, :3]


class PrefixBudgetTests(unittest.TestCase):
    def test_larger_budgets_extend_one_shared_parameter_pool(self):
        parameters = np.column_stack(
            (
                np.arange(6, dtype=float),
                np.full(6, 0.95),
                np.full(6, 0.30),
            )
        )
        distances = np.array([0.4, 0.1, 0.3, 0.2, 0.05, 0.6])

        posteriors = select_rejection_budget_posteriors(
            parameters,
            distances,
            budgets=(4, 6),
            acceptance_fraction=0.5,
        )

        np.testing.assert_allclose(
            posteriors[0]["samples"][:, 0],
            [1.0, 3.0],
        )
        np.testing.assert_allclose(
            posteriors[1]["samples"][:, 0],
            [4.0, 1.0, 3.0],
        )
        self.assertEqual(posteriors[0]["simulator_calls"], 4)
        self.assertEqual(posteriors[1]["simulator_calls"], 6)


class FixedBudgetAggregationTests(unittest.TestCase):
    def test_aggregation_keeps_method_stage_and_target_separate(self):
        raw = pd.DataFrame(
            {
                "method": ["A", "A"],
                "target": ["alpha", "alpha"],
                "target_type": ["parameter", "parameter"],
                "stage_type": ["budget", "budget"],
                "stage_index": [0, 0],
                "bias": [0.1, -0.1],
                "absolute_error": [0.1, 0.1],
                "scaled_absolute_error": [0.2, 0.2],
                "covered": [True, False],
                "interval_width": [0.4, 0.6],
                "simulator_calls": [2_000, 2_000],
                "runtime_seconds": [1.0, 2.0],
            }
        )

        summary = aggregate_fixed_budget_results(raw)
        row = summary.iloc[0]

        self.assertEqual(row["n_datasets"], 2)
        self.assertEqual(row["mean_simulator_calls"], 2_000)
        self.assertAlmostEqual(row["rmse"], 0.1)
        self.assertAlmostEqual(row["coverage_rate"], 0.5)
        self.assertAlmostEqual(row["median_scaled_absolute_error"], 0.2)


class FixedBudgetOutputTests(unittest.TestCase):
    def test_outputs_contain_three_tables_metadata_and_six_figures(self):
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
        for seed in (1, 2):
            for method_index, method in enumerate(methods):
                for stage_index, calls in enumerate((2_000, 5_000)):
                    for target_index, target in enumerate(targets):
                        error = 0.01 * (
                            1 + seed + method_index + stage_index + target_index
                        )
                        raw_rows.append(
                            {
                                "seed": seed,
                                "method": method,
                                "target": target,
                                "target_type": (
                                    "parameter" if target_index < 3 else "derived"
                                ),
                                "stage_type": "budget",
                                "stage_index": stage_index,
                                "bias": error,
                                "absolute_error": error,
                                "scaled_absolute_error": error,
                                "covered": True,
                                "interval_width": 0.5,
                                "simulator_calls": calls,
                                "runtime_seconds": 1.0,
                            }
                        )
        raw = pd.DataFrame(raw_rows)
        summary = aggregate_fixed_budget_results(raw)
        final = summary.sort_values("stage_index").groupby(
            ["method", "target"],
            as_index=False,
        ).tail(1)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            outputs = save_fixed_budget_outputs(
                raw,
                summary,
                final,
                {"seeds": [1, 2]},
                root / "tables",
                root / "figures",
                methods,
            )

            self.assertEqual(len(outputs["tables"]), 3)
            self.assertEqual(len(outputs["figures"]), 6)
            metadata = json.loads(
                outputs["metadata"].read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["seeds"], [1, 2])
            for path in outputs["tables"] + outputs["figures"]:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


class SmallFixedBudgetStudyTests(unittest.TestCase):
    def test_small_study_compares_rejection_prefixes_and_smc_populations(self):
        true_parameters = np.array([-0.5, 0.95, 0.30])
        observed = simulate_sv(
            true_parameters,
            50,
            np.random.default_rng(99),
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
            max_populations=2,
            epsilon_quantile=0.5,
            max_attempts_per_population=100,
            random_seed=1,
        )

        study = run_fixed_budget_study(
            observed,
            true_parameters,
            model_bundle,
            rejection_budgets=(20, 30),
            rejection_seeds=(1,),
            acceptance_fraction=0.20,
            scale_simulations=20,
            scale_seed=8,
            smc_config=smc_config,
            smc_seeds=(1,),
            smc_workers=1,
        )

        self.assertEqual(len(study["raw_results"]), 48)
        self.assertEqual(len(study["summary"]), 48)
        self.assertEqual(len(study["final_comparison"]), 24)
        self.assertEqual(
            sorted(
                study["raw_results"]
                .query("method == 'Manual ABC'")["simulator_calls"]
                .unique()
            ),
            [20, 30],
        )
        self.assertEqual(
            study["raw_results"].query("method == 'ABC-SMC'")[
                "stage_index"
            ].nunique(),
            2,
        )


if __name__ == "__main__":
    unittest.main()

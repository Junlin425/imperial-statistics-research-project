import tempfile
import unittest
from pathlib import Path

import numpy as np

from learned_summary_ablation import (
    FEATURE_GROUPS,
    run_learned_summary_ablation,
    save_ablation_outputs,
    volatility_half_life,
)
from sv_abc_core import simulate_sv


class HalfLifeTests(unittest.TestCase):
    def test_half_life_uses_the_standard_financial_definition(self):
        beta = np.array([0.50, 0.90, 0.95])
        expected = np.log(0.5) / np.log(beta)

        np.testing.assert_allclose(volatility_half_life(beta), expected)

    def test_half_life_rejects_nonstationary_values(self):
        for beta in [0.0, 1.0, 1.01]:
            with self.subTest(beta=beta):
                with self.assertRaises(ValueError):
                    volatility_half_life(beta)


class AblationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.true_parameters = np.array([-0.5, 0.95, 0.30])
        self.observed = simulate_sv(
            self.true_parameters,
            60,
            np.random.default_rng(71),
        )

    def test_small_ablation_compares_all_three_feature_groups(self):
        outputs = run_learned_summary_ablation(
            self.observed,
            self.true_parameters,
            training_simulations=80,
            training_series_length=60,
            inference_simulations=100,
            acceptance_fraction=0.10,
            random_seed=73,
            n_estimators=10,
        )

        self.assertEqual(set(outputs["models"]), set(FEATURE_GROUPS))
        self.assertEqual(set(outputs["posteriors"]), set(FEATURE_GROUPS))
        self.assertEqual(
            set(outputs["validation"]["feature_group"]),
            set(FEATURE_GROUPS),
        )
        self.assertEqual(len(outputs["validation"]), 9)
        self.assertEqual(len(outputs["parameter_recovery"]), 9)
        self.assertEqual(len(outputs["half_life_recovery"]), 3)
        self.assertIn(
            "normalised_absolute_error",
            outputs["parameter_recovery"].columns,
        )
        self.assertTrue(
            np.all(
                outputs["parameter_recovery"][
                    "normalised_absolute_error"
                ]
                >= 0.0
            )
        )
        for result in outputs["posteriors"].values():
            self.assertEqual(result.accepted.shape, (10, 4))

    def test_outputs_are_saved_as_tables_models_posteriors_and_figures(self):
        outputs = run_learned_summary_ablation(
            self.observed,
            self.true_parameters,
            feature_groups=("marginal",),
            training_simulations=60,
            training_series_length=60,
            inference_simulations=60,
            acceptance_fraction=0.10,
            random_seed=79,
            n_estimators=5,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            saved = save_ablation_outputs(
                outputs,
                output_dir=root / "results",
                figures_dir=root / "figures",
            )

            expected_names = {
                "learned_summary_ablation_validation.csv",
                "learned_summary_ablation_parameter_recovery.csv",
                "learned_summary_ablation_half_life.csv",
                "learned_summary_ablation_metadata.json",
                "rf_sv_summary_marginal.pkl",
                "abc_auto_ablation_marginal_synthetic.npy",
                "ablation_validation_r2.png",
                "ablation_parameter_error.png",
                "ablation_half_life_error.png",
            }
            self.assertEqual(
                {path.name for path in saved},
                expected_names,
            )
            for path in saved:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from financial_derived_quantities import (
    calculate_financial_quantities,
    run_financial_analysis,
)


class FinancialQuantityTests(unittest.TestCase):
    def test_formulas_match_hand_calculated_values(self):
        samples = np.array([[-0.5, 0.95, 0.30]])

        quantities = calculate_financial_quantities(samples)

        self.assertAlmostEqual(quantities["long_run_log_variance"][0], -10.0)
        self.assertAlmostEqual(
            quantities["half_life_days"][0],
            13.513407333964874,
        )
        self.assertAlmostEqual(
            quantities["stationary_log_volatility_variance"][0],
            0.9230769230769228,
        )

    def test_nonstationary_beta_is_rejected(self):
        samples = np.array([[-0.5, 1.0, 0.30]])

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            calculate_financial_quantities(samples)


class FinancialAnalysisWorkflowTests(unittest.TestCase):
    def test_workflow_saves_tables_metadata_and_separate_figures(self):
        samples_a = np.array(
            [
                [-0.50, 0.95, 0.30],
                [-0.45, 0.94, 0.28],
                [-0.55, 0.96, 0.32],
            ]
        )
        samples_b = np.array(
            [
                [-0.48, 0.95, 0.31],
                [-0.52, 0.95, 0.29],
                [-0.50, 0.94, 0.30],
            ]
        )
        equal_weights = np.full(3, 1.0 / 3.0)
        results = {
            "synthetic": {
                "Method A": {"samples": samples_a, "weights": equal_weights},
                "Method B": {"samples": samples_b, "weights": equal_weights},
            },
            "real": {
                "Method A": {"samples": samples_b, "weights": equal_weights},
                "Method B": {"samples": samples_a, "weights": equal_weights},
            },
        }

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            outputs = run_financial_analysis(
                results,
                root / "tables",
                root / "figures",
                method_order=("Method A", "Method B"),
            )

            summary = pd.read_csv(outputs["summary_table"])
            errors = pd.read_csv(outputs["synthetic_error_table"])
            metadata = json.loads(
                outputs["metadata"].read_text(encoding="utf-8")
            )
            self.assertEqual(len(summary), 12)
            self.assertEqual(len(errors), 2)
            self.assertEqual(len(outputs["figures"]), 7)
            self.assertEqual(metadata["posterior_transformation"], "draw_by_draw")
            for figure in outputs["figures"]:
                self.assertTrue(figure.exists())
                self.assertGreater(figure.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

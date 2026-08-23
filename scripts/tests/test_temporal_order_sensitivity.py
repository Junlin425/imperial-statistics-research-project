import tempfile
import unittest
from pathlib import Path

import numpy as np

from auto_summary import train_auto_summary_model
from sv_abc_core import simulate_sv
from temporal_order_sensitivity import (
    block_permute,
    evaluate_temporal_sensitivity,
    save_temporal_sensitivity_outputs,
    summarise_temporal_sensitivity,
)


class BlockPermutationTests(unittest.TestCase):
    def test_block_permutation_keeps_values_and_complete_blocks(self):
        values = np.arange(10)
        permuted = block_permute(
            values,
            block_length=2,
            rng=np.random.default_rng(81),
        )

        np.testing.assert_array_equal(np.sort(permuted), values)
        returned_blocks = {
            tuple(permuted[index : index + 2])
            for index in range(0, len(permuted), 2)
        }
        expected_blocks = {
            tuple(values[index : index + 2])
            for index in range(0, len(values), 2)
        }
        self.assertEqual(returned_blocks, expected_blocks)

    def test_invalid_block_length_is_rejected(self):
        with self.assertRaises(ValueError):
            block_permute(
                np.arange(10),
                block_length=0,
                rng=np.random.default_rng(83),
            )


class TemporalSensitivityWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.true_parameters = np.array([-0.5, 0.95, 0.30])
        cls.observed = simulate_sv(
            cls.true_parameters,
            60,
            np.random.default_rng(85),
        )
        cls.model_bundle, _, _ = train_auto_summary_model(
            n_simulations=80,
            series_length=60,
            feature_group="full",
            random_seed=87,
            n_estimators=10,
        )

    def test_wasserstein_is_zero_because_permutation_keeps_all_values(self):
        raw = evaluate_temporal_sensitivity(
            self.observed,
            self.model_bundle,
            manual_scale=np.ones(4),
            dataset_name="synthetic",
            block_lengths=(1, 5),
            n_permutations=4,
            random_seed=89,
        )

        self.assertEqual(len(raw), 8)
        self.assertEqual(set(raw["block_length"]), {1, 5})
        self.assertTrue(np.all(raw["wasserstein_distance"] < 1e-12))
        self.assertTrue(np.all(np.isfinite(raw["manual_distance"])))
        self.assertTrue(np.all(np.isfinite(raw["auto_distance"])))

    def test_summary_contains_three_methods_and_relative_distances(self):
        raw = evaluate_temporal_sensitivity(
            self.observed,
            self.model_bundle,
            manual_scale=np.ones(4),
            dataset_name="synthetic",
            block_lengths=(1, 5),
            n_permutations=4,
            random_seed=91,
        )

        summary = summarise_temporal_sensitivity(raw)

        self.assertEqual(
            set(summary["method"]),
            {"Manual ABC", "Auto ABC", "Wasserstein ABC"},
        )
        self.assertEqual(len(summary), 6)
        wasserstein = summary[summary["method"] == "Wasserstein ABC"]
        self.assertTrue(np.all(wasserstein["relative_distance"] == 0.0))
        for method in ["Manual ABC", "Auto ABC"]:
            method_rows = summary[summary["method"] == method]
            self.assertAlmostEqual(method_rows["relative_distance"].max(), 1.0)

    def test_tables_metadata_and_one_figure_are_saved(self):
        raw = evaluate_temporal_sensitivity(
            self.observed,
            self.model_bundle,
            manual_scale=np.ones(4),
            dataset_name="synthetic",
            block_lengths=(1, 5),
            n_permutations=3,
            random_seed=93,
        )
        summary = summarise_temporal_sensitivity(raw)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            saved = save_temporal_sensitivity_outputs(
                raw,
                summary,
                metadata={"n_permutations": 3},
                output_dir=root / "results",
                figures_dir=root / "figures",
            )

            expected_names = {
                "temporal_order_sensitivity_raw.csv",
                "temporal_order_sensitivity_summary.csv",
                "temporal_order_sensitivity_metadata.json",
                "temporal_sensitivity_synthetic.png",
            }
            self.assertEqual({path.name for path in saved}, expected_names)
            for path in saved:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

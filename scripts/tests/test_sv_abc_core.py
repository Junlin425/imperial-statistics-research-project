import unittest

import numpy as np

from sv_abc_core import (
    PRIOR_BOUNDS,
    estimate_prior_predictive_summary_scale,
    is_valid_return_series,
    make_distance_simulator,
    manual_summary,
    normalized_summary_distance,
    robust_summary_scale,
    sample_prior,
    simulate_sv,
    squared_return_acf,
    uniform_prior_log_density,
)


class PriorTests(unittest.TestCase):
    def test_samples_respect_all_prior_bounds(self):
        samples = sample_prior(np.random.default_rng(4), 500)

        self.assertEqual(samples.shape, (500, 3))
        self.assertTrue(np.all(samples >= PRIOR_BOUNDS[:, 0]))
        self.assertTrue(np.all(samples <= PRIOR_BOUNDS[:, 1]))

    def test_uniform_prior_log_density_distinguishes_inside_and_outside(self):
        values = uniform_prior_log_density(
            np.array(
                [
                    [-0.5, 0.95, 0.30],
                    [-2.0, 0.95, 0.30],
                ]
            )
        )

        self.assertTrue(np.isfinite(values[0]))
        self.assertEqual(values[1], -np.inf)


class SVSimulatorTests(unittest.TestCase):
    def test_fixed_seed_reproduces_simulated_returns(self):
        theta = np.array([-0.5, 0.95, 0.30])
        first = simulate_sv(theta, 100, np.random.default_rng(17))
        second = simulate_sv(theta, 100, np.random.default_rng(17))

        self.assertEqual(first.shape, (100,))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, second)


class ManualSummaryTests(unittest.TestCase):
    def test_squared_return_acf_matches_its_direct_formula(self):
        returns = np.linspace(-1.0, 2.0, 30)
        squared = returns**2
        centred = squared - np.mean(squared)
        denominator = np.dot(centred, centred)
        expected = np.array(
            [
                np.dot(centred[lag:], centred[:-lag]) / denominator
                for lag in [1, 5]
            ]
        )

        actual = squared_return_acf(returns, [1, 5])

        np.testing.assert_allclose(actual, expected)

    def test_shared_return_validity_rule_rejects_degenerate_series(self):
        valid = np.array([1.0, -2.0, 1.5, -1.0, 0.5, -0.4])
        tiny = 1e-9 * valid

        self.assertTrue(is_valid_return_series(valid))
        self.assertFalse(is_valid_return_series(tiny))
        self.assertFalse(
            is_valid_return_series(
                np.array([1.0, -2.0, np.nan, -1.0, 0.5, -0.4])
            )
        )

    def test_robust_scale_uses_componentwise_median_absolute_deviation(self):
        summaries = np.array(
            [
                [1.0, 10.0, 100.0, 1000.0],
                [2.0, 20.0, 200.0, 2000.0],
                [3.0, 30.0, 300.0, 3000.0],
                [1000.0, 10000.0, 100000.0, 1000000.0],
            ]
        )

        scale = robust_summary_scale(summaries)

        np.testing.assert_allclose(
            scale,
            1.4826 * np.array([1.0, 10.0, 100.0, 1000.0]),
        )

    def test_distance_uses_an_explicit_component_scale(self):
        distance = normalized_summary_distance(
            np.array([2.0, 4.0]),
            np.array([1.0, 2.0]),
            scale=np.array([2.0, 4.0]),
        )

        self.assertAlmostEqual(distance, np.sqrt(0.5))

    def test_prior_predictive_scale_is_reproducible_and_reports_counts(self):
        first_scale, first_counts = estimate_prior_predictive_summary_scale(
            length=60,
            n_simulations=20,
            random_seed=23,
        )
        second_scale, second_counts = estimate_prior_predictive_summary_scale(
            length=60,
            n_simulations=20,
            random_seed=23,
        )

        np.testing.assert_allclose(first_scale, second_scale)
        self.assertEqual(first_counts, second_counts)
        self.assertEqual(first_counts["valid"] + first_counts["invalid"], 20)
        self.assertTrue(np.all(first_scale > 0.0))

    def test_zero_minimum_variance_allows_tiny_nonconstant_ppc_series(self):
        values = 1e-9 * np.array(
            [1.0, -2.0, 1.5, -1.0, 0.5, -0.4, 0.8, -1.2]
        )

        self.assertIsNone(manual_summary(values))
        ppc_summary = manual_summary(values, minimum_variance=0.0)

        self.assertEqual(ppc_summary.shape, (4,))
        self.assertTrue(np.all(np.isfinite(ppc_summary)))

    def test_valid_nonconstant_series_has_four_finite_statistics(self):
        values = np.sin(np.linspace(0.0, 20.0, 200))
        summary = manual_summary(values)

        self.assertEqual(summary.shape, (4,))
        self.assertTrue(np.isfinite(summary).all())

    def test_constant_and_nonfinite_series_are_rejected(self):
        self.assertIsNone(manual_summary(np.ones(100)))
        self.assertIsNone(
            manual_summary(np.array([0.0, 1.0, np.nan, 2.0]))
        )

    def test_normalized_distance_is_finite_when_observed_component_is_zero(self):
        distance = normalized_summary_distance(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0.0, 2.0, 3.0, 4.0]),
        )

        self.assertTrue(np.isfinite(distance))
        self.assertGreater(distance, 0.0)

    def test_distance_factory_returns_a_finite_distance(self):
        observed = simulate_sv(
            np.array([-0.5, 0.95, 0.30]),
            100,
            np.random.default_rng(2),
        )
        callback = make_distance_simulator(observed)
        distance = callback(
            np.array([-0.5, 0.95, 0.30]),
            np.random.default_rng(3),
        )

        self.assertTrue(np.isfinite(distance))


if __name__ == "__main__":
    unittest.main()

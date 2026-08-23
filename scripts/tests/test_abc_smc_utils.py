import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from abc_smc_utils import (
    ABCSMCConfig,
    ABCSMCResult,
    compute_importance_weights,
    effective_sample_size,
    kernel_scales,
    run_abc_smc,
    save_abc_smc_result,
    sample_truncated_kernel,
    truncated_kernel_logpdf,
    weighted_quantile,
)


BOUNDS = np.array(
    [
        [-1.5, -0.1],
        [0.85, 0.999],
        [0.05, 0.50],
    ],
    dtype=float,
)


class WeightedStatisticsTests(unittest.TestCase):
    def test_weighted_quantile_matches_inverse_empirical_cdf(self):
        result = weighted_quantile(
            np.array([0.0, 10.0, 20.0]),
            np.array([0.1, 0.5, 0.9]),
            np.array([0.2, 0.6, 0.2]),
        )

        np.testing.assert_allclose(result, [0.0, 10.0, 20.0])

    def test_effective_sample_size_is_particle_count_for_equal_weights(self):
        self.assertAlmostEqual(
            effective_sample_size(np.full(5, 0.2)),
            5.0,
        )


class TruncatedKernelTests(unittest.TestCase):
    def test_samples_stay_inside_prior_bounds(self):
        rng = np.random.default_rng(7)
        samples = np.array(
            [
                sample_truncated_kernel(
                    np.array([-1.49, 0.998, 0.051]),
                    np.array([0.2, 0.02, 0.05]),
                    BOUNDS,
                    rng,
                )
                for _ in range(200)
            ]
        )

        self.assertTrue(np.all(samples >= BOUNDS[:, 0]))
        self.assertTrue(np.all(samples <= BOUNDS[:, 1]))

    def test_kernel_logpdf_is_finite_for_interior_point(self):
        result = truncated_kernel_logpdf(
            np.array([-0.5, 0.95, 0.30]),
            np.array(
                [
                    [-0.6, 0.94, 0.25],
                    [-0.4, 0.96, 0.35],
                ]
            ),
            np.array([0.2, 0.02, 0.05]),
            BOUNDS,
        )

        self.assertEqual(result.shape, (2,))
        self.assertTrue(np.isfinite(result).all())

    def test_kernel_scales_use_minimum_fraction_when_variance_collapses(self):
        particles = np.tile(np.array([-0.5, 0.95, 0.30]), (4, 1))
        scales = kernel_scales(
            particles,
            np.full(4, 0.25),
            BOUNDS,
            minimum_fraction=0.01,
        )

        np.testing.assert_allclose(
            scales,
            0.01 * (BOUNDS[:, 1] - BOUNDS[:, 0]),
        )


class ImportanceWeightTests(unittest.TestCase):
    def test_importance_weights_are_finite_and_normalized(self):
        def flat_prior_log_density(points):
            return np.zeros(len(np.atleast_2d(points)))

        weights = compute_importance_weights(
            particles=np.array(
                [
                    [-0.6, 0.94, 0.25],
                    [-0.4, 0.96, 0.35],
                ]
            ),
            previous_particles=np.array(
                [
                    [-0.7, 0.93, 0.20],
                    [-0.3, 0.97, 0.40],
                ]
            ),
            previous_weights=np.array([0.5, 0.5]),
            scales=np.array([0.2, 0.02, 0.05]),
            bounds=BOUNDS,
            prior_log_density=flat_prior_log_density,
        )

        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue(np.all(weights >= 0))
        self.assertAlmostEqual(float(weights.sum()), 1.0)


class ABCSMCConfigTests(unittest.TestCase):
    def test_formal_defaults_match_the_approved_design(self):
        config = ABCSMCConfig()

        self.assertEqual(config.n_particles, 500)
        self.assertEqual(config.n_pilot, 2000)
        self.assertEqual(config.max_populations, 5)
        self.assertEqual(config.epsilon_quantile, 0.50)
        self.assertEqual(config.max_attempts_per_population, 50_000)
        self.assertEqual(config.random_seed, 42)
        self.assertEqual(config.minimum_kernel_scale_fraction, 0.01)

    def test_invalid_configuration_is_rejected(self):
        invalid_arguments = [
            {"n_particles": 0},
            {"n_particles": 10, "n_pilot": 9},
            {"epsilon_quantile": 0.0},
            {"epsilon_quantile": 1.0},
            {"max_populations": 0},
            {"max_attempts_per_population": 0},
            {"minimum_kernel_scale_fraction": 0.0},
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    ABCSMCConfig(**arguments)


class ABCSMCRunnerTests(unittest.TestCase):
    @staticmethod
    def _prior_sampler(rng, n_samples):
        return rng.uniform(-2.0, 2.0, size=(n_samples, 3))

    @staticmethod
    def _prior_log_density(points):
        points = np.atleast_2d(points)
        inside = np.all((points >= -2.0) & (points <= 2.0), axis=1)
        return np.where(inside, 0.0, -np.inf)

    @staticmethod
    def _distance(theta, rng):
        del rng
        return float(np.linalg.norm(theta - np.array([0.2, -0.1, 0.3])))

    def _run_small_example(self, seed=123):
        return run_abc_smc(
            config=ABCSMCConfig(
                n_particles=10,
                n_pilot=40,
                max_populations=2,
                epsilon_quantile=0.5,
                max_attempts_per_population=1_000,
                random_seed=seed,
                minimum_kernel_scale_fraction=0.01,
            ),
            bounds=np.tile(np.array([-2.0, 2.0]), (3, 1)),
            prior_sampler=self._prior_sampler,
            prior_log_density=self._prior_log_density,
            distance_simulator=self._distance,
        )

    def test_two_population_run_has_valid_shapes_and_diagnostics(self):
        result = self._run_small_example()

        self.assertEqual(result.particles.shape, (2, 10, 3))
        self.assertEqual(result.weights.shape, (2, 10))
        self.assertEqual(result.distances.shape, (2, 10))
        np.testing.assert_allclose(result.weights.sum(axis=1), 1.0)
        self.assertLess(result.epsilons[1], result.epsilons[0])
        self.assertTrue(
            np.all((result.effective_sample_sizes >= 1.0))
        )
        self.assertTrue(
            np.all((result.effective_sample_sizes <= 10.0 + 1e-10))
        )
        self.assertEqual(result.stop_reason, "max_populations_reached")

    def test_fixed_seed_reproduces_complete_history(self):
        first = self._run_small_example(seed=321)
        second = self._run_small_example(seed=321)

        np.testing.assert_allclose(first.particles, second.particles)
        np.testing.assert_allclose(first.weights, second.weights)
        np.testing.assert_allclose(first.distances, second.distances)

    def test_attempt_limit_preserves_previous_complete_population(self):
        calls = {"count": 0}

        def distance_that_becomes_impossible(theta, rng):
            del rng
            calls["count"] += 1
            if calls["count"] <= 10:
                return float(np.linalg.norm(theta))
            return 1_000_000.0

        result = run_abc_smc(
            config=ABCSMCConfig(
                n_particles=5,
                n_pilot=10,
                max_populations=2,
                max_attempts_per_population=5,
                random_seed=9,
            ),
            bounds=np.tile(np.array([-2.0, 2.0]), (3, 1)),
            prior_sampler=self._prior_sampler,
            prior_log_density=self._prior_log_density,
            distance_simulator=distance_that_becomes_impossible,
        )

        self.assertEqual(result.completed_populations, 1)
        self.assertEqual(result.stop_reason, "max_attempts_reached")
        self.assertEqual(result.total_simulator_calls, 15)


class ResultSerializationTests(unittest.TestCase):
    def test_history_final_particles_and_metadata_are_consistent(self):
        result = ABCSMCResult(
            particles=np.array(
                [
                    [
                        [-0.6, 0.94, 0.25],
                        [-0.5, 0.95, 0.30],
                        [-0.4, 0.96, 0.35],
                    ]
                ]
            ),
            weights=np.array([[0.2, 0.5, 0.3]]),
            distances=np.array([[0.3, 0.1, 0.2]]),
            epsilons=np.array([0.5]),
            candidate_simulations=np.array([20]),
            eligible_counts=np.array([10]),
            acceptance_rates=np.array([0.5]),
            cumulative_simulator_calls=np.array([20]),
            effective_sample_sizes=np.array([1.0 / 0.38]),
            total_simulator_calls=20,
            stop_reason="max_populations_reached",
        )

        with TemporaryDirectory() as temporary_directory:
            output_prefix = Path(temporary_directory) / "abc_smc_test"
            paths = save_abc_smc_result(
                result,
                output_prefix,
                {"method": "ABC-SMC", "dataset": "test"},
            )

            with np.load(paths["history_file"]) as history:
                self.assertEqual(
                    set(history.files),
                    {
                        "particles",
                        "weights",
                        "distances",
                        "epsilons",
                        "candidate_simulations",
                        "eligible_counts",
                        "acceptance_rates",
                        "cumulative_simulator_calls",
                        "effective_sample_sizes",
                    },
                )
                self.assertEqual(history["particles"].shape, (1, 3, 3))

            final_particles = pd.read_csv(paths["final_file"])
            self.assertEqual(
                list(final_particles.columns),
                ["alpha", "beta", "sigma_eta", "distance", "weight"],
            )
            self.assertEqual(len(final_particles), 3)

            metadata = pd.read_json(paths["metadata_file"], typ="series")
            self.assertEqual(metadata["completed_populations"], 1)
            self.assertEqual(metadata["total_simulator_calls"], 20)
            self.assertEqual(
                metadata["stop_reason"],
                "max_populations_reached",
            )


if __name__ == "__main__":
    unittest.main()

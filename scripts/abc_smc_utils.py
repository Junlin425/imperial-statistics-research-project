"""Generic numerical helpers for adaptive ABC-SMC experiments."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtr
from scipy.stats import truncnorm

from abc_experiment_utils import save_run_metadata


@dataclass(frozen=True)
class ABCSMCConfig:
    """Validated controls for one adaptive ABC-SMC run."""

    n_particles: int = 500
    n_pilot: int = 2_000
    max_populations: int = 5
    epsilon_quantile: float = 0.50
    max_attempts_per_population: int = 50_000
    random_seed: int = 42
    minimum_kernel_scale_fraction: float = 0.01

    def __post_init__(self):
        if not isinstance(self.n_particles, int) or self.n_particles <= 0:
            raise ValueError("n_particles must be a positive integer")
        if not isinstance(self.n_pilot, int) or self.n_pilot < self.n_particles:
            raise ValueError("n_pilot must be at least n_particles")
        if (
            not isinstance(self.max_populations, int)
            or self.max_populations <= 0
        ):
            raise ValueError("max_populations must be a positive integer")
        if not 0 < self.epsilon_quantile < 1:
            raise ValueError("epsilon_quantile must lie in (0, 1)")
        if (
            not isinstance(self.max_attempts_per_population, int)
            or self.max_attempts_per_population <= 0
        ):
            raise ValueError(
                "max_attempts_per_population must be a positive integer"
            )
        if not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        if (
            not np.isfinite(self.minimum_kernel_scale_fraction)
            or self.minimum_kernel_scale_fraction <= 0
        ):
            raise ValueError(
                "minimum_kernel_scale_fraction must be positive"
            )


@dataclass
class ABCSMCResult:
    """Complete populations and diagnostics from an ABC-SMC run."""

    particles: np.ndarray
    weights: np.ndarray
    distances: np.ndarray
    epsilons: np.ndarray
    candidate_simulations: np.ndarray
    eligible_counts: np.ndarray
    acceptance_rates: np.ndarray
    cumulative_simulator_calls: np.ndarray
    effective_sample_sizes: np.ndarray
    total_simulator_calls: int
    stop_reason: str

    @property
    def completed_populations(self):
        return int(len(self.epsilons))


def _normalized_weights(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError(
            "weights must be a non-empty one-dimensional array"
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")

    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weights must have a positive sum")

    return weights / total


def weighted_quantile(values, quantiles, weights):
    """Evaluate inverse weighted empirical-CDF quantiles."""

    values = np.asarray(values, dtype=float)
    quantiles = np.atleast_1d(np.asarray(quantiles, dtype=float))
    weights = _normalized_weights(weights)

    if values.ndim != 1 or len(values) != len(weights):
        raise ValueError("values and weights must be equal-length vectors")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    if not np.all(np.isfinite(quantiles)):
        raise ValueError("quantiles must be finite")
    if np.any((quantiles < 0) | (quantiles > 1)):
        raise ValueError("quantiles must lie in [0, 1]")

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    indices = np.searchsorted(cumulative, quantiles, side="left")
    indices = np.clip(indices, 0, len(values) - 1)

    return sorted_values[indices]


def effective_sample_size(weights):
    """Return the standard importance-sampling effective sample size."""

    weights = _normalized_weights(weights)
    return float(1.0 / np.sum(weights**2))


def _validated_bounds(bounds, dimension=None):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (dimension, 2)")
    if dimension is not None and len(bounds) != dimension:
        raise ValueError("bounds dimension does not match the parameters")
    if not np.all(np.isfinite(bounds)):
        raise ValueError("bounds must be finite")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("every lower bound must be below its upper bound")
    return bounds


def kernel_scales(
    particles,
    weights,
    bounds,
    minimum_fraction=0.01,
):
    """Return adaptive diagonal kernel scales for one population."""

    particles = np.asarray(particles, dtype=float)
    if particles.ndim != 2 or len(particles) == 0:
        raise ValueError("particles must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(particles)):
        raise ValueError("particles must be finite")
    bounds = _validated_bounds(bounds, particles.shape[1])
    weights = _normalized_weights(weights)
    if len(weights) != len(particles):
        raise ValueError("weights must match the number of particles")
    if not np.isfinite(minimum_fraction) or minimum_fraction <= 0:
        raise ValueError("minimum_fraction must be positive")

    weighted_mean = np.sum(particles * weights[:, None], axis=0)
    weighted_variance = np.sum(
        weights[:, None] * (particles - weighted_mean) ** 2,
        axis=0,
    )
    minimum_scale = minimum_fraction * (bounds[:, 1] - bounds[:, 0])

    return np.maximum(np.sqrt(2.0 * weighted_variance), minimum_scale)


def _validated_kernel_inputs(parent, scales, bounds):
    parent = np.asarray(parent, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if parent.ndim != 1 or scales.shape != parent.shape:
        raise ValueError("parent and scales must be equal-length vectors")
    bounds = _validated_bounds(bounds, len(parent))
    if not np.all(np.isfinite(parent)):
        raise ValueError("parent must be finite")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("scales must be finite and positive")
    if np.any(parent < bounds[:, 0]) or np.any(parent > bounds[:, 1]):
        raise ValueError("parent must lie inside the bounds")
    return parent, scales, bounds


def sample_truncated_kernel(parent, scales, bounds, rng):
    """Draw one parameter vector from a diagonal truncated Gaussian kernel."""

    parent, scales, bounds = _validated_kernel_inputs(
        parent,
        scales,
        bounds,
    )
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    standardized_lower = (bounds[:, 0] - parent) / scales
    standardized_upper = (bounds[:, 1] - parent) / scales

    return np.asarray(
        truncnorm.rvs(
            standardized_lower,
            standardized_upper,
            loc=parent,
            scale=scales,
            random_state=rng,
        ),
        dtype=float,
    )


def truncated_kernel_logpdf(point, centres, scales, bounds):
    """Evaluate a diagonal truncated-Gaussian log density per centre."""

    point = np.asarray(point, dtype=float)
    centres = np.asarray(centres, dtype=float)
    scales = np.asarray(scales, dtype=float)

    if point.ndim != 1:
        raise ValueError("point must be a vector")
    if centres.ndim != 2 or centres.shape[1] != len(point):
        raise ValueError("centres must have shape (n, dimension)")
    bounds = _validated_bounds(bounds, len(point))
    if scales.shape != point.shape:
        raise ValueError("scales must match the point dimension")
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(centres)):
        raise ValueError("point and centres must be finite")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("scales must be finite and positive")

    if np.any(point < bounds[:, 0]) or np.any(point > bounds[:, 1]):
        return np.full(len(centres), -np.inf)
    if np.any(centres < bounds[:, 0]) or np.any(centres > bounds[:, 1]):
        raise ValueError("centres must lie inside the bounds")

    standardized_point = (point[None, :] - centres) / scales[None, :]
    lower = (bounds[:, 0][None, :] - centres) / scales[None, :]
    upper = (bounds[:, 1][None, :] - centres) / scales[None, :]
    normalization = ndtr(upper) - ndtr(lower)
    normalization = np.maximum(normalization, np.finfo(float).tiny)

    component_logpdf = (
        -0.5 * standardized_point**2
        - 0.5 * math.log(2.0 * math.pi)
        - np.log(scales)[None, :]
        - np.log(normalization)
    )

    return np.sum(component_logpdf, axis=1)


def compute_importance_weights(
    particles,
    previous_particles,
    previous_weights,
    scales,
    bounds,
    prior_log_density,
):
    """Calculate normalized ABC-SMC importance weights in log space."""

    particles = np.asarray(particles, dtype=float)
    previous_particles = np.asarray(previous_particles, dtype=float)
    if particles.ndim != 2 or previous_particles.ndim != 2:
        raise ValueError("particle inputs must be two-dimensional arrays")
    if particles.shape[1] != previous_particles.shape[1]:
        raise ValueError("particle inputs must use the same dimension")
    if len(particles) == 0 or len(previous_particles) == 0:
        raise ValueError("particle inputs must be non-empty")
    if not np.all(np.isfinite(particles)):
        raise ValueError("particles must be finite")

    bounds = _validated_bounds(bounds, particles.shape[1])
    previous_weights = _normalized_weights(previous_weights)
    if len(previous_weights) != len(previous_particles):
        raise ValueError(
            "previous_weights must match previous_particles"
        )

    prior_logs = np.asarray(prior_log_density(particles), dtype=float)
    if prior_logs.ndim == 0:
        prior_logs = np.full(len(particles), float(prior_logs))
    if prior_logs.shape != (len(particles),):
        raise ValueError(
            "prior_log_density must return one value per particle"
        )

    log_previous_weights = np.log(previous_weights)
    log_weights = np.empty(len(particles), dtype=float)
    for index, particle in enumerate(particles):
        component_logs = truncated_kernel_logpdf(
            particle,
            previous_particles,
            scales,
            bounds,
        )
        mixture_log_density = logsumexp(
            log_previous_weights + component_logs
        )
        log_weights[index] = prior_logs[index] - mixture_log_density

    normalizer = logsumexp(log_weights)
    if not np.isfinite(normalizer):
        raise FloatingPointError(
            "importance weights could not be normalized"
        )

    weights = np.exp(log_weights - normalizer)
    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0:
        raise FloatingPointError("importance weights are invalid")

    return weights / weights.sum()


def _validated_prior_draws(draws, n_samples, dimension, bounds):
    draws = np.asarray(draws, dtype=float)
    if draws.shape != (n_samples, dimension):
        raise ValueError(
            "prior_sampler must return shape (n_samples, dimension)"
        )
    if not np.all(np.isfinite(draws)):
        raise ValueError("prior_sampler returned non-finite values")
    if np.any(draws < bounds[:, 0]) or np.any(draws > bounds[:, 1]):
        raise ValueError("prior_sampler returned values outside the bounds")
    return draws


def _finite_distance(distance_simulator, theta, rng):
    distance = distance_simulator(theta, rng)
    if distance is None:
        return np.inf
    distance = np.asarray(distance, dtype=float)
    if distance.ndim != 0:
        raise ValueError("distance_simulator must return a scalar or None")
    distance = float(distance)
    if not np.isfinite(distance):
        return np.inf
    if distance < 0:
        raise ValueError("distance_simulator returned a negative distance")
    return distance


def run_abc_smc(
    config,
    bounds,
    prior_sampler,
    prior_log_density,
    distance_simulator,
    progress_callback=None,
):
    """Run adaptive ABC-SMC and return complete weighted populations."""

    if not isinstance(config, ABCSMCConfig):
        raise TypeError("config must be an ABCSMCConfig")
    bounds = _validated_bounds(bounds)
    dimension = len(bounds)
    rng = np.random.default_rng(config.random_seed)

    population_particles = []
    population_weights = []
    population_distances = []
    epsilons = []
    candidate_simulations = []
    eligible_counts = []
    acceptance_rates = []
    cumulative_simulator_calls = []
    effective_sample_sizes = []

    pilot_particles = _validated_prior_draws(
        prior_sampler(rng, config.n_pilot),
        config.n_pilot,
        dimension,
        bounds,
    )
    pilot_distances = np.array(
        [
            _finite_distance(distance_simulator, particle, rng)
            for particle in pilot_particles
        ],
        dtype=float,
    )
    finite_pilot = pilot_distances[np.isfinite(pilot_distances)]
    if len(finite_pilot) == 0:
        raise RuntimeError("pilot produced no finite distances")

    epsilon_0 = float(
        np.quantile(finite_pilot, config.epsilon_quantile)
    )
    eligible_indices = np.flatnonzero(
        np.isfinite(pilot_distances) & (pilot_distances <= epsilon_0)
    )
    eligible_particle_list = [
        pilot_particles[index].copy() for index in eligible_indices
    ]
    eligible_distance_list = [
        float(pilot_distances[index]) for index in eligible_indices
    ]
    population_zero_eligible = len(eligible_particle_list)
    additional_attempts = 0

    while (
        len(eligible_particle_list) < config.n_particles
        and additional_attempts < config.max_attempts_per_population
    ):
        candidate = _validated_prior_draws(
            prior_sampler(rng, 1),
            1,
            dimension,
            bounds,
        )[0]
        distance = _finite_distance(distance_simulator, candidate, rng)
        additional_attempts += 1
        if distance <= epsilon_0:
            eligible_particle_list.append(candidate)
            eligible_distance_list.append(distance)
            population_zero_eligible += 1

    if len(eligible_particle_list) < config.n_particles:
        raise RuntimeError(
            "Population 0 could not be completed within the attempt limit"
        )

    selected_indices = rng.choice(
        len(eligible_particle_list),
        size=config.n_particles,
        replace=False,
    )
    current_particles = np.asarray(
        [eligible_particle_list[index] for index in selected_indices],
        dtype=float,
    )
    current_distances = np.asarray(
        [eligible_distance_list[index] for index in selected_indices],
        dtype=float,
    )
    current_weights = np.full(
        config.n_particles,
        1.0 / config.n_particles,
    )

    population_zero_candidates = config.n_pilot + additional_attempts
    cumulative_calls = population_zero_candidates
    population_particles.append(current_particles)
    population_weights.append(current_weights)
    population_distances.append(current_distances)
    epsilons.append(epsilon_0)
    candidate_simulations.append(population_zero_candidates)
    eligible_counts.append(population_zero_eligible)
    acceptance_rates.append(
        population_zero_eligible / population_zero_candidates
    )
    cumulative_simulator_calls.append(cumulative_calls)
    effective_sample_sizes.append(
        effective_sample_size(current_weights)
    )

    if progress_callback is not None:
        progress_callback(
            0,
            epsilon_0,
            population_zero_candidates,
            acceptance_rates[-1],
            effective_sample_sizes[-1],
            cumulative_calls,
        )

    stop_reason = "max_populations_reached"
    for population_index in range(1, config.max_populations):
        next_epsilon = float(
            np.quantile(current_distances, config.epsilon_quantile)
        )
        if not next_epsilon < epsilons[-1]:
            stop_reason = "epsilon_not_decreasing"
            break

        scales = kernel_scales(
            current_particles,
            current_weights,
            bounds,
            config.minimum_kernel_scale_fraction,
        )
        accepted_particles = []
        accepted_distances = []
        attempts = 0

        while (
            len(accepted_particles) < config.n_particles
            and attempts < config.max_attempts_per_population
        ):
            ancestor_index = int(
                rng.choice(
                    config.n_particles,
                    p=current_weights,
                )
            )
            candidate = sample_truncated_kernel(
                current_particles[ancestor_index],
                scales,
                bounds,
                rng,
            )
            distance = _finite_distance(
                distance_simulator,
                candidate,
                rng,
            )
            attempts += 1
            if distance <= next_epsilon:
                accepted_particles.append(candidate)
                accepted_distances.append(distance)

        cumulative_calls += attempts
        if len(accepted_particles) < config.n_particles:
            stop_reason = "max_attempts_reached"
            break

        next_particles = np.asarray(accepted_particles, dtype=float)
        next_distances = np.asarray(accepted_distances, dtype=float)
        next_weights = compute_importance_weights(
            next_particles,
            current_particles,
            current_weights,
            scales,
            bounds,
            prior_log_density,
        )
        acceptance_rate = config.n_particles / attempts
        ess = effective_sample_size(next_weights)

        population_particles.append(next_particles)
        population_weights.append(next_weights)
        population_distances.append(next_distances)
        epsilons.append(next_epsilon)
        candidate_simulations.append(attempts)
        eligible_counts.append(config.n_particles)
        acceptance_rates.append(acceptance_rate)
        cumulative_simulator_calls.append(cumulative_calls)
        effective_sample_sizes.append(ess)

        current_particles = next_particles
        current_distances = next_distances
        current_weights = next_weights

        if progress_callback is not None:
            progress_callback(
                population_index,
                next_epsilon,
                attempts,
                acceptance_rate,
                ess,
                cumulative_calls,
            )

    return ABCSMCResult(
        particles=np.stack(population_particles),
        weights=np.stack(population_weights),
        distances=np.stack(population_distances),
        epsilons=np.asarray(epsilons, dtype=float),
        candidate_simulations=np.asarray(
            candidate_simulations,
            dtype=int,
        ),
        eligible_counts=np.asarray(eligible_counts, dtype=int),
        acceptance_rates=np.asarray(acceptance_rates, dtype=float),
        cumulative_simulator_calls=np.asarray(
            cumulative_simulator_calls,
            dtype=int,
        ),
        effective_sample_sizes=np.asarray(
            effective_sample_sizes,
            dtype=float,
        ),
        total_simulator_calls=int(cumulative_calls),
        stop_reason=stop_reason,
    )


def validate_abc_smc_result(result):
    """Validate the complete-population output contract."""

    if not isinstance(result, ABCSMCResult):
        raise TypeError("result must be an ABCSMCResult")

    particles = np.asarray(result.particles, dtype=float)
    if particles.ndim != 3 or particles.shape[2] != 3:
        raise ValueError("particles must have shape (G, N, 3)")
    populations, n_particles, _ = particles.shape
    if populations == 0 or n_particles == 0:
        raise ValueError("result must contain a complete population")

    expected_matrix_shape = (populations, n_particles)
    if np.asarray(result.weights).shape != expected_matrix_shape:
        raise ValueError("weights do not match particles")
    if np.asarray(result.distances).shape != expected_matrix_shape:
        raise ValueError("distances do not match particles")

    diagnostic_names = (
        "epsilons",
        "candidate_simulations",
        "eligible_counts",
        "acceptance_rates",
        "cumulative_simulator_calls",
        "effective_sample_sizes",
    )
    for name in diagnostic_names:
        if np.asarray(getattr(result, name)).shape != (populations,):
            raise ValueError(f"{name} must have one value per population")

    numeric_arrays = (
        particles,
        np.asarray(result.weights, dtype=float),
        np.asarray(result.distances, dtype=float),
        np.asarray(result.epsilons, dtype=float),
        np.asarray(result.acceptance_rates, dtype=float),
        np.asarray(result.effective_sample_sizes, dtype=float),
    )
    if not all(np.all(np.isfinite(array)) for array in numeric_arrays):
        raise ValueError("result contains non-finite values")
    if np.any(result.weights < 0):
        raise ValueError("weights must be non-negative")
    if not np.allclose(result.weights.sum(axis=1), 1.0):
        raise ValueError("every population's weights must sum to one")
    if populations > 1 and not np.all(np.diff(result.epsilons) < 0):
        raise ValueError("epsilons must be strictly decreasing")
    if np.any(result.candidate_simulations <= 0):
        raise ValueError("candidate_simulations must be positive")
    if np.any(result.cumulative_simulator_calls <= 0):
        raise ValueError("cumulative simulator calls must be positive")
    if (
        not isinstance(result.total_simulator_calls, int)
        or result.total_simulator_calls
        < int(result.cumulative_simulator_calls[-1])
    ):
        raise ValueError(
            "total_simulator_calls must include every attempted simulation"
        )


def save_abc_smc_result(result, output_prefix, metadata):
    """Save complete histories, final weighted particles, and metadata."""

    validate_abc_smc_result(result)
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    history_file = Path(f"{output_prefix}_history.npz")
    final_file = Path(f"{output_prefix}_final.csv")
    metadata_file = Path(f"{output_prefix}_metadata.json")

    np.savez_compressed(
        history_file,
        particles=result.particles,
        weights=result.weights,
        distances=result.distances,
        epsilons=result.epsilons,
        candidate_simulations=result.candidate_simulations,
        eligible_counts=result.eligible_counts,
        acceptance_rates=result.acceptance_rates,
        cumulative_simulator_calls=result.cumulative_simulator_calls,
        effective_sample_sizes=result.effective_sample_sizes,
    )

    final_particles = pd.DataFrame(
        np.column_stack(
            (
                result.particles[-1],
                result.distances[-1],
                result.weights[-1],
            )
        ),
        columns=(
            "alpha",
            "beta",
            "sigma_eta",
            "distance",
            "weight",
        ),
    )
    final_particles.to_csv(final_file, index=False)

    complete_metadata = dict(metadata)
    complete_metadata.update(
        {
            "completed_populations": result.completed_populations,
            "epsilon_sequence": result.epsilons.tolist(),
            "total_simulator_calls": result.total_simulator_calls,
            "final_ess": float(result.effective_sample_sizes[-1]),
            "final_acceptance_rate": float(result.acceptance_rates[-1]),
            "stop_reason": result.stop_reason,
            "history_file": str(history_file),
            "final_file": str(final_file),
        }
    )
    save_run_metadata(metadata_file, complete_metadata)

    return {
        "history_file": history_file,
        "final_file": final_file,
        "metadata_file": metadata_file,
    }

"""Shared one-dimensional Wasserstein rejection ABC runner."""

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.stats import wasserstein_distance

from abc_experiment_utils import save_run_metadata, select_top_fraction
from sv_abc_core import is_valid_return_series, sample_prior, simulate_sv


@dataclass(frozen=True)
class WassersteinABCResult:
    accepted: np.ndarray
    effective_epsilon: float
    valid_simulations: int
    invalid_simulations: int
    runtime_seconds: float


def run_wasserstein_rejection_abc(
    observed_values,
    n_simulations=10_000,
    acceptance_fraction=0.05,
    random_seed=42,
):
    """Run rejection ABC with the empirical one-dimensional W1 distance."""

    observed_values = np.asarray(observed_values, dtype=float)
    if (
        observed_values.ndim != 1
        or len(observed_values) < 2
        or not np.all(np.isfinite(observed_values))
    ):
        raise ValueError("observed_values must be a finite vector")
    if not isinstance(n_simulations, int) or n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer")
    if not 0.0 < acceptance_fraction <= 1.0:
        raise ValueError("acceptance_fraction must lie in (0, 1]")
    if not isinstance(random_seed, (int, np.integer)):
        raise TypeError("random_seed must be an integer")

    start_time = time.perf_counter()
    rng = np.random.default_rng(int(random_seed))
    parameters = sample_prior(rng, n_simulations)
    distances = np.full(n_simulations, np.finfo(float).max, dtype=float)
    valid_simulations = 0

    for index, theta in enumerate(parameters):
        simulated_values = simulate_sv(theta, len(observed_values), rng)
        if not is_valid_return_series(simulated_values):
            continue
        distance = float(wasserstein_distance(observed_values, simulated_values))
        if not np.isfinite(distance):
            continue
        distances[index] = distance
        valid_simulations += 1

    accepted_count = max(1, math.ceil(n_simulations * acceptance_fraction))
    if valid_simulations < accepted_count:
        raise RuntimeError(
            "not enough valid simulations to construct the requested posterior"
        )

    accepted, effective_epsilon = select_top_fraction(
        parameters,
        distances,
        acceptance_fraction,
    )
    runtime_seconds = time.perf_counter() - start_time

    return WassersteinABCResult(
        accepted=accepted,
        effective_epsilon=effective_epsilon,
        valid_simulations=int(valid_simulations),
        invalid_simulations=int(n_simulations - valid_simulations),
        runtime_seconds=float(runtime_seconds),
    )


def save_wasserstein_abc_result(
    result,
    sample_path,
    metadata_path,
    metadata,
):
    """Save one Wasserstein ABC posterior and its run metadata."""

    if not isinstance(result, WassersteinABCResult):
        raise TypeError("result must be a WassersteinABCResult")

    sample_path = str(sample_path)
    np.save(sample_path, result.accepted)
    complete_metadata = dict(metadata)
    complete_metadata.update(
        {
            "accepted_count": int(len(result.accepted)),
            "effective_epsilon": result.effective_epsilon,
            "valid_simulations": result.valid_simulations,
            "invalid_simulations": result.invalid_simulations,
            "runtime_seconds": result.runtime_seconds,
            "output_file": sample_path,
        }
    )
    save_run_metadata(metadata_path, complete_metadata)

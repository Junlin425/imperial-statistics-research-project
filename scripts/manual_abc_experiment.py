"""Shared Manual-summary rejection ABC experiment runner."""

import math
import time
from dataclasses import dataclass

import numpy as np

from abc_experiment_utils import save_run_metadata, select_top_fraction
from sv_abc_core import (
    estimate_prior_predictive_summary_scale,
    manual_summary,
    normalized_summary_distance,
    sample_prior,
    simulate_sv,
)


@dataclass(frozen=True)
class ManualABCResult:
    accepted: np.ndarray
    effective_epsilon: float
    summary_scale: np.ndarray
    valid_simulations: int
    invalid_simulations: int
    scale_valid_simulations: int
    scale_invalid_simulations: int
    scale_seed: int
    inference_seconds: float
    scale_seconds: float


def run_manual_rejection_abc(
    observed_values,
    n_simulations=10_000,
    acceptance_fraction=0.05,
    random_seed=42,
    scale_simulations=2_000,
    scale_seed=202607,
):
    """Run Manual rejection ABC with a robust prior-predictive scale."""

    observed_values = np.asarray(observed_values, dtype=float)
    observed_summary = manual_summary(observed_values)
    if observed_summary is None:
        raise ValueError("observed_values do not produce a valid summary")
    if not isinstance(n_simulations, int) or n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer")
    if not 0.0 < acceptance_fraction <= 1.0:
        raise ValueError("acceptance_fraction must lie in (0, 1]")
    if not isinstance(random_seed, (int, np.integer)):
        raise TypeError("random_seed must be an integer")

    scale_start = time.perf_counter()
    summary_scale, scale_counts = estimate_prior_predictive_summary_scale(
        length=len(observed_values),
        n_simulations=scale_simulations,
        random_seed=scale_seed,
    )
    scale_seconds = time.perf_counter() - scale_start

    inference_start = time.perf_counter()
    rng = np.random.default_rng(int(random_seed))
    parameters = sample_prior(rng, n_simulations)
    distances = np.full(n_simulations, np.finfo(float).max, dtype=float)
    valid_simulations = 0

    for index, theta in enumerate(parameters):
        simulated_values = simulate_sv(theta, len(observed_values), rng)
        with np.errstate(over="ignore", invalid="ignore"):
            simulated_summary = manual_summary(simulated_values)
        if simulated_summary is None:
            continue
        distances[index] = normalized_summary_distance(
            simulated_summary,
            observed_summary,
            scale=summary_scale,
        )
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
    inference_seconds = time.perf_counter() - inference_start

    return ManualABCResult(
        accepted=accepted,
        effective_epsilon=effective_epsilon,
        summary_scale=summary_scale,
        valid_simulations=int(valid_simulations),
        invalid_simulations=int(n_simulations - valid_simulations),
        scale_valid_simulations=int(scale_counts["valid"]),
        scale_invalid_simulations=int(scale_counts["invalid"]),
        scale_seed=int(scale_seed),
        inference_seconds=float(inference_seconds),
        scale_seconds=float(scale_seconds),
    )


def save_manual_abc_result(
    result,
    sample_path,
    metadata_path,
    metadata,
):
    """Save one Manual ABC posterior and its complete run metadata."""

    if not isinstance(result, ManualABCResult):
        raise TypeError("result must be a ManualABCResult")

    sample_path = str(sample_path)
    np.save(sample_path, result.accepted)

    complete_metadata = dict(metadata)
    complete_metadata.update(
        {
            "accepted_count": int(len(result.accepted)),
            "effective_epsilon": result.effective_epsilon,
            "valid_simulations": result.valid_simulations,
            "invalid_simulations": result.invalid_simulations,
            "scale_valid_simulations": result.scale_valid_simulations,
            "scale_invalid_simulations": result.scale_invalid_simulations,
            "scale_seed": result.scale_seed,
            "summary_scale": result.summary_scale.tolist(),
            "summary_scale_method": (
                "1.4826 times componentwise MAD from a fixed "
                "prior-predictive bank, with IQR/SD fallback"
            ),
            "inference_seconds": result.inference_seconds,
            "scale_seconds": result.scale_seconds,
            "output_file": sample_path,
        }
    )
    save_run_metadata(metadata_path, complete_metadata)

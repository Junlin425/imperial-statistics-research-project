"""Rejection ABC runner for the learned summaries."""

import math
import time
from dataclasses import dataclass

import numpy as np

from abc_experiment_utils import save_run_metadata, select_top_fraction
from auto_summary import extract_features
from sv_abc_core import sample_prior, simulate_sv


@dataclass(frozen=True)
class AutoABCResult:
    accepted: np.ndarray
    effective_epsilon: float
    valid_simulations: int
    invalid_simulations: int
    runtime_seconds: float
    prediction_scale: np.ndarray
    training_series_length: int
    target_series_length: int
    feature_group: str


def run_auto_rejection_abc(
    observed_values,
    model_bundle,
    n_simulations=10_000,
    acceptance_fraction=0.05,
    random_seed=42,
):
    """Run learned-summary ABC with a scaled prediction distance."""

    model = model_bundle["model"]
    feature_group = model_bundle["feature_group"]
    prediction_scale = np.asarray(
        model_bundle["prediction_scale"],
        dtype=float,
    )
    observed_features = extract_features(observed_values, feature_group)
    if observed_features is None:
        raise ValueError("observed_values do not produce valid features")
    if not isinstance(n_simulations, int) or n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer")
    if not 0.0 < acceptance_fraction <= 1.0:
        raise ValueError("acceptance_fraction must lie in (0, 1]")

    observed_prediction = model.predict(observed_features.reshape(1, -1))[0]
    rng = np.random.default_rng(random_seed)
    parameters = sample_prior(rng, n_simulations)
    distances = np.full(n_simulations, np.finfo(float).max, dtype=float)
    valid_indices = []
    feature_rows = []
    start_time = time.perf_counter()

    for index, theta in enumerate(parameters):
        simulated_values = simulate_sv(theta, len(observed_values), rng)
        row = extract_features(simulated_values, feature_group)
        if row is None:
            continue
        valid_indices.append(index)
        feature_rows.append(row)

    if feature_rows:
        predictions = model.predict(np.asarray(feature_rows, dtype=float))
        scaled_differences = (
            predictions - observed_prediction
        ) / prediction_scale
        distances[np.asarray(valid_indices)] = np.linalg.norm(
            scaled_differences,
            axis=1,
        )

    valid_count = len(valid_indices)
    accepted_count = max(1, math.ceil(n_simulations * acceptance_fraction))
    if valid_count < accepted_count:
        raise RuntimeError(
            "not enough valid simulations to construct the requested posterior"
        )

    accepted, epsilon = select_top_fraction(
        parameters,
        distances,
        acceptance_fraction,
    )
    return AutoABCResult(
        accepted=accepted,
        effective_epsilon=epsilon,
        valid_simulations=int(valid_count),
        invalid_simulations=int(n_simulations - valid_count),
        runtime_seconds=float(time.perf_counter() - start_time),
        prediction_scale=prediction_scale,
        training_series_length=int(model_bundle["series_length"]),
        target_series_length=int(len(observed_values)),
        feature_group=feature_group,
    )


def save_auto_abc_result(result, sample_path, metadata_path, metadata):
    """Save one Auto ABC posterior and its run information."""

    if not isinstance(result, AutoABCResult):
        raise TypeError("result must be an AutoABCResult")
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
            "feature_group": result.feature_group,
            "prediction_scale": result.prediction_scale.tolist(),
            "distance_scaling": "prediction MAD",
            "training_series_length": result.training_series_length,
            "target_series_length": result.target_series_length,
            "output_file": sample_path,
        }
    )
    save_run_metadata(metadata_path, complete_metadata)

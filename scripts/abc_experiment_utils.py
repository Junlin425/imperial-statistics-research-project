"""Shared helpers for reproducible ABC experiment comparisons."""

import json
import math
from pathlib import Path

import numpy as np

from project_paths import script_output


PARAMETER_NAMES = ("alpha", "beta", "sigma_eta")


def select_top_fraction(parameters, distances, fraction):
    """Return the exact lowest-distance fraction of parameter draws.

    The returned array has columns ``alpha``, ``beta``, ``sigma_eta``,
    and ``distance``, sorted from the smallest to the largest distance.
    """

    parameters = np.asarray(parameters, dtype=float)
    distances = np.asarray(distances, dtype=float)

    if parameters.ndim != 2 or parameters.shape[1] != 3:
        raise ValueError("parameters must have shape (n, 3)")
    if distances.ndim != 1:
        raise ValueError("distances must be a one-dimensional array")
    if len(parameters) != len(distances):
        raise ValueError("parameters and distances must have equal lengths")
    if len(parameters) == 0:
        raise ValueError("at least one simulation is required")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1]")
    if not np.all(np.isfinite(parameters)):
        raise ValueError("parameters must contain only finite values")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distances must contain only finite values")

    accepted_count = max(1, math.ceil(len(distances) * fraction))
    selected_indices = np.argsort(
        distances,
        kind="stable",
    )[:accepted_count]

    accepted = np.column_stack(
        (
            parameters[selected_indices],
            distances[selected_indices],
        )
    )
    effective_epsilon = float(accepted[-1, 3])

    return accepted, effective_epsilon


def evaluate_convergence_prefixes(
    parameters,
    distances,
    n_values,
    epsilon_values,
    truth,
    seed,
):
    """Evaluate an ABC convergence grid from prefixes of one simulation pool."""

    parameters = np.asarray(parameters, dtype=float)
    distances = np.asarray(distances, dtype=float)
    truth = np.asarray(truth, dtype=float)

    if parameters.ndim != 2 or parameters.shape[1] != 3:
        raise ValueError("parameters must have shape (n, 3)")
    if distances.ndim != 1 or len(distances) != len(parameters):
        raise ValueError("distances must match the number of parameter draws")
    if truth.shape != (3,):
        raise ValueError("truth must contain alpha, beta, and sigma_eta")

    rows = []

    for n_simulations in n_values:
        if n_simulations <= 0 or n_simulations > len(parameters):
            raise ValueError("each N must be within the simulation pool")

        prefix_parameters = parameters[:n_simulations]
        prefix_distances = distances[:n_simulations]

        for epsilon in epsilon_values:
            accepted = prefix_parameters[prefix_distances < epsilon]

            if len(accepted):
                means = accepted.mean(axis=0)
                absolute_errors = np.abs(means - truth)
            else:
                means = np.full(3, np.nan)
                absolute_errors = np.full(3, np.nan)

            row = {
                "seed": int(seed),
                "N": int(n_simulations),
                "epsilon": float(epsilon),
                "accepted": int(len(accepted)),
            }

            for index, name in enumerate(PARAMETER_NAMES):
                row[f"{name}_mean"] = float(means[index])
                row[f"{name}_abs_error"] = float(
                    absolute_errors[index]
                )

            rows.append(row)

    return rows


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def save_run_metadata(path, metadata):
    """Write reproducibility metadata as human-readable JSON."""

    path = Path(path)
    if not path.is_absolute():
        path = script_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
            default=_json_value,
        )
        + "\n",
        encoding="utf-8",
    )
    return path

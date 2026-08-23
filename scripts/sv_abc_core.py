"""Shared stochastic-volatility model operations for ABC-SMC."""

import numpy as np
from scipy.stats import kurtosis


PRIOR_BOUNDS = np.array(
    [
        [-1.5, -0.1],
        [0.85, 0.999],
        [0.05, 0.50],
    ],
    dtype=float,
)


def sample_prior(rng, n_samples):
    """Draw independent parameter vectors from the approved uniform prior."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    return rng.uniform(
        PRIOR_BOUNDS[:, 0],
        PRIOR_BOUNDS[:, 1],
        size=(n_samples, 3),
    )


def uniform_prior_log_density(theta):
    """Evaluate the joint uniform-prior log density."""

    theta = np.asarray(theta, dtype=float)
    scalar_input = theta.ndim == 1
    theta = np.atleast_2d(theta)
    if theta.ndim != 2 or theta.shape[1] != 3:
        raise ValueError("theta must have final dimension 3")

    inside = np.all(
        (theta >= PRIOR_BOUNDS[:, 0])
        & (theta <= PRIOR_BOUNDS[:, 1]),
        axis=1,
    )
    log_constant = -float(
        np.log(PRIOR_BOUNDS[:, 1] - PRIOR_BOUNDS[:, 0]).sum()
    )
    result = np.where(inside, log_constant, -np.inf)

    if scalar_input:
        return float(result[0])
    return result


def simulate_sv(theta, length, rng):
    """Simulate returns from the thesis stochastic-volatility model."""

    theta = np.asarray(theta, dtype=float)
    if theta.shape != (3,) or not np.all(np.isfinite(theta)):
        raise ValueError("theta must be a finite vector of length 3")
    if np.any(theta < PRIOR_BOUNDS[:, 0]) or np.any(
        theta > PRIOR_BOUNDS[:, 1]
    ):
        raise ValueError("theta must lie inside the prior bounds")
    if not isinstance(length, int) or length < 2:
        raise ValueError("length must be an integer of at least 2")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    alpha, beta, sigma_eta = theta
    log_sigma_squared = np.empty(length, dtype=float)
    log_sigma_squared[0] = alpha / (1.0 - beta)
    innovations = rng.normal(0.0, sigma_eta, size=length - 1)

    for index in range(1, length):
        log_sigma_squared[index] = (
            alpha
            + beta * log_sigma_squared[index - 1]
            + innovations[index - 1]
        )

    sigma = np.exp(np.clip(log_sigma_squared / 2.0, -745.0, 350.0))
    observation_errors = rng.normal(0.0, 1.0, size=length)
    returns = sigma * observation_errors

    return returns


def is_valid_return_series(values, minimum_variance=1e-12):
    """Apply one validity rule before any ABC data comparison."""

    values = np.asarray(values, dtype=float)
    if (
        not np.isscalar(minimum_variance)
        or not np.isfinite(minimum_variance)
        or minimum_variance < 0.0
    ):
        raise ValueError("minimum_variance must be a finite non-negative value")
    if values.ndim != 1 or len(values) < 6:
        return False
    if not np.all(np.isfinite(values)):
        return False

    with np.errstate(over="ignore", invalid="ignore"):
        variance = float(np.var(values))
        squared_variance = float(np.var(values**2))
    return bool(
        np.isfinite(variance)
        and np.isfinite(squared_variance)
        and variance > 0.0
        and squared_variance > 0.0
        and variance >= minimum_variance
        and squared_variance >= minimum_variance
    )


def squared_return_acf(values, lags):
    """Calculate squared-return autocorrelations at selected lags."""

    values = np.asarray(values, dtype=float)
    lags = tuple(lags)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite one-dimensional array")
    if not lags or any(
        not isinstance(lag, (int, np.integer))
        or lag <= 0
        or lag >= len(values)
        for lag in lags
    ):
        raise ValueError("lags must be positive integers below the series length")

    squared_values = values**2
    centred_values = squared_values - np.mean(squared_values)
    denominator = np.dot(centred_values, centred_values)
    if denominator <= 0.0:
        raise ValueError("squared returns must not be constant")

    return np.array(
        [
            np.dot(centred_values[lag:], centred_values[:-lag])
            / denominator
            for lag in lags
        ],
        dtype=float,
    )


def manual_summary(values, minimum_variance=1e-12):
    """Return Manual summaries, or None for invalid or degenerate data."""

    values = np.asarray(values, dtype=float)
    if not is_valid_return_series(values, minimum_variance=minimum_variance):
        return None

    variance = float(np.var(values))
    kurtosis_value = float(kurtosis(values, fisher=False))
    acf_1, acf_5 = squared_return_acf(values, [1, 5])
    summary = np.array(
        [
            variance,
            kurtosis_value,
            acf_1,
            acf_5,
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(summary)):
        return None
    return summary


def robust_summary_scale(summary_matrix, minimum_scale=1e-12):
    """Estimate component scales using a robust prior-predictive spread."""

    summary_matrix = np.asarray(summary_matrix, dtype=float)
    if summary_matrix.ndim != 2 or summary_matrix.shape[0] < 2:
        raise ValueError("summary_matrix must contain at least two rows")
    if not np.all(np.isfinite(summary_matrix)):
        raise ValueError("summary_matrix must contain only finite values")
    if (
        not np.isscalar(minimum_scale)
        or not np.isfinite(minimum_scale)
        or minimum_scale <= 0.0
    ):
        raise ValueError("minimum_scale must be a positive finite value")

    median = np.median(summary_matrix, axis=0)
    mad_scale = 1.4826 * np.median(
        np.abs(summary_matrix - median),
        axis=0,
    )
    q25, q75 = np.quantile(summary_matrix, [0.25, 0.75], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    standard_deviation = np.std(summary_matrix, axis=0, ddof=1)

    scale = np.where(mad_scale > minimum_scale, mad_scale, iqr_scale)
    scale = np.where(scale > minimum_scale, scale, standard_deviation)
    return np.maximum(scale, minimum_scale)


def estimate_prior_predictive_summary_scale(
    length,
    n_simulations=2_000,
    random_seed=202607,
    minimum_variance=1e-12,
):
    """Estimate Manual-summary scales from a fixed prior-predictive bank."""

    if not isinstance(length, int) or length < 6:
        raise ValueError("length must be an integer of at least 6")
    if not isinstance(n_simulations, int) or n_simulations < 2:
        raise ValueError("n_simulations must be an integer of at least 2")
    if not isinstance(random_seed, (int, np.integer)):
        raise TypeError("random_seed must be an integer")

    rng = np.random.default_rng(int(random_seed))
    parameters = sample_prior(rng, n_simulations)
    valid_summaries = []

    for theta in parameters:
        simulated_values = simulate_sv(theta, length, rng)
        with np.errstate(over="ignore", invalid="ignore"):
            summary = manual_summary(
                simulated_values,
                minimum_variance=minimum_variance,
            )
        if summary is not None:
            valid_summaries.append(summary)

    valid_count = len(valid_summaries)
    if valid_count < 2:
        raise RuntimeError(
            "fewer than two valid prior-predictive summaries were generated"
        )

    scale = robust_summary_scale(np.asarray(valid_summaries, dtype=float))
    counts = {
        "requested": int(n_simulations),
        "valid": int(valid_count),
        "invalid": int(n_simulations - valid_count),
        "random_seed": int(random_seed),
    }
    return scale, counts


def normalized_summary_distance(
    simulated_summary,
    observed_summary,
    scale=None,
):
    """Calculate a dimensionless Manual ABC Euclidean discrepancy."""

    simulated_summary = np.asarray(simulated_summary, dtype=float)
    observed_summary = np.asarray(observed_summary, dtype=float)
    if simulated_summary.shape != observed_summary.shape:
        raise ValueError("summary vectors must have the same shape")
    if simulated_summary.ndim != 1 or len(simulated_summary) == 0:
        raise ValueError("summary vectors must be non-empty vectors")
    if not np.all(np.isfinite(simulated_summary)) or not np.all(
        np.isfinite(observed_summary)
    ):
        raise ValueError("summary vectors must be finite")

    if scale is None:
        scale = np.maximum(np.abs(observed_summary), 1e-12)
    else:
        scale = np.asarray(scale, dtype=float)
        if scale.shape != observed_summary.shape:
            raise ValueError("scale must have the same shape as the summaries")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("scale components must be positive and finite")
    return float(
        np.sqrt(
            np.sum(
                ((observed_summary - simulated_summary) / scale) ** 2
            )
        )
    )


def make_distance_simulator(observed_values, summary_scale=None):
    """Build the distance callback consumed by the generic ABC-SMC runner."""

    observed_values = np.asarray(observed_values, dtype=float)
    observed_summary = manual_summary(observed_values)
    if observed_summary is None:
        raise ValueError("observed_values do not have a valid Manual summary")
    length = len(observed_values)

    def distance_simulator(theta, rng):
        simulated_values = simulate_sv(theta, length, rng)
        simulated_summary = manual_summary(simulated_values)
        if simulated_summary is None:
            return None
        return normalized_summary_distance(
            simulated_summary,
            observed_summary,
            scale=summary_scale,
        )

    return distance_simulator

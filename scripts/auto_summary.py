"""Features and training code for the learned ABC summaries."""

import time

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

from sv_abc_core import (
    is_valid_return_series,
    robust_summary_scale,
    sample_prior,
    simulate_sv,
    squared_return_acf,
)


MARGINAL_FEATURES = (
    "variance",
    "standard_deviation",
    "mean_absolute_return",
    "kurtosis",
    "quantile_01",
    "quantile_05",
    "quantile_25",
    "quantile_50",
    "quantile_75",
    "quantile_95",
    "quantile_99",
)

TEMPORAL_FEATURES = (
    "squared_acf_1",
    "squared_acf_2",
    "squared_acf_3",
    "squared_acf_5",
    "squared_acf_10",
)

PARAMETER_NAMES = ("alpha", "beta", "sigma_eta")
ACF_LAGS = (1, 2, 3, 5, 10)


def feature_names(feature_group="full"):
    """Return the feature names used by one learned-summary model."""

    if feature_group == "marginal":
        return MARGINAL_FEATURES
    if feature_group == "temporal":
        return TEMPORAL_FEATURES
    if feature_group == "full":
        return MARGINAL_FEATURES + TEMPORAL_FEATURES
    raise ValueError("feature_group must be marginal, temporal, or full")


def extract_features(returns, feature_group="full"):
    """Calculate a small, readable feature vector from one return series."""

    feature_names(feature_group)
    returns = np.asarray(returns, dtype=float)
    if not is_valid_return_series(returns):
        return None

    quantiles = np.quantile(
        returns,
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
    )
    marginal = np.array(
        [
            np.var(returns),
            np.std(returns),
            np.mean(np.abs(returns)),
            kurtosis(returns, fisher=False),
            *quantiles,
        ],
        dtype=float,
    )

    temporal = squared_return_acf(returns, ACF_LAGS)

    if feature_group == "marginal":
        features = marginal
    elif feature_group == "temporal":
        features = temporal
    else:
        features = np.concatenate([marginal, temporal])

    if not np.all(np.isfinite(features)):
        return None
    return features


def scaled_prediction_distance(first, second, scale):
    """Compare parameter predictions after putting them on equal scales."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if first.shape != (3,) or second.shape != (3,) or scale.shape != (3,):
        raise ValueError("predictions and scale must have length 3")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("scale must contain positive finite values")
    return float(np.linalg.norm((first - second) / scale))


def train_auto_summary_model(
    n_simulations=5_000,
    series_length=4_000,
    feature_group="full",
    random_seed=42,
    n_estimators=300,
):
    """Simulate training data and fit one multi-output random forest."""

    if not isinstance(n_simulations, int) or n_simulations < 20:
        raise ValueError("n_simulations must be an integer of at least 20")
    if not isinstance(series_length, int) or series_length < 11:
        raise ValueError("series_length must be an integer of at least 11")
    if not isinstance(n_estimators, int) or n_estimators <= 0:
        raise ValueError("n_estimators must be a positive integer")
    names = feature_names(feature_group)

    start_time = time.perf_counter()
    rng = np.random.default_rng(random_seed)
    parameter_bank = sample_prior(rng, n_simulations)
    features = []
    parameters = []

    for theta in parameter_bank:
        returns = simulate_sv(theta, series_length, rng)
        row = extract_features(returns, feature_group)
        if row is None:
            continue
        features.append(row)
        parameters.append(theta)

    feature_matrix = np.asarray(features, dtype=float)
    parameter_matrix = np.asarray(parameters, dtype=float)
    if len(feature_matrix) < 10:
        raise RuntimeError("too few valid simulations to train the model")

    x_train, x_validation, y_train, y_validation = train_test_split(
        feature_matrix,
        parameter_matrix,
        test_size=0.20,
        random_state=random_seed,
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_seed,
        n_jobs=-1,
    )
    training_start = time.perf_counter()
    model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - training_start

    validation_predictions = model.predict(x_validation)
    prediction_scale = robust_summary_scale(validation_predictions)
    rows = []
    for index, parameter in enumerate(PARAMETER_NAMES):
        rows.append(
            {
                "parameter": parameter,
                "r2": r2_score(
                    y_validation[:, index],
                    validation_predictions[:, index],
                ),
                "mae": mean_absolute_error(
                    y_validation[:, index],
                    validation_predictions[:, index],
                ),
                "rmse": root_mean_squared_error(
                    y_validation[:, index],
                    validation_predictions[:, index],
                ),
            }
        )

    bundle = {
        "model": model,
        "feature_group": feature_group,
        "feature_names": list(names),
        "prediction_scale": prediction_scale,
        "prediction_scale_source": "held-out validation predictions",
        "series_length": series_length,
        "training_simulations": n_simulations,
        "random_seed": random_seed,
    }
    details = {
        "valid_simulations": int(len(feature_matrix)),
        "invalid_simulations": int(n_simulations - len(feature_matrix)),
        "training_rows": int(len(x_train)),
        "validation_rows": int(len(x_validation)),
        "overall_validation_r2": float(
            r2_score(y_validation, validation_predictions)
        ),
        "training_seconds": float(training_seconds),
        "total_seconds": float(time.perf_counter() - start_time),
    }
    return bundle, pd.DataFrame(rows), details

"""Shared posterior predictive checks for the thesis SV experiments."""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from sv_abc_core import manual_summary, simulate_sv, squared_return_acf


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = PROJECT_DIR / "figures"
DATA_PATH = PROJECT_DIR / "data" / "processed" / "sp500_returns.csv"

DEFAULT_REPLICATIONS = 1_000
DEFAULT_SEED = 42
TEMPORAL_LAGS = tuple(range(1, 21))

METHOD_NAMES = (
    "Manual ABC",
    "Auto ABC",
    "Wasserstein ABC",
    "ABC-SMC",
)

STATISTIC_NAMES = (
    "Variance",
    "Kurtosis",
    "ACF1",
    "ACF5",
)

PREDICTIVE_FIGURE_NAMES = {
    "Variance": "ppc_variance.png",
    "Kurtosis": "ppc_kurtosis.png",
    "ACF1": "ppc_acf1.png",
    "ACF5": "ppc_acf5.png",
}

PREDICTIVE_DRAW_KEYS = {
    "Manual ABC": "manual",
    "Auto ABC": "auto",
    "Wasserstein ABC": "wasserstein",
    "ABC-SMC": "abc_smc",
}

TEMPORAL_FIGURE_NAMES = {
    "Manual ABC": "ppc_temporal_manual.png",
    "Auto ABC": "ppc_temporal_auto.png",
    "Wasserstein ABC": "ppc_temporal_wasserstein.png",
    "ABC-SMC": "ppc_temporal_abc_smc.png",
}


@dataclass(frozen=True)
class PosteriorSource:
    """Posterior particles and their optional sampling probabilities."""

    samples: np.ndarray
    probabilities: np.ndarray | None
    sampling_rule: str


def _validate_posterior_samples(samples, label):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] != 3:
        raise ValueError(f"{label} posterior must have shape (n, 3)")
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{label} posterior must contain finite values")
    return samples


def normalize_weights(weights):
    """Validate non-negative importance weights and normalize them."""

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a non-empty vector")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain finite values")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must have a positive finite sum")
    return weights / total


def load_posterior_sources(scripts_dir=SCRIPT_DIR):
    """Load the three rejection posteriors and weighted ABC-SMC particles."""

    scripts_dir = Path(scripts_dir).resolve()
    rejection_files = (
        ("Manual ABC", "abc_manual_sv.npy"),
        ("Auto ABC", "abc_auto_sv.npy"),
        ("Wasserstein ABC", "abc_wasserstein_sv.npy"),
    )

    sources = {}
    for method_name, filename in rejection_files:
        path = scripts_dir / filename
        stored_samples = np.load(path, allow_pickle=False)
        if stored_samples.ndim == 2 and stored_samples.shape[1] == 4:
            stored_samples = stored_samples[:, :3]
        samples = _validate_posterior_samples(
            stored_samples,
            method_name,
        )
        sources[method_name] = PosteriorSource(
            samples=samples,
            probabilities=None,
            sampling_rule="uniform",
        )

    smc_path = scripts_dir / "abc_smc_sv_final.csv"
    smc_frame = pd.read_csv(smc_path)
    required_columns = ["alpha", "beta", "sigma_eta", "weight"]
    missing_columns = [
        column for column in required_columns if column not in smc_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "ABC-SMC posterior is missing columns: "
            + ", ".join(missing_columns)
        )
    smc_samples = _validate_posterior_samples(
        smc_frame[["alpha", "beta", "sigma_eta"]].to_numpy(),
        "ABC-SMC",
    )
    smc_probabilities = normalize_weights(smc_frame["weight"].to_numpy())
    if len(smc_probabilities) != len(smc_samples):
        raise ValueError("ABC-SMC weights must match the posterior sample count")
    sources["ABC-SMC"] = PosteriorSource(
        samples=smc_samples,
        probabilities=smc_probabilities,
        sampling_rule="importance-weighted",
    )

    return sources


def sample_parameter_vectors(source, rng, n_replications):
    """Sample posterior parameter vectors using a source's sampling rule."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not isinstance(n_replications, int) or n_replications <= 0:
        raise ValueError("n_replications must be a positive integer")

    samples = _validate_posterior_samples(source.samples, "Input")
    probabilities = source.probabilities
    if probabilities is not None:
        probabilities = normalize_weights(probabilities)
        if len(probabilities) != len(samples):
            raise ValueError("weights must match the posterior sample count")

    indices = rng.choice(
        len(samples),
        size=n_replications,
        replace=True,
        p=probabilities,
    )
    return samples[indices]


def _validate_observed_summary(observed_summary):
    observed_summary = np.asarray(observed_summary, dtype=float)
    if observed_summary.shape != (len(STATISTIC_NAMES),):
        raise ValueError("observed_summary must contain four statistics")
    if not np.all(np.isfinite(observed_summary)):
        raise ValueError("observed_summary must contain finite values")
    return observed_summary


def _validate_predictive_draws(predictive_draws):
    if tuple(predictive_draws) != METHOD_NAMES:
        raise ValueError("predictive_draws must contain all four methods in order")

    validated = {}
    replication_count = None
    for method_name in METHOD_NAMES:
        values = np.asarray(predictive_draws[method_name], dtype=float)
        if (
            values.ndim != 2
            or values.shape[0] == 0
            or values.shape[1] != len(STATISTIC_NAMES)
        ):
            raise ValueError(
                f"{method_name} predictive draws must have shape (n, 4)"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{method_name} predictive draws must contain finite values"
            )
        if replication_count is None:
            replication_count = len(values)
        elif len(values) != replication_count:
            raise ValueError("all methods must use the same replication count")
        validated[method_name] = values

    return validated


def compute_predictive_draws(
    observed_returns,
    sources,
    n_replications=DEFAULT_REPLICATIONS,
    seed=DEFAULT_SEED,
):
    """Simulate replicated summaries from each method's posterior."""

    observed_returns = np.asarray(observed_returns, dtype=float)
    if observed_returns.ndim != 1 or len(observed_returns) < 6:
        raise ValueError("observed_returns must be a vector of at least length 6")
    if not np.all(np.isfinite(observed_returns)):
        raise ValueError("observed_returns must contain finite values")
    if not isinstance(n_replications, int) or n_replications <= 0:
        raise ValueError("n_replications must be a positive integer")
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if tuple(sources) != METHOD_NAMES:
        raise ValueError("sources must contain all four methods in order")

    observed_summary = manual_summary(
        observed_returns,
        minimum_variance=0.0,
    )
    if observed_summary is None:
        raise ValueError("observed_returns do not produce valid PPC statistics")

    child_sequences = np.random.SeedSequence(int(seed)).spawn(
        len(METHOD_NAMES)
    )
    predictive_draws = {}

    for method_name, child_sequence in zip(METHOD_NAMES, child_sequences):
        rng = np.random.default_rng(child_sequence)
        parameter_vectors = sample_parameter_vectors(
            sources[method_name],
            rng,
            n_replications,
        )
        method_draws = np.empty(
            (n_replications, len(STATISTIC_NAMES)),
            dtype=float,
        )
        for index, theta in enumerate(parameter_vectors):
            simulated_returns = simulate_sv(theta, len(observed_returns), rng)
            simulated_summary = manual_summary(
                simulated_returns,
                minimum_variance=0.0,
            )
            if simulated_summary is None:
                raise RuntimeError(
                    f"{method_name} produced an invalid summary at "
                    f"replication {index}"
                )
            method_draws[index] = simulated_summary
        predictive_draws[method_name] = method_draws

    return observed_summary, predictive_draws


def compute_temporal_acf(returns, lags=TEMPORAL_LAGS):
    """Calculate squared-return ACF values used to assess clustering."""

    returns = np.asarray(returns, dtype=float)
    lags = tuple(lags)
    if returns.ndim != 1 or not np.all(np.isfinite(returns)):
        raise ValueError("returns must be a finite one-dimensional array")
    if len(returns) <= max(lags, default=0):
        raise ValueError("returns must be longer than the largest lag")
    return squared_return_acf(returns, lags)


def _validate_temporal_draws(observed_acf, predictive_acf):
    observed_acf = np.asarray(observed_acf, dtype=float)
    if observed_acf.shape != (len(TEMPORAL_LAGS),):
        raise ValueError("observed_acf must contain lags 1 to 20")
    if not np.all(np.isfinite(observed_acf)):
        raise ValueError("observed_acf must contain finite values")
    if tuple(predictive_acf) != METHOD_NAMES:
        raise ValueError("predictive_acf must contain all four methods in order")

    validated = {}
    replication_count = None
    for method_name in METHOD_NAMES:
        values = np.asarray(predictive_acf[method_name], dtype=float)
        if values.ndim != 2 or values.shape[1] != len(TEMPORAL_LAGS):
            raise ValueError(
                f"{method_name} temporal draws must have shape (n, 20)"
            )
        if len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError(
                f"{method_name} temporal draws must be non-empty and finite"
            )
        if replication_count is None:
            replication_count = len(values)
        elif len(values) != replication_count:
            raise ValueError("all methods must use the same replication count")
        validated[method_name] = values
    return observed_acf, validated


def compute_complete_predictive_draws(
    observed_returns,
    sources,
    n_replications=DEFAULT_REPLICATIONS,
    seed=DEFAULT_SEED,
):
    """Simulate the original summaries and ACF lags 1 to 20 once."""

    observed_returns = np.asarray(observed_returns, dtype=float)
    if observed_returns.ndim != 1 or len(observed_returns) <= max(TEMPORAL_LAGS):
        raise ValueError("observed_returns must be longer than 20 observations")
    if not np.all(np.isfinite(observed_returns)):
        raise ValueError("observed_returns must contain finite values")
    if not isinstance(n_replications, int) or n_replications <= 0:
        raise ValueError("n_replications must be a positive integer")
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if tuple(sources) != METHOD_NAMES:
        raise ValueError("sources must contain all four methods in order")

    observed_summary = manual_summary(observed_returns, minimum_variance=0.0)
    if observed_summary is None:
        raise ValueError("observed_returns do not produce valid PPC statistics")
    observed_acf = compute_temporal_acf(observed_returns)

    child_sequences = np.random.SeedSequence(int(seed)).spawn(
        len(METHOD_NAMES)
    )
    predictive_draws = {}
    predictive_acf = {}
    for method_name, child_sequence in zip(METHOD_NAMES, child_sequences):
        rng = np.random.default_rng(child_sequence)
        parameter_vectors = sample_parameter_vectors(
            sources[method_name], rng, n_replications
        )
        method_draws = np.empty((n_replications, len(STATISTIC_NAMES)))
        method_acf = np.empty((n_replications, len(TEMPORAL_LAGS)))
        for index, theta in enumerate(parameter_vectors):
            simulated_returns = simulate_sv(theta, len(observed_returns), rng)
            simulated_summary = manual_summary(
                simulated_returns, minimum_variance=0.0
            )
            if simulated_summary is None:
                raise RuntimeError(
                    f"{method_name} produced an invalid summary at "
                    f"replication {index}"
                )
            method_draws[index] = simulated_summary
            method_acf[index] = compute_temporal_acf(simulated_returns)
        predictive_draws[method_name] = method_draws
        predictive_acf[method_name] = method_acf

    return observed_summary, observed_acf, predictive_draws, predictive_acf


def build_temporal_acf_summary(observed_acf, predictive_acf):
    """Summarize predictive squared-return ACF values at lags 1 to 20."""

    observed_acf, predictive_acf = _validate_temporal_draws(
        observed_acf, predictive_acf
    )
    rows = []
    for method_name in METHOD_NAMES:
        values = predictive_acf[method_name]
        for index, lag in enumerate(TEMPORAL_LAGS):
            lag_values = values[:, index]
            rows.append(
                {
                    "Method": method_name,
                    "Lag": lag,
                    "Observed": observed_acf[index],
                    "Predictive mean": float(np.mean(lag_values)),
                    "Predictive SD": float(np.std(lag_values, ddof=0)),
                    "Predictive median": float(np.median(lag_values)),
                    "Predictive q2.5": float(np.quantile(lag_values, 0.025)),
                    "Predictive q97.5": float(np.quantile(lag_values, 0.975)),
                    "Posterior Predictive p-value": float(
                        np.mean(lag_values >= observed_acf[index])
                    ),
                    "Replications": len(lag_values),
                }
            )
    return pd.DataFrame(rows)


def build_temporal_rmse_table(observed_acf, predictive_acf):
    """Measure the gap between observed and predictive ACF curves."""

    observed_acf, predictive_acf = _validate_temporal_draws(
        observed_acf, predictive_acf
    )
    rows = []
    for method_name in METHOD_NAMES:
        values = predictive_acf[method_name]
        draw_rmse = np.sqrt(np.mean((values - observed_acf) ** 2, axis=1))
        median_curve = np.median(values, axis=0)
        rows.append(
            {
                "Method": method_name,
                "Median-curve ACF RMSE": float(
                    np.sqrt(np.mean((median_curve - observed_acf) ** 2))
                ),
                "Mean draw ACF RMSE": float(np.mean(draw_rmse)),
                "Draw ACF RMSE SD": float(np.std(draw_rmse, ddof=0)),
                "Draw ACF RMSE q2.5": float(np.quantile(draw_rmse, 0.025)),
                "Draw ACF RMSE q97.5": float(np.quantile(draw_rmse, 0.975)),
                "Replications": len(values),
            }
        )
    return pd.DataFrame(rows)


def plot_temporal_acf(observed_acf, predictive_acf, output_paths):
    """Save one readable temporal PPC figure for each ABC method."""

    observed_acf, predictive_acf = _validate_temporal_draws(
        observed_acf, predictive_acf
    )
    output_paths = {
        method_name: Path(path).resolve()
        for method_name, path in output_paths.items()
    }
    if tuple(output_paths) != METHOD_NAMES:
        raise ValueError("output_paths must contain all four methods in order")

    colours = ["#4C78A8", "#F58518", "#54A24B", "#7A5195"]
    for method_name, colour in zip(METHOD_NAMES, colours):
        output_paths[method_name].parent.mkdir(parents=True, exist_ok=True)
        values = predictive_acf[method_name]
        median = np.median(values, axis=0)
        lower = np.quantile(values, 0.025, axis=0)
        upper = np.quantile(values, 0.975, axis=0)

        figure, axis = plt.subplots(figsize=(8.8, 5.6))
        axis.fill_between(
            TEMPORAL_LAGS, lower, upper, color=colour, alpha=0.22,
            label="95% predictive interval"
        )
        axis.plot(
            TEMPORAL_LAGS, median, color=colour, marker="o", linewidth=2,
            markersize=4, label="Predictive median"
        )
        axis.plot(
            TEMPORAL_LAGS, observed_acf, color="#222222", marker="s",
            linewidth=2, markersize=4, label="Observed S&P 500"
        )
        axis.axhline(0.0, color="#777777", linewidth=1)
        axis.set_title(f"Temporal posterior predictive check: {method_name}")
        axis.set_xlabel("Lag")
        axis.set_ylabel("Squared-return autocorrelation")
        axis.set_xticks(range(1, 21))
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(
            output_paths[method_name], dpi=220, bbox_inches="tight",
            pad_inches=0.08
        )
        plt.close(figure)
    return output_paths


def build_summary_table(observed_summary, predictive_draws):
    """Summarize each method's posterior predictive distribution."""

    observed_summary = _validate_observed_summary(observed_summary)
    predictive_draws = _validate_predictive_draws(predictive_draws)
    rows = []

    for method_name in METHOD_NAMES:
        values = predictive_draws[method_name]
        for statistic_index, statistic_name in enumerate(STATISTIC_NAMES):
            statistic_values = values[:, statistic_index]
            rows.append(
                {
                    "Method": method_name,
                    "Statistic": statistic_name,
                    "Observed": observed_summary[statistic_index],
                    "Predictive mean": float(np.mean(statistic_values)),
                    "Predictive SD": float(np.std(statistic_values, ddof=0)),
                    "Predictive median": float(np.median(statistic_values)),
                    "Predictive q2.5": float(
                        np.quantile(statistic_values, 0.025)
                    ),
                    "Predictive q97.5": float(
                        np.quantile(statistic_values, 0.975)
                    ),
                    "Replications": len(statistic_values),
                }
            )

    return pd.DataFrame(rows)


def build_pvalue_table(observed_summary, predictive_draws):
    """Calculate one-sided posterior predictive p-values."""

    observed_summary = _validate_observed_summary(observed_summary)
    predictive_draws = _validate_predictive_draws(predictive_draws)
    rows = []

    for method_name in METHOD_NAMES:
        values = predictive_draws[method_name]
        for statistic_index, statistic_name in enumerate(STATISTIC_NAMES):
            pvalue = float(
                np.mean(
                    values[:, statistic_index]
                    >= observed_summary[statistic_index]
                )
            )
            rows.append(
                {
                    "Method": method_name,
                    "Statistic": statistic_name,
                    "Posterior Predictive p-value": pvalue,
                }
            )

    return pd.DataFrame(rows)


def build_wide_check_table(observed_summary, predictive_draws):
    """Build the backward-compatible table of predictive means."""

    observed_summary = _validate_observed_summary(observed_summary)
    predictive_draws = _validate_predictive_draws(predictive_draws)
    column_names = {
        "Manual ABC": "Manual",
        "Auto ABC": "Auto",
        "Wasserstein ABC": "Wasserstein",
        "ABC-SMC": "ABC-SMC",
    }
    data = {
        "Statistic": list(STATISTIC_NAMES),
        "Real": observed_summary,
    }
    for method_name in METHOD_NAMES:
        data[column_names[method_name]] = np.mean(
            predictive_draws[method_name],
            axis=0,
        )
    return pd.DataFrame(data)


def prepare_plot_statistic(
    statistic_name,
    method_values,
    observed_value,
    variance_log_floor=-12.0,
):
    """Prepare readable plot values without changing saved PPC statistics."""

    values = [np.asarray(item, dtype=float) for item in method_values]
    if any(item.ndim != 1 or not np.all(np.isfinite(item)) for item in values):
        raise ValueError("plot values must be finite one-dimensional arrays")
    if not np.isfinite(observed_value):
        raise ValueError("observed plot value must be finite")

    if statistic_name == "Variance":
        if observed_value <= 0.0 or any(np.any(item <= 0.0) for item in values):
            raise ValueError("variance plot values must be positive")
        transformed = [
            np.maximum(np.log10(item), variance_log_floor)
            for item in values
        ]
        transformed_observed = max(
            float(np.log10(observed_value)),
            variance_log_floor,
        )
        axis_label = (
            "log10 posterior predictive variance "
            f"(values below {variance_log_floor:g} clipped)"
        )
        title = "Variance (log10)"
        return transformed, transformed_observed, axis_label, title

    return (
        values,
        float(observed_value),
        "Posterior predictive statistic",
        statistic_name,
    )


def plot_predictive_distributions(
    observed_summary,
    predictive_draws,
    output_paths,
):
    """Plot each posterior predictive statistic in a separate figure."""

    observed_summary = _validate_observed_summary(observed_summary)
    predictive_draws = _validate_predictive_draws(predictive_draws)
    output_paths = {
        statistic_name: Path(path).resolve()
        for statistic_name, path in output_paths.items()
    }
    if set(output_paths) != set(STATISTIC_NAMES):
        raise ValueError("output_paths must contain one path per statistic")
    for output_path in output_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    colours = ["#4C78A8", "#F58518", "#54A24B", "#7A5195"]
    short_names = ["Manual", "Auto", "Wasserstein", "ABC-SMC"]

    for statistic_index, statistic_name in enumerate(STATISTIC_NAMES):
        figure, axis = plt.subplots(figsize=(8.8, 5.6))
        values = [
            predictive_draws[method_name][:, statistic_index]
            for method_name in METHOD_NAMES
        ]
        values, plot_observed, axis_label, plot_title = (
            prepare_plot_statistic(
                statistic_name,
                values,
                observed_summary[statistic_index],
            )
        )
        boxplot = axis.boxplot(
            values,
            tick_labels=short_names,
            patch_artist=True,
            showfliers=False,
            widths=0.65,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, colour in zip(boxplot["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.78)
        axis.axhline(
            plot_observed,
            color="#222222",
            linestyle="--",
            linewidth=1.8,
            label="Observed",
        )
        axis.set_title(
            f"Posterior predictive distribution: {plot_title}",
            fontsize=14,
            fontweight="bold",
        )
        axis.set_ylabel(axis_label, fontsize=12)
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(axis="x", labelrotation=0, labelsize=11)
        axis.tick_params(axis="y", labelsize=11)
        axis.legend(loc="best", frameon=False, fontsize=11)
        figure.tight_layout()
        figure.savefig(
            output_paths[statistic_name],
            dpi=220,
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close(figure)

    return output_paths


def _load_observed_returns(data_path):
    data_path = Path(data_path).resolve()
    frame = pd.read_csv(data_path)
    if "Return" not in frame.columns:
        raise ValueError("observed data must contain a Return column")
    returns = frame["Return"].to_numpy(dtype=float)
    if returns.ndim != 1 or len(returns) < 6:
        raise ValueError("observed Return column must contain at least 6 values")
    if not np.all(np.isfinite(returns)):
        raise ValueError("observed Return column must contain finite values")
    return returns


def _input_record(path):
    path = Path(path).resolve()
    modified = datetime.fromtimestamp(
        path.stat().st_mtime,
        timezone.utc,
    ).isoformat()
    return {
        "path": str(path),
        "last_modified_utc": modified,
    }


def run_posterior_predictive_analysis(
    n_replications=DEFAULT_REPLICATIONS,
    seed=DEFAULT_SEED,
    scripts_dir=SCRIPT_DIR,
    figures_dir=FIGURES_DIR,
    data_path=DATA_PATH,
):
    """Run and save the complete four-method posterior predictive analysis."""

    if not isinstance(n_replications, int) or n_replications <= 0:
        raise ValueError("n_replications must be a positive integer")
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")

    start_time = time.perf_counter()
    scripts_dir = Path(scripts_dir).resolve()
    figures_dir = Path(figures_dir).resolve()
    data_path = Path(data_path).resolve()

    sources = load_posterior_sources(scripts_dir)
    observed_returns = _load_observed_returns(data_path)
    (
        observed_summary,
        observed_acf,
        predictive_draws,
        predictive_acf,
    ) = compute_complete_predictive_draws(
        observed_returns,
        sources,
        n_replications=n_replications,
        seed=seed,
    )
    wide_table = build_wide_check_table(
        observed_summary,
        predictive_draws,
    )
    summary_table = build_summary_table(
        observed_summary,
        predictive_draws,
    )
    pvalue_table = build_pvalue_table(
        observed_summary,
        predictive_draws,
    )
    temporal_acf_table = build_temporal_acf_summary(
        observed_acf, predictive_acf
    )
    temporal_rmse_table = build_temporal_rmse_table(
        observed_acf, predictive_acf
    )

    paths = {
        "wide_table": (
            scripts_dir / "posterior_predictive_check.csv"
        ).resolve(),
        "summary_table": (
            scripts_dir / "posterior_predictive_summary.csv"
        ).resolve(),
        "pvalue_table": (
            scripts_dir / "posterior_predictive_pvalues.csv"
        ).resolve(),
        "metadata": (
            scripts_dir / "posterior_predictive_metadata.json"
        ).resolve(),
        "draws": (
            scripts_dir / "posterior_predictive_draws.npz"
        ).resolve(),
        "temporal_acf_table": (
            scripts_dir / "posterior_predictive_acf.csv"
        ).resolve(),
        "temporal_rmse_table": (
            scripts_dir / "posterior_predictive_acf_rmse.csv"
        ).resolve(),
        "figures": {
            statistic_name: (
                figures_dir / PREDICTIVE_FIGURE_NAMES[statistic_name]
            ).resolve()
            for statistic_name in STATISTIC_NAMES
        },
        "temporal_figures": {
            method_name: (
                figures_dir / TEMPORAL_FIGURE_NAMES[method_name]
            ).resolve()
            for method_name in METHOD_NAMES
        },
    }

    scripts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_predictive_distributions(
        observed_summary,
        predictive_draws,
        paths["figures"],
    )
    plot_temporal_acf(
        observed_acf,
        predictive_acf,
        paths["temporal_figures"],
    )
    wide_table.to_csv(paths["wide_table"], index=False)
    summary_table.to_csv(paths["summary_table"], index=False)
    pvalue_table.to_csv(paths["pvalue_table"], index=False)
    temporal_acf_table.to_csv(paths["temporal_acf_table"], index=False)
    temporal_rmse_table.to_csv(paths["temporal_rmse_table"], index=False)
    np.savez_compressed(
        paths["draws"],
        observed_summary=observed_summary,
        observed_acf=observed_acf,
        **{
            PREDICTIVE_DRAW_KEYS[method_name]: values
            for method_name, values in predictive_draws.items()
        },
        **{
            f"{PREDICTIVE_DRAW_KEYS[method_name]}_acf": values
            for method_name, values in predictive_acf.items()
        },
    )

    input_paths = {
        "observed_returns": data_path,
        "Manual ABC": scripts_dir / "abc_manual_sv.npy",
        "Auto ABC": scripts_dir / "abc_auto_sv.npy",
        "Wasserstein ABC": scripts_dir / "abc_wasserstein_sv.npy",
        "ABC-SMC": scripts_dir / "abc_smc_sv_final.csv",
    }
    runtime_seconds = time.perf_counter() - start_time
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "replications_per_method": n_replications,
        "runtime_seconds": runtime_seconds,
        "statistics": list(STATISTIC_NAMES),
        "temporal_lags": list(TEMPORAL_LAGS),
        "methods": list(METHOD_NAMES),
        "sampling_rules": {
            method_name: sources[method_name].sampling_rule
            for method_name in METHOD_NAMES
        },
        "posterior_predictive_pvalue_definition": (
            "Pr(T(y_rep) >= T(y_obs) | y_obs)"
        ),
        "acf_rmse_definition": (
            "Root mean squared gap across squared-return ACF lags 1 to 20"
        ),
        "inputs": {
            name: _input_record(path)
            for name, path in input_paths.items()
        },
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "observed_summary": observed_summary,
        "predictive_draws": predictive_draws,
        "wide_table": wide_table,
        "summary_table": summary_table,
        "pvalue_table": pvalue_table,
        "temporal_acf_table": temporal_acf_table,
        "temporal_rmse_table": temporal_rmse_table,
        "observed_acf": observed_acf,
        "predictive_acf": predictive_acf,
        "metadata": metadata,
        "paths": paths,
        "runtime_seconds": runtime_seconds,
    }

"""Repeated synthetic experiments for stochastic-volatility ABC methods."""

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abc_experiment_utils import select_top_fraction
from abc_smc_comparison import weighted_posterior_summary
from abc_smc_utils import ABCSMCConfig, run_abc_smc
from auto_summary import extract_features
from financial_derived_quantities import calculate_financial_quantities
from sv_abc_core import (
    PRIOR_BOUNDS,
    estimate_prior_predictive_summary_scale,
    make_distance_simulator,
    manual_summary,
    sample_prior,
    simulate_sv,
    uniform_prior_log_density,
)


METHOD_ORDER = (
    "Manual ABC",
    "Auto ABC",
    "Wasserstein ABC",
    "ABC-SMC",
)


PARAMETER_NAMES = ("alpha", "beta", "sigma_eta")
DERIVED_NAMES = (
    "long_run_log_variance",
    "half_life_days",
    "stationary_log_volatility_variance",
)
TARGET_NAMES = PARAMETER_NAMES + DERIVED_NAMES

TARGET_LABELS = {
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "sigma_eta": r"$\sigma_\eta$",
    "long_run_log_variance": "Long-run log variance",
    "half_life_days": "Volatility shock half-life",
    "stationary_log_volatility_variance": (
        "Stationary log-volatility variance"
    ),
}


def build_regime_table(
    long_run_log_variance=-10.0,
    beta_values=(0.90, 0.95, 0.98),
    sigma_eta_values=(0.15, 0.30),
    data_seeds=(202601, 202602, 202603, 202604, 202605),
):
    """Create the registered six-regime, five-seed experiment table."""

    rows = []
    for beta in beta_values:
        for sigma_eta in sigma_eta_values:
            alpha = long_run_log_variance * (1.0 - beta)
            for replicate, data_seed in enumerate(data_seeds, start=1):
                rows.append(
                    {
                        "dataset_id": (
                            f"beta_{beta:.2f}_sigma_{sigma_eta:.2f}_"
                            f"seed_{data_seed}"
                        ),
                        "replicate": replicate,
                        "data_seed": int(data_seed),
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "sigma_eta": float(sigma_eta),
                        "beta_regime": float(beta),
                        "sigma_eta_regime": float(sigma_eta),
                        "long_run_log_variance": float(
                            long_run_log_variance
                        ),
                    }
                )
    return pd.DataFrame(rows)


def select_shared_bank_posteriors(
    parameters,
    distances_by_method,
    acceptance_fraction,
):
    """Select each rejection posterior from one parameter bank."""

    parameters = np.asarray(parameters, dtype=float)
    posteriors = {}
    for method, distances in distances_by_method.items():
        accepted, epsilon = select_top_fraction(
            parameters,
            distances,
            acceptance_fraction,
        )
        samples = accepted[:, :3]
        posteriors[method] = {
            "samples": samples,
            "weights": np.full(len(samples), 1.0 / len(samples)),
            "epsilon": epsilon,
        }
    return posteriors


def build_shared_rejection_bank(
    model_bundle,
    series_length,
    n_simulations,
    random_seed,
):
    """Simulate one prior bank used by all rejection ABC comparisons."""

    if not isinstance(series_length, int) or series_length < 11:
        raise ValueError("series_length must be an integer of at least 11")
    if not isinstance(n_simulations, int) or n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer")

    rng = np.random.default_rng(random_seed)
    parameters = sample_prior(rng, n_simulations)
    manual_summaries = np.full((n_simulations, 4), np.nan)
    auto_predictions = np.full((n_simulations, 3), np.nan)
    sorted_returns = np.empty(
        (n_simulations, series_length),
        dtype=np.float32,
    )
    valid_indices = []
    feature_rows = []

    for index, theta in enumerate(parameters):
        simulated_returns = simulate_sv(theta, series_length, rng)
        with np.errstate(over="ignore", invalid="ignore"):
            summary = manual_summary(simulated_returns)
            features = extract_features(
                simulated_returns,
                model_bundle["feature_group"],
            )
        if summary is None or features is None:
            continue

        valid_position = len(valid_indices)
        valid_indices.append(index)
        feature_rows.append(features)
        manual_summaries[index] = summary
        sorted_returns[valid_position] = np.sort(simulated_returns)

    if not valid_indices:
        raise RuntimeError("the shared rejection bank has no valid simulations")

    valid_indices = np.asarray(valid_indices, dtype=int)
    predictions = model_bundle["model"].predict(
        np.asarray(feature_rows, dtype=float)
    )
    auto_predictions[valid_indices] = predictions

    return {
        "parameters": parameters,
        "valid_indices": valid_indices,
        "manual_summaries": manual_summaries,
        "auto_predictions": auto_predictions,
        "sorted_returns": sorted_returns[: len(valid_indices)].copy(),
    }


def rejection_distances_from_bank(
    observed_returns,
    bank,
    model_bundle,
    summary_scale,
    wasserstein_chunk_size=250,
):
    """Calculate three distances without resimulating the candidate bank."""

    observed_returns = np.asarray(observed_returns, dtype=float)
    observed_summary = manual_summary(observed_returns)
    observed_features = extract_features(
        observed_returns,
        model_bundle["feature_group"],
    )
    if observed_summary is None or observed_features is None:
        raise ValueError("observed_returns do not produce valid comparisons")

    observed_prediction = model_bundle["model"].predict(
        observed_features.reshape(1, -1)
    )[0]
    prediction_scale = np.asarray(
        model_bundle["prediction_scale"],
        dtype=float,
    )
    summary_scale = np.asarray(summary_scale, dtype=float)
    valid_indices = bank["valid_indices"]
    maximum_distance = np.finfo(float).max
    n_simulations = len(bank["parameters"])

    manual_distances = np.full(n_simulations, maximum_distance)
    manual_differences = (
        bank["manual_summaries"][valid_indices] - observed_summary
    ) / summary_scale
    manual_distances[valid_indices] = np.linalg.norm(
        manual_differences,
        axis=1,
    )

    auto_distances = np.full(n_simulations, maximum_distance)
    auto_differences = (
        bank["auto_predictions"][valid_indices] - observed_prediction
    ) / prediction_scale
    auto_distances[valid_indices] = np.linalg.norm(
        auto_differences,
        axis=1,
    )

    wasserstein_distances = np.full(n_simulations, maximum_distance)
    observed_sorted = np.sort(observed_returns)
    valid_distances = np.empty(len(valid_indices), dtype=float)
    for start in range(0, len(valid_indices), wasserstein_chunk_size):
        stop = min(start + wasserstein_chunk_size, len(valid_indices))
        simulated_sorted = bank["sorted_returns"][start:stop]
        valid_distances[start:stop] = np.mean(
            np.abs(simulated_sorted - observed_sorted),
            axis=1,
        )
    wasserstein_distances[valid_indices] = valid_distances

    return {
        "Manual ABC": manual_distances,
        "Auto ABC": auto_distances,
        "Wasserstein ABC": wasserstein_distances,
    }


def _run_one_smc(observed_returns, summary_scale, config):
    start_time = time.perf_counter()
    result = run_abc_smc(
        config=config,
        bounds=PRIOR_BOUNDS,
        prior_sampler=sample_prior,
        prior_log_density=uniform_prior_log_density,
        distance_simulator=make_distance_simulator(
            observed_returns,
            summary_scale=summary_scale,
        ),
    )
    return {
        "samples": result.particles[-1],
        "weights": result.weights[-1],
        "simulator_calls": result.total_simulator_calls,
        "runtime_seconds": time.perf_counter() - start_time,
        "completed_populations": result.completed_populations,
        "final_epsilon": float(result.epsilons[-1]),
        "final_ess": float(result.effective_sample_sizes[-1]),
        "stop_reason": result.stop_reason,
    }


def _run_smc_jobs(observed_datasets, summary_scale, config, workers):
    if workers == 1:
        return [
            _run_one_smc(observed, summary_scale, config)
            for observed in observed_datasets
        ]

    arguments = [
        (observed, summary_scale, config) for observed in observed_datasets
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_run_one_smc, *arguments_row)
            for arguments_row in arguments
        ]
        results = []
        for index, future in enumerate(futures, start=1):
            result = future.result()
            print(
                f"Completed ABC-SMC dataset {index}/{len(futures)} "
                f"with {result['simulator_calls']} calls",
                flush=True,
            )
            results.append(result)
    return results


def posterior_recovery_rows(
    samples,
    weights,
    true_parameters,
    dataset_information,
    method,
    simulator_calls,
    runtime_seconds=np.nan,
):
    """Create parameter and financial recovery rows for one posterior."""

    samples = np.asarray(samples, dtype=float)
    true_parameters = np.asarray(true_parameters, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) == 0:
        raise ValueError("samples must have shape (n, 3)")
    if true_parameters.shape != (3,):
        raise ValueError("true_parameters must contain three values")

    quantity_samples = calculate_financial_quantities(samples)
    true_quantities = calculate_financial_quantities(
        true_parameters.reshape(1, 3)
    )
    targets = {
        **{
            name: samples[:, index]
            for index, name in enumerate(PARAMETER_NAMES)
        },
        **quantity_samples,
    }
    true_values = {
        **{
            name: float(true_parameters[index])
            for index, name in enumerate(PARAMETER_NAMES)
        },
        **{
            name: float(values[0])
            for name, values in true_quantities.items()
        },
    }

    rows = []
    for target in TARGET_NAMES:
        summary = weighted_posterior_summary(targets[target], weights)
        true_value = true_values[target]
        bias = summary["mean"] - true_value
        absolute_error = abs(bias)
        if target in PARAMETER_NAMES:
            parameter_index = PARAMETER_NAMES.index(target)
            scale = (
                PRIOR_BOUNDS[parameter_index, 1]
                - PRIOR_BOUNDS[parameter_index, 0]
            )
            target_type = "parameter"
        else:
            scale = abs(true_value)
            target_type = "derived"

        rows.append(
            {
                **dataset_information,
                "method": method,
                "target": target,
                "target_type": target_type,
                **summary,
                "true_value": true_value,
                "bias": bias,
                "absolute_error": absolute_error,
                "scaled_absolute_error": absolute_error / scale,
                "covered": (
                    summary["q2.5"] <= true_value <= summary["q97.5"]
                ),
                "interval_width": summary["q97.5"] - summary["q2.5"],
                "simulator_calls": int(simulator_calls),
                "runtime_seconds": float(runtime_seconds),
            }
        )
    return rows


def run_repeated_regime_study(
    regimes,
    model_bundle,
    series_length=4_000,
    rejection_simulations=10_000,
    acceptance_fraction=0.05,
    rejection_seed=42,
    scale_simulations=2_000,
    scale_seed=202607,
    smc_config=ABCSMCConfig(),
    smc_workers=1,
):
    """Run all four ABC methods over the registered synthetic regimes."""

    regimes = pd.DataFrame(regimes).reset_index(drop=True)
    accepted_count = max(
        1,
        math.ceil(rejection_simulations * acceptance_fraction),
    )
    if accepted_count != smc_config.n_particles:
        raise ValueError(
            "rejection accepted count must equal the ABC-SMC particle count"
        )
    if not isinstance(smc_workers, int) or smc_workers <= 0:
        raise ValueError("smc_workers must be a positive integer")

    observed_datasets = []
    for row in regimes.itertuples(index=False):
        true_parameters = np.array(
            [row.alpha, row.beta, row.sigma_eta],
            dtype=float,
        )
        observed_datasets.append(
            simulate_sv(
                true_parameters,
                series_length,
                np.random.default_rng(row.data_seed),
            )
        )

    scale_start = time.perf_counter()
    summary_scale, scale_counts = estimate_prior_predictive_summary_scale(
        length=series_length,
        n_simulations=scale_simulations,
        random_seed=scale_seed,
    )
    scale_seconds = time.perf_counter() - scale_start

    bank_start = time.perf_counter()
    bank = build_shared_rejection_bank(
        model_bundle,
        series_length,
        rejection_simulations,
        rejection_seed,
    )
    bank_seconds = time.perf_counter() - bank_start
    print(
        f"Built shared rejection bank with {len(bank['valid_indices'])} "
        f"valid draws in {bank_seconds:.1f} seconds",
        flush=True,
    )

    rejection_results = []
    distance_seconds_total = 0.0
    for dataset_index, observed in enumerate(observed_datasets, start=1):
        distance_start = time.perf_counter()
        distance_sets = rejection_distances_from_bank(
            observed,
            bank,
            model_bundle,
            summary_scale,
        )
        elapsed = time.perf_counter() - distance_start
        distance_seconds_total += elapsed
        rejection_results.append(
            select_shared_bank_posteriors(
                bank["parameters"],
                distance_sets,
                acceptance_fraction,
            )
        )
        print(
            f"Completed rejection distances {dataset_index}/"
            f"{len(observed_datasets)}",
            flush=True,
        )

    smc_results = _run_smc_jobs(
        observed_datasets,
        summary_scale,
        smc_config,
        smc_workers,
    )

    posterior_samples = np.empty(
        (len(regimes), len(METHOD_ORDER), accepted_count, 3),
        dtype=float,
    )
    posterior_weights = np.empty(
        (len(regimes), len(METHOD_ORDER), accepted_count),
        dtype=float,
    )
    raw_rows = []
    smc_diagnostics = []

    for dataset_index, regime in regimes.iterrows():
        true_parameters = regime[list(PARAMETER_NAMES)].to_numpy(dtype=float)
        dataset_information = {
            "dataset_id": regime["dataset_id"],
            "replicate": int(regime["replicate"]),
            "data_seed": int(regime["data_seed"]),
            "beta_regime": float(regime["beta_regime"]),
            "sigma_eta_regime": float(regime["sigma_eta_regime"]),
        }
        method_results = dict(rejection_results[dataset_index])
        method_results["ABC-SMC"] = smc_results[dataset_index]

        for method_index, method in enumerate(METHOD_ORDER):
            result = method_results[method]
            samples = np.asarray(result["samples"], dtype=float)
            weights = np.asarray(result["weights"], dtype=float)
            posterior_samples[dataset_index, method_index] = samples
            posterior_weights[dataset_index, method_index] = weights
            if method == "ABC-SMC":
                simulator_calls = result["simulator_calls"]
                runtime_seconds = result["runtime_seconds"]
            else:
                simulator_calls = rejection_simulations
                runtime_seconds = np.nan
            raw_rows.extend(
                posterior_recovery_rows(
                    samples,
                    weights,
                    true_parameters,
                    dataset_information,
                    method,
                    simulator_calls,
                    runtime_seconds,
                )
            )

        smc_diagnostics.append(
            {
                **dataset_information,
                **{
                    key: smc_results[dataset_index][key]
                    for key in (
                        "simulator_calls",
                        "runtime_seconds",
                        "completed_populations",
                        "final_epsilon",
                        "final_ess",
                        "stop_reason",
                    )
                },
            }
        )

    raw_results = pd.DataFrame(raw_rows)
    regime_summary = aggregate_recovery_results(
        raw_results,
        group_columns=(
            "beta_regime",
            "sigma_eta_regime",
            "method",
            "target",
            "target_type",
        ),
    )
    overall_summary = aggregate_recovery_results(
        raw_results,
        group_columns=("method", "target", "target_type"),
    )
    win_rates = calculate_method_win_rates(raw_results, METHOD_ORDER)
    block_comparisons = calculate_block_paired_comparisons(
        raw_results,
        METHOD_ORDER,
    )

    metadata = {
        "dataset_count": int(len(regimes)),
        "series_length": int(series_length),
        "rejection_simulations": int(rejection_simulations),
        "acceptance_fraction": float(acceptance_fraction),
        "accepted_count": int(accepted_count),
        "rejection_seed": int(rejection_seed),
        "shared_rejection_bank": True,
        "common_random_number_blocks": int(regimes["data_seed"].nunique()),
        "common_random_number_design": (
            "the same data seed is reused across the six regimes to form "
            "a paired comparison block"
        ),
        "shared_bank_valid_draws": int(len(bank["valid_indices"])),
        "shared_bank_invalid_draws": int(
            rejection_simulations - len(bank["valid_indices"])
        ),
        "shared_bank_seconds": float(bank_seconds),
        "shared_rejection_distance_seconds": float(
            distance_seconds_total
        ),
        "scale_simulations": int(scale_simulations),
        "scale_seed": int(scale_seed),
        "scale_valid_draws": int(scale_counts["valid"]),
        "scale_invalid_draws": int(scale_counts["invalid"]),
        "scale_seconds": float(scale_seconds),
        "summary_scale": summary_scale.tolist(),
        "smc_configuration": {
            key: getattr(smc_config, key)
            for key in smc_config.__dataclass_fields__
        },
        "smc_workers": int(smc_workers),
        "smc_diagnostics": smc_diagnostics,
        "regimes": regimes.to_dict(orient="records"),
    }
    return {
        "raw_results": raw_results,
        "regime_summary": regime_summary,
        "overall_summary": overall_summary,
        "win_rates": win_rates,
        "block_comparisons": block_comparisons,
        "posterior_samples": posterior_samples,
        "posterior_weights": posterior_weights,
        "metadata": metadata,
    }


def aggregate_recovery_results(raw_results, group_columns):
    """Aggregate recovery errors, coverage, and interval widths."""

    raw_results = pd.DataFrame(raw_results)
    rows = []
    for group_values, group in raw_results.groupby(
        list(group_columns),
        sort=False,
        dropna=False,
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_columns, group_values))
        scaled_errors = group["scaled_absolute_error"].to_numpy(dtype=float)
        biases = group["bias"].to_numpy(dtype=float)
        q25, q75 = np.quantile(scaled_errors, [0.25, 0.75])
        row.update(
            {
                "n_datasets": int(len(group)),
                "mean_bias": float(np.mean(biases)),
                "rmse": float(np.sqrt(np.mean(biases ** 2))),
                "mean_absolute_error": float(
                    group["absolute_error"].mean()
                ),
                "mean_scaled_absolute_error": float(
                    np.mean(scaled_errors)
                ),
                "sd_scaled_absolute_error": float(
                    np.std(scaled_errors, ddof=1)
                    if len(scaled_errors) > 1
                    else 0.0
                ),
                "median_scaled_absolute_error": float(
                    np.median(scaled_errors)
                ),
                "q25_scaled_absolute_error": float(q25),
                "q75_scaled_absolute_error": float(q75),
                "scaled_error_iqr": float(q75 - q25),
                "coverage_rate": float(group["covered"].mean()),
                "mean_interval_width": float(
                    group["interval_width"].mean()
                ),
                "mean_simulator_calls": float(
                    group["simulator_calls"].mean()
                ),
                "mean_runtime_seconds": float(
                    group["runtime_seconds"].mean()
                )
                if "runtime_seconds" in group
                else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_method_win_rates(raw_results, method_order):
    """Count the lowest scaled error for every dataset and target."""

    raw_results = pd.DataFrame(raw_results)
    winner_indices = raw_results.groupby(
        ["dataset_id", "target"],
        sort=False,
    )["scaled_absolute_error"].idxmin()
    winners = raw_results.loc[winner_indices, ["target", "method"]]

    rows = []
    targets = [
        target for target in TARGET_NAMES if target in raw_results["target"].values
    ]
    for target in targets:
        target_winners = winners[winners["target"] == target]
        total = len(target_winners)
        for method in method_order:
            wins = int((target_winners["method"] == method).sum())
            rows.append(
                {
                    "target": target,
                    "method": method,
                    "wins": wins,
                    "datasets": total,
                    "win_rate": wins / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def calculate_block_paired_comparisons(
    raw_results,
    method_order,
    n_bootstrap=20_000,
    random_seed=202608,
):
    """Compare the two lowest-error methods by common seed blocks.

    One seed is reused across all volatility regimes. Resampling whole seed
    blocks keeps this pairing and avoids treating all datasets as independent.
    """

    raw_results = pd.DataFrame(raw_results)
    required = {
        "dataset_id",
        "data_seed",
        "method",
        "target",
        "scaled_absolute_error",
    }
    missing = required.difference(raw_results.columns)
    if missing:
        raise ValueError(f"raw_results is missing columns: {sorted(missing)}")
    if not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")

    rng = np.random.default_rng(random_seed)
    method_rank = {name: index for index, name in enumerate(method_order)}
    rows = []
    targets = [
        target
        for target in TARGET_NAMES
        if target in raw_results["target"].values
    ]
    for target in targets:
        target_rows = raw_results[raw_results["target"] == target]
        method_means = target_rows.groupby("method")[
            "scaled_absolute_error"
        ].mean()
        available_methods = sorted(
            method_means.index,
            key=lambda name: (
                method_means[name],
                method_rank.get(name, 999),
            ),
        )
        if len(available_methods) < 2:
            continue
        best_method, comparison_method = available_methods[:2]

        paired = target_rows.pivot_table(
            index=["dataset_id", "data_seed"],
            columns="method",
            values="scaled_absolute_error",
            aggfunc="first",
        ).dropna(subset=[best_method, comparison_method])
        paired["difference"] = (
            paired[best_method] - paired[comparison_method]
        )
        block_differences = (
            paired.reset_index()
            .groupby("data_seed")["difference"]
            .mean()
            .to_numpy(dtype=float)
        )
        if len(block_differences) == 0:
            continue

        sampled_indices = rng.integers(
            0,
            len(block_differences),
            size=(n_bootstrap, len(block_differences)),
        )
        bootstrap_means = block_differences[sampled_indices].mean(axis=1)
        ci_lower, ci_upper = np.quantile(bootstrap_means, [0.025, 0.975])
        rows.append(
            {
                "target": target,
                "best_method": best_method,
                "comparison_method": comparison_method,
                "mean_paired_difference": float(
                    np.mean(block_differences)
                ),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "seed_blocks": int(len(block_differences)),
                "bootstrap_resamples": int(n_bootstrap),
            }
        )
    return pd.DataFrame(rows)


def _save_regime_figure(regime_summary, target, method_order, output_path):
    rows = regime_summary[regime_summary["target"] == target].copy()
    regimes = (
        rows[["beta_regime", "sigma_eta_regime"]]
        .drop_duplicates()
        .sort_values(["beta_regime", "sigma_eta_regime"])
    )
    regime_pairs = list(regimes.itertuples(index=False, name=None))
    positions = np.arange(len(regime_pairs), dtype=float)
    width = 0.16

    figure, axis = plt.subplots(figsize=(10.0, 5.4))
    for method_index, method in enumerate(method_order):
        method_rows = rows[rows["method"] == method].set_index(
            ["beta_regime", "sigma_eta_regime"]
        )
        method_rows = method_rows.reindex(regime_pairs)
        medians = method_rows["median_scaled_absolute_error"].to_numpy()
        lower = medians - method_rows["q25_scaled_absolute_error"].to_numpy()
        upper = method_rows["q75_scaled_absolute_error"].to_numpy() - medians
        offset = (method_index - (len(method_order) - 1) / 2) * width
        axis.errorbar(
            positions + offset,
            medians,
            yerr=np.vstack((lower, upper)),
            fmt="o",
            capsize=4,
            linewidth=1.8,
            label=method,
        )

    labels = [
        f"$\\beta$={beta:.2f}\n$\\sigma_\\eta$={sigma_eta:.2f}"
        for beta, sigma_eta in regime_pairs
    ]
    axis.set_xticks(positions, labels)
    axis.set_ylabel(
        "Prior-normalised absolute error"
        if target in PARAMETER_NAMES
        else "Relative absolute error"
    )
    axis.set_title(f"Repeated recovery: {TARGET_LABELS[target]}")
    axis.margins(y=0.12)
    axis.legend(fontsize=9, ncol=2)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _save_win_rate_figure(win_rates, method_order, output_path):
    targets = [
        target for target in TARGET_NAMES if target in win_rates["target"].values
    ]
    positions = np.arange(len(targets), dtype=float)
    width = 0.75 / len(method_order)

    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    for method_index, method in enumerate(method_order):
        rows = win_rates[win_rates["method"] == method].set_index("target")
        values = rows.reindex(targets)["win_rate"].to_numpy()
        offset = (method_index - (len(method_order) - 1) / 2) * width
        axis.bar(positions + offset, values, width=width, label=method)

    labels = [TARGET_LABELS[target] for target in targets]
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylabel("Method win rate")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Lowest recovery error across synthetic datasets")
    axis.legend(fontsize=9)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_repeated_study_outputs(
    raw_results,
    regime_summary,
    overall_summary,
    win_rates,
    posterior_samples,
    posterior_weights,
    metadata,
    output_dir,
    figures_dir,
    method_order,
    block_comparisons=None,
):
    """Save complete tables, posterior arrays, metadata, and figures."""

    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if block_comparisons is None:
        block_comparisons = calculate_block_paired_comparisons(
            raw_results,
            method_order,
        )

    table_data = {
        "repeated_regime_raw_results.csv": raw_results,
        "repeated_regime_summary.csv": regime_summary,
        "repeated_regime_overall_summary.csv": overall_summary,
        "repeated_regime_win_rates.csv": win_rates,
        "repeated_regime_block_comparisons.csv": block_comparisons,
    }
    table_paths = []
    for filename, frame in table_data.items():
        path = output_dir / filename
        pd.DataFrame(frame).to_csv(path, index=False)
        table_paths.append(path)

    posterior_path = output_dir / "repeated_regime_posteriors.npz"
    np.savez_compressed(
        posterior_path,
        samples=np.asarray(posterior_samples, dtype=float),
        weights=np.asarray(posterior_weights, dtype=float),
    )

    figure_paths = []
    for target in TARGET_NAMES:
        figure_path = figures_dir / f"repeated_regime_{target}.png"
        _save_regime_figure(
            regime_summary,
            target,
            method_order,
            figure_path,
        )
        figure_paths.append(figure_path)
    win_rate_path = figures_dir / "repeated_regime_win_rates.png"
    _save_win_rate_figure(win_rates, method_order, win_rate_path)
    figure_paths.append(win_rate_path)

    metadata_path = output_dir / "repeated_regime_metadata.json"
    complete_metadata = dict(metadata)
    complete_metadata.update(
        {
            "method_order": list(method_order),
            "tables": [str(path) for path in table_paths],
            "posterior_file": str(posterior_path),
            "figures": [str(path) for path in figure_paths],
        }
    )
    metadata_path.write_text(
        json.dumps(complete_metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "tables": table_paths,
        "posterior_file": posterior_path,
        "metadata": metadata_path,
        "figures": figure_paths,
    }

"""Comparison tables and figures for rejection ABC and ABC-SMC."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from abc_smc_utils import weighted_quantile


PARAMETERS = ("alpha", "beta", "sigma_eta")
METHOD_ORDER = (
    "Manual ABC",
    "Auto ABC",
    "Wasserstein ABC",
    "ABC-SMC",
)
TRUE_PARAMETERS = {
    "alpha": -0.5,
    "beta": 0.95,
    "sigma_eta": 0.30,
}


def _normalized_weights(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("weights must be a non-empty vector")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    return weights / total


def weighted_posterior_summary(values, weights):
    """Return weighted mean, spread, median, and central 95% interval."""

    values = np.asarray(values, dtype=float)
    weights = _normalized_weights(weights)
    if values.ndim != 1 or len(values) != len(weights):
        raise ValueError("values and weights must be equal-length vectors")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")

    mean = float(np.sum(weights * values))
    variance = float(np.sum(weights * (values - mean) ** 2))
    q025, median, q975 = weighted_quantile(
        values,
        np.array([0.025, 0.5, 0.975]),
        weights,
    )

    return {
        "mean": mean,
        "sd": float(np.sqrt(variance)),
        "q2.5": float(q025),
        "median": float(median),
        "q97.5": float(q975),
    }


def _read_metadata(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_rejection_result(scripts_dir, npy_name, metadata_name):
    samples = np.asarray(np.load(scripts_dir / npy_name), dtype=float)
    if samples.ndim != 2 or samples.shape[1] < 3 or len(samples) == 0:
        raise ValueError(f"malformed rejection result: {npy_name}")
    samples = samples[:, :3]
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"non-finite rejection result: {npy_name}")
    return {
        "samples": samples,
        "weights": np.full(len(samples), 1.0 / len(samples)),
        "metadata": _read_metadata(scripts_dir / metadata_name),
        "history": None,
    }


def _load_smc_result(scripts_dir, prefix):
    history_path = scripts_dir / f"{prefix}_history.npz"
    with np.load(history_path) as history_file:
        history = {name: history_file[name].copy() for name in history_file.files}
    required = {
        "particles",
        "weights",
        "distances",
        "epsilons",
        "candidate_simulations",
        "eligible_counts",
        "acceptance_rates",
        "cumulative_simulator_calls",
        "effective_sample_sizes",
    }
    if not required.issubset(history):
        raise ValueError(f"SMC history is missing keys: {prefix}")
    if history["particles"].ndim != 3 or history["particles"].shape[2] != 3:
        raise ValueError(f"malformed SMC particles: {prefix}")
    if history["weights"].shape != history["particles"].shape[:2]:
        raise ValueError(f"malformed SMC weights: {prefix}")

    final_weights = _normalized_weights(history["weights"][-1])
    if not np.allclose(final_weights, history["weights"][-1]):
        raise ValueError(f"SMC weights are not normalized: {prefix}")

    return {
        "samples": np.asarray(history["particles"][-1], dtype=float),
        "weights": final_weights,
        "metadata": _read_metadata(
            scripts_dir / f"{prefix}_metadata.json"
        ),
        "history": history,
    }


def load_all_results(scripts_dir):
    """Load all saved rejection ABC and ABC-SMC posteriors."""

    rejection_specs = {
        "real": {
            "Manual ABC": ("abc_manual_sv.npy", "abc_manual_sv_metadata.json"),
            "Auto ABC": ("abc_auto_sv.npy", "abc_auto_sv_metadata.json"),
            "Wasserstein ABC": (
                "abc_wasserstein_sv.npy",
                "abc_wasserstein_sv_metadata.json",
            ),
        },
        "synthetic": {
            "Manual ABC": (
                "abc_manual_synthetic_sv.npy",
                "abc_manual_synthetic_sv_metadata.json",
            ),
            "Auto ABC": (
                "abc_auto_synthetic_sv.npy",
                "abc_auto_synthetic_sv_metadata.json",
            ),
            "Wasserstein ABC": (
                "abc_wasserstein_synthetic_sv.npy",
                "abc_wasserstein_synthetic_sv_metadata.json",
            ),
        },
    }

    results = {"real": {}, "synthetic": {}}
    for dataset, methods in rejection_specs.items():
        for method, (npy_name, metadata_name) in methods.items():
            results[dataset][method] = _load_rejection_result(
                scripts_dir,
                npy_name,
                metadata_name,
            )
    results["real"]["ABC-SMC"] = _load_smc_result(
        scripts_dir,
        "abc_smc_sv",
    )
    results["synthetic"]["ABC-SMC"] = _load_smc_result(
        scripts_dir,
        "abc_smc_synthetic_sv",
    )
    return results


def _runtime_and_calls(metadata):
    runtime = metadata.get(
        "runtime_seconds",
        metadata.get("inference_seconds", np.nan),
    )
    calls = metadata.get(
        "simulation_budget",
        metadata.get("total_simulator_calls", np.nan),
    )
    return float(runtime), int(calls) if np.isfinite(calls) else np.nan


def _build_comparison_table(results):
    rows = []
    for dataset in ("real", "synthetic"):
        for method in METHOD_ORDER:
            result = results[dataset][method]
            runtime, calls = _runtime_and_calls(result["metadata"])
            for index, parameter in enumerate(PARAMETERS):
                summary = weighted_posterior_summary(
                    result["samples"][:, index],
                    result["weights"],
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "parameter": parameter,
                        **summary,
                        "runtime_seconds": runtime,
                        "simulator_calls": calls,
                    }
                )
    return pd.DataFrame(rows)


def _build_synthetic_errors(comparison):
    synthetic = comparison[comparison["dataset"] == "synthetic"]
    means = synthetic.pivot(
        index="method",
        columns="parameter",
        values="mean",
    ).reindex(METHOD_ORDER)
    errors = pd.DataFrame(index=means.index)
    for parameter in PARAMETERS:
        errors[f"{parameter}_error"] = (
            means[parameter] - TRUE_PARAMETERS[parameter]
        ).abs()
    errors["mean_abs_error"] = errors.mean(axis=1)
    return errors.reset_index()


def _build_real_means(comparison):
    """Return one row of real-data posterior means for each method."""

    real_results = comparison[comparison["dataset"] == "real"]
    means = real_results.pivot(
        index="method",
        columns="parameter",
        values="mean",
    )
    means = means.reindex(index=METHOD_ORDER, columns=PARAMETERS)
    return means.reset_index()


def _format_axis(axis):
    axis.grid(alpha=0.22)
    axis.tick_params(axis="both", labelsize=11)
    axis.xaxis.label.set_size(13)
    axis.yaxis.label.set_size(13)
    axis.title.set_size(14)


def _save_figure(figure, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(figure)


def _plot_posterior_parameter(
    results,
    dataset,
    parameter_index,
    parameter,
    output_path,
):
    labels = {
        "alpha": r"$\alpha$",
        "beta": r"$\beta$",
        "sigma_eta": r"$\sigma_\eta$",
    }
    figure, axis = plt.subplots(figsize=(8.8, 5.6))
    all_values = np.concatenate(
        [
            results[dataset][method]["samples"][:, parameter_index]
            for method in METHOD_ORDER
        ]
    )
    x_values = np.linspace(all_values.min(), all_values.max(), 500)
    for method in METHOD_ORDER:
        result = results[dataset][method]
        density = gaussian_kde(
            result["samples"][:, parameter_index],
            weights=result["weights"],
        )
        axis.plot(
            x_values,
            density(x_values),
            linewidth=2.4,
            label=method,
        )
    if dataset == "synthetic":
        axis.axvline(
            TRUE_PARAMETERS[parameter],
            color="black",
            linestyle="--",
            linewidth=2.2,
            label="True value",
        )
    dataset_label = "Synthetic data" if dataset == "synthetic" else "S&P 500"
    axis.set_xlabel(labels[parameter])
    axis.set_ylabel("Density")
    axis.set_title(f"{dataset_label}: posterior for {labels[parameter]}")
    axis.legend(fontsize=11, frameon=True)
    _format_axis(axis)
    _save_figure(figure, output_path)


def _plot_epsilon_diagnostics(results, output_path):
    figure, axis = plt.subplots(figsize=(8.8, 5.6))
    for dataset, label in (("synthetic", "Synthetic"), ("real", "S&P 500")):
        epsilon = results[dataset]["ABC-SMC"]["history"]["epsilons"]
        axis.plot(
            np.arange(len(epsilon)),
            epsilon,
            marker="o",
            linewidth=2.4,
            label=label,
        )
    axis.set_xlabel("Population")
    axis.set_ylabel(r"Tolerance $\epsilon$")
    axis.set_title("Adaptive ABC-SMC tolerance sequence")
    axis.set_xticks(range(max(
        len(results["synthetic"]["ABC-SMC"]["history"]["epsilons"]),
        len(results["real"]["ABC-SMC"]["history"]["epsilons"]),
    )))
    axis.legend(fontsize=11)
    _format_axis(axis)
    _save_figure(figure, output_path)


def _plot_acceptance_rate_diagnostics(results, output_path):
    figure, axis = plt.subplots(figsize=(8.8, 5.6))
    for dataset, label in (("synthetic", "Synthetic"), ("real", "S&P 500")):
        history = results[dataset]["ABC-SMC"]["history"]
        populations = np.arange(len(history["epsilons"]))
        axis.plot(
            populations,
            history["acceptance_rates"],
            marker="o",
            linewidth=2.4,
            label=label,
        )
    axis.set_title("ABC-SMC acceptance rate")
    axis.set_xlabel("Population")
    axis.set_ylabel("Accepted / proposed")
    axis.legend(fontsize=11)
    _format_axis(axis)
    _save_figure(figure, output_path)


def _plot_ess_diagnostics(results, output_path):
    figure, axis = plt.subplots(figsize=(8.8, 5.6))
    for dataset, label in (("synthetic", "Synthetic"), ("real", "S&P 500")):
        history = results[dataset]["ABC-SMC"]["history"]
        populations = np.arange(len(history["epsilons"]))
        axis.plot(
            populations,
            history["effective_sample_sizes"],
            marker="o",
            linewidth=2.4,
            label=label,
        )
    axis.set_title("ABC-SMC effective sample size")
    axis.set_xlabel("Population")
    axis.set_ylabel("ESS")
    axis.legend(fontsize=11)
    _format_axis(axis)
    _save_figure(figure, output_path)


def run_comparison(scripts_dir, figures_dir):
    """Generate all four-method tables and ABC-SMC diagnostic figures."""

    scripts_dir = Path(scripts_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results = load_all_results(scripts_dir)

    comparison = _build_comparison_table(results)
    errors = _build_synthetic_errors(comparison)
    real_means = _build_real_means(comparison)
    comparison_path = scripts_dir / "abc_smc_method_comparison.csv"
    error_path = scripts_dir / "abc_smc_synthetic_errors.csv"
    legacy_summary_path = scripts_dir / "posterior_summary_updated.csv"
    legacy_real_path = scripts_dir / "real_data_results_table.csv"
    legacy_error_path = scripts_dir / "synthetic_error_table.csv"
    comparison.to_csv(comparison_path, index=False)
    errors.to_csv(error_path, index=False)
    comparison.to_csv(legacy_summary_path, index=False)
    real_means.to_csv(legacy_real_path, index=False)
    errors.to_csv(legacy_error_path, index=False)

    figure_paths = []
    for dataset in ("synthetic", "real"):
        for parameter_index, parameter in enumerate(PARAMETERS):
            figure_path = (
                figures_dir / f"{dataset}_posterior_{parameter}.png"
            )
            _plot_posterior_parameter(
                results,
                dataset,
                parameter_index,
                parameter,
                figure_path,
            )
            figure_paths.append(figure_path)

    epsilon_path = figures_dir / "smc_epsilon.png"
    acceptance_path = figures_dir / "smc_acceptance_rate.png"
    ess_path = figures_dir / "smc_effective_sample_size.png"
    _plot_epsilon_diagnostics(results, epsilon_path)
    _plot_acceptance_rate_diagnostics(results, acceptance_path)
    _plot_ess_diagnostics(results, ess_path)
    figure_paths.extend([epsilon_path, acceptance_path, ess_path])

    return {
        "comparison_table": comparison_path,
        "synthetic_error_table": error_path,
        "legacy_summary_table": legacy_summary_path,
        "legacy_real_table": legacy_real_path,
        "legacy_error_table": legacy_error_path,
        "figures": figure_paths,
    }

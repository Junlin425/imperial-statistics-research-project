"""Calculate financial interpretations of stochastic-volatility posteriors."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abc_smc_comparison import weighted_posterior_summary


QUANTITY_NAMES = (
    "long_run_log_variance",
    "half_life_days",
    "stationary_log_volatility_variance",
)

QUANTITY_LABELS = {
    "long_run_log_variance": "Long-run log variance",
    "half_life_days": "Volatility shock half-life (days)",
    "stationary_log_volatility_variance": (
        "Stationary log-volatility variance"
    ),
}

TRUE_PARAMETERS = np.array([-0.5, 0.95, 0.30])


def calculate_financial_quantities(parameter_samples):
    """Transform posterior draws into three financial quantities."""

    samples = np.asarray(parameter_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] < 3 or len(samples) == 0:
        raise ValueError("parameter_samples must be a non-empty matrix")
    samples = samples[:, :3]
    if not np.all(np.isfinite(samples)):
        raise ValueError("parameter_samples must be finite")

    alpha = samples[:, 0]
    beta = samples[:, 1]
    sigma_eta = samples[:, 2]
    if np.any(beta <= 0.0) or np.any(beta >= 1.0):
        raise ValueError("beta must lie between 0 and 1")
    if np.any(sigma_eta <= 0.0):
        raise ValueError("sigma_eta must be positive")

    return {
        "long_run_log_variance": alpha / (1.0 - beta),
        "half_life_days": np.log(0.5) / np.log(beta),
        "stationary_log_volatility_variance": (
            sigma_eta ** 2 / (1.0 - beta ** 2)
        ),
    }


def _true_quantity_values(true_parameters):
    true_samples = np.asarray(true_parameters, dtype=float).reshape(1, 3)
    quantities = calculate_financial_quantities(true_samples)
    return {name: float(values[0]) for name, values in quantities.items()}


def build_financial_summary(
    results,
    method_order,
    true_parameters=TRUE_PARAMETERS,
):
    """Summarise transformed posterior draws for synthetic and real data."""

    true_values = _true_quantity_values(true_parameters)
    rows = []
    for dataset in ("synthetic", "real"):
        for method in method_order:
            result = results[dataset][method]
            quantities = calculate_financial_quantities(result["samples"])
            for quantity in QUANTITY_NAMES:
                row = {
                    "dataset": dataset,
                    "method": method,
                    "quantity": quantity,
                }
                row.update(
                    weighted_posterior_summary(
                        quantities[quantity],
                        result["weights"],
                    )
                )
                if dataset == "synthetic":
                    true_value = true_values[quantity]
                    absolute_error = abs(row["mean"] - true_value)
                    row.update(
                        {
                            "true_value": true_value,
                            "bias": row["mean"] - true_value,
                            "absolute_error": absolute_error,
                            "relative_absolute_error": (
                                absolute_error / abs(true_value)
                            ),
                            "covered": (
                                row["q2.5"] <= true_value <= row["q97.5"]
                            ),
                        }
                    )
                else:
                    row.update(
                        {
                            "true_value": np.nan,
                            "bias": np.nan,
                            "absolute_error": np.nan,
                            "relative_absolute_error": np.nan,
                            "covered": np.nan,
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _build_synthetic_error_table(summary, method_order):
    synthetic = summary[summary["dataset"] == "synthetic"]
    errors = pd.DataFrame({"method": list(method_order)})
    for quantity in QUANTITY_NAMES:
        quantity_rows = synthetic[synthetic["quantity"] == quantity]
        quantity_rows = quantity_rows.set_index("method").reindex(method_order)
        errors[f"{quantity}_absolute_error"] = (
            quantity_rows["absolute_error"].to_numpy()
        )
        errors[f"{quantity}_relative_error"] = (
            quantity_rows["relative_absolute_error"].to_numpy()
        )
        errors[f"{quantity}_covered"] = (
            quantity_rows["covered"].to_numpy()
        )

    relative_columns = [
        f"{quantity}_relative_error" for quantity in QUANTITY_NAMES
    ]
    errors["mean_relative_error"] = errors[relative_columns].mean(axis=1)
    return errors


def _save_interval_figure(rows, dataset, quantity, output_path):
    """Plot posterior medians and central 95% intervals for one quantity."""

    positions = np.arange(len(rows))
    medians = rows["median"].to_numpy()
    lower_errors = medians - rows["q2.5"].to_numpy()
    upper_errors = rows["q97.5"].to_numpy() - medians

    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    axis.errorbar(
        medians,
        positions,
        xerr=np.vstack((lower_errors, upper_errors)),
        fmt="o",
        markersize=7,
        capsize=5,
        linewidth=2,
    )
    if dataset == "synthetic":
        axis.axvline(
            rows["true_value"].iloc[0],
            color="black",
            linestyle="--",
            linewidth=2,
            label="True value",
        )
        axis.legend()

    axis.set_yticks(positions, rows["method"])
    axis.invert_yaxis()
    axis.set_xlabel(QUANTITY_LABELS[quantity])
    dataset_label = "Synthetic data" if dataset == "synthetic" else "S&P 500"
    axis.set_title(f"{dataset_label}: posterior median and 95% interval")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _save_relative_error_figure(errors, output_path):
    """Compare dimensionless synthetic errors across financial quantities."""

    positions = np.arange(len(errors))
    width = 0.24
    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    for index, quantity in enumerate(QUANTITY_NAMES):
        offset = (index - 1) * width
        axis.bar(
            positions + offset,
            errors[f"{quantity}_relative_error"],
            width=width,
            label=QUANTITY_LABELS[quantity],
        )
    axis.set_xticks(positions, errors["method"])
    axis.set_ylabel("Relative absolute error")
    axis.set_title("Synthetic recovery of financial quantities")
    axis.legend(fontsize=9)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def run_financial_analysis(
    results,
    output_dir,
    figures_dir,
    method_order,
    true_parameters=TRUE_PARAMETERS,
):
    """Create tables, metadata, and standalone figures."""

    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = build_financial_summary(
        results,
        method_order,
        true_parameters,
    )
    errors = _build_synthetic_error_table(summary, method_order)
    summary_path = output_dir / "financial_derived_quantities_summary.csv"
    error_path = output_dir / "financial_derived_quantities_errors.csv"
    metadata_path = output_dir / "financial_derived_quantities_metadata.json"
    summary.to_csv(summary_path, index=False)
    errors.to_csv(error_path, index=False)

    figure_paths = []
    for dataset in ("synthetic", "real"):
        for quantity in QUANTITY_NAMES:
            rows = summary[
                (summary["dataset"] == dataset)
                & (summary["quantity"] == quantity)
            ].copy()
            rows["method"] = pd.Categorical(
                rows["method"],
                categories=method_order,
                ordered=True,
            )
            rows = rows.sort_values("method")
            figure_path = figures_dir / f"derived_{dataset}_{quantity}.png"
            _save_interval_figure(rows, dataset, quantity, figure_path)
            figure_paths.append(figure_path)

    relative_error_path = figures_dir / "derived_synthetic_relative_errors.png"
    _save_relative_error_figure(errors, relative_error_path)
    figure_paths.append(relative_error_path)

    metadata = {
        "posterior_transformation": "draw_by_draw",
        "method_order": list(method_order),
        "true_parameters": np.asarray(true_parameters).tolist(),
        "formulas": {
            "long_run_log_variance": "alpha / (1 - beta)",
            "half_life_days": "log(0.5) / log(beta)",
            "stationary_log_volatility_variance": (
                "sigma_eta^2 / (1 - beta^2)"
            ),
        },
        "summary_table": str(summary_path),
        "synthetic_error_table": str(error_path),
        "figures": [str(path) for path in figure_paths],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "summary_table": summary_path,
        "synthetic_error_table": error_path,
        "metadata": metadata_path,
        "figures": figure_paths,
    }

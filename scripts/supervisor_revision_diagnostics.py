"""Create the extra diagnostics requested after supervisor feedback."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from project_paths import FIGURES_DIR, SCRIPT_DIR
from sv_abc_core import (
    PRIOR_BOUNDS,
    is_valid_return_series,
    sample_prior,
    simulate_sv,
)


PARAMETER_ERROR_COLUMNS = (
    "alpha_error",
    "beta_error",
    "sigma_eta_error",
)


def calculate_prior_scaled_errors(error_table, prior_bounds=PRIOR_BOUNDS):
    """Divide each absolute parameter error by its prior width."""

    error_table = error_table.copy()
    prior_bounds = np.asarray(prior_bounds, dtype=float)
    if prior_bounds.shape != (3, 2):
        raise ValueError("prior_bounds must have shape (3, 2)")

    prior_widths = prior_bounds[:, 1] - prior_bounds[:, 0]
    if np.any(prior_widths <= 0.0):
        raise ValueError("every prior upper bound must exceed its lower bound")

    scaled_columns = []
    for error_column, width in zip(PARAMETER_ERROR_COLUMNS, prior_widths):
        if error_column not in error_table:
            raise ValueError(f"missing required column: {error_column}")
        scaled_column = error_column.replace("_error", "_scaled_error")
        error_table[scaled_column] = error_table[error_column] / width
        scaled_columns.append(scaled_column)

    error_table["mean_scaled_error"] = error_table[scaled_columns].mean(axis=1)
    return error_table[["method", *scaled_columns, "mean_scaled_error"]]


def simulate_prior_validity(
    n_draws=20_000,
    series_length=4_000,
    random_seed=20260827,
):
    """Draw from the nominal prior and record whether each series is valid."""

    if not isinstance(n_draws, int) or n_draws <= 0:
        raise ValueError("n_draws must be a positive integer")
    if not isinstance(series_length, int) or series_length < 6:
        raise ValueError("series_length must be an integer of at least 6")

    rng = np.random.default_rng(random_seed)
    parameters = sample_prior(rng, n_draws)
    valid = np.zeros(n_draws, dtype=bool)

    for index, theta in enumerate(parameters):
        returns = simulate_sv(theta, series_length, rng)
        valid[index] = is_valid_return_series(returns)

    return parameters, valid


def summarize_induced_prior(parameters, valid):
    """Summarise the nominal draws and the same draws after validity filtering."""

    parameters = np.asarray(parameters, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if parameters.ndim != 2 or parameters.shape[1] != 3:
        raise ValueError("parameters must have shape (n_draws, 3)")
    if valid.shape != (len(parameters),):
        raise ValueError("valid must contain one flag per parameter draw")
    if not np.any(valid):
        raise ValueError("at least one draw must be valid")

    long_run_level = parameters[:, 0] / (1.0 - parameters[:, 1])
    values = {
        "alpha": parameters[:, 0],
        "beta": parameters[:, 1],
        "sigma_eta": parameters[:, 2],
        "long_run_log_variance": long_run_level,
    }

    rows = []
    for distribution, mask in (
        ("Nominal prior", np.ones(len(parameters), dtype=bool)),
        ("Valid-only prior", valid),
    ):
        for variable, variable_values in values.items():
            selected = variable_values[mask]
            q025, median, q975 = np.quantile(selected, [0.025, 0.5, 0.975])
            rows.append(
                {
                    "distribution": distribution,
                    "variable": variable,
                    "n_draws": int(len(selected)),
                    "mean": float(np.mean(selected)),
                    "standard_deviation": float(np.std(selected, ddof=1)),
                    "q025": float(q025),
                    "median": float(median),
                    "q975": float(q975),
                }
            )
    return pd.DataFrame(rows)


def summarize_wasserstein_ridge(samples, true_mu_h=-10.0):
    """Measure the alpha-beta ridge in the stored Wasserstein posterior."""

    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] < 3:
        raise ValueError("samples must have at least alpha, beta and sigma_eta")

    alpha = samples[:, 0]
    beta = samples[:, 1]
    mu_h = alpha / (1.0 - beta)
    q025, median, q975 = np.quantile(mu_h, [0.025, 0.5, 0.975])
    return {
        "n_draws": int(len(samples)),
        "mean_alpha": float(np.mean(alpha)),
        "mean_beta": float(np.mean(beta)),
        "mean_sigma_eta": float(np.mean(samples[:, 2])),
        "alpha_beta_correlation": float(np.corrcoef(alpha, beta)[0, 1]),
        "mean_long_run_log_variance": float(np.mean(mu_h)),
        "median_long_run_log_variance": float(median),
        "q025_long_run_log_variance": float(q025),
        "q975_long_run_log_variance": float(q975),
        "true_long_run_log_variance": float(true_mu_h),
        "absolute_mu_h_error": float(abs(np.mean(mu_h) - true_mu_h)),
    }


def run_supervisor_diagnostics(
    error_table_path,
    wasserstein_samples_path,
    output_directory,
    figure_directory,
    n_prior_draws=20_000,
    series_length=4_000,
    random_seed=20260827,
):
    """Run all supervisor-requested diagnostics and save reusable outputs."""

    output_directory = Path(output_directory)
    figure_directory = Path(figure_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    errors = pd.read_csv(error_table_path)
    scaled_errors = calculate_prior_scaled_errors(errors)
    scaled_path = output_directory / "synthetic_scaled_error_table.csv"
    scaled_errors.to_csv(scaled_path, index=False)

    wasserstein_samples = np.load(wasserstein_samples_path)
    ridge_summary = summarize_wasserstein_ridge(wasserstein_samples)
    ridge_path = output_directory / "wasserstein_ridge_summary.csv"
    pd.DataFrame([ridge_summary]).to_csv(ridge_path, index=False)

    parameters, valid = simulate_prior_validity(
        n_draws=n_prior_draws,
        series_length=series_length,
        random_seed=random_seed,
    )
    prior_summary = summarize_induced_prior(parameters, valid)
    prior_summary_path = output_directory / "induced_prior_summary.csv"
    prior_summary.to_csv(prior_summary_path, index=False)

    prior_draws = pd.DataFrame(
        parameters,
        columns=["alpha", "beta", "sigma_eta"],
    )
    prior_draws["long_run_log_variance"] = (
        prior_draws["alpha"] / (1.0 - prior_draws["beta"])
    )
    prior_draws["valid"] = valid
    prior_draws_path = output_directory / "induced_prior_draws.csv"
    prior_draws.to_csv(prior_draws_path, index=False)

    figures = []
    ridge_figure = figure_directory / "wasserstein_alpha_beta_ridge.png"
    alpha = wasserstein_samples[:, 0]
    beta = wasserstein_samples[:, 1]
    mu_h = alpha / (1.0 - beta)
    beta_grid = np.linspace(PRIOR_BOUNDS[1, 0], PRIOR_BOUNDS[1, 1], 300)
    plt.figure(figsize=(7.2, 5.0))
    points = plt.scatter(beta, alpha, c=mu_h, s=20, alpha=0.65, cmap="viridis")
    plt.plot(beta_grid, -10.0 * (1.0 - beta_grid), "k--", linewidth=2,
             label=r"True $\mu_h=-10$ ridge")
    plt.scatter([0.95], [-0.5], marker="*", s=160, color="red", label="Truth")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\alpha$")
    plt.title("Wasserstein posterior: alpha-beta identification ridge")
    plt.colorbar(points, label=r"$\mu_h=\alpha/(1-\beta)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ridge_figure, dpi=220)
    plt.close()
    figures.append(ridge_figure)

    variable_labels = {
        "alpha": r"$\alpha$",
        "beta": r"$\beta$",
        "sigma_eta": r"$\sigma_\eta$",
        "long_run_log_variance": r"$\mu_h=\alpha/(1-\beta)$",
    }
    for variable, label in variable_labels.items():
        figure_path = figure_directory / f"induced_prior_{variable}.png"
        plt.figure(figsize=(7.2, 4.8))
        plt.hist(prior_draws[variable], bins=50, density=True, alpha=0.45,
                 color="#4C78A8", label="Nominal prior")
        plt.hist(prior_draws.loc[valid, variable], bins=50, density=True,
                 alpha=0.55, color="#F58518", label="Valid-only prior")
        plt.xlabel(label)
        plt.ylabel("Density")
        plt.title(f"Numerical validity and the induced prior for {variable}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_path, dpi=220)
        plt.close()
        figures.append(figure_path)

    validity_figure = figure_directory / "induced_prior_validity_alpha_beta.png"
    plt.figure(figsize=(7.2, 5.0))
    plt.scatter(
        prior_draws.loc[valid, "beta"],
        prior_draws.loc[valid, "alpha"],
        s=8,
        alpha=0.25,
        color="#54A24B",
        label="Valid",
    )
    plt.scatter(
        prior_draws.loc[~valid, "beta"],
        prior_draws.loc[~valid, "alpha"],
        s=8,
        alpha=0.25,
        color="#E45756",
        label="Invalid",
    )
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\alpha$")
    plt.title("Numerical validity across the nominal alpha-beta prior")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(validity_figure, dpi=220)
    plt.close()
    figures.append(validity_figure)

    metadata = {
        "n_prior_draws": int(n_prior_draws),
        "series_length": int(series_length),
        "random_seed": int(random_seed),
        "valid_draws": int(np.sum(valid)),
        "invalid_draws": int(np.sum(~valid)),
        "invalid_fraction": float(np.mean(~valid)),
        "validity_interpretation": (
            "The effective parameter distribution is proportional to the "
            "nominal prior times the probability of a numerically valid series."
        ),
        "wasserstein_ridge": ridge_summary,
    }
    metadata_path = output_directory / "supervisor_revision_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "scaled_errors": scaled_path,
        "wasserstein_ridge_summary": ridge_path,
        "induced_prior_summary": prior_summary_path,
        "induced_prior_draws": prior_draws_path,
        "metadata": metadata_path,
        "figures": figures,
    }


def main():
    outputs = run_supervisor_diagnostics(
        error_table_path=SCRIPT_DIR / "synthetic_error_table.csv",
        wasserstein_samples_path=(
            SCRIPT_DIR / "abc_wasserstein_synthetic_sv.npy"
        ),
        output_directory=SCRIPT_DIR,
        figure_directory=FIGURES_DIR,
    )
    print(pd.read_csv(outputs["scaled_errors"]).to_string(index=False))
    print(pd.read_csv(outputs["wasserstein_ridge_summary"]).to_string(index=False))
    print(pd.read_csv(outputs["induced_prior_summary"]).to_string(index=False))


if __name__ == "__main__":
    main()

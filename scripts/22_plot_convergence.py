"""Create separate, publication-ready Manual ABC convergence figures."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR.parent / "figures"
RESULTS_PATH = SCRIPT_DIR / "convergence_study.csv"

PLOTS = (
    (
        "accepted",
        "accepted_std",
        "Accepted samples",
        "Manual ABC accepted samples",
        "convergence_accepted_samples.png",
    ),
    (
        "alpha_abs_error",
        "alpha_abs_error_std",
        r"Absolute error for $\alpha$",
        r"Manual ABC convergence for $\alpha$",
        "convergence_alpha_error.png",
    ),
    (
        "beta_abs_error",
        "beta_abs_error_std",
        r"Absolute error for $\beta$",
        r"Manual ABC convergence for $\beta$",
        "convergence_beta_error.png",
    ),
    (
        "sigma_eta_abs_error",
        "sigma_eta_abs_error_std",
        r"Absolute error for $\sigma_\eta$",
        r"Manual ABC convergence for $\sigma_\eta$",
        "convergence_sigma_eta_error.png",
    ),
)


def _plot_measure(frame, value_column, std_column, ylabel, title, output_path):
    figure, axis = plt.subplots(figsize=(8.8, 5.6))
    for simulation_count in sorted(frame["N"].unique()):
        subset = frame[frame["N"] == simulation_count].sort_values(
            "epsilon",
            ascending=False,
        )
        epsilon = subset["epsilon"].to_numpy(dtype=float)
        values = subset[value_column].to_numpy(dtype=float)
        standard_deviation = subset[std_column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        epsilon = epsilon[finite]
        values = values[finite]
        standard_deviation = np.nan_to_num(
            standard_deviation[finite],
            nan=0.0,
        )
        line = axis.plot(
            epsilon,
            values,
            marker="o",
            linewidth=2.3,
            label=f"N={simulation_count:,}",
        )[0]
        lower = np.maximum(values - standard_deviation, 0.0)
        upper = values + standard_deviation
        axis.fill_between(
            epsilon,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.12,
        )

    axis.invert_xaxis()
    axis.set_xlabel(r"Tolerance $\epsilon$", fontsize=13)
    axis.set_ylabel(ylabel, fontsize=13)
    axis.set_title(title, fontsize=14)
    axis.tick_params(axis="both", labelsize=11)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=10, ncol=2)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(figure)


def main():
    frame = pd.read_csv(RESULTS_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for value_column, std_column, ylabel, title, filename in PLOTS:
        output_path = FIGURES_DIR / filename
        _plot_measure(
            frame,
            value_column,
            std_column,
            ylabel,
            title,
            output_path,
        )
        outputs.append(output_path)

    print("Saved convergence figures:")
    for output_path in outputs:
        print(output_path)


if __name__ == "__main__":
    main()

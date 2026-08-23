"""Regenerate and export all thesis plots as separate image files."""

from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from abc_smc_comparison import run_comparison
from posterior_predictive_utils import (
    METHOD_NAMES,
    PREDICTIVE_DRAW_KEYS,
    PREDICTIVE_FIGURE_NAMES,
    plot_predictive_distributions,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CODE_FIGURES_DIR = PROJECT_DIR / "figures"
REPORT_FIGURES_DIR = (
    PROJECT_DIR.parent.parent / "Report" / "final" / "final figure"
)

CONVERGENCE_FIGURE_NAMES = (
    "convergence_accepted_samples.png",
    "convergence_alpha_error.png",
    "convergence_beta_error.png",
    "convergence_sigma_eta_error.png",
)

OLD_COMPOSITE_NAMES = (
    "synthetic_posteriors.png",
    "real_posteriors.png",
    "convergence.png",
    "smc_efficiency.png",
    "ppc_distributions.png",
)


def _load_saved_predictive_draws():
    draws_path = SCRIPT_DIR / "posterior_predictive_draws.npz"
    if not draws_path.exists():
        raise FileNotFoundError(
            "posterior_predictive_draws.npz is missing; run "
            "20_posterior_predictive_check.py first"
        )
    with np.load(draws_path) as saved:
        observed = saved["observed_summary"].copy()
        predictive_draws = {
            method_name: saved[PREDICTIVE_DRAW_KEYS[method_name]].copy()
            for method_name in METHOD_NAMES
        }
    return observed, predictive_draws


def _generate_ppc_figures():
    observed, predictive_draws = _load_saved_predictive_draws()
    output_paths = {
        statistic: CODE_FIGURES_DIR / filename
        for statistic, filename in PREDICTIVE_FIGURE_NAMES.items()
    }
    return list(
        plot_predictive_distributions(
            observed,
            predictive_draws,
            output_paths,
        ).values()
    )


def _generate_convergence_figures():
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "22_plot_convergence.py")],
        cwd=SCRIPT_DIR,
        check=True,
    )
    return [
        CODE_FIGURES_DIR / filename
        for filename in CONVERGENCE_FIGURE_NAMES
    ]


def _export_to_report(figure_paths):
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for filename in OLD_COMPOSITE_NAMES:
        old_path = REPORT_FIGURES_DIR / filename
        if old_path.exists():
            old_path.unlink()

    exported = []
    for source_path in figure_paths:
        destination = REPORT_FIGURES_DIR / source_path.name
        shutil.copy2(source_path, destination)
        exported.append(destination)
    return exported


def main():
    CODE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    comparison_outputs = run_comparison(
        scripts_dir=SCRIPT_DIR,
        figures_dir=CODE_FIGURES_DIR,
    )
    figure_paths = [
        *comparison_outputs["figures"],
        *_generate_convergence_figures(),
        *_generate_ppc_figures(),
    ]
    exported = _export_to_report(figure_paths)

    print("Exported separate thesis figures:")
    for path in exported:
        print(path)


if __name__ == "__main__":
    main()

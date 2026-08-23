"""Create small audit tables and refresh figures used in the report."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from repeated_synthetic_regimes import (
    METHOD_ORDER,
    TARGET_NAMES,
    _save_regime_figure,
    _save_win_rate_figure,
    calculate_block_paired_comparisons,
)


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR.parent / "figures"
REPORT_FIGURES_DIR = (
    SCRIPT_DIR.parents[2] / "Report" / "final" / "final figure"
)


def make_validity_row(
    stage,
    nominal_draws,
    valid_draws,
    accepted_draws=None,
):
    """Summarise the draws kept by the common numerical validity rule."""

    invalid_draws = nominal_draws - valid_draws
    row = {
        "stage": stage,
        "nominal_draws": nominal_draws,
        "valid_draws": valid_draws,
        "invalid_draws": invalid_draws,
        "invalid_fraction": invalid_draws / nominal_draws,
        "accepted_draws": np.nan,
        "accepted_fraction_of_nominal": np.nan,
        "accepted_fraction_of_valid": np.nan,
    }
    if accepted_draws is not None:
        row.update(
            {
                "accepted_draws": accepted_draws,
                "accepted_fraction_of_nominal": (
                    accepted_draws / nominal_draws
                ),
                "accepted_fraction_of_valid": accepted_draws / valid_draws,
            }
        )
    return row


def build_validity_table():
    """Read saved metadata and build the simulation-validity table."""

    auto = json.loads(
        (SCRIPT_DIR / "auto_summary_training_metadata.json").read_text()
    )
    repeated = json.loads(
        (SCRIPT_DIR / "repeated_regime_metadata.json").read_text()
    )
    fixed = json.loads(
        (SCRIPT_DIR / "fixed_budget_metadata.json").read_text()
    )
    mean_fixed_valid = np.mean(
        [run["valid_draws"] for run in fixed["bank_runs"]]
    )

    rows = [
        make_validity_row(
            "Auto summary training bank",
            auto["training_simulations"],
            auto["valid_simulations"],
        ),
        make_validity_row(
            "Manual summary scale bank",
            repeated["scale_simulations"],
            repeated["scale_valid_draws"],
        ),
        make_validity_row(
            "Shared 10000-draw rejection bank",
            repeated["rejection_simulations"],
            repeated["shared_bank_valid_draws"],
            repeated["accepted_count"],
        ),
        make_validity_row(
            "Mean cost-study 20000-draw bank",
            max(fixed["rejection_budgets"]),
            mean_fixed_valid,
            max(fixed["rejection_budgets"])
            * fixed["acceptance_fraction"],
        ),
    ]
    return pd.DataFrame(rows)


def refresh_repeated_study_outputs():
    """Rebuild the paired table and repeated-study figures from saved CSVs."""

    raw = pd.read_csv(SCRIPT_DIR / "repeated_regime_raw_results.csv")
    summary = pd.read_csv(SCRIPT_DIR / "repeated_regime_summary.csv")
    win_rates = pd.read_csv(SCRIPT_DIR / "repeated_regime_win_rates.csv")
    comparisons = calculate_block_paired_comparisons(raw, METHOD_ORDER)
    comparisons.to_csv(
        SCRIPT_DIR / "repeated_regime_block_comparisons.csv",
        index=False,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGET_NAMES:
        path = FIGURES_DIR / f"repeated_regime_{target}.png"
        _save_regime_figure(summary, target, METHOD_ORDER, path)
        if (REPORT_FIGURES_DIR / path.name).exists():
            shutil.copy2(path, REPORT_FIGURES_DIR / path.name)
    win_path = FIGURES_DIR / "repeated_regime_win_rates.png"
    _save_win_rate_figure(win_rates, METHOD_ORDER, win_path)
    shutil.copy2(win_path, REPORT_FIGURES_DIR / win_path.name)
    return comparisons


def main():
    validity = build_validity_table()
    validity.to_csv(SCRIPT_DIR / "simulation_validity_summary.csv", index=False)
    comparisons = refresh_repeated_study_outputs()
    print(validity.to_string(index=False))
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()

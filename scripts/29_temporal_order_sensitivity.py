"""Run the formal temporal-order sensitivity experiment."""

import shutil
import time

import joblib
import numpy as np
import pandas as pd

from project_paths import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_DIR,
    SCRIPT_DIR,
    script_output,
)
from sv_abc_core import estimate_prior_predictive_summary_scale, manual_summary
from temporal_order_sensitivity import (
    evaluate_temporal_sensitivity,
    save_temporal_sensitivity_outputs,
    summarise_temporal_sensitivity,
)


BLOCK_LENGTHS = (1, 5, 20, 100)
N_PERMUTATIONS = 100
SCALE_SIMULATIONS = 2_000
SYNTHETIC_SEED = 42
REAL_SEED = 43
SCALE_SEED = 202607

FIGURE_NAMES = (
    "temporal_sensitivity_synthetic.png",
    "temporal_sensitivity_sp500.png",
)


def _analyse_dataset(returns, dataset_name, model_bundle, seed, scale_seed):
    """Estimate one Manual scale and analyse one return series."""

    manual_scale, scale_counts = estimate_prior_predictive_summary_scale(
        length=len(returns),
        n_simulations=SCALE_SIMULATIONS,
        random_seed=scale_seed,
    )
    raw_results = evaluate_temporal_sensitivity(
        returns,
        model_bundle,
        manual_scale=manual_scale,
        dataset_name=dataset_name,
        block_lengths=BLOCK_LENGTHS,
        n_permutations=N_PERMUTATIONS,
        random_seed=seed,
    )
    scale_information = {
        "scale": manual_scale.tolist(),
        "counts": scale_counts,
        "series_length": len(returns),
        "permutation_seed": seed,
        "observed_manual_summary": dict(
            zip(
                ["variance", "kurtosis", "squared_acf_1", "squared_acf_5"],
                manual_summary(returns).tolist(),
            )
        ),
    }
    return raw_results, scale_information


def main():
    """Analyse synthetic and S&P 500 returns and save all outputs."""

    start_time = time.perf_counter()
    model_path = script_output("rf_sv_summary.pkl")
    model_bundle = joblib.load(model_path)
    synthetic_returns = np.load(script_output("synthetic_sv_returns.npy"))
    real_frame = pd.read_csv(PROCESSED_DATA_DIR / "sp500_returns.csv")
    real_returns = real_frame["Return"].to_numpy(dtype=float)

    synthetic_raw, synthetic_scale = _analyse_dataset(
        synthetic_returns,
        "synthetic",
        model_bundle,
        SYNTHETIC_SEED,
        SCALE_SEED,
    )
    real_raw, real_scale = _analyse_dataset(
        real_returns,
        "sp500",
        model_bundle,
        REAL_SEED,
        SCALE_SEED + 1,
    )
    raw_results = pd.concat([synthetic_raw, real_raw], ignore_index=True)
    summary = summarise_temporal_sensitivity(raw_results)

    metadata = {
        "datasets": {
            "synthetic": synthetic_scale,
            "sp500": real_scale,
        },
        "block_lengths": list(BLOCK_LENGTHS),
        "permutations_per_block_length": N_PERMUTATIONS,
        "methods": ["Manual ABC", "Auto ABC", "Wasserstein ABC"],
        "auto_model_file": str(model_path),
        "auto_feature_group": model_bundle["feature_group"],
        "manual_scale_simulations": SCALE_SIMULATIONS,
        "relative_distance_definition": (
            "mean distance divided by the largest mean distance for each "
            "dataset and method; zero when all distances are numerical zero"
        ),
        "block_permutation_definition": (
            "consecutive blocks are kept intact and their order is randomly shuffled"
        ),
        "wasserstein_expectation": (
            "zero because every permutation contains the same return values"
        ),
        "posterior_shift_experiment": "not run; distance response is the primary test",
        "runtime_seconds": time.perf_counter() - start_time,
    }
    saved_paths = save_temporal_sensitivity_outputs(
        raw_results,
        summary,
        metadata=metadata,
        output_dir=SCRIPT_DIR,
        figures_dir=FIGURES_DIR,
    )

    report_figures_dir = (
        PROJECT_DIR.parent.parent / "Report" / "draft" / "figures"
    )
    report_figures_dir.mkdir(parents=True, exist_ok=True)
    for filename in FIGURE_NAMES:
        shutil.copy2(FIGURES_DIR / filename, report_figures_dir / filename)

    columns = [
        "dataset",
        "method",
        "block_length",
        "mean_distance",
        "relative_distance",
    ]
    print(summary[columns].round(6).to_string(index=False))
    print(f"\nSaved {len(saved_paths)} result files.")
    return raw_results, summary


if __name__ == "__main__":
    main()

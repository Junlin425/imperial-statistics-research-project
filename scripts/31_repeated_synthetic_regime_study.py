"""Run the formal repeated synthetic volatility-regime experiment."""

import platform
from pathlib import Path

import joblib
import sklearn

from abc_smc_utils import ABCSMCConfig
from project_paths import FIGURES_DIR, SCRIPT_DIR
from repeated_synthetic_regimes import (
    build_regime_table,
    run_repeated_regime_study,
    save_repeated_study_outputs,
)


SERIES_LENGTH = 4_000
REJECTION_SIMULATIONS = 10_000
ACCEPTANCE_FRACTION = 0.05
REJECTION_SEED = 42
SCALE_SIMULATIONS = 2_000
SCALE_SEED = 202607
SMC_WORKERS = 3

FORMAL_SMC_CONFIG = ABCSMCConfig(
    n_particles=500,
    n_pilot=2_000,
    max_populations=5,
    epsilon_quantile=0.50,
    max_attempts_per_population=50_000,
    random_seed=42,
    minimum_kernel_scale_fraction=0.01,
)


def main():
    """Run 30 datasets and save complete recovery results."""

    model_path = SCRIPT_DIR / "rf_sv_summary.pkl"
    model_bundle = joblib.load(model_path)
    if model_bundle["series_length"] != SERIES_LENGTH:
        raise ValueError("learned-summary model has the wrong series length")

    regimes = build_regime_table()
    study = run_repeated_regime_study(
        regimes,
        model_bundle,
        series_length=SERIES_LENGTH,
        rejection_simulations=REJECTION_SIMULATIONS,
        acceptance_fraction=ACCEPTANCE_FRACTION,
        rejection_seed=REJECTION_SEED,
        scale_simulations=SCALE_SIMULATIONS,
        scale_seed=SCALE_SEED,
        smc_config=FORMAL_SMC_CONFIG,
        smc_workers=SMC_WORKERS,
    )
    study["metadata"].update(
        {
            "model_file": str(model_path),
            "model_feature_group": model_bundle["feature_group"],
            "model_training_simulations": model_bundle[
                "training_simulations"
            ],
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
        }
    )

    outputs = save_repeated_study_outputs(
        study["raw_results"],
        study["regime_summary"],
        study["overall_summary"],
        study["win_rates"],
        study["posterior_samples"],
        study["posterior_weights"],
        study["metadata"],
        SCRIPT_DIR,
        FIGURES_DIR,
        method_order=(
            "Manual ABC",
            "Auto ABC",
            "Wasserstein ABC",
            "ABC-SMC",
        ),
        block_comparisons=study["block_comparisons"],
    )

    for table in outputs["tables"]:
        print(f"Saved: {table}")
    print(f"Saved: {outputs['posterior_file']}")
    print(f"Saved: {outputs['metadata']}")
    for figure in outputs["figures"]:
        print(f"Saved: {figure}")


if __name__ == "__main__":
    main()

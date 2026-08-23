"""Run the formal four-method fixed-budget comparison."""

import platform

import joblib
import numpy as np
import sklearn

from abc_smc_utils import ABCSMCConfig
from fixed_budget_study import (
    run_fixed_budget_study,
    save_fixed_budget_outputs,
)
from project_paths import FIGURES_DIR, SCRIPT_DIR
from repeated_synthetic_regimes import METHOD_ORDER


TRUE_PARAMETERS = np.array([-0.5, 0.95, 0.30])
REJECTION_BUDGETS = (2_000, 5_000, 10_000, 20_000)
SEEDS = tuple(range(10))
ACCEPTANCE_FRACTION = 0.05
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
    """Run ten repeated prefixes and ten ABC-SMC experiments."""

    observed_path = SCRIPT_DIR / "synthetic_sv_returns.npy"
    model_path = SCRIPT_DIR / "rf_sv_summary.pkl"
    observed_returns = np.load(observed_path)
    model_bundle = joblib.load(model_path)

    study = run_fixed_budget_study(
        observed_returns,
        TRUE_PARAMETERS,
        model_bundle,
        rejection_budgets=REJECTION_BUDGETS,
        rejection_seeds=SEEDS,
        acceptance_fraction=ACCEPTANCE_FRACTION,
        scale_simulations=SCALE_SIMULATIONS,
        scale_seed=SCALE_SEED,
        smc_config=FORMAL_SMC_CONFIG,
        smc_seeds=SEEDS,
        smc_workers=SMC_WORKERS,
    )
    study["metadata"].update(
        {
            "observed_file": str(observed_path),
            "true_parameters": TRUE_PARAMETERS.tolist(),
            "model_file": str(model_path),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
        }
    )

    outputs = save_fixed_budget_outputs(
        study["raw_results"],
        study["summary"],
        study["final_comparison"],
        study["metadata"],
        SCRIPT_DIR,
        FIGURES_DIR,
        METHOD_ORDER,
    )
    for table in outputs["tables"]:
        print(f"Saved: {table}")
    print(f"Saved: {outputs['metadata']}")
    for figure in outputs["figures"]:
        print(f"Saved: {figure}")


if __name__ == "__main__":
    main()

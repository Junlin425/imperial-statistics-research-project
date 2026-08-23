"""Run the formal learned-summary ablation experiment."""

import shutil

import numpy as np

from learned_summary_ablation import (
    run_learned_summary_ablation,
    save_ablation_outputs,
)
from project_paths import FIGURES_DIR, PROJECT_DIR, SCRIPT_DIR, script_output


TRAINING_SIMULATIONS = 5_000
TRAINING_SERIES_LENGTH = 4_000
INFERENCE_SIMULATIONS = 10_000
ACCEPTANCE_FRACTION = 0.05
RANDOM_SEED = 42
N_ESTIMATORS = 300
TRUE_PARAMETERS = np.array([-0.5, 0.95, 0.30])

ABLATION_FIGURES = (
    "ablation_validation_r2.png",
    "ablation_parameter_error.png",
    "ablation_half_life_error.png",
)


def main():
    """Run all three feature groups and save their results."""

    observed_returns = np.load(script_output("synthetic_sv_returns.npy"))
    outputs = run_learned_summary_ablation(
        observed_returns,
        TRUE_PARAMETERS,
        training_simulations=TRAINING_SIMULATIONS,
        training_series_length=TRAINING_SERIES_LENGTH,
        inference_simulations=INFERENCE_SIMULATIONS,
        acceptance_fraction=ACCEPTANCE_FRACTION,
        random_seed=RANDOM_SEED,
        n_estimators=N_ESTIMATORS,
    )
    saved_paths = save_ablation_outputs(
        outputs,
        output_dir=SCRIPT_DIR,
        figures_dir=FIGURES_DIR,
    )

    report_figures_dir = (
        PROJECT_DIR.parent.parent / "Report" / "draft" / "figures"
    )
    report_figures_dir.mkdir(parents=True, exist_ok=True)
    for filename in ABLATION_FIGURES:
        shutil.copy2(
            FIGURES_DIR / filename,
            report_figures_dir / filename,
        )

    print("Validation results:")
    print(outputs["validation"].round(4).to_string(index=False))
    print("\nSynthetic parameter recovery:")
    columns = [
        "feature_group",
        "parameter",
        "posterior_mean",
        "absolute_error",
        "covered",
    ]
    print(outputs["parameter_recovery"][columns].round(4).to_string(index=False))
    print("\nVolatility half-life recovery:")
    print(outputs["half_life_recovery"].round(4).to_string(index=False))
    print(f"\nSaved {len(saved_paths)} result files.")
    return outputs


if __name__ == "__main__":
    main()

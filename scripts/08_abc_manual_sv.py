"""Run Manual-summary rejection ABC on S&P 500 returns."""

import numpy as np
import pandas as pd

from manual_abc_experiment import (
    run_manual_rejection_abc,
    save_manual_abc_result,
)
from project_paths import PROCESSED_DATA_DIR, script_output


N_SIMULATIONS = 10_000
ACCEPTANCE_FRACTION = 0.05
RANDOM_SEED = 42
SCALE_SIMULATIONS = 2_000
SCALE_SEED = 202607


def run_real_experiment(
    input_path=None,
    sample_path=None,
    metadata_path=None,
):
    """Run and save the formal real-data Manual ABC experiment."""

    if input_path is None:
        input_path = PROCESSED_DATA_DIR / "sp500_returns.csv"
    if sample_path is None:
        sample_path = script_output("abc_manual_sv.npy")
    if metadata_path is None:
        metadata_path = script_output("abc_manual_sv_metadata.json")

    frame = pd.read_csv(input_path)
    if "Return" not in frame.columns:
        raise ValueError("real-data file must contain a Return column")
    observed_returns = frame["Return"].to_numpy(dtype=float)

    result = run_manual_rejection_abc(
        observed_returns,
        n_simulations=N_SIMULATIONS,
        acceptance_fraction=ACCEPTANCE_FRACTION,
        random_seed=RANDOM_SEED,
        scale_simulations=SCALE_SIMULATIONS,
        scale_seed=SCALE_SEED,
    )
    save_manual_abc_result(
        result,
        sample_path=sample_path,
        metadata_path=metadata_path,
        metadata={
            "method": "Manual ABC",
            "dataset": "S&P 500 log returns",
            "random_seed": RANDOM_SEED,
            "simulation_budget": N_SIMULATIONS,
            "scale_simulations": SCALE_SIMULATIONS,
            "total_simulator_calls": N_SIMULATIONS + SCALE_SIMULATIONS,
            "acceptance_fraction": ACCEPTANCE_FRACTION,
        },
    )

    posterior_mean = result.accepted[:, :3].mean(axis=0)
    print(f"Accepted samples: {len(result.accepted)}")
    print(f"Effective epsilon: {result.effective_epsilon:.8g}")
    print(f"alpha: {posterior_mean[0]:.8f}")
    print(f"beta: {posterior_mean[1]:.8f}")
    print(f"sigma_eta: {posterior_mean[2]:.8f}")
    print(
        "Valid/invalid inference simulations: "
        f"{result.valid_simulations}/{result.invalid_simulations}"
    )
    return result


if __name__ == "__main__":
    run_real_experiment()

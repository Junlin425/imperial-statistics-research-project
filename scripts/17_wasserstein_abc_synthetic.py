"""Run one-dimensional Wasserstein ABC on the stored synthetic SV series."""

import numpy as np

from project_paths import script_output
from wasserstein_abc_experiment import (
    run_wasserstein_rejection_abc,
    save_wasserstein_abc_result,
)


N_SIMULATIONS = 10_000
ACCEPTANCE_FRACTION = 0.05
RANDOM_SEED = 42


def run_synthetic_experiment(
    input_path=None,
    sample_path=None,
    metadata_path=None,
):
    """Run and save the formal synthetic Wasserstein ABC experiment."""

    if input_path is None:
        input_path = script_output("synthetic_sv_returns.npy")
    if sample_path is None:
        sample_path = script_output("abc_wasserstein_synthetic_sv.npy")
    if metadata_path is None:
        metadata_path = script_output(
            "abc_wasserstein_synthetic_sv_metadata.json"
        )

    observed_returns = np.asarray(np.load(input_path), dtype=float)
    result = run_wasserstein_rejection_abc(
        observed_returns,
        n_simulations=N_SIMULATIONS,
        acceptance_fraction=ACCEPTANCE_FRACTION,
        random_seed=RANDOM_SEED,
    )
    save_wasserstein_abc_result(
        result,
        sample_path=sample_path,
        metadata_path=metadata_path,
        metadata={
            "method": "Wasserstein ABC",
            "dataset": "synthetic SV data",
            "random_seed": RANDOM_SEED,
            "simulation_budget": N_SIMULATIONS,
            "total_simulator_calls": N_SIMULATIONS,
            "acceptance_fraction": ACCEPTANCE_FRACTION,
            "true_parameters": {
                "alpha": -0.5,
                "beta": 0.95,
                "sigma_eta": 0.30,
            },
        },
    )

    posterior_mean = result.accepted[:, :3].mean(axis=0)
    print(f"Accepted samples: {len(result.accepted)}")
    print(f"Effective epsilon: {result.effective_epsilon:.8g}")
    print(f"alpha: {posterior_mean[0]:.8f}")
    print(f"beta: {posterior_mean[1]:.8f}")
    print(f"sigma_eta: {posterior_mean[2]:.8f}")
    print(
        "Valid/invalid simulations: "
        f"{result.valid_simulations}/{result.invalid_simulations}"
    )
    return result


if __name__ == "__main__":
    run_synthetic_experiment()

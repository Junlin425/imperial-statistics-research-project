"""Repeated Manual ABC convergence study on one shared synthetic dataset."""

import time

import numpy as np
import pandas as pd

from abc_experiment_utils import (
    evaluate_convergence_prefixes,
    save_run_metadata,
)
from project_paths import script_output
from sv_abc_core import (
    estimate_prior_predictive_summary_scale,
    manual_summary,
    normalized_summary_distance,
    sample_prior,
    simulate_sv,
)


N_list = [1_000, 2_000, 5_000, 10_000, 20_000]
epsilon_list = [2.0, 1.0, 0.5, 0.2, 0.1]
CONVERGENCE_SEEDS = list(range(10))
TRUE_PARAMETERS = np.array([-0.5, 0.95, 0.30])
SCALE_SIMULATIONS = 2_000
SCALE_SEED = 202607


def run_convergence_study(input_path=None):
    """Generate repeated prefix-based Manual ABC convergence results."""

    if input_path is None:
        input_path = script_output("synthetic_sv_returns.npy")
    observed_returns = np.asarray(np.load(input_path), dtype=float)
    observed_summary = manual_summary(observed_returns)
    if observed_summary is None:
        raise ValueError("synthetic returns do not produce a valid summary")

    summary_scale, scale_counts = estimate_prior_predictive_summary_scale(
        length=len(observed_returns),
        n_simulations=SCALE_SIMULATIONS,
        random_seed=SCALE_SEED,
    )

    max_n = max(N_list)
    results = []
    start_time = time.perf_counter()

    for seed in CONVERGENCE_SEEDS:
        rng = np.random.default_rng(seed)
        parameters = sample_prior(rng, max_n)
        distances = np.full(max_n, np.finfo(float).max, dtype=float)

        for index, theta in enumerate(parameters):
            simulated_returns = simulate_sv(
                theta,
                len(observed_returns),
                rng,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                simulated_summary = manual_summary(simulated_returns)
            if simulated_summary is None:
                continue
            distances[index] = normalized_summary_distance(
                simulated_summary,
                observed_summary,
                scale=summary_scale,
            )

        results.extend(
            evaluate_convergence_prefixes(
                parameters=parameters,
                distances=distances,
                n_values=N_list,
                epsilon_values=epsilon_list,
                truth=TRUE_PARAMETERS,
                seed=seed,
            )
        )
        print(f"Completed seed {seed}: {max_n} simulator calls")

    raw_results = pd.DataFrame(results)
    raw_path = script_output("convergence_study_raw.csv")
    raw_results.to_csv(raw_path, index=False)

    summary = raw_results.groupby(
        ["N", "epsilon"],
        as_index=False,
    ).agg(
        accepted=("accepted", "mean"),
        accepted_std=("accepted", "std"),
        alpha_mean=("alpha_mean", "mean"),
        alpha_mean_std=("alpha_mean", "std"),
        beta_mean=("beta_mean", "mean"),
        beta_mean_std=("beta_mean", "std"),
        sigma_eta_mean=("sigma_eta_mean", "mean"),
        sigma_eta_mean_std=("sigma_eta_mean", "std"),
        alpha_abs_error=("alpha_abs_error", "mean"),
        alpha_abs_error_std=("alpha_abs_error", "std"),
        beta_abs_error=("beta_abs_error", "mean"),
        beta_abs_error_std=("beta_abs_error", "std"),
        sigma_eta_abs_error=("sigma_eta_abs_error", "mean"),
        sigma_eta_abs_error_std=("sigma_eta_abs_error", "std"),
    )
    summary_path = script_output("convergence_study.csv")
    summary.to_csv(summary_path, index=False)

    inference_calls = len(CONVERGENCE_SEEDS) * max_n
    save_run_metadata(
        script_output("convergence_study_metadata.json"),
        {
            "method": "Manual ABC convergence study",
            "random_seeds": CONVERGENCE_SEEDS,
            "simulation_pool_per_seed": max_n,
            "inference_simulator_calls": inference_calls,
            "scale_simulations": SCALE_SIMULATIONS,
            "total_simulator_calls": inference_calls + SCALE_SIMULATIONS,
            "scale_seed": SCALE_SEED,
            "scale_valid_simulations": scale_counts["valid"],
            "scale_invalid_simulations": scale_counts["invalid"],
            "summary_scale": summary_scale.tolist(),
            "summary_scale_method": (
                "1.4826 times componentwise MAD from a fixed "
                "prior-predictive bank, with IQR/SD fallback"
            ),
            "N_values": N_list,
            "epsilon_values": epsilon_list,
            "runtime_seconds": time.perf_counter() - start_time,
            "raw_output_file": str(raw_path),
            "summary_output_file": str(summary_path),
        },
    )
    return raw_results, summary


if __name__ == "__main__":
    run_convergence_study()

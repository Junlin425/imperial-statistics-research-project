"""Run adaptive Manual-summary ABC-SMC on real S&P 500 returns."""

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from abc_smc_utils import (
    ABCSMCConfig,
    run_abc_smc,
    save_abc_smc_result,
)
from sv_abc_core import (
    PRIOR_BOUNDS,
    estimate_prior_predictive_summary_scale,
    make_distance_simulator,
    sample_prior,
    uniform_prior_log_density,
)


N_PARTICLES = 500
N_PILOT = 2_000
MAX_POPULATIONS = 5
EPSILON_QUANTILE = 0.50
MAX_ATTEMPTS_PER_POPULATION = 50_000
RANDOM_SEED = 42
MIN_KERNEL_SCALE_FRACTION = 0.01
SCALE_SIMULATIONS = 2_000
SCALE_SEED = 202607

SCRIPT_DIR = Path(__file__).resolve().parent
FORMAL_CONFIG = ABCSMCConfig(
    n_particles=N_PARTICLES,
    n_pilot=N_PILOT,
    max_populations=MAX_POPULATIONS,
    epsilon_quantile=EPSILON_QUANTILE,
    max_attempts_per_population=MAX_ATTEMPTS_PER_POPULATION,
    random_seed=RANDOM_SEED,
    minimum_kernel_scale_fraction=MIN_KERNEL_SCALE_FRACTION,
)


def _print_progress(
    population,
    epsilon,
    candidates,
    acceptance_rate,
    ess,
    cumulative_calls,
):
    print(
        f"Population {population}: "
        f"epsilon={epsilon:.8g}, "
        f"candidates={candidates}, "
        f"acceptance_rate={acceptance_rate:.4f}, "
        f"ESS={ess:.2f}, "
        f"cumulative_calls={cumulative_calls}",
        flush=True,
    )


def run_real_experiment(
    config=FORMAL_CONFIG,
    input_path=None,
    output_prefix=None,
):
    """Run and save one real-data ABC-SMC experiment."""

    if input_path is None:
        input_path = (
            SCRIPT_DIR.parent
            / "data"
            / "processed"
            / "sp500_returns.csv"
        )
    if output_prefix is None:
        output_prefix = SCRIPT_DIR / "abc_smc_sv"

    frame = pd.read_csv(input_path)
    if "Return" not in frame.columns:
        raise ValueError("real-data file must contain a Return column")
    observed_returns = frame["Return"].to_numpy(dtype=float)
    if (
        observed_returns.ndim != 1
        or len(observed_returns) < 6
        or not np.all(np.isfinite(observed_returns))
    ):
        raise ValueError("real returns must be a finite vector")

    scale_start = time.perf_counter()
    summary_scale, scale_counts = estimate_prior_predictive_summary_scale(
        length=len(observed_returns),
        n_simulations=SCALE_SIMULATIONS,
        random_seed=SCALE_SEED,
    )
    scale_seconds = time.perf_counter() - scale_start

    start_time = time.perf_counter()
    result = run_abc_smc(
        config=config,
        bounds=PRIOR_BOUNDS,
        prior_sampler=sample_prior,
        prior_log_density=uniform_prior_log_density,
        distance_simulator=make_distance_simulator(
            observed_returns,
            summary_scale=summary_scale,
        ),
        progress_callback=_print_progress,
    )
    runtime_seconds = time.perf_counter() - start_time

    paths = save_abc_smc_result(
        result,
        output_prefix,
        {
            "method": "ABC-SMC with Manual summaries",
            "dataset": "S&P 500 log returns",
            "runtime_seconds": runtime_seconds,
            "scale_seconds": scale_seconds,
            "random_seed": config.random_seed,
            "scale_seed": SCALE_SEED,
            "scale_simulations": SCALE_SIMULATIONS,
            "scale_valid_simulations": scale_counts["valid"],
            "scale_invalid_simulations": scale_counts["invalid"],
            "summary_scale": summary_scale.tolist(),
            "summary_scale_method": (
                "1.4826 times componentwise MAD from a fixed "
                "prior-predictive bank, with IQR/SD fallback"
            ),
            "total_simulator_calls_including_scale": (
                result.total_simulator_calls + SCALE_SIMULATIONS
            ),
            "configuration": asdict(config),
        },
    )

    final_mean = np.sum(
        result.particles[-1] * result.weights[-1, :, None],
        axis=0,
    )
    print("\nFinal weighted posterior means")
    print(f"alpha: {final_mean[0]:.8f}")
    print(f"beta: {final_mean[1]:.8f}")
    print(f"sigma_eta: {final_mean[2]:.8f}")
    print(f"runtime_seconds: {runtime_seconds:.3f}")
    print(f"scale_seconds: {scale_seconds:.3f}")
    print(f"stop_reason: {result.stop_reason}")

    return result, paths


if __name__ == "__main__":
    run_real_experiment()

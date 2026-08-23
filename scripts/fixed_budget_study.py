"""Fixed-simulator-budget comparison for the four ABC methods."""

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abc_experiment_utils import select_top_fraction
from abc_smc_utils import ABCSMCConfig, run_abc_smc
from repeated_synthetic_regimes import (
    METHOD_ORDER,
    PARAMETER_NAMES,
    TARGET_LABELS,
    TARGET_NAMES,
    aggregate_recovery_results,
    build_shared_rejection_bank,
    posterior_recovery_rows,
    rejection_distances_from_bank,
)
from sv_abc_core import (
    PRIOR_BOUNDS,
    estimate_prior_predictive_summary_scale,
    make_distance_simulator,
    sample_prior,
    uniform_prior_log_density,
)


def select_rejection_budget_posteriors(
    parameters,
    distances,
    budgets,
    acceptance_fraction,
):
    """Select fixed-fraction posteriors from prefixes of one bank."""

    parameters = np.asarray(parameters, dtype=float)
    distances = np.asarray(distances, dtype=float)
    posteriors = []
    for budget in budgets:
        if not isinstance(budget, int) or budget <= 0 or budget > len(parameters):
            raise ValueError("each budget must lie inside the simulation bank")
        accepted, epsilon = select_top_fraction(
            parameters[:budget],
            distances[:budget],
            acceptance_fraction,
        )
        samples = accepted[:, :3]
        posteriors.append(
            {
                "samples": samples,
                "weights": np.full(len(samples), 1.0 / len(samples)),
                "epsilon": epsilon,
                "simulator_calls": budget,
            }
        )
    return posteriors


def aggregate_fixed_budget_results(raw_results):
    """Summarise repeated errors at every budget or SMC population."""

    return aggregate_recovery_results(
        raw_results,
        group_columns=(
            "method",
            "target",
            "target_type",
            "stage_type",
            "stage_index",
        ),
    )


def _run_smc_history(observed_returns, summary_scale, config, seed):
    seeded_config = replace(config, random_seed=int(seed))
    start_time = time.perf_counter()
    result = run_abc_smc(
        config=seeded_config,
        bounds=PRIOR_BOUNDS,
        prior_sampler=sample_prior,
        prior_log_density=uniform_prior_log_density,
        distance_simulator=make_distance_simulator(
            observed_returns,
            summary_scale=summary_scale,
        ),
    )
    return result, time.perf_counter() - start_time, int(seed)


def _run_smc_seed_jobs(
    observed_returns,
    summary_scale,
    config,
    seeds,
    workers,
):
    if workers == 1:
        return [
            _run_smc_history(
                observed_returns,
                summary_scale,
                config,
                seed,
            )
            for seed in seeds
        ]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_smc_history,
                observed_returns,
                summary_scale,
                config,
                seed,
            )
            for seed in seeds
        ]
        results = []
        for index, future in enumerate(futures, start=1):
            result = future.result()
            print(
                f"Completed fixed-budget ABC-SMC seed {index}/"
                f"{len(futures)}",
                flush=True,
            )
            results.append(result)
    return results


def run_fixed_budget_study(
    observed_returns,
    true_parameters,
    model_bundle,
    rejection_budgets=(2_000, 5_000, 10_000, 20_000),
    rejection_seeds=tuple(range(10)),
    acceptance_fraction=0.05,
    scale_simulations=2_000,
    scale_seed=202607,
    smc_config=ABCSMCConfig(),
    smc_seeds=tuple(range(10)),
    smc_workers=1,
):
    """Compare rejection prefixes with ABC-SMC population costs."""

    observed_returns = np.asarray(observed_returns, dtype=float)
    true_parameters = np.asarray(true_parameters, dtype=float)
    if observed_returns.ndim != 1 or len(observed_returns) < 11:
        raise ValueError("observed_returns must contain at least 11 values")
    if true_parameters.shape != (3,):
        raise ValueError("true_parameters must contain three values")
    if not rejection_budgets or tuple(sorted(rejection_budgets)) != tuple(
        rejection_budgets
    ):
        raise ValueError("rejection_budgets must be increasing")
    if not rejection_seeds or not smc_seeds:
        raise ValueError("both method families need at least one seed")
    if not isinstance(smc_workers, int) or smc_workers <= 0:
        raise ValueError("smc_workers must be a positive integer")

    summary_scale, scale_counts = estimate_prior_predictive_summary_scale(
        length=len(observed_returns),
        n_simulations=scale_simulations,
        random_seed=scale_seed,
    )
    raw_rows = []
    bank_timings = []
    maximum_budget = max(rejection_budgets)

    for seed_index, seed in enumerate(rejection_seeds, start=1):
        start_time = time.perf_counter()
        bank = build_shared_rejection_bank(
            model_bundle,
            len(observed_returns),
            maximum_budget,
            int(seed),
        )
        distances_by_method = rejection_distances_from_bank(
            observed_returns,
            bank,
            model_bundle,
            summary_scale,
        )
        elapsed = time.perf_counter() - start_time
        bank_timings.append(
            {
                "seed": int(seed),
                "valid_draws": int(len(bank["valid_indices"])),
                "invalid_draws": int(
                    maximum_budget - len(bank["valid_indices"])
                ),
                "runtime_seconds": float(elapsed),
            }
        )

        for method in METHOD_ORDER[:3]:
            posteriors = select_rejection_budget_posteriors(
                bank["parameters"],
                distances_by_method[method],
                rejection_budgets,
                acceptance_fraction,
            )
            for stage_index, posterior in enumerate(posteriors):
                raw_rows.extend(
                    posterior_recovery_rows(
                        posterior["samples"],
                        posterior["weights"],
                        true_parameters,
                        {
                            "seed": int(seed),
                            "stage_type": "rejection_budget",
                            "stage_index": int(stage_index),
                        },
                        method,
                        posterior["simulator_calls"],
                    )
                )
        print(
            f"Completed fixed-budget rejection seed {seed_index}/"
            f"{len(rejection_seeds)}",
            flush=True,
        )

    smc_runs = _run_smc_seed_jobs(
        observed_returns,
        summary_scale,
        smc_config,
        smc_seeds,
        smc_workers,
    )
    smc_diagnostics = []
    for result, runtime_seconds, seed in smc_runs:
        for population in range(result.completed_populations):
            raw_rows.extend(
                posterior_recovery_rows(
                    result.particles[population],
                    result.weights[population],
                    true_parameters,
                    {
                        "seed": seed,
                        "stage_type": "smc_population",
                        "stage_index": population,
                    },
                    "ABC-SMC",
                    int(result.cumulative_simulator_calls[population]),
                    runtime_seconds,
                )
            )
        smc_diagnostics.append(
            {
                "seed": seed,
                "completed_populations": result.completed_populations,
                "total_simulator_calls": result.total_simulator_calls,
                "runtime_seconds": runtime_seconds,
                "stop_reason": result.stop_reason,
                "final_ess": float(result.effective_sample_sizes[-1]),
                "final_epsilon": float(result.epsilons[-1]),
            }
        )

    raw_results = pd.DataFrame(raw_rows)
    training_calls = int(model_bundle.get("training_simulations", 0))
    raw_results["training_inclusive_calls"] = raw_results[
        "simulator_calls"
    ]
    auto_rows = raw_results["method"] == "Auto ABC"
    raw_results.loc[auto_rows, "training_inclusive_calls"] += training_calls

    summary = aggregate_fixed_budget_results(raw_results)
    final_comparison = (
        summary.sort_values("stage_index")
        .groupby(["method", "target"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    metadata = {
        "rejection_budgets": list(rejection_budgets),
        "rejection_seeds": [int(seed) for seed in rejection_seeds],
        "acceptance_fraction": float(acceptance_fraction),
        "shared_prefix_per_seed": True,
        "shared_bank_across_rejection_methods": True,
        "bank_runs": bank_timings,
        "learned_summary_training_calls": training_calls,
        "scale_simulations": int(scale_simulations),
        "scale_seed": int(scale_seed),
        "scale_valid_draws": int(scale_counts["valid"]),
        "scale_invalid_draws": int(scale_counts["invalid"]),
        "summary_scale": summary_scale.tolist(),
        "smc_seeds": [int(seed) for seed in smc_seeds],
        "smc_configuration": {
            key: getattr(smc_config, key)
            for key in smc_config.__dataclass_fields__
        },
        "smc_diagnostics": smc_diagnostics,
    }
    return {
        "raw_results": raw_results,
        "summary": summary,
        "final_comparison": final_comparison,
        "metadata": metadata,
    }


def _save_budget_figure(summary, target, method_order, output_path):
    target_rows = summary[summary["target"] == target]
    figure, axis = plt.subplots(figsize=(8.8, 5.6))

    for method in method_order:
        rows = target_rows[target_rows["method"] == method].sort_values(
            "mean_simulator_calls"
        )
        if rows.empty:
            continue
        medians = rows["median_scaled_absolute_error"].to_numpy()
        lower = medians - rows["q25_scaled_absolute_error"].to_numpy()
        upper = rows["q75_scaled_absolute_error"].to_numpy() - medians
        axis.errorbar(
            rows["mean_simulator_calls"],
            medians,
            yerr=np.vstack((lower, upper)),
            marker="o",
            capsize=4,
            linewidth=2,
            label=method,
        )

    axis.set_xscale("log")
    axis.set_xlabel("Inference simulator calls")
    axis.set_ylabel(
        "Prior-normalised absolute error"
        if target in PARAMETER_NAMES
        else "Relative absolute error"
    )
    axis.set_title(f"Fixed-budget recovery: {TARGET_LABELS[target]}")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_fixed_budget_outputs(
    raw_results,
    summary,
    final_comparison,
    metadata,
    output_dir,
    figures_dir,
    method_order,
):
    """Save fixed-budget tables, metadata, and separate figures."""

    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table_data = {
        "fixed_budget_raw_results.csv": raw_results,
        "fixed_budget_summary.csv": summary,
        "fixed_budget_final_comparison.csv": final_comparison,
    }
    table_paths = []
    for filename, frame in table_data.items():
        path = output_dir / filename
        pd.DataFrame(frame).to_csv(path, index=False)
        table_paths.append(path)

    figure_paths = []
    for target in TARGET_NAMES:
        path = figures_dir / f"fixed_budget_{target}.png"
        _save_budget_figure(summary, target, method_order, path)
        figure_paths.append(path)

    metadata_path = output_dir / "fixed_budget_metadata.json"
    complete_metadata = dict(metadata)
    complete_metadata.update(
        {
            "tables": [str(path) for path in table_paths],
            "figures": [str(path) for path in figure_paths],
            "method_order": list(method_order),
        }
    )
    metadata_path.write_text(
        json.dumps(complete_metadata, indent=2),
        encoding="utf-8",
    )
    return {
        "tables": table_paths,
        "metadata": metadata_path,
        "figures": figure_paths,
    }

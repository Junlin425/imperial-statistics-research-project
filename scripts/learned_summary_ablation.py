"""Compare marginal, temporal, and full learned summaries."""

from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abc_experiment_utils import save_run_metadata
from auto_abc_experiment import run_auto_rejection_abc
from auto_summary import train_auto_summary_model
from sv_abc_core import PRIOR_BOUNDS


FEATURE_GROUPS = ("marginal", "temporal", "full")
PARAMETER_NAMES = ("alpha", "beta", "sigma_eta")


def volatility_half_life(beta):
    """Convert volatility persistence into shock half-life in days."""

    beta = np.asarray(beta, dtype=float)
    if not np.all(np.isfinite(beta)) or np.any(beta <= 0.0) or np.any(beta >= 1.0):
        raise ValueError("beta must be finite and lie between 0 and 1")
    return np.log(0.5) / np.log(beta)


def _posterior_row(values, true_value):
    """Return the main recovery statistics for one posterior quantity."""

    lower, upper = np.quantile(values, [0.025, 0.975])
    posterior_mean = np.mean(values)
    return {
        "true_value": float(true_value),
        "posterior_mean": float(posterior_mean),
        "posterior_median": float(np.median(values)),
        "posterior_sd": float(np.std(values, ddof=1)),
        "q025": float(lower),
        "q975": float(upper),
        "bias": float(posterior_mean - true_value),
        "absolute_error": float(abs(posterior_mean - true_value)),
        "covered": bool(lower <= true_value <= upper),
    }


def _recovery_tables(accepted, feature_group, true_parameters):
    """Summarise raw parameters and volatility half-life."""

    samples = np.asarray(accepted, dtype=float)[:, :3]
    parameter_rows = []
    for index, parameter in enumerate(PARAMETER_NAMES):
        row = {
            "feature_group": feature_group,
            "parameter": parameter,
        }
        row.update(_posterior_row(samples[:, index], true_parameters[index]))
        prior_width = PRIOR_BOUNDS[index, 1] - PRIOR_BOUNDS[index, 0]
        row["normalised_absolute_error"] = (
            row["absolute_error"] / prior_width
        )
        parameter_rows.append(row)

    half_life_values = volatility_half_life(samples[:, 1])
    true_half_life = volatility_half_life(true_parameters[1]).item()
    half_life_row = {"feature_group": feature_group}
    half_life_row.update(_posterior_row(half_life_values, true_half_life))
    return parameter_rows, half_life_row


def run_learned_summary_ablation(
    observed_returns,
    true_parameters,
    feature_groups=FEATURE_GROUPS,
    training_simulations=5_000,
    training_series_length=4_000,
    inference_simulations=10_000,
    acceptance_fraction=0.05,
    random_seed=42,
    n_estimators=300,
):
    """Train and test each learned-summary feature group fairly."""

    true_parameters = np.asarray(true_parameters, dtype=float)
    if true_parameters.shape != (3,):
        raise ValueError("true_parameters must contain alpha, beta, and sigma_eta")
    if not feature_groups:
        raise ValueError("at least one feature group is required")

    models = {}
    posteriors = {}
    validation_frames = []
    parameter_rows = []
    half_life_rows = []
    training_details = {}

    for feature_group in feature_groups:
        bundle, metrics, details = train_auto_summary_model(
            n_simulations=training_simulations,
            series_length=training_series_length,
            feature_group=feature_group,
            random_seed=random_seed,
            n_estimators=n_estimators,
        )
        metrics.insert(0, "feature_group", feature_group)
        result = run_auto_rejection_abc(
            observed_returns,
            bundle,
            n_simulations=inference_simulations,
            acceptance_fraction=acceptance_fraction,
            random_seed=random_seed,
        )
        group_parameter_rows, half_life_row = _recovery_tables(
            result.accepted,
            feature_group,
            true_parameters,
        )

        models[feature_group] = bundle
        posteriors[feature_group] = result
        validation_frames.append(metrics)
        parameter_rows.extend(group_parameter_rows)
        half_life_rows.append(half_life_row)
        training_details[feature_group] = details

    return {
        "models": models,
        "posteriors": posteriors,
        "validation": pd.concat(validation_frames, ignore_index=True),
        "parameter_recovery": pd.DataFrame(parameter_rows),
        "half_life_recovery": pd.DataFrame(half_life_rows),
        "training_details": training_details,
        "settings": {
            "feature_groups": list(feature_groups),
            "true_parameters": true_parameters.tolist(),
            "training_simulations": training_simulations,
            "training_series_length": training_series_length,
            "inference_simulations": inference_simulations,
            "acceptance_fraction": acceptance_fraction,
            "random_seed": random_seed,
            "n_estimators": n_estimators,
        },
    }


def _grouped_bar_plot(frame, value_column, ylabel, output_path):
    """Save one clear grouped bar chart."""

    table = frame.pivot(
        index="parameter",
        columns="feature_group",
        values=value_column,
    )
    table = table.reindex([name for name in PARAMETER_NAMES if name in table.index])
    groups = [group for group in FEATURE_GROUPS if group in table.columns]
    table = table.reindex(columns=groups)
    x_positions = np.arange(len(table.index))
    width = 0.75 / len(groups)

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for group_index, group in enumerate(groups):
        offset = (group_index - (len(groups) - 1) / 2) * width
        axis.bar(
            x_positions + offset,
            table[group],
            width=width,
            label=group,
        )
    axis.set_xticks(x_positions, table.index)
    axis.set_ylabel(ylabel)
    axis.legend(title="Feature group")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _half_life_plot(frame, output_path):
    """Save the half-life absolute errors as a standalone chart."""

    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.bar(frame["feature_group"], frame["absolute_error"])
    axis.set_xlabel("Feature group")
    axis.set_ylabel("Half-life absolute error (days)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_ablation_outputs(outputs, output_dir, figures_dir):
    """Save ablation tables, models, posteriors, metadata, and figures."""

    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    table_files = {
        "validation": "learned_summary_ablation_validation.csv",
        "parameter_recovery": "learned_summary_ablation_parameter_recovery.csv",
        "half_life_recovery": "learned_summary_ablation_half_life.csv",
    }
    for key, filename in table_files.items():
        path = output_dir / filename
        outputs[key].to_csv(path, index=False)
        saved_paths.append(path)

    group_metadata = {}
    for feature_group, bundle in outputs["models"].items():
        model_path = output_dir / f"rf_sv_summary_{feature_group}.pkl"
        posterior_path = (
            output_dir
            / f"abc_auto_ablation_{feature_group}_synthetic.npy"
        )
        joblib.dump(bundle, model_path)
        result = outputs["posteriors"][feature_group]
        np.save(posterior_path, result.accepted)
        saved_paths.extend([model_path, posterior_path])
        group_metadata[feature_group] = {
            **outputs["training_details"][feature_group],
            "valid_inference_simulations": result.valid_simulations,
            "invalid_inference_simulations": result.invalid_simulations,
            "accepted_count": len(result.accepted),
            "effective_epsilon": result.effective_epsilon,
            "inference_seconds": result.runtime_seconds,
            "model_file": str(model_path),
            "posterior_file": str(posterior_path),
        }

    metadata_path = output_dir / "learned_summary_ablation_metadata.json"
    save_run_metadata(
        metadata_path,
        {
            **outputs["settings"],
            "groups": group_metadata,
        },
    )
    saved_paths.append(metadata_path)

    validation_figure = figures_dir / "ablation_validation_r2.png"
    error_figure = figures_dir / "ablation_parameter_error.png"
    half_life_figure = figures_dir / "ablation_half_life_error.png"
    _grouped_bar_plot(
        outputs["validation"],
        "r2",
        "Validation R-squared",
        validation_figure,
    )
    _grouped_bar_plot(
        outputs["parameter_recovery"],
        "normalised_absolute_error",
        "Prior-range normalised error",
        error_figure,
    )
    _half_life_plot(outputs["half_life_recovery"], half_life_figure)
    saved_paths.extend(
        [validation_figure, error_figure, half_life_figure]
    )
    return saved_paths

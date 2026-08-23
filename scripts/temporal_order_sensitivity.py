"""Test whether ABC distances respond to changes in time order."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abc_experiment_utils import save_run_metadata
from auto_summary import extract_features, scaled_prediction_distance
from sv_abc_core import manual_summary


METHOD_COLUMNS = {
    "manual_distance": "Manual ABC",
    "auto_distance": "Auto ABC",
    "wasserstein_distance": "Wasserstein ABC",
}

DATASET_TITLES = {
    "synthetic": "Synthetic returns",
    "sp500": "S&P 500 returns",
}


def block_permute(values, block_length, rng):
    """Randomly reorder consecutive blocks while keeping each block intact."""

    values = np.asarray(values)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not isinstance(block_length, (int, np.integer)) or block_length <= 0:
        raise ValueError("block_length must be a positive integer")
    if block_length > len(values):
        raise ValueError("block_length cannot exceed the series length")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    blocks = [
        values[start : start + block_length]
        for start in range(0, len(values), block_length)
    ]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[index] for index in order])


def evaluate_temporal_sensitivity(
    observed_returns,
    model_bundle,
    manual_scale,
    dataset_name,
    block_lengths=(1, 5, 20, 100),
    n_permutations=100,
    random_seed=42,
):
    """Measure three ABC distances after repeated block permutations."""

    observed_returns = np.asarray(observed_returns, dtype=float)
    manual_scale = np.asarray(manual_scale, dtype=float)
    if manual_scale.shape != (4,) or np.any(manual_scale <= 0.0):
        raise ValueError("manual_scale must contain four positive values")
    if not isinstance(n_permutations, int) or n_permutations <= 0:
        raise ValueError("n_permutations must be a positive integer")

    observed_manual = manual_summary(observed_returns)
    feature_group = model_bundle["feature_group"]
    observed_features = extract_features(observed_returns, feature_group)
    if observed_manual is None or observed_features is None:
        raise ValueError("observed_returns do not produce valid summaries")

    model = model_bundle["model"]
    prediction_scale = np.asarray(model_bundle["prediction_scale"], dtype=float)
    observed_prediction = model.predict(observed_features.reshape(1, -1))[0]
    rng = np.random.default_rng(random_seed)
    rows = []
    feature_rows = []

    for block_length in block_lengths:
        for permutation in range(1, n_permutations + 1):
            permuted = block_permute(observed_returns, block_length, rng)
            permuted_manual = manual_summary(permuted)
            permuted_features = extract_features(permuted, feature_group)
            if permuted_manual is None or permuted_features is None:
                raise RuntimeError("a permutation produced invalid summaries")

            rows.append(
                {
                    "dataset": dataset_name,
                    "block_length": int(block_length),
                    "permutation": permutation,
                    "manual_distance": float(
                        np.linalg.norm(
                            (permuted_manual - observed_manual) / manual_scale
                        )
                    ),
                    "auto_distance": np.nan,
                    "wasserstein_distance": float(
                        wasserstein_distance(observed_returns, permuted)
                    ),
                    "squared_acf_1": float(permuted_manual[2]),
                    "squared_acf_5": float(permuted_manual[3]),
                }
            )
            feature_rows.append(permuted_features)

    predictions = model.predict(np.asarray(feature_rows, dtype=float))
    for row, prediction in zip(rows, predictions):
        row["auto_distance"] = scaled_prediction_distance(
            prediction,
            observed_prediction,
            prediction_scale,
        )
    return pd.DataFrame(rows)


def summarise_temporal_sensitivity(raw_results):
    """Summarise permutation distances and calculate relative responses."""

    frames = []
    group_columns = ["dataset", "block_length"]
    for distance_column, method_name in METHOD_COLUMNS.items():
        grouped = raw_results.groupby(group_columns)[distance_column]
        frame = grouped.agg(
            mean_distance="mean",
            sd_distance="std",
            median_distance="median",
            q025=lambda values: values.quantile(0.025),
            q975=lambda values: values.quantile(0.975),
        ).reset_index()
        frame.insert(1, "method", method_name)
        frames.append(frame)

    summary = pd.concat(frames, ignore_index=True)
    summary["relative_distance"] = 0.0
    summary["relative_q025"] = 0.0
    summary["relative_q975"] = 0.0

    grouped_indices = summary.groupby(["dataset", "method"]).groups.values()
    for indices in grouped_indices:
        normalising_distance = summary.loc[indices, "mean_distance"].max()
        if normalising_distance <= 1e-12:
            continue
        summary.loc[indices, "relative_distance"] = (
            summary.loc[indices, "mean_distance"] / normalising_distance
        )
        summary.loc[indices, "relative_q025"] = (
            summary.loc[indices, "q025"] / normalising_distance
        )
        summary.loc[indices, "relative_q975"] = (
            summary.loc[indices, "q975"] / normalising_distance
        )
    return summary.sort_values(
        ["dataset", "method", "block_length"]
    ).reset_index(drop=True)


def _plot_dataset(summary, dataset_name, output_path):
    """Plot relative distance against block length for one dataset."""

    dataset_rows = summary[summary["dataset"] == dataset_name]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for method_name in METHOD_COLUMNS.values():
        rows = dataset_rows[dataset_rows["method"] == method_name].sort_values(
            "block_length"
        )
        x_values = rows["block_length"].to_numpy(dtype=float)
        y_values = rows["relative_distance"].to_numpy(dtype=float)
        axis.plot(x_values, y_values, marker="o", label=method_name)
        axis.fill_between(
            x_values,
            rows["relative_q025"].to_numpy(dtype=float),
            rows["relative_q975"].to_numpy(dtype=float),
            alpha=0.12,
        )

    block_lengths = sorted(dataset_rows["block_length"].unique())
    axis.set_xscale("log")
    axis.set_xticks(block_lengths, block_lengths)
    axis.set_xlabel("Block length")
    axis.set_ylabel("Relative distance")
    dataset_title = DATASET_TITLES.get(dataset_name, dataset_name)
    axis.set_title(f"Temporal-order sensitivity: {dataset_title}")
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_temporal_sensitivity_outputs(
    raw_results,
    summary,
    metadata,
    output_dir,
    figures_dir,
):
    """Save sensitivity tables, metadata, and one figure per dataset."""

    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "temporal_order_sensitivity_raw.csv"
    summary_path = output_dir / "temporal_order_sensitivity_summary.csv"
    metadata_path = output_dir / "temporal_order_sensitivity_metadata.json"
    raw_results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_run_metadata(metadata_path, metadata)
    saved_paths = [raw_path, summary_path, metadata_path]

    for dataset_name in sorted(summary["dataset"].unique()):
        safe_name = str(dataset_name).lower().replace(" ", "_")
        figure_path = figures_dir / f"temporal_sensitivity_{safe_name}.png"
        _plot_dataset(summary, dataset_name, figure_path)
        saved_paths.append(figure_path)
    return saved_paths

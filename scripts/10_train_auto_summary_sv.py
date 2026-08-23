"""Train the random forest used by Auto ABC."""

import joblib

from abc_experiment_utils import save_run_metadata
from auto_summary import train_auto_summary_model
from project_paths import script_output


N_SIMULATIONS = 5_000
TRAINING_SERIES_LENGTH = 4_000
FEATURE_GROUP = "full"
N_ESTIMATORS = 300
RANDOM_SEED = 42


def main():
    """Train and save the full-feature learned-summary model."""

    bundle, metrics, details = train_auto_summary_model(
        n_simulations=N_SIMULATIONS,
        series_length=TRAINING_SERIES_LENGTH,
        feature_group=FEATURE_GROUP,
        random_seed=RANDOM_SEED,
        n_estimators=N_ESTIMATORS,
    )

    model_path = script_output("rf_sv_summary.pkl")
    metrics_path = script_output("auto_summary_validation_metrics.csv")
    metadata_path = script_output("auto_summary_training_metadata.json")
    joblib.dump(bundle, model_path)
    metrics.to_csv(metrics_path, index=False)

    save_run_metadata(
        metadata_path,
        {
            "method": "Random-forest learned summary",
            "feature_group": FEATURE_GROUP,
            "feature_names": bundle["feature_names"],
            "random_seed": RANDOM_SEED,
            "training_simulations": N_SIMULATIONS,
            "training_series_length": TRAINING_SERIES_LENGTH,
            "n_estimators": N_ESTIMATORS,
            "prediction_scale": bundle["prediction_scale"].tolist(),
            "prediction_scale_source": bundle[
                "prediction_scale_source"
            ],
            "distance_scaling": (
                "robust scale of held-out validation predictions"
            ),
            "model_file": str(model_path),
            "metrics_file": str(metrics_path),
            **details,
        },
    )

    print(metrics.round(4).to_string(index=False))
    print(f"Valid simulations: {details['valid_simulations']}")
    print(f"Invalid simulations: {details['invalid_simulations']}")
    print(f"Saved model: {model_path}")
    return bundle, metrics, details


if __name__ == "__main__":
    main()

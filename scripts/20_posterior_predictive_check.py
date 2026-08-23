"""Run the canonical four-method posterior predictive analysis."""

from posterior_predictive_utils import (
    DEFAULT_REPLICATIONS,
    DEFAULT_SEED,
    run_posterior_predictive_analysis,
)


def main():
    result = run_posterior_predictive_analysis(
        n_replications=DEFAULT_REPLICATIONS,
        seed=DEFAULT_SEED,
    )

    print("\n" + "=" * 72)
    print("POSTERIOR PREDICTIVE SUMMARY")
    print("=" * 72)
    print(result["summary_table"].round(6).to_string(index=False))

    print("\n" + "=" * 72)
    print("POSTERIOR PREDICTIVE P-VALUES")
    print("=" * 72)
    print(result["pvalue_table"].round(6).to_string(index=False))

    print(
        f"\nCompleted {DEFAULT_REPLICATIONS:,} replications per method "
        f"with seed {DEFAULT_SEED} in "
        f"{result['runtime_seconds']:.2f} seconds."
    )
    print("\nSaved outputs:")
    for output_path in result["paths"].values():
        print(output_path)


if __name__ == "__main__":
    main()

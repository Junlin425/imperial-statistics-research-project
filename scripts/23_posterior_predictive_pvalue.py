"""Compatibility entry point for posterior predictive p-values."""

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
    print("POSTERIOR PREDICTIVE P-VALUES")
    print("=" * 72)
    print(result["pvalue_table"].round(6).to_string(index=False))
    print(
        f"\nCompleted {DEFAULT_REPLICATIONS:,} replications per method "
        f"with seed {DEFAULT_SEED}."
    )
    print(f"Saved: {result['paths']['pvalue_table']}")


if __name__ == "__main__":
    main()

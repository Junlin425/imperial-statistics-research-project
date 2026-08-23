"""Generate financial interpretations of the saved ABC posteriors."""

from abc_smc_comparison import METHOD_ORDER, load_all_results
from financial_derived_quantities import run_financial_analysis
from project_paths import FIGURES_DIR, SCRIPT_DIR


def main():
    """Transform each posterior draw and save the resulting analysis."""

    results = load_all_results(SCRIPT_DIR)
    outputs = run_financial_analysis(
        results,
        SCRIPT_DIR,
        FIGURES_DIR,
        METHOD_ORDER,
    )
    print(f"Saved: {outputs['summary_table']}")
    print(f"Saved: {outputs['synthetic_error_table']}")
    print(f"Saved: {outputs['metadata']}")
    for figure in outputs["figures"]:
        print(f"Saved: {figure}")


if __name__ == "__main__":
    main()

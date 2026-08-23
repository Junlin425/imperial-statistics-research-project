"""Regenerate result tables from the saved posterior files."""

from abc_smc_comparison import run_comparison
from project_paths import FIGURES_DIR, SCRIPT_DIR


def main():
    """Build current tables instead of using hard-coded posterior means."""

    outputs = run_comparison(SCRIPT_DIR, FIGURES_DIR)
    print(f"Saved: {outputs['comparison_table']}")
    print(f"Saved: {outputs['synthetic_error_table']}")
    return outputs


if __name__ == "__main__":
    main()

"""Generate comparison tables and figures after both ABC-SMC runs."""

from pathlib import Path

from abc_smc_comparison import run_comparison


SCRIPT_DIR = Path(__file__).resolve().parent


if __name__ == "__main__":
    outputs = run_comparison(
        scripts_dir=SCRIPT_DIR,
        figures_dir=SCRIPT_DIR.parent / "figures",
    )
    print("Saved comparison outputs:")
    print(outputs["comparison_table"])
    print(outputs["synthetic_error_table"])
    for figure in outputs["figures"]:
        print(figure)

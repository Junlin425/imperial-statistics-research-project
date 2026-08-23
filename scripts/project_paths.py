"""Stable filesystem locations for the stochastic-volatility project."""

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_DIR / "figures"


def script_output(filename):
    """Return a file path inside ``scripts`` independent of the CWD."""

    filename = Path(filename)
    if filename.is_absolute() or len(filename.parts) != 1:
        raise ValueError("filename must be one file name without directories")
    return SCRIPT_DIR / filename

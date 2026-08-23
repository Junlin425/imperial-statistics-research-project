import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import acf

from project_paths import PROCESSED_DATA_DIR

# =====================================
# Summary Statistics Function
# =====================================

def summary_statistics(y):

    variance = np.var(y)

    kurt = kurtosis(
        y,
        fisher=False
    )

    squared_returns = y**2

    acf_values = acf(
        squared_returns,
        nlags=5,
        fft=False
    )

    acf_lag1 = acf_values[1]

    acf_lag5 = acf_values[5]

    summary = np.array([
        variance,
        kurt,
        acf_lag1,
        acf_lag5
    ])

    return summary


# =====================================
# Load Real Data
# =====================================

df = pd.read_csv(
    PROCESSED_DATA_DIR / "sp500_returns.csv"
)

returns = df["Return"].values

# =====================================
# Compute Summary
# =====================================

S_real = summary_statistics(
    returns
)

print("\nS&P500 Summary Statistics")

print("--------------------------------")

print("Variance :", S_real[0])

print("Kurtosis :", S_real[1])

print("ACF Lag1 :", S_real[2])

print("ACF Lag5 :", S_real[3])

import pandas as pd
import numpy as np

from project_paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

# =====================================
# Load raw data
# =====================================

df = pd.read_csv(
    RAW_DATA_DIR / "sp500_raw.csv",
)

# =====================================
# Log Returns
# =====================================

df["Return"] = np.log(
    df["Close"] /
    df["Close"].shift(1)
)

# =====================================
# Remove NA
# =====================================

df = df.dropna()

# =====================================
# Keep useful columns
# =====================================

returns = df[["Date", "Return"]]

# =====================================
# Save
# =====================================

returns.to_csv(
    PROCESSED_DATA_DIR / "sp500_returns.csv",
    index=False
)

print(returns.head())

print("\nObservations:", len(returns))

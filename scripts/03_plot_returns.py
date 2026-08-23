import pandas as pd
import matplotlib.pyplot as plt

from project_paths import PROCESSED_DATA_DIR

# =====================================
# Load returns
# =====================================

df = pd.read_csv(
    PROCESSED_DATA_DIR / "sp500_returns.csv"
)

# =====================================
# Plot returns
# =====================================

plt.figure(figsize=(12,5))

plt.plot(
    df["Return"]
)

plt.title(
    "S&P500 Daily Log Returns"
)

plt.xlabel("Time")

plt.ylabel("Return")

plt.tight_layout()

plt.show()

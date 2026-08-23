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
# Squared returns
# =====================================

df["Squared_Return"] = (
    df["Return"] ** 2
)

# =====================================
# Plot
# =====================================

plt.figure(figsize=(12,5))

plt.plot(
    df["Squared_Return"]
)

plt.title(
    "Squared Returns"
)

plt.xlabel("Time")

plt.ylabel("Squared Return")

plt.tight_layout()

plt.show()

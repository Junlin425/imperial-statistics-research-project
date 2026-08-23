import yfinance as yf
import pandas as pd

from project_paths import RAW_DATA_DIR

# Download S&P500
sp500 = yf.download(
    "^GSPC",
    start="2010-01-01",
    end="2025-01-01",
    auto_adjust=True
)

# Only keep the 'Close' column
sp500 = sp500[["Close"]]

# Turn into a regular DataFrame
sp500 = sp500.reset_index()

# If the columns are a MultiIndex, flatten them
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.get_level_values(0)

# Save raw data
sp500.to_csv(
    RAW_DATA_DIR / "sp500_raw.csv",
    index=False
)

print(sp500.head())

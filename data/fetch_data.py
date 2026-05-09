"""
fetch_data.py
Downloads BANKBEES.NS historical OHLCV data from Yahoo Finance via yfinance.
Mirrors the exact logic used in BankBeES_Predictor.ipynb Cell 0.
"""

import yfinance as yf
import pandas as pd
import os

TICKER = "BANKBEES.NS"
START_DATE = "2020-01-01"
END_DATE = "2026-05-01"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "bankbees_historical.csv")


def fetch_data(ticker=TICKER, start=START_DATE, end=END_DATE, use_cache=False):
    """
    Download BANKBEES ETF data from Yahoo Finance.

    Parameters:
        ticker   : str  — Yahoo Finance ticker (default: BANKBEES.NS)
        start    : str  — Start date in YYYY-MM-DD format
        end      : str  — End date in YYYY-MM-DD format
        use_cache: bool — Load from local CSV if available

    Returns:
        pd.DataFrame with columns: Open, High, Low, Close, Volume
    """
    if use_cache and os.path.exists(CACHE_PATH):
        print(f"Loading cached data from {CACHE_PATH}")
        data = pd.read_csv(CACHE_PATH, index_col="Date", parse_dates=True)
        return data

    print(f"Fetching data for {ticker} from {start} to {end} ...")
    data = yf.download(ticker, start=start, end=end)

    # Flatten MultiIndex columns (yfinance >= 0.2.x)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    # Save to cache
    data.to_csv(CACHE_PATH)
    print(f"Data saved to {CACHE_PATH}")
    print(f"Total rows fetched: {len(data)}")

    return data


if __name__ == "__main__":
    df = fetch_data()
    print(df.head())
  

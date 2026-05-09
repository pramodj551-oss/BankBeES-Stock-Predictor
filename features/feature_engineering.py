"""
feature_engineering.py
Builds features used in BankBeES_Predictor.ipynb (Cell 1).

Features created:
    MA_10        — 10-day Moving Average of Close price
    MA_50        — 50-day Moving Average of Close price
    Daily_Return — Percentage change in Close vs previous day
    Target       — Tomorrow's Close price (shifted by -1, used as label)

Final feature columns fed to the model:
    ['Close', 'Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50', 'Daily_Return']
"""

import pandas as pd

FEATURE_COLUMNS = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50', 'Daily_Return']


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators and target variable to raw OHLCV data.

    Parameters:
        data : pd.DataFrame — Raw OHLCV dataframe from yfinance

    Returns:
        pd.DataFrame — Cleaned dataframe with features + Target column
    """
    df = data.copy()

    # Flatten MultiIndex if needed (yfinance >= 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # 1. Moving Averages
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # 2. Daily Return — % change vs previous close
    df['Daily_Return'] = df['Close'].pct_change()

    # 3. Target — Tomorrow's closing price (supervised label)
    df['Target'] = df['Close'].shift(-1)

    # Drop NaN rows caused by rolling (50 rows) and shift (1 row)
    df = df.dropna().copy()

    print(f"Feature engineering complete. Rows after cleaning: {len(df)}")
    return df


def get_X_y(df: pd.DataFrame):
    """
    Split engineered dataframe into features (X) and target (y).

    Returns:
        X : pd.DataFrame — Feature matrix
        y : pd.Series   — Target vector (next day's close)
    """
    X = df[FEATURE_COLUMNS]
    y = df['Target']
    return X, y
  

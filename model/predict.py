"""
predict.py
Loads the saved Random Forest model and predicts the next day's
BANKBEES closing price from a single day's OHLCV + feature row.
"""

import pandas as pd
import numpy as np
from model.train_model import load_model
from features.feature_engineering import FEATURE_COLUMNS


def predict_next_price(latest_row: pd.DataFrame) -> float:
    """
    Predict tomorrow's closing price given today's feature row.

    Parameters:
        latest_row : pd.DataFrame — Single row with columns matching FEATURE_COLUMNS

    Returns:
        float — Predicted next-day closing price in ₹
    """
    model = load_model()

    # Ensure correct column order
    X = latest_row[FEATURE_COLUMNS]
    prediction = model.predict(X)[0]
    return round(float(prediction), 2)


def predict_from_df(df_features: pd.DataFrame) -> pd.Series:
    """
    Predict prices for an entire feature dataframe (used in Streamlit app).

    Parameters:
        df_features : pd.DataFrame — Feature matrix (same columns as training)

    Returns:
        pd.Series — Predicted prices indexed by date
    """
    model = load_model()
    X = df_features[FEATURE_COLUMNS]
    predictions = model.predict(X)
    return pd.Series(predictions, index=df_features.index, name="Predicted_Price")

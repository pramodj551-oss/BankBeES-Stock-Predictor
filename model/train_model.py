"""
train_model.py
Trains a Random Forest Regressor on BANKBEES ETF data.
Mirrors BankBeES_Predictor.ipynb Cells 2 & 3 exactly.

Results from notebook run:
    Training rows : 1,212
    Test rows     : 304
    MAE           : ₹34.09
    RMSE          : ₹41.90
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model.pkl")


def train(X: pd.DataFrame, y: pd.Series, n_estimators=100, random_state=42):
    """
    Train RandomForestRegressor and evaluate on test split.

    Parameters:
        X             : pd.DataFrame — Feature matrix
        y             : pd.Series   — Target (next day close)
        n_estimators  : int         — Number of trees (notebook used 100)
        random_state  : int         — Reproducibility seed

    Returns:
        model   : trained RandomForestRegressor
        results : pd.DataFrame with Actual_Price and Predicted_Price
        metrics : dict with MAE, RMSE
    """
    # Time-series split — NO shuffle (critical for time series!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"Rows available for training: {X_train.shape[0]}")
    print(f"Rows available for testing : {X_test.shape[0]}")

    # Train model
    print("\nTraining the model... please wait.")
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    print("Model training completed!")

    # Predict
    predictions = model.predict(X_test)

    # Results dataframe
    results = pd.DataFrame({
        'Actual_Price': y_test,
        'Predicted_Price': predictions
    }, index=X_test.index)

    # Metrics
    mae  = mean_absolute_error(results['Actual_Price'], results['Predicted_Price'])
    rmse = np.sqrt(mean_squared_error(results['Actual_Price'], results['Predicted_Price']))

    print("\n--- Model Evaluation (Regression Metrics) ---")
    print(f"Mean Absolute Error (MAE) : ₹{mae:.2f}")
    print(f"Root Mean Squared Error   : ₹{rmse:.2f}")

    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2)}

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return model, results, metrics


def load_model():
    """Load saved model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No saved model found at {MODEL_PATH}. "
            "Run train_model.py first."
        )
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from data.fetch_data import fetch_data
    from features.feature_engineering import add_features, get_X_y

    raw   = fetch_data()
    df    = add_features(raw)
    X, y  = get_X_y(df)
    model, results, metrics = train(X, y)
    print("\nSample predictions:")
    print(results.head())

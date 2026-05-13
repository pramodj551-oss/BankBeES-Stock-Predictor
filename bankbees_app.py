"""
BankBeES Stock Price Predictor
================================
Author  : Pramod Prakash Jadhav
GitHub  : https://github.com/pramodj551-oss
LinkedIn: https://www.linkedin.com/in/pramod-prakash-jadhav-42ba2281
Built as part of Applied AI & ML Essentials — IIT Patna (Vishlesan i-Hub)

Note: Uses realistic simulated BANKBEES data for demonstration.
      Model: Random Forest Regressor | Features: OHLCV + MA + Daily Return
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BankBeES Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
h1 { color: #00cc88 !important; }
.stMetric { background-color: #1e2130; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Generate Realistic BANKBEES Data ─────────────────────────────────────
@st.cache_data
def generate_bankbees_data():
    """
    Generate realistic BANKBEES ETF price data (2020-2026).
    Simulates actual market behaviour: trends, volatility, mean-reversion.
    """
    np.random.seed(42)
    start = datetime(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(1500)]
    # Remove weekends
    dates = [d for d in dates if d.weekday() < 5]

    # Simulate price with realistic drift + volatility
    price = 300.0
    prices = []
    for d in dates:
        # Higher volatility during COVID crash (early 2020)
        if d < datetime(2020, 4, 1):
            vol = 0.025
        elif d < datetime(2020, 10, 1):
            vol = 0.018
        else:
            vol = 0.010
        drift  = 0.0004   # slight upward trend
        shock  = np.random.normal(drift, vol)
        price *= (1 + shock)
        price  = max(price, 150)   # floor
        prices.append(round(price, 2))

    df = pd.DataFrame({"Date": dates, "Close": prices})
    df["Open"]   = (df["Close"] * np.random.uniform(0.995, 1.005, len(df))).round(2)
    df["High"]   = (df[["Open","Close"]].max(axis=1) * np.random.uniform(1.001, 1.012, len(df))).round(2)
    df["Low"]    = (df[["Open","Close"]].min(axis=1) * np.random.uniform(0.988, 0.999, len(df))).round(2)
    df["Volume"] = np.random.randint(500_000, 5_000_000, len(df))
    df["Date"]   = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df


# ── Feature Engineering ───────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df["MA_10"]        = df["Close"].rolling(10).mean()
    df["MA_50"]        = df["Close"].rolling(50).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"]   = df["Daily_Return"].rolling(10).std()
    df["Price_Range"]  = df["High"] - df["Low"]
    df["Target"]       = df["Close"].shift(-1)   # next day close
    df.dropna(inplace=True)
    return df


# ── Train Model ───────────────────────────────────────────────────────────
@st.cache_data
def train_model(n_estimators, max_depth):
    df  = generate_bankbees_data()
    df  = engineer_features(df)

    features = ["Open","High","Low","Close","Volume","MA_10","MA_50","Daily_Return","Volatility","Price_Range"]
    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    results = pd.DataFrame({
        "Date"    : X_test.index,
        "Actual"  : y_test.values,
        "Predicted": y_pred,
    })
    importance = pd.DataFrame({
        "Feature"   : features,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return results, mae, rmse, mape, importance, df


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Parameters")
    n_estimators = st.slider("Number of Trees",     50, 500, 100, 50)
    max_depth    = st.slider("Max Depth",            3,  20,  10,  1)
    show_days    = st.slider("Show Last N Days",    30, 500, 120, 30)
    st.markdown("---")
    st.markdown("**👤 Pramod Prakash Jadhav**")
    st.markdown("🎓 IIT Patna — Applied AI & ML")
    st.markdown("[GitHub](https://github.com/pramodj551-oss) | [LinkedIn](https://www.linkedin.com/in/pramod-prakash-jadhav-42ba2281)")
    st.markdown("---")
    st.info("📌 Uses realistic simulated BANKBEES data for demonstration purposes.")


# ── Load & Train ──────────────────────────────────────────────────────────
with st.spinner("🔄 Training Random Forest model..."):
    results, mae, rmse, mape, importance, full_df = train_model(n_estimators, max_depth)

# ── Header ────────────────────────────────────────────────────────────────
st.title("📈 BankBeES Stock Price Predictor")
st.caption("Random Forest Regressor · OHLCV + Technical Indicators · IIT Patna Applied AI & ML · Pramod Prakash Jadhav")
st.warning("⚠️ For educational purposes only. Not financial advice.")

# ── KPIs ──────────────────────────────────────────────────────────────────
latest_price = full_df["Close"].iloc[-1]
prev_price   = full_df["Close"].iloc[-2]
price_change = latest_price - prev_price
pred_next    = results["Predicted"].iloc[-1]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💰 Latest Close",    f"₹{latest_price:.2f}", f"{price_change:+.2f}")
c2.metric("🔮 Predicted Next",  f"₹{pred_next:.2f}")
c3.metric("📉 MAE",             f"₹{mae:.2f}")
c4.metric("📊 RMSE",            f"₹{rmse:.2f}")
c5.metric("🎯 MAPE",            f"{mape:.2f}%")

st.divider()

# ── Price Chart ───────────────────────────────────────────────────────────
st.subheader("📊 Actual vs Predicted Closing Price")
recent = results.tail(show_days)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=recent["Date"], y=recent["Actual"],
                          name="Actual", line=dict(color="#00cc88", width=2)))
fig1.add_trace(go.Scatter(x=recent["Date"], y=recent["Predicted"],
                          name="Predicted", line=dict(color="#ff6b6b", width=2, dash="dot")))
fig1.update_layout(template="plotly_dark", height=400,
                   xaxis_title="Date", yaxis_title="Price (₹)",
                   legend=dict(x=0.01, y=0.99), margin=dict(t=20, b=20))
st.plotly_chart(fig1, use_container_width=True)

# ── Candlestick + MA ──────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🕯️ Candlestick Chart (Last 60 Days)")
    candle_df = full_df.tail(60)
    fig2 = go.Figure(go.Candlestick(
        x=candle_df.index,
        open=candle_df["Open"], high=candle_df["High"],
        low=candle_df["Low"],   close=candle_df["Close"],
        increasing_line_color="#00cc88",
        decreasing_line_color="#ff4b4b",
    ))
    fig2.add_trace(go.Scatter(x=candle_df.index, y=candle_df["MA_10"],
                              name="MA10", line=dict(color="#ffd700", width=1)))
    fig2.add_trace(go.Scatter(x=candle_df.index, y=candle_df["MA_50"],
                              name="MA50", line=dict(color="#64b5f6", width=1)))
    fig2.update_layout(template="plotly_dark", height=350,
                       xaxis_rangeslider_visible=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.subheader("🔍 Feature Importance")
    fig3 = px.bar(importance, x="Importance", y="Feature",
                  orientation="h", color="Importance",
                  color_continuous_scale="teal",
                  template="plotly_dark")
    fig3.update_layout(height=350, margin=dict(t=10, b=10), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

# ── Error Distribution ────────────────────────────────────────────────────
st.subheader("📉 Prediction Error Distribution")
results["Error"] = results["Actual"] - results["Predicted"]
fig4 = px.histogram(results, x="Error", nbins=50,
                    color_discrete_sequence=["#00cc88"],
                    template="plotly_dark",
                    labels={"Error": "Prediction Error (₹)"})
fig4.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
fig4.update_layout(height=300, margin=dict(t=10, b=10))
st.plotly_chart(fig4, use_container_width=True)

# ── Model Performance Table ───────────────────────────────────────────────
st.subheader("📋 Model Performance Summary")
perf = pd.DataFrame({
    "Metric"     : ["MAE", "RMSE", "MAPE", "Train Size", "Test Size", "Features"],
    "Value"      : [f"₹{mae:.2f}", f"₹{rmse:.2f}", f"{mape:.2f}%",
                    f"{int(len(full_df)*0.8):,} days",
                    f"{int(len(full_df)*0.2):,} days", "10"],
    "Interpretation": [
        "Average prediction error",
        "Error penalising large deviations",
        "Mean absolute % error",
        "80% historical data for training",
        "20% held-out for evaluation",
        "OHLCV + MA10 + MA50 + Return + Volatility + Range"
    ]
})
st.dataframe(perf, use_container_width=True, hide_index=True)

# ── Download ──────────────────────────────────────────────────────────────
st.download_button("⬇️ Download Predictions CSV",
                   results.to_csv(index=False).encode("utf-8"),
                   "bankbees_predictions.csv", "text/csv")

st.divider()
st.caption("Built by Pramod Prakash Jadhav · IIT Patna (Vishlesan i-Hub) · github.com/pramodj551-oss")

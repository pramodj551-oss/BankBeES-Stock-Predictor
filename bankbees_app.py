"""
BankBeES Stock Price Predictor — Full Production App
======================================================
Author  : Pramod Prakash Jadhav
GitHub  : https://github.com/pramodj551-oss
LinkedIn: https://linkedin.com/in/pramod-prakash-jadhav-42ba2281
Built as part of Applied AI & ML Essentials — IIT Patna (Vishlesan i-Hub)

Models     : Random Forest | Linear Regression | ARIMA | Prophet | LSTM
Indicators : Bollinger Bands | MACD | RSI
Data       : Live Yahoo Finance  OR  Realistic Simulated BANKBEES data
"""

import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be FIRST Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BankBeES Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
h1  { color: #00cc88 !important; }
h2  { color: #64b5f6 !important; }
.stMetric          { background-color: #1e2130; border-radius: 8px; padding: 10px; }
.stMetricLabel     { font-size: 0.80rem !important; }
.stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    data_source = st.radio(
        "Data Source",
        ["📡 Live (Yahoo Finance)", "🧪 Simulated (Demo)"],
        index=0,
    )
    use_live = data_source.startswith("📡")

    if use_live:
        ticker = st.text_input("Ticker Symbol", value="BANKBEES.NS")
        period = st.selectbox("Data Period", ["6mo", "1y", "2y", "5y"], index=1)
    else:
        st.info("Using realistic simulated BANKBEES data (2020–2026).")
        ticker, period = "BANKBEES.NS", "simulated"

    st.markdown("---")
    st.markdown("### 🤖 Model Selection")
    model_choice = st.selectbox(
        "Forecast / Prediction Model",
        ["Random Forest", "Linear Regression", "ARIMA", "Prophet", "LSTM"],
    )

    if model_choice == "Random Forest":
        n_estimators = st.slider("Number of Trees",  50, 500, 100, 50)
        max_depth    = st.slider("Max Tree Depth",    3,  20,  10,   1)
        show_days    = st.slider("Show Last N Days", 30, 500, 120,  30)
    else:
        forecast_days = st.slider("Forecast Days (Future)", 7, 60, 30)
        # defaults so RF tab always works
        n_estimators, max_depth, show_days = 100, 10, 120

    st.markdown("---")
    st.markdown("**👤 Pramod Prakash Jadhav**")
    st.markdown("🎓 IIT Patna — Applied AI & ML (Vishlesan i-Hub)")
    st.markdown(
        "[GitHub](https://github.com/pramodj551-oss) | "
        "[LinkedIn](https://www.linkedin.com/in/pramod-prakash-jadhav-42ba2281)"
    )
    st.markdown("---")
    st.warning("📌 Educational purposes only.\nNot financial advice.")


# ─────────────────────────────────────────────────────────────────────────────
# DATA  —  LIVE (Yahoo Finance)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_live_data(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    for _ in range(3):
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if not df.empty:
                df = df.reset_index()
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df["Date"] = pd.to_datetime(df["Date"])
                return df.sort_values("Date").reset_index(drop=True)
        except Exception:
            pass
        time.sleep(3)
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# DATA  —  SIMULATED (realistic BANKBEES 2020-2026)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_simulated_data() -> pd.DataFrame:
    np.random.seed(42)
    start = datetime(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(1500)]
    dates = [d for d in dates if d.weekday() < 5]   # business days only

    price = 300.0
    prices = []
    for d in dates:
        if   d < datetime(2020, 4,  1): vol = 0.025   # COVID crash
        elif d < datetime(2020, 10, 1): vol = 0.018   # recovery
        else:                            vol = 0.010   # normal
        drift = 0.0004
        price *= (1 + np.random.normal(drift, vol))
        price  = max(price, 150)
        prices.append(round(price, 2))

    df = pd.DataFrame({"Date": pd.to_datetime(dates), "Close": prices})
    df["Open"]   = (df["Close"] * np.random.uniform(0.995, 1.005, len(df))).round(2)
    df["High"]   = (df[["Open","Close"]].max(axis=1) * np.random.uniform(1.001, 1.012, len(df))).round(2)
    df["Low"]    = (df[["Open","Close"]].min(axis=1) * np.random.uniform(0.988, 0.999, len(df))).round(2)
    df["Volume"] = np.random.randint(500_000, 5_000_000, len(df))
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS  (also adds RF feature columns)
# ─────────────────────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()

    # Moving Averages
    df["MA_10"] = close.rolling(10).mean()
    df["MA_50"] = close.rolling(50).mean()

    # Bollinger Bands (20-day ±2σ)
    df["BB_Mid"]   = close.rolling(20).mean()
    _std           = close.rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * _std
    df["BB_Lower"] = df["BB_Mid"] - 2 * _std

    # RSI (14-day)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # MACD (12, 26, signal=9)
    ema12             = close.ewm(span=12, adjust=False).mean()
    ema26             = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # Extra features for Random Forest
    df["Daily_Return"] = close.pct_change()
    df["Volatility"]   = df["Daily_Return"].rolling(10).std()
    df["Price_Range"]  = df["High"] - df["Low"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FORECAST MODELS
# ─────────────────────────────────────────────────────────────────────────────
def _future_bdays(df: pd.DataFrame, days: int):
    return pd.date_range(
        start=df["Date"].max() + timedelta(days=1), periods=days, freq="B"
    )


# ── 1. Random Forest ─────────────────────────────────────────────────────────
def run_random_forest(df: pd.DataFrame, n_estimators: int, max_depth: int):
    work = df.copy()
    work["Target"] = work["Close"].shift(-1)
    work.dropna(inplace=True)

    FEATURES = [
        "Open", "High", "Low", "Close", "Volume",
        "MA_10", "MA_50", "Daily_Return", "Volatility", "Price_Range",
    ]
    X = work[FEATURES]
    y = work["Target"].values.flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

    results = pd.DataFrame({
        "Date":      work["Date"].iloc[-len(y_test):].values,
        "Actual":    y_test,
        "Predicted": y_pred,
    })
    importance = pd.DataFrame({
        "Feature":    FEATURES,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return results, mae, rmse, mape, importance


# ── 2. Linear Regression ─────────────────────────────────────────────────────
def forecast_lr(df: pd.DataFrame, days: int):
    X = np.array([d.toordinal() for d in df["Date"]]).reshape(-1, 1)
    y = df["Close"].values.flatten()
    model = LinearRegression().fit(X, y)
    fd    = _future_bdays(df, days)
    X_fut = np.array([d.toordinal() for d in fd]).reshape(-1, 1)
    return fd, model.predict(X_fut), None, None


# ── 3. ARIMA (5,1,0) ─────────────────────────────────────────────────────────
def forecast_arima(df: pd.DataFrame, days: int):
    from statsmodels.tsa.arima.model import ARIMA
    close  = df["Close"].values.flatten()
    result = ARIMA(close, order=(5, 1, 0)).fit()
    pred   = result.get_forecast(steps=days)
    fc     = pred.predicted_mean
    ci     = pred.conf_int(alpha=0.10)
    return _future_bdays(df, days), fc, ci.iloc[:, 0].values, ci.iloc[:, 1].values


# ── 4. Prophet ────────────────────────────────────────────────────────────────
def forecast_prophet(df: pd.DataFrame, days: int):
    from prophet import Prophet
    prop_df      = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    prop_df["y"] = prop_df["y"].values.flatten()
    m = Prophet(
        daily_seasonality=False, weekly_seasonality=True,
        yearly_seasonality=True, interval_width=0.90,
    )
    m.fit(prop_df)
    future   = m.make_future_dataframe(periods=days, freq="B")
    forecast = m.predict(future)
    fut = forecast[forecast["ds"] > df["Date"].max()][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ]
    return fut["ds"].values, fut["yhat"].values, fut["yhat_lower"].values, fut["yhat_upper"].values


# ── 5. LSTM ───────────────────────────────────────────────────────────────────
def forecast_lstm(df: pd.DataFrame, days: int):
    import tensorflow as tf

    close  = df["Close"].values.flatten().reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)
    SEQ    = min(60, len(scaled) - 1)

    X_seq, y_seq = [], []
    for i in range(SEQ, len(scaled)):
        X_seq.append(scaled[i - SEQ : i])
        y_seq.append(scaled[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(SEQ, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_seq, y_seq, epochs=20, batch_size=32, verbose=0)

    seq, preds = scaled[-SEQ:].copy(), []
    for _ in range(days):
        inp = seq[-SEQ:].reshape(1, SEQ, 1)
        p   = model.predict(inp, verbose=0)[0][0]
        preds.append(p)
        seq = np.append(seq, [[p]], axis=0)

    predictions = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return _future_bdays(df, days), predictions, None, None


FORECAST_DISPATCH = {
    "Linear Regression": forecast_lr,
    "ARIMA":             forecast_arima,
    "Prophet":           forecast_prophet,
    "LSTM":              forecast_lstm,
}


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def chart_technical(df, future_dates=None, preds=None,
                    model_name="", conf_low=None, conf_high=None):
    title_row1 = "Price · MA10 · MA50 · Bollinger Bands"
    if preds is not None:
        title_row1 += f"  +  {model_name} Forecast"

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=[title_row1, "MACD (12 / 26 / 9)", "RSI (14)"],
        vertical_spacing=0.06,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(),   close=df["Close"].squeeze(),
        name="OHLC",
        increasing_line_color="#00cc88", decreasing_line_color="#ff4b4b",
    ), row=1, col=1)

    # MA lines
    for col_name, color, label in [("MA_10","#ffd700","MA10"), ("MA_50","#64b5f6","MA50")]:
        fig.add_trace(go.Scatter(x=df["Date"], y=df[col_name], name=label,
                                 line=dict(color=color, width=1)), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], name="BB Upper",
                             line=dict(color="rgba(100,149,237,0.5)", dash="dot", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], name="BB Lower",
                             line=dict(color="rgba(100,149,237,0.5)", dash="dot", width=1),
                             fill="tonexty", fillcolor="rgba(100,149,237,0.07)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Mid"], name="BB Mid",
                             line=dict(color="rgba(100,149,237,0.9)", width=1)), row=1, col=1)

    # Forecast overlay
    if preds is not None and future_dates is not None:
        fig.add_trace(go.Scatter(
            x=list(future_dates), y=list(preds),
            name=f"{model_name} Forecast",
            line=dict(color="orange", dash="dash", width=2),
        ), row=1, col=1)
        if conf_low is not None:
            fd = list(future_dates)
            fig.add_trace(go.Scatter(
                x=fd + fd[::-1],
                y=list(conf_high) + list(conf_low)[::-1],
                fill="toself", fillcolor="rgba(255,165,0,0.12)",
                line=dict(color="rgba(255,165,0,0)"), name="90% CI",
            ), row=1, col=1)

    # MACD
    bar_colors = ["#00cc88" if v >= 0 else "#ff4b4b" for v in df["MACD_Hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_Hist"],
                         marker_color=bar_colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"],
                             name="MACD", line=dict(color="#2196f3", width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"],
                             name="Signal", line=dict(color="#ff9800", width=1.2)), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"],
                             name="RSI", line=dict(color="#ce93d8", width=1.5)), row=3, col=1)
    for lvl, clr in [(70, "red"), (30, "green")]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=clr, opacity=0.5, row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=820,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="MACD",      row=2, col=1)
    fig.update_yaxes(title_text="RSI",       row=3, col=1, range=[0, 100])
    return fig


def chart_rf_actual_vs_pred(results, show_days):
    recent = results.tail(show_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["Date"], y=recent["Actual"],
                             name="Actual", line=dict(color="#00cc88", width=2)))
    fig.add_trace(go.Scatter(x=recent["Date"], y=recent["Predicted"],
                             name="Predicted", line=dict(color="#ff6b6b", width=2, dash="dot")))
    fig.update_layout(
        template="plotly_dark", height=380,
        xaxis_title="Date", yaxis_title="Price (₹)",
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=20, b=20),
    )
    return fig


def chart_feature_importance(importance):
    fig = px.bar(
        importance, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="teal", template="plotly_dark",
    )
    fig.update_layout(height=350, margin=dict(t=10, b=10), showlegend=False)
    return fig


def chart_error_dist(results):
    err = results.copy()
    err["Error"] = err["Actual"] - err["Predicted"]
    fig = px.histogram(
        err, x="Error", nbins=50,
        color_discrete_sequence=["#00cc88"],
        template="plotly_dark",
        labels={"Error": "Prediction Error (₹)"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 BankBeES Stock Price Predictor")
st.caption(
    "Random Forest · LR · ARIMA · Prophet · LSTM  |  "
    "Bollinger Bands · MACD · RSI  |  "
    "IIT Patna Applied AI & ML · Pramod Prakash Jadhav"
)

# ── Load data ─────────────────────────────────────────────────────────────────
if use_live:
    with st.spinner(f"📡 Fetching {ticker} ({period}) from Yahoo Finance …"):
        df = load_live_data(ticker, period)
    if df.empty:
        st.error(
            "⚠️ Could not fetch live data. Yahoo Finance may be rate-limiting. "
            "Switch to **Simulated** mode in the sidebar, or wait ~60 s and reload."
        )
        st.stop()
else:
    with st.spinner("🧪 Generating simulated BANKBEES data …"):
        df = load_simulated_data()

df = compute_indicators(df)

# ── KPI row ───────────────────────────────────────────────────────────────────
close_vals = df["Close"].values.flatten()
latest     = float(close_vals[-1])
prev       = float(close_vals[-2])
change     = latest - prev
pct        = change / prev * 100
rsi_now    = float(df["RSI"].iloc[-1])
rsi_label  = "🔴 Overbought" if rsi_now > 70 else ("🟢 Oversold" if rsi_now < 30 else "🟡 Neutral")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💰 Last Close",  f"₹{latest:.2f}",               f"{change:+.2f} ({pct:+.2f}%)")
c2.metric("📈 52W High",    f"₹{float(df['High'].max()):.2f}")
c3.metric("📉 52W Low",     f"₹{float(df['Low'].min()):.2f}")
c4.metric("📊 Avg Volume",  f"{int(df['Volume'].mean()):,}")
c5.metric("⚡ RSI (14)",    f"{rsi_now:.1f}",                rsi_label)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Technical Analysis + Forecast",
    "🌲 Random Forest Evaluation",
    "📋 Data & Downloads",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Technical Analysis + Forecast
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if model_choice == "Random Forest":
        st.info(
            "Random Forest predicts **next-day close** on historical data — "
            "full evaluation is in the **🌲 Random Forest Evaluation** tab. "
            "Select LR / ARIMA / Prophet / LSTM for future forecasting."
        )
        st.plotly_chart(chart_technical(df), use_container_width=True)

    else:
        with st.spinner(f"⚙️ Running {model_choice} ({forecast_days}-day forecast) …"):
            try:
                future_dates, preds, conf_low, conf_high = FORECAST_DISPATCH[model_choice](
                    df, forecast_days
       

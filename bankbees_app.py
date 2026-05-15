"""
BankBeES Stock Price Predictor
Author  : Pramod Prakash Jadhav
GitHub  : https://github.com/pramodj551-oss
IIT Patna - Applied AI & ML Essentials (Vishlesan i-Hub)
Models  : Random Forest | Linear Regression | ARIMA
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
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BankBeES Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
h1 { color: #00cc88 !important; }
h2 { color: #64b5f6 !important; }
.stMetric { background-color: #1e2130; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")

    data_source = st.radio(
        "Data Source",
        ["Live (Yahoo Finance)", "Simulated (Demo)"],
        index=0,
    )
    use_live = data_source == "Live (Yahoo Finance)"

    if use_live:
        ticker = st.text_input("Ticker Symbol", value="BANKBEES.NS")
        period = st.selectbox("Data Period", ["6mo", "1y", "2y", "5y"], index=1)
    else:
        st.info("Using simulated BANKBEES data (2020-2026).")
        ticker = "BANKBEES.NS"
        period = "simulated"

    st.markdown("---")
    st.markdown("### Model Selection")
    model_choice = st.selectbox(
        "Model",
        ["Random Forest", "Linear Regression", "ARIMA"],
    )

    if model_choice == "Random Forest":
        n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
        max_depth    = st.slider("Max Tree Depth",   3,  20,  10,   1)
        show_days    = st.slider("Show Last N Days", 30, 500, 120, 30)
    else:
        forecast_days = st.slider("Forecast Days", 7, 60, 30)
        n_estimators  = 100
        max_depth     = 10
        show_days     = 120

    st.markdown("---")
    st.markdown("**Pramod Prakash Jadhav**")
    st.markdown("IIT Patna - Applied AI & ML")
    st.markdown("[GitHub](https://github.com/pramodj551-oss) | [LinkedIn](https://www.linkedin.com/in/pramod-prakash-jadhav-42ba2281)")
    st.markdown("---")
    st.warning("Educational purposes only. Not financial advice.")

# ── Data Loading ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_live_data(tkr, per):
    import yfinance as yf
    for _ in range(3):
        try:
            df = yf.download(tkr, period=per, auto_adjust=True, progress=False)
            if not df.empty:
                df = df.reset_index()
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df["Date"] = pd.to_datetime(df["Date"])
                return df.sort_values("Date").reset_index(drop=True)
        except Exception:
            pass
        time.sleep(3)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_simulated_data():
    np.random.seed(42)
    start = datetime(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(1500)]
    dates = [d for d in dates if d.weekday() < 5]

    price = 300.0
    prices = []
    for d in dates:
        if d < datetime(2020, 4, 1):
            vol = 0.025
        elif d < datetime(2020, 10, 1):
            vol = 0.018
        else:
            vol = 0.010
        price *= (1 + np.random.normal(0.0004, vol))
        price  = max(price, 150)
        prices.append(round(price, 2))

    df = pd.DataFrame({"Date": pd.to_datetime(dates), "Close": prices})
    df["Open"]   = (df["Close"] * np.random.uniform(0.995, 1.005, len(df))).round(2)
    df["High"]   = (df[["Open","Close"]].max(axis=1) * np.random.uniform(1.001, 1.012, len(df))).round(2)
    df["Low"]    = (df[["Open","Close"]].min(axis=1) * np.random.uniform(0.988, 0.999, len(df))).round(2)
    df["Volume"] = np.random.randint(500_000, 5_000_000, len(df))
    return df.reset_index(drop=True)


# ── Indicators ─────────────────────────────────────────────────────────────────
def compute_indicators(df):
    close = df["Close"].squeeze()
    df["MA_10"]       = close.rolling(10).mean()
    df["MA_50"]       = close.rolling(50).mean()
    df["BB_Mid"]      = close.rolling(20).mean()
    std               = close.rolling(20).std()
    df["BB_Upper"]    = df["BB_Mid"] + 2 * std
    df["BB_Lower"]    = df["BB_Mid"] - 2 * std
    delta             = close.diff()
    gain              = delta.clip(lower=0).rolling(14).mean()
    loss              = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]         = 100 - (100 / (1 + gain / loss))
    ema12             = close.ewm(span=12, adjust=False).mean()
    ema26             = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    df["Daily_Return"]= close.pct_change()
    df["Volatility"]  = df["Daily_Return"].rolling(10).std()
    df["Price_Range"] = df["High"] - df["Low"]
    return df


# ── Future Dates ───────────────────────────────────────────────────────────────
def future_bdays(df, days):
    return pd.date_range(
        start=df["Date"].max() + timedelta(days=1),
        periods=days,
        freq="B",
    )


# ── Models ─────────────────────────────────────────────────────────────────────
def run_random_forest(df, n_est, m_depth):
    work = df.copy()
    work["Target"] = work["Close"].shift(-1)
    work = work.dropna()

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
        n_estimators=n_est, max_depth=m_depth, random_state=42, n_jobs=-1
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


def forecast_lr(df, days):
    X = np.array([d.toordinal() for d in df["Date"]]).reshape(-1, 1)
    y = df["Close"].values.flatten()
    model = LinearRegression().fit(X, y)
    fd    = future_bdays(df, days)
    X_fut = np.array([d.toordinal() for d in fd]).reshape(-1, 1)
    return fd, model.predict(X_fut), None, None


def forecast_arima(df, days):
    close  = df["Close"].values.flatten()
    result = ARIMA(close, order=(5, 1, 0)).fit()
    pred   = result.get_forecast(steps=days)
    fc     = pred.predicted_mean
    ci     = pred.conf_int(alpha=0.10)
    return future_bdays(df, days), fc, ci.iloc[:, 0].values, ci.iloc[:, 1].values


FORECAST_FNS = {
    "Linear Regression": forecast_lr,
    "ARIMA":             forecast_arima,
}


# ── Charts ─────────────────────────────────────────────────────────────────────
def chart_technical(df, future_dates=None, preds=None, model_name="", conf_low=None, conf_high=None):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=["Price + Bollinger Bands", "MACD", "RSI (14)"],
        vertical_spacing=0.06,
    )

    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(),   close=df["Close"].squeeze(),
        name="OHLC",
        increasing_line_color="#00cc88", decreasing_line_color="#ff4b4b",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_10"], name="MA10",
                             line=dict(color="#ffd700", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_50"], name="MA50",
                             line=dict(color="#64b5f6", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], name="BB Upper",
                             line=dict(color="rgba(100,149,237,0.5)", dash="dot", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], name="BB Lower",
                             line=dict(color="rgba(100,149,237,0.5)", dash="dot", width=1),
                             fill="tonexty", fillcolor="rgba(100,149,237,0.07)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Mid"], name="BB Mid",
                             line=dict(color="rgba(100,149,237,0.9)", width=1)), row=1, col=1)

    if preds is not None and future_dates is not None:
        fig.add_trace(go.Scatter(
            x=list(future_dates), y=list(preds),
            name=model_name + " Forecast",
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

    bar_colors = ["#00cc88" if v >= 0 else "#ff4b4b" for v in df["MACD_Hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_Hist"],
                         marker_color=bar_colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD",
                             line=dict(color="#2196f3", width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"], name="Signal",
                             line=dict(color="#ff9800", width=1.2)), row=2, col=1)

    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI",
                             line=dict(color="#ce93d8", width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red",   opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", opacity=0.5, row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=820,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_yaxes(title_text="Price (Rs)", row=1, col=1)
    fig.update_yaxes(title_text="MACD",       row=2, col=1)
    fig.update_yaxes(title_text="RSI",        row=3, col=1, range=[0, 100])
    return fig


def chart_rf_vs_actual(results, show_days):
    recent = results.tail(show_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["Date"], y=recent["Actual"],
                             name="Actual", line=dict(color="#00cc88", width=2)))
    fig.add_trace(go.Scatter(x=recent["Date"], y=recent["Predicted"],
                             name="Predicted", line=dict(color="#ff6b6b", width=2, dash="dot")))
    fig.update_layout(template="plotly_dark", height=380,
                      xaxis_title="Date", yaxis_title="Price (Rs)",
                      margin=dict(t=20, b=20))
    return fig


def chart_importance(importance):
    fig = px.bar(importance, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="teal", template="plotly_dark")
    fig.update_layout(height=350, margin=dict(t=10, b=10), showlegend=False)
    return fig


def chart_errors(results):
    err = results.copy()
    err["Error"] = err["Actual"] - err["Predicted"]
    fig = px.histogram(err, x="Error", nbins=50,
                       color_discrete_sequence=["#00cc88"], template="plotly_dark",
                       labels={"Error": "Prediction Error (Rs)"})
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    return fig


# ── Main App ───────────────────────────────────────────────────────────────────
st.title("BankBeES Stock Price Predictor")
st.caption("Random Forest | LR | ARIMA  |  Bollinger Bands | MACD | RSI  |  IIT Patna - Pramod Prakash Jadhav")

# Load data
if use_live:
    with st.spinner("Fetching data from Yahoo Finance..."):
        df = load_live_data(ticker, period)
    if df.empty:
        st.error("Could not fetch live data. Switch to Simulated mode or wait 60s and reload.")
        st.stop()
else:
    with st.spinner("Generating simulated data..."):
        df = load_simulated_data()

df = compute_indicators(df)

# KPIs
close_vals = df["Close"].values.flatten()
latest     = float(close_vals[-1])
prev       = float(close_vals[-2])
change     = latest - prev
pct        = change / prev * 100
rsi_now    = float(df["RSI"].iloc[-1])
rsi_label  = "Overbought" if rsi_now > 70 else ("Oversold" if rsi_now < 30 else "Neutral")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last Close",  "Rs " + str(round(latest, 2)), str(round(change, 2)) + " (" + str(round(pct, 2)) + "%)")
c2.metric("52W High",    "Rs " + str(round(float(df["High"].max()), 2)))
c3.metric("52W Low",     "Rs " + str(round(float(df["Low"].min()), 2)))
c4.metric("Avg Volume",  str(int(df["Volume"].mean())))
c5.metric("RSI (14)",    str(round(rsi_now, 1)), rsi_label)

st.divider()

# Tabs
tab1, tab2, tab3 = st.tabs([
    "Technical Analysis + Forecast",
    "Random Forest Evaluation",
    "Data and Downloads",
])

# Tab 1
with tab1:
    if model_choice == "Random Forest":
        st.info("Random Forest shows next-day prediction on historical data. See the Random Forest Evaluation tab for full details.")
        st.plotly_chart(chart_technical(df), use_container_width=True)
    else:
        with st.spinner("Running " + model_choice + " forecast..."):
            try:
                fn = FORECAST_FNS[model_choice]
                future_dates, preds, conf_low, conf_high = fn(df, forecast_days)
            except Exception as e:
                st.error("Forecast failed: " + str(e))
                st.stop()

        st.plotly_chart(
            chart_technical(df, future_dates, preds, model_choice, conf_low, conf_high),
            use_container_width=True,
        )

        with st.expander("Forecast Table"):
            fdf = pd.DataFrame({
                "Date":               future_dates,
                "Predicted Price Rs": np.round(preds, 2),
            })
            if conf_low is not None:
                fdf["Lower 90%"] = np.round(conf_low, 2)
                fdf["Upper 90%"] = np.round(conf_high, 2)
            st.dataframe(fdf.set_index("Date"), use_container_width=True)

# Tab 2
with tab2:
    st.subheader("Random Forest - Next Day Close Prediction")
    st.caption("Train 80% | Test 20% | Features: OHLCV + MA10 + MA50 + Return + Volatility + Range")

    with st.spinner("Training Random Forest..."):
        try:
            rf_results, mae, rmse, mape, importance = run_random_forest(df, n_estimators, max_depth)
        except Exception as e:
            st.error("Training failed: " + str(e))
            st.stop()

    pred_next = float(rf_results["Predicted"].iloc[-1])
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Predicted Next Close", "Rs " + str(round(pred_next, 2)))
    r2.metric("MAE",  "Rs " + str(round(mae, 2)))
    r3.metric("RMSE", "Rs " + str(round(rmse, 2)))
    r4.metric("MAPE", str(round(mape, 2)) + "%")

    st.markdown("---")
    st.subheader("Actual vs Predicted (Test Set)")
    st.plotly_chart(chart_rf_vs_actual(rf_results, show_days), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Feature Importance")
        st.plotly_chart(chart_importance(importance), use_container_width=True)
    with col_b:
        st.subheader("Prediction Error Distribution")
        st.plotly_chart(chart_errors(rf_results), use_container_width=True)

    perf_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "MAPE", "Train Size", "Test Size", "Features", "Trees", "Max Depth"],
        "Value": [
            "Rs " + str(round(mae, 2)),
            "Rs " + str(round(rmse, 2)),
            str(round(mape, 2)) + "%",
            str(int(len(df) * 0.8)) + " days",
            str(int(len(df) * 0.2)) + " days",
            "10",
            str(n_estimators),
            str(max_depth),
        ],
        "Interpretation": [
            "Average prediction error in rupees",
            "Penalises large deviations more than MAE",
            "Mean absolute percentage error",
            "80% historical data used for training",
            "20% held-out for unbiased evaluation",
            "OHLCV + MA10 + MA50 + Return + Volatility + Range",
            "Number of decision trees",
            "Maximum depth of each tree",
        ],
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

# Tab 3
with tab3:
    st.subheader("Recent OHLCV + Indicators (last 30 rows)")
    display_cols = ["Date", "Open", "High", "Low", "Close", "Volume",
                    "MA_10", "MA_50", "RSI", "MACD", "BB_Upper", "BB_Lower"]
    st.dataframe(df[display_cols].tail(30).set_index("Date").round(2), use_container_width=True)

    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "Download OHLCV + Indicators CSV",
            df[display_cols].to_csv(index=False).encode("utf-8"),
            "bankbees_data.csv",
            "text/csv",
        )
    with col_dl2:
        try:
            st.download_button(
                "Download RF Predictions CSV",
                rf_results.to_csv(index=False).encode("utf-8"),
                "bankbees_rf_predictions.csv",
                "text/csv",
            )
        except NameError:
            st.info("Visit the Random Forest tab first to enable this download.")

st.divider()
st.caption("Built by Pramod Prakash Jadhav | IIT Patna (Vishlesan i-Hub) | github.com/pramodj551-oss")
    

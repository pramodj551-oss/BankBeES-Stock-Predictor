import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from plotly import graph_objs as go
from sklearn.ensemble import RandomForestRegressor          # ✅ Fixed: LinearRegression → RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import date

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="BankBeES Stock Predictor", page_icon="📈")
st.title("📈 BankBeES Stock Price Predictor")
st.markdown("Analyze historical trends and predict future prices using **Random Forest**.")
st.markdown("> ⚠️ For educational purposes only. Not financial advice.")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
stock_symbol = st.sidebar.text_input("Enter Stock Ticker", "BANKBEES.NS")
n_estimators = st.sidebar.slider("Number of Trees (RandomForest)", 50, 300, 100, 50)
start_date   = "2020-01-01"                                # ✅ Fixed: 2015 → 2020 (matches notebook)
today        = date.today().strftime("%Y-%m-%d")

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(ticker):
    df = yf.download(ticker, start=start_date, end=today)
    if df.empty:
        return None
    df.reset_index(inplace=True)
    # ✅ Fixed: Flatten MultiIndex columns (yfinance >= 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

data = load_data(stock_symbol)

if data is None:
    st.error("Could not fetch data. Please check ticker symbol or reload after 1 minute.")
    st.stop()

# ─── Feature Engineering ──────────────────────────────────────────────────────
# ✅ Added: Exactly matches BankBeES_Predictor.ipynb Cell 1
def add_features(df):
    df = df.copy()
    df['MA_10']        = df['Close'].rolling(window=10).mean()
    df['MA_50']        = df['Close'].rolling(window=50).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Target']       = df['Close'].shift(-1)             # Tomorrow's price
    return df.dropna()

df = add_features(data)
FEATURES = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50', 'Daily_Return']

# ─── Model Training ───────────────────────────────────────────────────────────
# ✅ Fixed: RandomForest, shuffle=False (time-series safe — matches notebook Cell 2)
@st.cache_resource
def train_model(dataframe, n_trees):
    X = dataframe[FEATURES]
    y = dataframe['Target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False               # shuffle=False critical for time series
    )
    model = RandomForestRegressor(n_estimators=n_trees, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results = pd.DataFrame({
        'Date':            X_test.index,
        'Actual_Price':    y_test.values,
        'Predicted_Price': predictions
    })
    mae  = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    return model, results, mae, rmse, importance

model, results, mae, rmse, importance = train_model(df, n_estimators)

# ─── Metrics Row ──────────────────────────────────────────────────────────────
st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE",           f"₹{mae:.2f}",       help="Avg prediction error")
c2.metric("RMSE",          f"₹{rmse:.2f}")
c3.metric("Training Rows", f"{int(len(df)*0.8):,}")
c4.metric("Test Rows",     f"{len(results):,}")
st.divider()

# ─── Historical Data Table ────────────────────────────────────────────────────
st.subheader("📋 Historical Data (Last 5 Days)")
st.write(data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail())

# ─── Chart 1: Historical Price Trend ─────────────────────────────────────────
st.subheader("📅 Historical Price Trend with Moving Averages")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=data['Date'], y=data['Close'],
    name="Close Price", line=dict(color='royalblue', width=2)
))
fig1.add_trace(go.Scatter(
    x=df['Date'] if 'Date' in df.columns else df.index,
    y=df['MA_10'],
    name="MA 10", line=dict(color='orange', width=1.5, dash='dot')
))
fig1.add_trace(go.Scatter(
    x=df['Date'] if 'Date' in df.columns else df.index,
    y=df['MA_50'],
    name="MA 50", line=dict(color='violet', width=1.5, dash='dash')
))
fig1.update_layout(
    title="Historical Price Trend",
    xaxis_rangeslider_visible=True,
    template="plotly_dark",
    hovermode="x unified"
)
st.plotly_chart(fig1, use_container_width=True)

# ─── Chart 2: Actual vs Predicted ────────────────────────────────────────────
# ✅ Added: Was completely missing in original file
st.subheader("🎯 Actual vs Predicted Price (Test Set: Last 20%)")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=results['Date'], y=results['Actual_Price'],
    name="Actual Price", line=dict(color='royalblue', width=2)
))
fig2.add_trace(go.Scatter(
    x=results['Date'], y=results['Predicted_Price'],
    name="Predicted Price", line=dict(color='tomato', width=2, dash='dot')
))
fig2.update_layout(
    title="Actual vs Predicted — BANKBEES.NS",
    template="plotly_dark",
    hovermode="x unified",
    height=420
)
st.plotly_chart(fig2, use_container_width=True)

# ─── Chart 3: Feature Importance ─────────────────────────────────────────────
# ✅ Added: Shows which features matter most
st.subheader("🔍 Feature Importance (Random Forest)")
fig3 = go.Figure(go.Bar(
    x=importance.sort_values().values,
    y=importance.sort_values().index,
    orientation='h',
    marker_color='steelblue'
))
fig3.update_layout(template="plotly_dark", height=300, xaxis_title="Importance Score")
st.plotly_chart(fig3, use_container_width=True)

# ─── Live Prediction Panel ────────────────────────────────────────────────────
# ✅ Added: Interactive next-day prediction
st.divider()
st.subheader("🔮 Predict Tomorrow's Price")
st.markdown("Enter today's market values:")

col1, col2 = st.columns(2)
with col1:
    inp_close  = st.number_input("Close  (₹)", value=float(data['Close'].iloc[-1]),  step=1.0)
    inp_open   = st.number_input("Open   (₹)", value=float(data['Open'].iloc[-1]),   step=1.0)
    inp_high   = st.number_input("High   (₹)", value=float(data['High'].iloc[-1]),   step=1.0)
    inp_low    = st.number_input("Low    (₹)", value=float(data['Low'].iloc[-1]),    step=1.0)
with col2:
    inp_volume = st.number_input("Volume",      value=float(data['Volume'].iloc[-1]), step=1000.0)
    inp_ma10   = st.number_input("MA_10",       value=float(df['MA_10'].iloc[-1]),    step=1.0)
    inp_ma50   = st.number_input("MA_50",       value=float(df['MA_50'].iloc[-1]),    step=1.0)
    inp_ret    = st.number_input("Daily Return",value=float(df['Daily_Return'].iloc[-1]), step=0.001, format="%.4f")

if st.button("🚀 Predict Tomorrow's Price", type="primary"):
    row = pd.DataFrame([{
        'Close': inp_close, 'Open': inp_open, 'High': inp_high, 'Low': inp_low,
        'Volume': inp_volume, 'MA_10': inp_ma10, 'MA_50': inp_ma50, 'Daily_Return': inp_ret
    }])
    pred  = model.predict(row[FEATURES])[0]
    delta = pred - inp_close
    st.success(f"**Predicted Next-Day Close: ₹{pred:.2f}**")
    direction = "📈 UP" if delta > 0 else "📉 DOWN"
    st.info(f"Direction: **{direction}** by ₹{abs(delta):.2f} vs today's close")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built by Pramod · IIT Patna Applied AI & ML Program · Data: Yahoo Finance · Model: Random Forest")

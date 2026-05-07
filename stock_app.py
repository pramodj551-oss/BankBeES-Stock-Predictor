import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta

# --- Page Configuration ---
st.set_page_config(page_title="BankBeES Stock Predictor", page_icon="📈")
st.title("📈 BankBeES Stock Price Predictor")
st.markdown("Analyze historical trends and predict future prices.")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
stock_symbol = st.sidebar.text_input("Enter Stock Ticker", "BANKBEES.NS")
start_date = "2015-01-01"
today = date.today().strftime("%Y-%m-%d")

# --- Function to Load Data ---
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start=start_date, end=today)
    if df.empty:
        return None
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

# Fetch data
data = load_data(stock_symbol)

# --- Error Check ---
if data is None:
    st.error("Error: Could not fetch data. Check the Ticker symbol.")
    st.stop()

# --- 1. Historical Data Display ---
st.subheader("Historical Data Overview (Last 5 Days)")
st.write(data.tail())

# --- 2. Historical Price Chart ---
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price", line=dict(color='royalblue')))
fig1.layout.update(title_text="Historical Price Trend", xaxis_rangeslider_visible=True, template="plotly_dark")
st.plotly_chart(fig1)

# --- 3. Machine Learning Prediction ---
st.subheader("Stock Price Prediction")

# Data preparation
data['Date_Ordinal'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)
X = data['Date_Ordinal'].values.reshape(-1, 1)
y = data['Close'].values.reshape(-1, 1)

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next 30 days
last_date = data['Date'].max()
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
future_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
predictions = model.predict(future_ordinal)

# Display Forecast Data
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions.flatten()})
st.write("Predicted Prices for the next 30 days:")
st.dataframe(forecast_df.head(10))

# --- 4. Forecast Chart ---
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Historical Close", line=dict(color='gray')))
fig2.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted Price'], name="Predicted Price", line=dict(color='orange', dash='dash')))
fig2.layout.update(title_text="Price Forecast (Next 30 Days)", template="plotly_dark")
st.plotly_chart(fig2)

st.sidebar.markdown("---")
st.sidebar.info("Educational purposes only. No financial advice.")
x=data['Date'], y=data['Close'], name="Historical Close", line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted Price'], name="Predicted Price", line=dict(color='orange', dash='dash')))
    fig.layout.update(title_text="Price Forecast (Next 30 Days)", template="plotly_dark")
    st.plotly_chart(fig)

plot_forecast()

st.sidebar.markdown("---")
st.sidebar.info("This app uses Linear Regression for trend estimation. Not financial advice.")

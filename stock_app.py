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
st.markdown("Analyze historical trends and predict future stock prices using Machine Learning.")

# --- Sidebar for User Input ---
st.sidebar.header("Configuration")
stock_symbol = st.sidebar.text_input("Enter Stock Ticker", "BANKBEES.NS")
start_date = "2015-01-01"
today = date.today().strftime("%Y-%m-%d")

# --- Load Data ---
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start=start_date, end=today)
    if df.empty:
        return None
    df.reset_index(inplace=True)
    # Cleaning: Handle Multi-index columns if any and drop NaNs
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df

data_load_state = st.text("Loading data...")
data = load_data(stock_symbol)

if data is None or len(data) == 0:
    st.error("Error: Could not fetch data. Please check the Ticker symbol.")
    st.stop()

data_load_state.text("Loading data... Done!")

# --- Raw Data Overview ---
st.subheader("Historical Data Overview (Last 5 Days)")
st.write(data.tail())

# --- Plotting Historical Data ---
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price", line=dict(color='royalblue')))
    fig.layout.update(title_text="Historical Price Trend", xaxis_rangeslider_visible=True, template="plotly_dark")
    st.plotly_chart(fig)

plot_raw_data()

# --- Machine Learning: Prediction Logic ---
st.subheader("Stock Price Prediction")

try:
    # Proper Data preparation
    data['Date_Ordinal'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)
    
    # Ensuring X and y are properly shaped and contain no NaNs
    X = data['Date_Ordinal'].values.reshape(-1, 1)
    y = data['Close'].values.reshape(-1, 1)

    # Training the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict for the next 30 days
    last_date = data['Date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
    future_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    predictions = model.predict(future_ordinal)

    # Displaying Predictions
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions.flatten()})
    st.write("Predicted Prices for the next 30 days:")
    st.dataframe(forecast_df.head(10))

    # --- Plotting Predictions ---
    def plot_forecast():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Historical Close", line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted Price'], name="Predicted Price", line=dict(color='orange', dash='dash')))
        fig.layout.update(title_text="Price Forecast (Next 30 Days)", template="plotly_dark")
        st.plotly_chart(fig)

    plot_forecast()

except Exception as e:
    st.error(f"An error occurred during prediction: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Note: Linear Regression is for trend estimation. Stock markets are volatile.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price", line=dict(color='royalblue')))
    fig.layout.update(title_text="Historical Price Trend", xaxis_rangeslider_visible=True, template="plotly_dark")
    st.plotly_chart(fig)

plot_raw_data()

# --- Machine Learning: Prediction Logic ---
st.subheader("Stock Price Prediction")

# Data preparation for Linear Regression
data['Date_Ordinal'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)
X = np.array(data['Date_Ordinal']).reshape(-1, 1)
y = np.array(data['Close']).reshape(-1, 1)

# Training the model
model = LinearRegression()
model.fit(X, y)

# Predict for the next 30 days
future_dates = pd.date_range(start=today, periods=30)
future_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
predictions = model.predict(future_ordinal)

# Displaying Predictions
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions.flatten()})
st.write("Predicted Prices for the next 30 days:")
st.dataframe(forecast_df.head(10))

# --- Plotting Predictions ---
def plot_forecast():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Historical Close", line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted Price'], name="Predicted Price", line=dict(color='orange', dash='dash')))
    fig.layout.update(title_text="Price Forecast (Next 30 Days)", template="plotly_dark")
    st.plotly_chart(fig)

plot_forecast()

st.sidebar.markdown("---")
st.sidebar.info("This app uses Linear Regression for trend estimation. Not financial advice.")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta

st.set_page_config(page_title="BankBeES Stock Predictor", page_icon="📈")
st.title("📈 BankBeES Stock Price Predictor")
st.markdown("Analyze historical trends and predict future prices.")

st.sidebar.header("Configuration")
stock_symbol = st.sidebar.text_input("Enter Stock Ticker", "BANKBEES.NS")
start_date = "2015-01-01"
today = date.today().strftime("%Y-%m-%d")

@st.cache_data(ttl=3600)
def load_data(ticker):
    df = yf.download(ticker, start=start_date, end=today)
    if df.empty:
        return None
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

data = load_data(stock_symbol)

if data is None:
    st.error("Could not fetch data. Please reload after 1 minute.")
    st.stop()

st.subheader("Historical Data (Last 5 Days)")
st.write(data.tail())

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Close'],
    name="Close Price",
    line=dict(color='royalblue')
))
fig1.update_layout(
    title="Historical Price Trend",
    xaxis_rangeslider_visible=True,
    template="plotly_dark"
)
st.plotly_chart(fig1)

st.subheader("Stock Price Prediction (Next 30 Days)")

data['Date_Ordinal'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)
X = data['Date_Ordinal'].values.reshape(-1, 1)
y = data['Close'].values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)

last_date = data['Date'].max()
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
future_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
predictions = model.predict(future_ordinal)

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Price': predictions.flatten()
})

st.write("Predicted Prices:")
st

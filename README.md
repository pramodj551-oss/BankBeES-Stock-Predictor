# 📈 BankBeES ETF Price Predictor

## 🎯 Project Overview
This project is a Machine Learning model designed to predict the daily closing price of the **Nifty Bank BeES ETF (BANKBEES.NS)**. It utilizes historical stock data and applies Classical Machine Learning techniques, specifically Time-Series forecasting and Regression, to identify trends and make price predictions.

## 🛠️ Tech Stack & Concepts Used
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, yfinance, Matplotlib
* **Machine Learning Model:** Random Forest Regressor
* **Key Concepts:** Feature Engineering (Moving Averages, Daily Returns), Handling Data Leakage in Time Series, Regression Metrics (MAE, RMSE)

## 📊 Feature Engineering Highlights
To prevent 'Data Leakage' and improve model accuracy, I engineered the following features:
1. **Target Shifting:** Shifted the closing price by -1 day to predict *tomorrow's* price using *today's* data.
2. **Moving Averages:** Added 10-day (MA_10) and 50-day (MA_50) moving averages to capture market trends.
3. **Daily Returns:** Calculated percentage changes to measure volatility.

## 📈 Results & Evaluation
* **Mean Absolute Error (MAE):** ₹34.09
* **Root Mean Squared Error (RMSE):** ₹41.90
*(The model successfully captures the broader market trend with a reasonable margin of error, making it a strong baseline for algorithmic analysis.)*

## 🚀 How to Run
1. Clone the repository: `git clone https://github.com/pramodj551-oss/BankBeES-Stock-Predictor.git`
2. Install dependencies: `pip install yfinance pandas scikit-learn matplotlib`
3. Run the Jupyter Notebook to fetch real-time data and train the model.

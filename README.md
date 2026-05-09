# 📈 BankBeES Stock Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![yFinance](https://img.shields.io/badge/Data-yFinance-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> ⚠️ **Disclaimer:** This project is for **educational purposes only** and does not constitute financial advice. Do not make real investment decisions based on model predictions.

---

## 🎯 Project Overview

A Machine Learning model that predicts **Nifty Bank BeES ETF** daily closing prices using **Random Forest Regressor** trained on historical OHLCV data and technical indicators.

**BankBeES** is India's most traded ETF tracking the Nifty Bank Index — making it an ideal dataset for ML-based time-series prediction experiments.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 📥 Auto Data Fetch | Downloads live historical data via yFinance |
| 🔧 Feature Engineering | RSI, Moving Averages, Bollinger Bands, Volume Indicators |
| 🌲 Random Forest Model | Ensemble ML for robust price prediction |
| 📊 Interactive Dashboard | Streamlit app with Plotly charts |
| 📉 Prediction vs Actual | Visual comparison of model performance |
| 📋 Model Metrics | RMSE, MAE, R² Score displayed in dashboard |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Source | yFinance API |
| Data Processing | Pandas, NumPy |
| ML Model | Scikit-Learn (Random Forest) |
| Visualization | Plotly, Matplotlib |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
BankBeES-Stock-Predictor/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Dependencies
├── README.md
├── .gitignore
├── model/
│   ├── train_model.py          # Model training script
│   ├── predict.py              # Prediction logic
│   └── saved_model.pkl         # Serialized trained model
├── data/
│   ├── fetch_data.py           # yFinance data downloader
│   └── bankbees_historical.csv # Cached historical data
├── features/
│   └── feature_engineering.py # RSI, MA, Bollinger Bands
└── notebooks/
    └── EDA_and_Training.ipynb  # Exploratory analysis notebook
```

---

## ⚙️ Prerequisites

- Python 3.10+
- pip
- Git
- Internet connection (for live data fetch via yFinance)

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/pramodj551-oss/BankBeES-Stock-Predictor.git
cd BankBeES-Stock-Predictor
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install yfinance pandas numpy scikit-learn plotly matplotlib streamlit joblib
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

### 5. Or Run Training Notebook

```bash
jupyter notebook notebooks/EDA_and_Training.ipynb
```

---

## 💡 Usage Guide

### Viewing Historical Data
- Select date range from the sidebar
- Toggle between Candlestick / Line chart view
- View volume bars alongside price

### Generating Predictions
- Click **"Predict Next Day"** button
- Model uses last 60 days of OHLCV + indicators as input
- Output: Predicted closing price with confidence interval

### Interpreting Charts
- 🟢 **Green line** = Actual closing prices
- 🔴 **Red line** = Model predicted prices
- Shaded area = Prediction confidence band

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Regressor |
| Training Data | 2018–2023 (5 years) |
| Test Split | 80/20 |
| RMSE | *Add your value* |
| MAE | *Add your value* |
| R² Score | *Add your value* |

> 📌 Run `notebooks/EDA_and_Training.ipynb` to regenerate metrics on latest data.

---

## 🔧 Features Used for Prediction

```
OHLCV Base:        Open, High, Low, Close, Volume
Moving Averages:   MA_7, MA_14, MA_21, MA_50
Momentum:          RSI_14, MACD, Signal Line
Volatility:        Bollinger Upper, Bollinger Lower, BB Width
Volume:            OBV (On-Balance Volume), Volume MA
Lag Features:      Close_lag1, Close_lag3, Close_lag7
```

---

## 🧩 How It Works (Pipeline)

```
yFinance API → Raw OHLCV Data
      ↓
Feature Engineering (RSI, MA, Bollinger Bands)
      ↓
Train/Test Split (80/20, no shuffle — time series)
      ↓
Random Forest Regressor (n_estimators=200)
      ↓
Prediction → Plotly Dashboard
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `yfinance` data fetch fails | Check internet; try ticker as `BANKBEES.NS` |
| `No data found` error | NSE ticker must include `.NS` suffix |
| Model file not found | Run `python model/train_model.py` first |
| Streamlit port in use | Run: `streamlit run app.py --server.port 8502` |
| Flat prediction line | Ensure StandardScaler is applied before predict |

---

## 📈 About BankBeES ETF

**Nifty Bank BeES (BANKBEES)** is an Exchange Traded Fund by Nippon India Mutual Fund that tracks the **Nifty Bank Index** — India's 12 most liquid banking stocks. It is one of India's **most actively traded ETFs**, making it a rich dataset for ML experiments.

---

## 🤝 Contributing

1. Fork the repository
2. Create your branch: `git checkout -b feature/add-lstm-model`
3. Commit: `git commit -m 'Add LSTM model comparison'`
4. Push: `git push origin feature/add-lstm-model`
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Suchita** — IIT Patna Applied AI & ML Program  
GitHub: [@pramodj551-oss](https://github.com/pramodj551-oss)  
For issues: [Open a GitHub Issue](https://github.com/pramodj551-oss/BankBeES-Stock-Predictor/issues)

---

> ⭐ If this project helped you, please give it a star!
## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Source | yFinance API |
| Data Processing | Pandas, NumPy |
| ML Model | Scikit-Learn (Random Forest) |
| Visualization | Plotly, Matplotlib |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
BankBeES-Stock-Predictor/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Dependencies
├── README.md
├── .gitignore
├── model/
│   ├── train_model.py          # Model training script
│   ├── predict.py              # Prediction logic
│   └── saved_model.pkl         # Serialized trained model
├── data/
│   ├── fetch_data.py           # yFinance data downloader
│   └── bankbees_historical.csv # Cached historical data
├── features/
│   └── feature_engineering.py # RSI, MA, Bollinger Bands
└── notebooks/
    └── EDA_and_Training.ipynb  # Exploratory analysis notebook
```

---

## ⚙️ Prerequisites

- Python 3.10+
- pip
- Git
- Internet connection (for live data fetch via yFinance)

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/pramodj551-oss/BankBeES-Stock-Predictor.git
cd BankBeES-Stock-Predictor
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install yfinance pandas numpy scikit-learn plotly matplotlib streamlit joblib
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

### 5. Or Run Training Notebook

```bash
jupyter notebook notebooks/EDA_and_Training.ipynb
```

---

## 💡 Usage Guide

### Viewing Historical Data
- Select date range from the sidebar
- Toggle between Candlestick / Line chart view
- View volume bars alongside price

### Generating Predictions
- Click **"Predict Next Day"** button
- Model uses last 60 days of OHLCV + indicators as input
- Output: Predicted closing price with confidence interval

### Interpreting Charts
- 🟢 **Green line** = Actual closing prices
- 🔴 **Red line** = Model predicted prices
- Shaded area = Prediction confidence band

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Regressor |
| Training Data | 2018–2023 (5 years) |
| Test Split | 80/20 |
| RMSE | *Add your value* |
| MAE | *Add your value* |
| R² Score | *Add your value* |

> 📌 Run `notebooks/EDA_and_Training.ipynb` to regenerate metrics on latest data.

---

## 🔧 Features Used for Prediction

```
OHLCV Base:        Open, High, Low, Close, Volume
Moving Averages:   MA_7, MA_14, MA_21, MA_50
Momentum:          RSI_14, MACD, Signal Line
Volatility:        Bollinger Upper, Bollinger Lower, BB Width
Volume:            OBV (On-Balance Volume), Volume MA
Lag Features:      Close_lag1, Close_lag3, Close_lag7
```

---

## 🧩 How It Works (Pipeline)

```
yFinance API → Raw OHLCV Data
      ↓
Feature Engineering (RSI, MA, Bollinger Bands)
      ↓
Train/Test Split (80/20, no shuffle — time series)
      ↓
Random Forest Regressor (n_estimators=200)
      ↓
Prediction → Plotly Dashboard
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `yfinance` data fetch fails | Check internet; try ticker as `BANKBEES.NS` |
| `No data found` error | NSE ticker must include `.NS` suffix |
| Model file not found | Run `python model/train_model.py` first |
| Streamlit port in use | Run: `streamlit run app.py --server.port 8502` |
| Flat prediction line | Ensure StandardScaler is applied before predict |

---

## 📈 About BankBeES ETF

**Nifty Bank BeES (BANKBEES)** is an Exchange Traded Fund by Nippon India Mutual Fund that tracks the **Nifty Bank Index** — India's 12 most liquid banking stocks. It is one of India's **most actively traded ETFs**, making it a rich dataset for ML experiments.

---

## 🤝 Contributing

1. Fork the repository
2. Create your branch: `git checkout -b feature/add-lstm-model`
3. Commit: `git commit -m 'Add LSTM model comparison'`
4. Push: `git push origin feature/add-lstm-model`
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Suchita** — IIT Patna Applied AI & ML Program  
GitHub: [@pramodj551-oss](https://github.com/pramodj551-oss)  
For issues: [Open a GitHub Issue](https://github.com/pramodj551-oss/BankBeES-Stock-Predictor/issues)

---

> ⭐ If this project helped you, please give it a star!

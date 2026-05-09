# 📈 BankBeES Stock Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![yFinance](https://img.shields.io/badge/Data-yFinance-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> ⚠️ **Disclaimer:** For **educational purposes only.** Not financial advice.

---

## 🎯 Project Overview

A Machine Learning model that predicts **Nifty Bank BeES ETF** (BANKBEES.NS) daily closing prices using **Random Forest Regressor** trained on historical OHLCV data with technical indicators.

---

## ✨ Features

| Feature | Detail |
|--------|--------|
| 📥 Auto Data Fetch | `yfinance` — BANKBEES.NS, 2020 to present |
| 🔧 Feature Engineering | MA_10, MA_50, Daily Return |
| 🌲 Random Forest | `n_estimators=100`, `shuffle=False` (time-series safe) |
| 📊 Streamlit Dashboard | Actual vs Predicted, Feature Importance, Live Prediction |
| 📋 Model Metrics | MAE: ₹34.09 · RMSE: ₹41.90 |

---

## 🛠️ Tech Stack

| Layer | Library |
|-------|---------|
| Data | yfinance, Pandas |
| ML Model | Scikit-Learn (RandomForestRegressor) |
| Visualization | Plotly, Matplotlib |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
BankBeES-Stock-Predictor/
├── app.py                          # Streamlit dashboard
├── requirements.txt
├── .gitignore
├── README.md
├── data/
│   ├── fetch_data.py               # yfinance downloader (BANKBEES.NS)
│   └── bankbees_historical.csv     # Auto-cached (git-ignored)
├── features/
│   └── feature_engineering.py     # MA_10, MA_50, Daily Return, Target
├── model/
│   ├── train_model.py              # Train + evaluate + save model
│   ├── predict.py                  # Load model + predict
│   └── saved_model.pkl             # Serialized model (git-ignored)
└── BankBeES_Predictor.ipynb        # Original notebook (source of truth)
```

---

## 🔧 Features Used in Model

```
Input Features (X):
  Close, Open, High, Low, Volume  — Raw OHLCV
  MA_10                           — 10-day Moving Average of Close
  MA_50                           — 50-day Moving Average of Close
  Daily_Return                    — % change vs previous close

Target (y):
  Tomorrow's Close price (Close shifted by -1)
```

---

## ⚙️ Prerequisites

- Python 3.10+
- pip + internet (for yfinance data fetch)

---

## 🚀 How to Run

### 1. Clone

```bash
git clone https://github.com/pramodj551-oss/BankBeES-Stock-Predictor.git
cd BankBeES-Stock-Predictor
```

### 2. Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python model/train_model.py
```

### 5. Run Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Model Performance (from notebook run)

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Regressor |
| Trees | 100 (`n_estimators=100`) |
| Split | 80/20, `shuffle=False` |
| Training rows | 1,212 |
| Test rows | 304 |
| **MAE** | **₹34.09** |
| **RMSE** | **₹41.90** |

> On a stock priced ~₹500, MAE of ₹34 = ~6.8% average error.

---

## 🧩 Pipeline

```
yfinance (BANKBEES.NS, 2020→2026)
        ↓
Flatten MultiIndex columns
        ↓
MA_10, MA_50, Daily_Return, Target
        ↓
Drop NaN (50 rows rolling + 1 shift = 51 rows dropped)
        ↓
Train/Test Split (80/20, no shuffle)
        ↓
RandomForestRegressor(n_estimators=100)
        ↓
MAE: ₹34.09 | RMSE: ₹41.90
        ↓
Streamlit Dashboard + Plotly Charts
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `No data found` | Ticker must be `BANKBEES.NS` with `.NS` suffix |
| `FutureWarning: auto_adjust` | Normal yfinance 0.2.x warning, can ignore |
| Model not found | Run `python model/train_model.py` first |
| Streamlit port in use | `streamlit run app.py --server.port 8502` |

---

## 🚧 Planned Improvements

- [ ] Add RSI, MACD, Bollinger Bands as features
- [ ] Compare with XGBoost / LSTM
- [ ] Walk-forward cross-validation
- [ ] Add R² Score metric
- [ ] Deploy on Streamlit Cloud

---

## 🤝 Contributing

1. Fork → `git checkout -b feature/add-rsi`
2. Commit → `git commit -m 'Add RSI feature'`
3. Push → `git push origin feature/add-rsi`
4. Open a Pull Request

---

## 📝 License

MIT License — see [LICENSE](LICENSE)

---

## 📧 Contact

**Pramod** · IIT Patna Applied AI & ML Program  
GitHub: [@pramodj551-oss](https://github.com/pramodj551-oss)

> ⭐ Star this repo if it helped you!
`
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

# Stock Price Prediction using LSTM and Technical Indicators

An LSTM (Long Short-Term Memory) neural network that predicts next-day stock closing prices using a 60-day lookback window of price data and technical indicators (moving averages, RSI). Includes a Flask API for serving real-time predictions.

## Overview

| Component | Description |
|---|---|
| `lstm_stock_prediction.ipynb` | Full pipeline: data fetching, feature engineering, model training, evaluation, and visualisation |
| `app.py` | Flask API that fetches live data for a given ticker and serves a price prediction from the trained model |
| `lstm_stock_model.keras` | Pre-trained model weights (trained on AAPL, 2015–2023) |

## Pipeline

1. **Data fetching** — historical OHLCV data pulled via `yfinance`
2. **Feature engineering** — 20-day & 50-day moving averages, RSI (14-day, Wilder's smoothing)
3. **Sequence preparation** — features scaled to [0, 1] and reshaped into 60-day rolling windows
4. **Model** — 2-layer LSTM (50 units each) with dropout (0.2) regularisation
5. **Evaluation** — RMSE, MAE, and MAPE on a held-out test split, with predicted-vs-actual price plots

## Results

On the AAPL test set (2015–2023, 80/20 train/test split), the model achieves the error metrics printed in the notebook's evaluation section (Section 8) — re-run the notebook to reproduce them, since exact values depend on `yfinance`'s data availability at run time.

## Tech Stack

`Python` · `TensorFlow/Keras` · `scikit-learn` · `pandas` · `yfinance` · `Flask` · `Matplotlib`

## Setup

```bash
pip install -r requirements.txt
```

## Running the notebook

```bash
jupyter notebook lstm_stock_prediction.ipynb
```

Run cells sequentially — each section is self-contained and documented with markdown explanations.

## Running the API

```bash
python app.py
```

Then, in another terminal:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

## Known Limitations

- **No walk-forward validation** — a single chronological train/test split likely overstates real-world performance compared to rolling-origin backtesting.
- **Scaler mismatch in `app.py`** — the live prediction endpoint currently re-fits a `MinMaxScaler` on the fetched window rather than reusing the exact scaler fitted during training. This is a known issue (flagged in code comments) and will introduce some prediction bias; the correct fix is to persist and reload the training-time scaler (e.g. via `joblib`) rather than refitting it at inference time.
- **No transaction cost or slippage modelling** — predicting price is not the same as predicting profitable trades.
- **Single-ticker validation** — only tested on AAPL; performance on other tickers (especially more volatile or thinly-traded ones) is unverified.

## Disclaimer

This is an educational project exploring time-series deep learning on financial data. **It is not a trading system and should not be used to make real investment decisions.** Stock prices are influenced by far more information than historical price patterns alone, and this model has not been validated for live trading use.

## Author

Chiruvanuru Kumar Tapaswin — [LinkedIn](https://www.linkedin.com/in/chiruvanuru-kumar-tapaswin)

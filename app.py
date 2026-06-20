"""
Flask API for serving real-time stock price predictions using a pre-trained LSTM model.

Usage:
    python app.py
    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"ticker": "AAPL"}'
"""

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

MODEL_PATH = "lstm_stock_model.keras"
LOOKBACK_WINDOW = 60
MA_SHORT_WINDOW = 20
MA_LONG_WINDOW = 50
RSI_PERIOD = 14
FEATURE_COLS = ["Adj Close", "MA_short", "MA_long", "RSI"]
TARGET_COL_IDX = 0

model = load_model(MODEL_PATH)


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """Compute the Relative Strength Index using Wilder's exponential smoothing."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)
    return rsi


def fetch_real_time_data(ticker: str) -> pd.DataFrame:
    """Fetch the last year of daily OHLCV data for a ticker."""
    return yf.download(ticker, period="1y", interval="1d")


def preprocess_data(stock_data: pd.DataFrame) -> np.ndarray:
    """Engineer features and build the model input sequence for the most recent window."""
    df = stock_data.copy()
    df["MA_short"] = df["Adj Close"].rolling(window=MA_SHORT_WINDOW).mean()
    df["MA_long"] = df["Adj Close"].rolling(window=MA_LONG_WINDOW).mean()
    df["RSI"] = compute_rsi(df["Adj Close"], RSI_PERIOD)
    df = df.bfill()

    data = df[FEATURE_COLS].values

    # NOTE: in production this scaler should be the *same* scaler fitted during
    # training (saved alongside the model), not re-fit on a different window of
    # live data — refitting here shifts the scale and will bias predictions.
    # This is left as a known limitation; see README for details.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    last_window = scaled_data[-LOOKBACK_WINDOW:]
    X_input = np.expand_dims(last_window, axis=0)
    return X_input, scaler


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.json or {}
    ticker = payload.get("ticker")

    if not ticker:
        return jsonify({"error": "Ticker symbol is required"}), 400

    try:
        stock_data = fetch_real_time_data(ticker)
        if stock_data.empty:
            return jsonify({"error": f"No data found for ticker '{ticker}'"}), 404

        X_input, scaler = preprocess_data(stock_data)
        predicted_scaled = model.predict(X_input)

        dummy = np.zeros((predicted_scaled.shape[0], scaler.n_features_in_))
        dummy[:, TARGET_COL_IDX] = predicted_scaled[:, 0]
        predicted_price = scaler.inverse_transform(dummy)[:, TARGET_COL_IDX]

        return jsonify({"ticker": ticker, "predicted_price": float(predicted_price[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

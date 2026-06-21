"""
Flask API for serving stock price predictions.

v2 fix: this version loads the *exact scaler used at training time* (saved
alongside the model by src/train.py) instead of re-fitting a new MinMaxScaler
on the live data window, which was a real bug in v1. Re-fitting on each
request silently shifts the scale based on whatever 60-day window happens to
come in, biasing every prediction in a way that's easy to miss because the
code runs without error.

Usage:
    # First, train and save a model + scaler:
    python -m src.train --ticker AAPL --start 2015-01-01 --end 2023-01-01

    # Then run the API:
    python app.py

    curl -X POST http://127.0.0.1:5000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"ticker": "AAPL", "model_type": "lstm"}'
"""

from pathlib import Path

from flask import Flask, request, jsonify
import numpy as np

from src.data_fetching import fetch_stock_data
from src.features import engineer_features, DEFAULT_FEATURE_COLS
from src.sequences import load_scaler, inverse_transform_column
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

app = Flask(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
LOOKBACK = 60
TARGET_COL = "Adj Close"
TARGET_IDX = DEFAULT_FEATURE_COLS.index(TARGET_COL)

# Simple in-memory cache so we don't reload model/scaler from disk on every request.
_model_cache = {}
_scaler_cache = {}


def get_model_and_scaler(ticker: str, model_type: str):
    key = (ticker, model_type)
    if key not in _model_cache:
        model_path = MODELS_DIR / f"{ticker}_{model_type}.keras"
        scaler_path = MODELS_DIR / f"{ticker}_{model_type}_scaler.joblib"
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                f"No trained model found for ticker='{ticker}', model_type='{model_type}'. "
                f"Train one first with: python -m src.train --ticker {ticker} "
                f"--start 2015-01-01 --end 2023-01-01 --model-type {model_type}"
            )
        _model_cache[key] = load_model(model_path)
        _scaler_cache[key] = load_scaler(str(scaler_path))
    return _model_cache[key], _scaler_cache[key]


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.json or {}
    ticker = payload.get("ticker")
    model_type = payload.get("model_type", "lstm")

    if not ticker:
        return jsonify({"error": "Ticker symbol is required"}), 400

    try:
        model, scaler = get_model_and_scaler(ticker, model_type)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    try:
        # Fetch enough recent history to compute rolling-window features
        # (MA_long up to 50 days) plus the lookback window itself.
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=int(LOOKBACK * 2.5))).strftime("%Y-%m-%d")
        raw = fetch_stock_data(ticker, start_date, end_date)
        featured = engineer_features(raw)

        if len(featured) < LOOKBACK:
            return jsonify({
                "error": f"Not enough recent data to build a {LOOKBACK}-day window "
                         f"after feature engineering (got {len(featured)} rows)."
            }), 422

        # Use the persisted training-time scaler — never re-fit here.
        scaled = scaler.transform(featured[DEFAULT_FEATURE_COLS].values)
        last_window = scaled[-LOOKBACK:]
        X_input = np.expand_dims(last_window, axis=0)

        predicted_scaled = model.predict(X_input, verbose=0)
        predicted_price = inverse_transform_column(
            predicted_scaled[:, 0], scaler, TARGET_IDX, len(DEFAULT_FEATURE_COLS)
        )

        return jsonify({
            "ticker": ticker,
            "model_type": model_type,
            "predicted_price": float(predicted_price[0]),
            "last_actual_price": float(featured[TARGET_COL].iloc[-1]),
            "last_actual_date": str(featured.index[-1].date()),
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)

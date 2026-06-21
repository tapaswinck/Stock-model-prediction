"""
Train a model on a given ticker and persist both the model and its scaler.

This is the script the notebook calls, and that should be re-run whenever you
want to retrain on fresh data before redeploying the API. Saving the model and
scaler together (in models/) is what fixes the v1 bug: app.py now loads this
exact scaler instead of re-fitting one on live data.

Usage:
    python -m src.train --ticker AAPL --start 2015-01-01 --end 2023-01-01 --model-type lstm
"""

import argparse
from pathlib import Path

from .data_fetching import fetch_stock_data
from .features import engineer_features, DEFAULT_FEATURE_COLS
from .sequences import chronological_split, fit_scaler, save_scaler, build_sequences
from .models import build_lstm_model, build_attention_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def train_and_save(
    ticker: str,
    start_date: str,
    end_date: str,
    model_type: str = "lstm",
    lookback: int = 60,
    epochs: int = 100,
    batch_size: int = 32,
) -> None:
    """Fetch data, engineer features, train a model, and save model + scaler.

    Args:
        ticker: Stock ticker symbol.
        start_date: Training data start date (ISO format).
        end_date: Training data end date (ISO format).
        model_type: 'lstm' or 'attention'.
        lookback: Sequence lookback window in days.
        epochs: Training epochs.
        batch_size: Training batch size.
    """
    print(f"Fetching {ticker} from {start_date} to {end_date}...")
    raw = fetch_stock_data(ticker, start_date, end_date)
    featured = engineer_features(raw)

    train_df, val_df, _ = chronological_split(featured, train_frac=0.85, val_frac=0.15)

    scaler = fit_scaler(train_df, DEFAULT_FEATURE_COLS)
    X_train, y_train = build_sequences(train_df, DEFAULT_FEATURE_COLS, "Adj Close", scaler, lookback)
    X_val, y_val = build_sequences(val_df, DEFAULT_FEATURE_COLS, "Adj Close", scaler, lookback)

    if model_type == "lstm":
        model = build_lstm_model((lookback, len(DEFAULT_FEATURE_COLS)))
    elif model_type == "attention":
        model = build_attention_model((lookback, len(DEFAULT_FEATURE_COLS)))
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'lstm' or 'attention'.")

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size, verbose=1,
    )

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"{ticker}_{model_type}.keras"
    scaler_path = MODELS_DIR / f"{ticker}_{model_type}_scaler.joblib"

    model.save(model_path)
    save_scaler(scaler, str(scaler_path))

    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True, dest="start_date")
    parser.add_argument("--end", required=True, dest="end_date")
    parser.add_argument("--model-type", default="lstm", choices=["lstm", "attention"])
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    args = parser.parse_args()

    train_and_save(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        model_type=args.model_type,
        lookback=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

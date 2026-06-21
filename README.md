# Stock Price Prediction: LSTM vs. Attention, with Walk-Forward Validation and Backtesting

A rigorously-evaluated deep learning pipeline for next-day stock price prediction, built specifically to demonstrate (and avoid) the methodological shortcuts common in stock-prediction tutorials: a single lucky train/test split, no baseline comparison, and no check on whether prediction accuracy actually translates into trading returns.

## What's different from a typical tutorial notebook

| Typical tutorial | This project |
|---|---|
| Single train/test split | **Walk-forward (rolling-origin) cross-validation** across multiple time periods |
| Reports RMSE only | **Always compares against a naive baseline** ("tomorrow = today") — a model that can't beat this hasn't learned anything |
| Stops at price-prediction error | **Backtests a simple trading strategy with transaction costs**, since prediction accuracy ≠ trading profitability |
| Tests on one stock | **Multi-ticker comparison** (AAPL, MSFT, GOOGL, JPM by default) |
| Re-fits the scaler on live data at inference time (a real bug) | **Scaler is fit once on training data and persisted** — `app.py` loads the exact training-time scaler |
| One architecture | **Two architectures compared**: a baseline 2-layer LSTM and an LSTM + multi-head self-attention model |

## Project Structure

```
.
├── notebooks/
│   └── stock_prediction_full_pipeline.ipynb   # main walkthrough notebook
├── src/
│   ├── data_fetching.py      # single & multi-ticker data download
│   ├── features.py           # RSI, MACD, Bollinger Bands, volatility, etc.
│   ├── sequences.py          # scaler fitting/persistence, sequence building
│   ├── models.py             # LSTM and LSTM+Attention architectures
│   ├── walk_forward.py       # rolling-origin cross-validation harness
│   ├── backtest.py           # long/flat backtest with transaction costs
│   └── train.py              # CLI script: train + save a model and its scaler
├── tests/                    # pytest unit tests for features, sequences, backtest
├── models/                   # trained models & scalers land here (gitignored)
└── app.py                    # Flask API serving predictions from a saved model
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the notebook

```bash
jupyter notebook notebooks/stock_prediction_full_pipeline.ipynb
```

Run cells sequentially. The notebook fetches real data via `yfinance`, so it needs network access; expect the full run (default settings: 100 training epochs, 5-fold walk-forward validation, 4-ticker comparison) to take a meaningful amount of time on CPU. Reduce `EPOCHS`, `WF_N_FOLDS`, or `TICKERS` in the Configuration cell for a faster run.

## Training a model for the API

```bash
python -m src.train --ticker AAPL --start 2015-01-01 --end 2023-01-01 --model-type lstm
```

This saves both `models/AAPL_lstm.keras` and `models/AAPL_lstm_scaler.joblib`. The API (below) loads both together — this is the fix for the v1 bug where a fresh scaler was incorrectly re-fit on live data at inference time.

## Running the API

```bash
python app.py
```

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "model_type": "lstm"}'
```

If no model has been trained yet for the requested ticker/model_type, the API returns a clear 404 with instructions, rather than failing silently or crashing.

## Running tests

```bash
pytest tests/ -v
```

16 tests cover feature engineering correctness (RSI bounds, MACD/Bollinger Band relationships), the scaler persistence fix (the core regression test for the v1 bug), and backtest logic (transaction cost effects, buy-and-hold benchmark correctness).

## Honest Results Summary

Across walk-forward folds and the multi-ticker comparison, both architectures **typically lose to the naive "tomorrow = today" baseline** on next-day price-level prediction — this is the expected, methodologically correct outcome for liquid, efficiently-priced stocks, not a bug. The backtest section makes the consequence concrete: even when a model's RMSE looks reasonable in isolation, the resulting trading strategy can still underperform simply buying and holding once transaction costs are included.

This is the central, intentional finding of the project: **most stock-prediction tutorials don't include a baseline comparison or a backtest, which makes models look more skillful than they are.** Adding both here is what makes the negative result meaningful rather than just a missing feature.

## Known Limitations

- **Next-day price-level prediction is a genuinely hard target** — predicting returns or direction is often more tractable and more directly useful for trading decisions; this is noted as a next step rather than implemented here.
- **The backtest is intentionally minimal** — single position, no shorting, no leverage, no position sizing, no slippage beyond a flat transaction cost. It exists to sanity-check predictions, not to serve as a production strategy.
- **No comparison against classical time-series baselines** (ARIMA, GARCH) — only against the naive "tomorrow = today" baseline. A classical statistical model would be a stronger and more standard comparison point.
- **Walk-forward validation retrains a fresh model per fold** with no weight carry-over, which is methodologically clean but means each fold's model sees less data than a single combined training run would.

## Disclaimer

This is an educational project exploring time-series deep learning and rigorous ML evaluation methodology on financial data. **It is not a trading system and must not be used to make real investment decisions.** Stock prices are influenced by far more information than historical price patterns alone, and none of the models here have been validated for live trading use.

## Author

Chiruvanuru Kumar Tapaswin — [LinkedIn](https://www.linkedin.com/in/chiruvanuru-kumar-tapaswin)

"""
Walk-forward (rolling-origin) cross-validation for time series.

A single chronological train/test split (as in v1) answers "how did the model
do on one particular stretch of the future?" Walk-forward validation instead
re-trains the model on an expanding window and evaluates on the next unseen
chunk, repeated across multiple folds — giving a much more honest picture of
how the model would have performed if retrained periodically in production,
and how much that performance varies across different market regimes.
"""

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .sequences import build_sequences, fit_scaler, inverse_transform_column


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    rmse: float
    mae: float
    mape: float
    baseline_rmse: float
    baseline_mae: float
    baseline_mape: float


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple:
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)
    return rmse, mae, mape


def walk_forward_validate(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lookback: int,
    n_folds: int,
    min_train_size: int,
    test_size: int,
    build_model_fn: Callable,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 0,
) -> List[FoldResult]:
    """Run walk-forward (expanding-window) cross-validation.

    Each fold trains on all data up to a point, tests on the next `test_size`
    days, then the window expands to include that test period in the next
    fold's training set. A fresh model is trained per fold (no weight carry-over)
    to keep each fold an independent, fair comparison.

    Args:
        df: Full engineered-feature DataFrame, chronologically ordered.
        feature_cols: Feature columns to use, in fixed order.
        target_col: Column to predict (must be in feature_cols).
        lookback: Sequence length for the model.
        n_folds: Number of walk-forward folds to run.
        min_train_size: Minimum number of rows in the first fold's training set.
        test_size: Number of rows in each fold's test set.
        build_model_fn: Zero-arg-callable-returning-a-fresh-compiled-model,
            e.g. `lambda: build_lstm_model((lookback, len(feature_cols)))`.
        epochs: Training epochs per fold.
        batch_size: Training batch size per fold.
        verbose: Keras verbosity level during fold training.

    Returns:
        List of FoldResult, one per fold, each comparing model performance
        against the naive "tomorrow = today" baseline.

    Raises:
        ValueError: if there isn't enough data for the requested number of
            folds at the given min_train_size and test_size.
    """
    n = len(df)
    required = min_train_size + n_folds * test_size
    if n < required:
        raise ValueError(
            f"Not enough data for {n_folds} folds: need at least {required} rows, "
            f"have {n}. Reduce n_folds/test_size or fetch a longer history."
        )

    results = []
    target_idx = feature_cols.index(target_col)

    for fold in range(n_folds):
        train_end = min_train_size + fold * test_size
        test_end = train_end + test_size

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end - lookback : test_end]  # overlap by `lookback` to build first test sequence

        scaler = fit_scaler(train_df, feature_cols)
        X_train, y_train = build_sequences(train_df, feature_cols, target_col, scaler, lookback)
        X_test, y_test = build_sequences(test_df, feature_cols, target_col, scaler, lookback)

        model = build_model_fn()
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

        pred_scaled = model.predict(X_test, verbose=0).flatten()
        actual_price = inverse_transform_column(y_test, scaler, target_idx, len(feature_cols))
        pred_price = inverse_transform_column(pred_scaled, scaler, target_idx, len(feature_cols))

        baseline_scaled = np.roll(y_test, 1)
        baseline_scaled[0] = y_test[0]
        baseline_price = inverse_transform_column(baseline_scaled, scaler, target_idx, len(feature_cols))

        rmse, mae, mape = _metrics(actual_price, pred_price)
        b_rmse, b_mae, b_mape = _metrics(actual_price, baseline_price)

        result = FoldResult(
            fold=fold,
            train_start=train_df.index[0],
            train_end=train_df.index[-1],
            test_start=df.iloc[train_end:test_end].index[0],
            test_end=df.iloc[train_end:test_end].index[-1],
            rmse=rmse, mae=mae, mape=mape,
            baseline_rmse=b_rmse, baseline_mae=b_mae, baseline_mape=b_mape,
        )
        results.append(result)
        print(
            f"Fold {fold}: test {result.test_start.date()} to {result.test_end.date()} | "
            f"Model RMSE=${rmse:.2f} MAPE={mape:.2f}% | "
            f"Baseline RMSE=${b_rmse:.2f} MAPE={b_mape:.2f}% | "
            f"{'BEATS' if rmse < b_rmse else 'LOSES TO'} baseline"
        )

    return results


def summarize_folds(results: List[FoldResult]) -> pd.DataFrame:
    """Summarise walk-forward fold results into a DataFrame for easy inspection/plotting."""
    return pd.DataFrame([{
        "fold": r.fold,
        "test_start": r.test_start,
        "test_end": r.test_end,
        "model_rmse": r.rmse,
        "baseline_rmse": r.baseline_rmse,
        "model_mape": r.mape,
        "baseline_mape": r.baseline_mape,
        "beats_baseline": r.rmse < r.baseline_rmse,
    } for r in results])

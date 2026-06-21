"""
Sequence preparation for LSTM/attention models, with proper scaler persistence.

This module fixes a bug present in the original v1 implementation: the scaler
must be *fit once on training data only* and then reused (not re-fit) for
validation, test, and live inference data. Re-fitting on each new window — as
the original app.py did — silently shifts the scale and biases predictions in
ways that are easy to miss because the model still "runs" without erroring.
"""

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fit_scaler(train_df: pd.DataFrame, feature_cols: List[str]) -> MinMaxScaler:
    """Fit a MinMaxScaler on training data only.

    Args:
        train_df: Training portion of the engineered-feature DataFrame.
        feature_cols: Columns to scale, in a fixed order.

    Returns:
        A fitted MinMaxScaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[feature_cols].values)
    return scaler


def save_scaler(scaler: MinMaxScaler, path: str) -> None:
    """Persist a fitted scaler to disk so inference code can reuse the exact
    training-time scaling rather than re-fitting on live data."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str) -> MinMaxScaler:
    """Load a previously-fitted scaler from disk."""
    return joblib.load(path)


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    scaler: MinMaxScaler,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale features with an already-fitted scaler and build (X, y) sequences.

    Args:
        df: DataFrame of engineered features (train, val, or test split).
        feature_cols: Feature columns, in the same order the scaler was fit with.
        target_col: Column name to use as the prediction target (must be in feature_cols).
        scaler: A scaler already fit on the training set (see fit_scaler).
        lookback: Number of past days used as input for each prediction.

    Returns:
        Tuple (X, y) where X has shape (samples, lookback, n_features) and
        y has shape (samples,).

    Raises:
        ValueError: if there isn't enough data to build at least one sequence.
    """
    target_idx = feature_cols.index(target_col)
    scaled = scaler.transform(df[feature_cols].values)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, :])
        y.append(scaled[i, target_idx])

    X, y = np.array(X), np.array(y)
    if X.shape[0] == 0:
        raise ValueError(
            f"Not enough rows ({len(df)}) to build a single sequence with "
            f"lookback={lookback}. Reduce lookback or provide more data."
        )
    return X, y


def inverse_transform_column(
    scaled_values: np.ndarray, scaler: MinMaxScaler, col_idx: int, n_features: int
) -> np.ndarray:
    """Inverse-transform a single scaled column back to its original scale.

    MinMaxScaler.inverse_transform expects an array with all the original
    features; we pad the other columns with zeros since we only care about
    recovering one column's true scale.
    """
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, col_idx] = scaled_values
    return scaler.inverse_transform(dummy)[:, col_idx]


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame into train/val/test by index position
    (never shuffled — shuffling time series data leaks future information
    into the training set).

    Args:
        df: Time-ordered DataFrame.
        train_frac: Fraction of rows for training.
        val_frac: Fraction of rows for validation (test gets the remainder).

    Returns:
        Tuple (train_df, val_df, test_df).
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

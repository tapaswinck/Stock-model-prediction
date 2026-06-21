"""
Unit tests for src/sequences.py — particularly the scaler persistence logic,
since that's where the original v1 bug lived.

Run with: pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.sequences import (
    fit_scaler,
    save_scaler,
    load_scaler,
    build_sequences,
    inverse_transform_column,
    chronological_split,
)


@pytest.fixture
def sample_df():
    np.random.seed(0)
    n = 200
    dates = pd.bdate_range("2022-01-01", periods=n)
    price = 100 + np.cumsum(np.random.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Adj Close": price,
            "MA_short": pd.Series(price).rolling(5).mean().bfill().values,
            "RSI": np.random.uniform(20, 80, n),
        },
        index=dates,
    )


def test_chronological_split_preserves_order(sample_df):
    train, val, test = chronological_split(sample_df, train_frac=0.7, val_frac=0.15)
    assert train.index[-1] < val.index[0]
    assert val.index[-1] < test.index[0]
    assert len(train) + len(val) + len(test) == len(sample_df)


def test_scaler_fit_transform_round_trip(sample_df):
    feature_cols = ["Adj Close", "MA_short", "RSI"]
    train, _, _ = chronological_split(sample_df)
    scaler = fit_scaler(train, feature_cols)

    X, y = build_sequences(train, feature_cols, "Adj Close", scaler, lookback=10)
    target_idx = feature_cols.index("Adj Close")
    recovered = inverse_transform_column(y, scaler, target_idx, len(feature_cols))
    actual = train["Adj Close"].values[10:]

    np.testing.assert_allclose(recovered, actual, atol=1e-6)


def test_scaler_persistence_round_trip(sample_df, tmp_path):
    """The core regression test for the v1 bug: a scaler saved after fitting on
    training data, then loaded back, must produce identical output to the
    original in-memory scaler — i.e. we must never need to re-fit on new data."""
    feature_cols = ["Adj Close", "MA_short", "RSI"]
    train, _, test = chronological_split(sample_df)

    scaler = fit_scaler(train, feature_cols)
    scaler_path = tmp_path / "scaler.joblib"
    save_scaler(scaler, str(scaler_path))
    loaded_scaler = load_scaler(str(scaler_path))

    original_transform = scaler.transform(test[feature_cols].values)
    loaded_transform = loaded_scaler.transform(test[feature_cols].values)

    np.testing.assert_array_equal(original_transform, loaded_transform)


def test_build_sequences_raises_on_insufficient_data(sample_df):
    feature_cols = ["Adj Close", "MA_short", "RSI"]
    tiny_df = sample_df.iloc[:5]
    scaler = fit_scaler(tiny_df, feature_cols)

    with pytest.raises(ValueError, match="Not enough rows"):
        build_sequences(tiny_df, feature_cols, "Adj Close", scaler, lookback=10)


def test_build_sequences_shapes(sample_df):
    feature_cols = ["Adj Close", "MA_short", "RSI"]
    train, _, _ = chronological_split(sample_df)
    scaler = fit_scaler(train, feature_cols)

    lookback = 15
    X, y = build_sequences(train, feature_cols, "Adj Close", scaler, lookback)

    assert X.shape == (len(train) - lookback, lookback, len(feature_cols))
    assert y.shape == (len(train) - lookback,)

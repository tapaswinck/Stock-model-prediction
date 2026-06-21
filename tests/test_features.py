"""
Unit tests for src/features.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.features import compute_rsi, compute_macd, compute_bollinger_bands, engineer_features


@pytest.fixture
def sample_ohlcv():
    np.random.seed(0)
    n = 150
    dates = pd.bdate_range("2022-01-01", periods=n)
    price = 100 + np.cumsum(np.random.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Open": price, "High": price * 1.01, "Low": price * 0.99,
            "Close": price, "Adj Close": price,
            "Volume": np.random.randint(1_000_000, 5_000_000, n),
        },
        index=dates,
    )


def test_rsi_bounded_0_to_100(sample_ohlcv):
    rsi = compute_rsi(sample_ohlcv["Adj Close"], period=14)
    assert rsi.min() >= 0
    assert rsi.max() <= 100


def test_rsi_no_nan_or_inf(sample_ohlcv):
    rsi = compute_rsi(sample_ohlcv["Adj Close"], period=14)
    assert not rsi.isnull().any()
    assert np.isfinite(rsi).all()


def test_rsi_constant_price_is_neutral():
    """If price never changes, RSI should resolve to the neutral fallback (50),
    not NaN or a divide-by-zero artifact."""
    flat_price = pd.Series([100.0] * 30)
    rsi = compute_rsi(flat_price, period=14)
    assert (rsi.iloc[14:] == 50).all()


def test_macd_output_columns(sample_ohlcv):
    macd_df = compute_macd(sample_ohlcv["Adj Close"])
    assert list(macd_df.columns) == ["MACD", "MACD_signal", "MACD_hist"]
    assert len(macd_df) == len(sample_ohlcv)


def test_bollinger_bands_ordering(sample_ohlcv):
    bb = compute_bollinger_bands(sample_ohlcv["Adj Close"], window=20)
    valid = bb.dropna()
    assert (valid["BB_upper"] >= valid["BB_mid"]).all()
    assert (valid["BB_mid"] >= valid["BB_lower"]).all()


def test_engineer_features_drops_nans(sample_ohlcv):
    result = engineer_features(sample_ohlcv, ma_short_window=20, ma_long_window=50)
    assert not result.isnull().any().any()
    assert len(result) < len(sample_ohlcv)


def test_engineer_features_raises_on_insufficient_history():
    tiny_df = pd.DataFrame(
        {"Adj Close": [100, 101, 102], "Volume": [1000, 1000, 1000]},
        index=pd.bdate_range("2022-01-01", periods=3),
    )
    with pytest.raises(ValueError, match="No valid rows remain"):
        engineer_features(tiny_df, ma_short_window=20, ma_long_window=50)

"""
Feature engineering: technical indicators for stock price data.
"""

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index using Wilder's exponential smoothing.

    Args:
        series: Price series (typically 'Adj Close').
        period: Lookback period in days.

    Returns:
        RSI series in [0, 100], with NaN/inf values from zero-division filled at 50 (neutral).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD line, signal line, and histogram.

    Args:
        series: Price series.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line EMA period.

    Returns:
        DataFrame with columns ['MACD', 'MACD_signal', 'MACD_hist'].
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"MACD": macd_line, "MACD_signal": signal_line, "MACD_hist": histogram}
    )


def compute_bollinger_bands(series: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Compute Bollinger Bands.

    Args:
        series: Price series.
        window: Rolling window for the moving average and std dev.
        n_std: Number of standard deviations for the bands.

    Returns:
        DataFrame with columns ['BB_mid', 'BB_upper', 'BB_lower', 'BB_width'].
    """
    mid = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid
    return pd.DataFrame({"BB_mid": mid, "BB_upper": upper, "BB_lower": lower, "BB_width": width})


def compute_daily_return(series: pd.Series) -> pd.Series:
    """Compute simple daily percentage return."""
    return series.pct_change()


def compute_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling standard deviation of daily returns (a simple volatility proxy)."""
    return series.pct_change().rolling(window=window).std()


def engineer_features(
    stock_data: pd.DataFrame,
    ma_short_window: int = 20,
    ma_long_window: int = 50,
    rsi_period: int = 14,
    include_macd: bool = True,
    include_bollinger: bool = True,
    include_volume_features: bool = True,
) -> pd.DataFrame:
    """Add a configurable set of technical indicator features to OHLCV data.

    All rolling-window features introduce NaNs at the start of the series; these
    rows are dropped before returning.

    Args:
        stock_data: DataFrame with at least an 'Adj Close' column (and 'Volume' if
            include_volume_features is True).
        ma_short_window: Short moving average window in days.
        ma_long_window: Long moving average window in days.
        rsi_period: RSI lookback period in days.
        include_macd: Whether to add MACD line/signal/histogram.
        include_bollinger: Whether to add Bollinger Band features.
        include_volume_features: Whether to add volume-based features.

    Returns:
        DataFrame with original columns plus engineered features, NaN rows dropped.

    Raises:
        ValueError: if no rows remain after dropping NaNs (e.g. insufficient history
            for the chosen window sizes).
    """
    df = stock_data.copy()
    close = df["Adj Close"]

    df["MA_short"] = close.rolling(window=ma_short_window).mean()
    df["MA_long"] = close.rolling(window=ma_long_window).mean()
    df["RSI"] = compute_rsi(close, rsi_period)
    df["Daily_return"] = compute_daily_return(close)
    df["Volatility"] = compute_volatility(close, window=ma_short_window)

    if include_macd:
        df = pd.concat([df, compute_macd(close)], axis=1)

    if include_bollinger:
        df = pd.concat([df, compute_bollinger_bands(close, window=ma_short_window)], axis=1)

    if include_volume_features and "Volume" in df.columns:
        df["Volume_MA"] = df["Volume"].rolling(window=ma_short_window).mean()
        df["Volume_change"] = df["Volume"].pct_change()

    rows_before = len(df)
    df = df.dropna()
    print(f"Dropped {rows_before - len(df)} rows with NaN values from rolling-window warm-up")
    print(f"{len(df)} rows remaining after feature engineering")

    if df.empty:
        raise ValueError(
            "No valid rows remain after feature engineering. "
            "Reduce window sizes or fetch a longer date range."
        )
    return df


DEFAULT_FEATURE_COLS = [
    "Adj Close",
    "MA_short",
    "MA_long",
    "RSI",
    "Daily_return",
    "Volatility",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "BB_width",
]

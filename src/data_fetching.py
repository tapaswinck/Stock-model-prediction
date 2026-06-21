"""
Data fetching utilities for historical stock data.
"""

from typing import List, Dict
import pandas as pd
import yfinance as yf


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical OHLCV data for a single ticker from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL'.
        start_date: ISO date string, e.g. '2015-01-01'.
        end_date: ISO date string, e.g. '2023-01-01'.

    Returns:
        DataFrame indexed by date with columns Open, High, Low, Close, Adj Close, Volume.

    Raises:
        ValueError: if no data was returned for the ticker (e.g. invalid symbol or delisted).
    """
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' between {start_date} and {end_date}. "
            "Check the ticker symbol and date range."
        )
    # yfinance occasionally returns a MultiIndex column structure for single tickers
    # depending on version; normalise to flat columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def fetch_multi_ticker_data(
    tickers: List[str], start_date: str, end_date: str
) -> Dict[str, pd.DataFrame]:
    """Download historical OHLCV data for multiple tickers.

    Tickers that fail to download (e.g. invalid symbol, no data in range) are skipped
    with a warning printed to stdout, rather than failing the entire batch.

    Args:
        tickers: List of ticker symbols.
        start_date: ISO date string.
        end_date: ISO date string.

    Returns:
        Dict mapping ticker -> DataFrame, containing only tickers that downloaded successfully.
    """
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = fetch_stock_data(ticker, start_date, end_date)
            print(f"[OK] {ticker}: {len(results[ticker])} rows")
        except ValueError as e:
            print(f"[SKIP] {ticker}: {e}")
    return results

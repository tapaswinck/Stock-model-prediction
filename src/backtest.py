"""
A minimal long/flat backtest to translate price predictions into a trading
simulation, including transaction costs.

This is intentionally simple — a single position, no leverage, no shorting —
because the point is not to build a production trading strategy. The point is
to honestly check whether "the model's predicted price is directionally
correct most days" translates into "a strategy based on this would have made
money after costs," which is a much higher and more relevant bar than RMSE
alone tells you.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    n_trades: int
    win_rate_pct: float
    equity_curve: pd.Series


def backtest_directional_strategy(
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    dates: pd.DatetimeIndex,
    transaction_cost_bps: float = 5.0,
    trading_days_per_year: int = 252,
) -> BacktestResult:
    """Backtest a simple long/flat strategy: go long when the model predicts
    tomorrow's price is higher than today's, otherwise stay in cash.

    Args:
        actual_prices: True prices for the test period, length N.
        predicted_prices: Model's predicted prices for the same period, length N.
            predicted_prices[i] is the prediction for actual_prices[i] (i.e.
            already aligned — this function looks at predicted_prices[i] vs.
            actual_prices[i-1] to decide whether to be long on day i).
        dates: DatetimeIndex aligned with actual_prices, length N.
        transaction_cost_bps: Round-trip cost in basis points charged whenever
            the position changes (long -> flat or flat -> long).
        trading_days_per_year: Used to annualise the return.

    Returns:
        BacktestResult with summary statistics and the equity curve.

    Raises:
        ValueError: if input arrays have mismatched lengths or fewer than 2 points.
    """
    if not (len(actual_prices) == len(predicted_prices) == len(dates)):
        raise ValueError("actual_prices, predicted_prices, and dates must be the same length.")
    if len(actual_prices) < 2:
        raise ValueError("Need at least 2 data points to backtest.")

    n = len(actual_prices)
    cost = transaction_cost_bps / 10_000.0

    position = 0  # 0 = flat (cash), 1 = long
    equity = [1.0]
    n_trades = 0
    winning_trades = 0
    trade_entry_price = None

    for i in range(1, n):
        predicted_move_up = predicted_prices[i] > actual_prices[i - 1]
        new_position = 1 if predicted_move_up else 0

        daily_return = (actual_prices[i] / actual_prices[i - 1]) - 1
        equity_today = equity[-1] * (1 + position * daily_return)

        if new_position != position:
            equity_today *= (1 - cost)
            n_trades += 1
            if new_position == 1:
                trade_entry_price = actual_prices[i]
            elif trade_entry_price is not None:
                if actual_prices[i] > trade_entry_price:
                    winning_trades += 1

        equity.append(equity_today)
        position = new_position

    equity_curve = pd.Series(equity, index=dates)
    total_return_pct = (equity_curve.iloc[-1] - 1) * 100

    n_days = len(equity_curve)
    years = n_days / trading_days_per_year
    annualized_return_pct = (
        ((equity_curve.iloc[-1]) ** (1 / years) - 1) * 100 if years > 0 else float("nan")
    )

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown_pct = drawdown.min() * 100

    daily_returns = equity_curve.pct_change().dropna()
    sharpe_ratio = (
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(trading_days_per_year)
        if daily_returns.std() > 0 else 0.0
    )

    win_rate_pct = (winning_trades / n_trades * 100) if n_trades > 0 else float("nan")

    return BacktestResult(
        total_return_pct=total_return_pct,
        annualized_return_pct=annualized_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=float(sharpe_ratio),
        n_trades=n_trades,
        win_rate_pct=win_rate_pct,
        equity_curve=equity_curve,
    )


def backtest_buy_and_hold(actual_prices: np.ndarray, dates: pd.DatetimeIndex) -> BacktestResult:
    """Buy-and-hold benchmark: buy on day 0, hold for the entire period, no trading costs.

    This is the benchmark every active strategy must beat to be worth the extra
    complexity and risk — if a model-driven strategy can't outperform simply
    holding the asset, the model isn't adding value, regardless of its RMSE.
    """
    equity_curve = pd.Series(actual_prices / actual_prices[0], index=dates)
    total_return_pct = (equity_curve.iloc[-1] - 1) * 100

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown_pct = drawdown.min() * 100

    daily_returns = equity_curve.pct_change().dropna()
    sharpe_ratio = (
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() > 0 else 0.0
    )

    return BacktestResult(
        total_return_pct=total_return_pct,
        annualized_return_pct=float("nan"),
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=float(sharpe_ratio),
        n_trades=1,
        win_rate_pct=float("nan"),
        equity_curve=equity_curve,
    )

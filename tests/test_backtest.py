"""
Unit tests for src/backtest.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_directional_strategy, backtest_buy_and_hold


@pytest.fixture
def trending_up_prices():
    """A steadily rising price series — a perfect predictor should go long
    immediately and stay long, beating buy-and-hold only by avoiding costs
    (there's nothing to avoid here, so a perfect model should roughly match
    buy-and-hold, modulo the first day's lag)."""
    dates = pd.bdate_range("2022-01-01", periods=100)
    prices = 100 * (1.001 ** np.arange(100))
    return prices, dates


def test_perfect_foresight_beats_or_matches_buy_and_hold(trending_up_prices):
    actual, dates = trending_up_prices
    # "Perfect" predictions: predicted_prices[i] correctly anticipates the
    # direction of tomorrow's move relative to today.
    predicted = np.roll(actual, -1)
    predicted[-1] = actual[-1]

    strat = backtest_directional_strategy(actual, predicted, dates, transaction_cost_bps=0.0)
    bh = backtest_buy_and_hold(actual, dates)

    # With zero costs and a steadily rising series, going long every day
    # should produce (approximately) the same return as buy-and-hold.
    assert strat.total_return_pct == pytest.approx(bh.total_return_pct, rel=0.05)


def test_transaction_costs_reduce_returns(trending_up_prices):
    actual, dates = trending_up_prices
    np.random.seed(0)
    noisy_predicted = actual * (1 + np.random.normal(0, 0.02, len(actual)))

    no_cost = backtest_directional_strategy(actual, noisy_predicted, dates, transaction_cost_bps=0.0)
    with_cost = backtest_directional_strategy(actual, noisy_predicted, dates, transaction_cost_bps=20.0)

    assert with_cost.total_return_pct <= no_cost.total_return_pct


def test_backtest_raises_on_mismatched_lengths():
    actual = np.array([100, 101, 102])
    predicted = np.array([100, 101])
    dates = pd.bdate_range("2022-01-01", periods=3)

    with pytest.raises(ValueError, match="same length"):
        backtest_directional_strategy(actual, predicted, dates)


def test_buy_and_hold_matches_simple_return(trending_up_prices):
    actual, dates = trending_up_prices
    bh = backtest_buy_and_hold(actual, dates)
    expected_return_pct = (actual[-1] / actual[0] - 1) * 100
    assert bh.total_return_pct == pytest.approx(expected_return_pct, rel=1e-6)

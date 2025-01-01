import pytest  # type: ignore
import pandas as pd
import numpy as np
from backtester_framework.stats import core_metrics, performance_metrics, tail_metrics


def test_annualized_return(returns_data):
    """Annualized return test"""
    frequencies = {
        "1min": 525600,
        "5min": 105120,
        "15min": 35040,
        "30min": 17520,
        "1H": 8760,
        "4H": 2190,
        "D": 365,
        "W": 52,
        "M": 12,
    }

    for freq, N in frequencies.items():
        result = core_metrics.annualized_return(returns_data, N=N)
        assert isinstance(result, float)
        assert not np.isnan(result)


def test_annualized_std(returns_data):
    """Annualized volatility test"""
    result = core_metrics.annualized_std(returns_data, N=252)
    assert isinstance(result, float)
    assert result >= 0
    assert not np.isnan(result)


def test_sharpe_ratio():
    """Sharpe ratio test with different scenarios"""
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    risk_free_rates = [0.0, 0.02, 0.05]

    for rf in risk_free_rates:
        result = performance_metrics.sharpe_ratio(returns, risk_free_rate=rf, N=252)
        assert isinstance(result, float)
        assert not np.isnan(result)


def test_sortino_ratio(returns_data):
    """Sortino ratio test"""
    result = performance_metrics.sortino_ratio(returns_data, target_return=0, N=252)
    assert isinstance(result, float)
    assert not np.isnan(result)


def test_max_drawdown():
    """Maximum drawdown test with different scenarios"""
    # Test case 1: Simple drawdown
    nav1 = pd.Series([100, 95, 90, 95, 100, 85, 90])
    result1 = tail_metrics.max_drawdown(nav1)
    assert abs(result1 - (-0.15)) < 1e-6

    # Test case 2: No drawdown
    nav2 = pd.Series([100, 101, 102, 103, 104])
    result2 = tail_metrics.max_drawdown(nav2)
    assert result2 == 0.0

    # Test case 3: Multiple drawdowns
    nav3 = pd.Series([100, 90, 95, 85, 80, 85, 90, 85, 80])
    result3 = tail_metrics.max_drawdown(nav3)
    assert abs(result3 - (-0.20)) < 1e-6


def test_count_trades():
    """Trade counting test with different scenarios"""
    pos1 = pd.Series([0, 1, 1, -1, -1, 0])
    assert core_metrics.count_trades(pos1) == 4

    pos2 = pd.Series([1])
    assert core_metrics.count_trades(pos2) == 1

    pos3 = pd.Series([0, 1, -1, 0, 1, -1])
    assert core_metrics.count_trades(pos3) == 6


def test_winning_trades():
    """Win rate test with different scenarios"""
    pos1 = pd.Series([0, 1, 1, 0])
    ret1 = pd.Series([0.01, 0.02, 0.03, 0.01])
    result1 = core_metrics.winning_trades_percentage(pos1, ret1)
    assert isinstance(result1, float)
    assert 0 <= result1 <= 1

    pos2 = pd.Series([0, -1, -1, 0])
    ret2 = pd.Series([0.01, 0.02, 0.03, 0.01])
    result2 = core_metrics.winning_trades_percentage(pos2, ret2)
    assert result2 == 0.0

    pos3 = pd.Series([0, 1, -1, 0])
    ret3 = pd.Series([0.01, 0.02, -0.02, -0.01])
    result3 = core_metrics.winning_trades_percentage(pos3, ret3)
    assert 0 <= result3 <= 1

import pytest  # type: ignore
import pandas as pd
import numpy as np
from backtester_framework.strategies.strategy_constructor import Strategy, strategy
from backtester_framework.strategies.moving_average import ma_crossover
from backtester_framework.strategies.RSI import rsi_strategy


def test_strategy_decorator():
    """Strategy decorator test"""

    @strategy(name="TestStrategy")
    def test_strategy(historical_data: pd.DataFrame, current_position: float) -> float:
        return 1.0 if historical_data["close"].iloc[-1] > 100 else -1.0

    assert test_strategy.__name__ == "TestStrategy"
    strategy_instance = test_strategy()
    assert isinstance(strategy_instance, Strategy)


def test_ma_crossover(multi_sample_data):
    """MA Crossover strategy test"""
    strategy = ma_crossover(short_window=20, long_window=50)

    # Test with insufficient data
    position = strategy.get_position(multi_sample_data[0].iloc[:10], 0)
    assert position == 0.0

    # Test with full data
    for data in multi_sample_data:
        position = strategy.get_position(data, 0)
        assert isinstance(position, float)
        assert -1.0 <= position <= 1.0


def test_rsi_strategy(multi_sample_data):
    """RSI strategy test"""
    strategy = rsi_strategy(rsi_period=14, overbought=70, oversold=30)

    # Test with insufficient data
    position = strategy.get_position(multi_sample_data[0].iloc[:10], 0)
    assert position == 0.0

    # Test with full data
    for data in multi_sample_data:
        position = strategy.get_position(data, 0)
        assert isinstance(position, float)
        assert -1.0 <= position <= 1.0


def test_strategy_position_bounds(multi_sample_data):
    """Test if positions are all in [-1,1]"""
    strategies = [ma_crossover(), rsi_strategy()]

    for strategy in strategies:
        for data in multi_sample_data:
            position = strategy.get_position(data, 0)
            assert -1.0 <= position <= 1.0


def test_strategy_parameters():
    """Test strategy parameters initialization"""
    # Test MA Crossover parameters
    ma = ma_crossover(short_window=10, long_window=30)
    assert ma.short_window == 10
    assert ma.long_window == 30

    # Test RSI parameters
    rsi = rsi_strategy(rsi_period=10, overbought=75, oversold=25)
    assert rsi.rsi_period == 10
    assert rsi.overbought == 75
    assert rsi.oversold == 25


def test_strategy_with_missing_data(multi_sample_data):
    """Test strategies behavior with missing data"""
    # Create data with missing values
    data = multi_sample_data[0].copy()
    data.loc[data.index[5], "close"] = np.nan

    strategies = [ma_crossover(), rsi_strategy()]

    # Both strategies should handle missing data gracefully
    for strategy in strategies:
        position = strategy.get_position(data, 0)
        assert isinstance(position, float)
        assert -1.0 <= position <= 1.0


def test_custom_strategy(multi_sample_data):
    """Test custom strategy creation with decorator"""

    @strategy(name="CustomStrategy")
    def custom_strategy(
        historical_data: pd.DataFrame, current_position: float, threshold: float = 0.5
    ) -> float:
        returns = historical_data["close"].pct_change()
        if returns.iloc[-1] > threshold:
            return 1.0
        elif returns.iloc[-1] < -threshold:
            return -1.0
        return current_position

    strat = custom_strategy(threshold=0.01)

    for data in multi_sample_data:
        position = strat.get_position(data, 0)
        assert isinstance(position, float)
        assert -1.0 <= position <= 1.0

import pytest  # type: ignore
import pandas as pd
from backtester_framework.backtester.backtester import Backtester
from backtester_framework.strategies.moving_average import ma_crossover
from backtester_framework.strategies.RSI import rsi_strategy


def test_backtester_initialization(multi_sample_data):
    """Test backtester initiation"""
    backtester = Backtester(multi_sample_data, names=["Asset1", "Asset2"])
    assert len(backtester.data_dict) == 2
    assert backtester.initial_capital == 10000.0
    assert backtester.commission == 0.001
    assert backtester.slippage == 0.0
    assert backtester.rebalancing_frequency == "D"


def test_backtester_with_strategy(multi_sample_data):
    """Strategy execution test"""
    backtester = Backtester(multi_sample_data, names=["Asset1", "Asset2"])
    strategy = ma_crossover(short_window=20, long_window=50)
    result = backtester.run(strategy)

    assert all(isinstance(nav, pd.Series) for nav in result.nav.values())
    assert all(nav.iloc[0] == backtester.initial_capital for nav in result.nav.values())
    assert all(
        len(nav) == len(data)
        for nav, data in zip(result.nav.values(), multi_sample_data)
    )


def test_backtester_with_multiple_strategies(multi_sample_data):
    """Test with different strategies for different assets"""
    backtester = Backtester(multi_sample_data, names=["Asset1", "Asset2"])
    strategies = {
        "Asset1": ma_crossover(short_window=20, long_window=50),
        "Asset2": rsi_strategy(rsi_period=14),
    }
    result = backtester.run(strategies)

    assert set(result.nav.keys()) == {"Asset1", "Asset2"}
    assert all(isinstance(nav, pd.Series) for nav in result.nav.values())
    assert all(nav.iloc[0] == backtester.initial_capital for nav in result.nav.values())


def test_backtester_rebalancing_daily(daily_data):
    """Test for several rebalancing frequencies with daily data"""
    frequencies = ["D", "W", "M"]
    strategy = ma_crossover()

    for freq in frequencies:
        backtester = Backtester(daily_data, rebalancing_frequency=freq)
        result = backtester.run(strategy)
        for asset_positions in result.positions.values():
            assert len(asset_positions) == len(daily_data)


def test_backtester_rebalancing_intraday(intraday_data):
    """Test for several rebalancing frequencies with intraday data"""
    frequencies = ["5min", "15min", "30min", "1H", "4H"]
    strategy = ma_crossover()

    for freq in frequencies:
        backtester = Backtester(intraday_data, rebalancing_frequency=freq)
        result = backtester.run(strategy)
        for asset_positions in result.positions.values():
            assert len(asset_positions) == len(intraday_data)


def test_backtester_invalid_frequency(daily_data):
    """Test if the backtester reject invalid frequencies"""
    with pytest.raises(ValueError):
        Backtester(daily_data, rebalancing_frequency="Y")


def test_backtester_frequency_validation(daily_data):
    """Frequency validation test according to the data frequency"""
    with pytest.raises(ValueError):
        Backtester(daily_data, rebalancing_frequency="1min")


def test_backtester_with_costs(daily_data):
    """Transaction cost impact test"""
    strategy = ma_crossover()

    backtester_no_costs = Backtester(daily_data, commission=0, slippage=0)
    result_no_costs = backtester_no_costs.run(strategy)

    backtester_with_costs = Backtester(daily_data, commission=0.001, slippage=0.001)
    result_with_costs = backtester_with_costs.run(strategy)

    # Vérifier que les coûts ont un impact négatif pour chaque actif
    for asset_name in result_with_costs.nav.keys():
        assert (
            result_with_costs.nav[asset_name].iloc[-1]
            <= result_no_costs.nav[asset_name].iloc[-1]
        )

import pytest  # type: ignore
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_data():
    """Creates a small dataset to run tests"""
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    data = {
        "close": [float(x) for x in [100, 102, 101, 103, 102, 104, 103, 105, 106, 104]],
        "volume": [1000] * 10,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def multi_sample_data(sample_data):
    """Creates multiple datasets for testing"""
    data1 = sample_data.copy()
    data2 = sample_data.copy() * 1.1
    return [data1, data2]


@pytest.fixture
def daily_data():
    """Creates simulated daily data for tests"""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    prices = np.sin(np.linspace(0, 4 * np.pi, 100)) * 10 + 100
    return pd.DataFrame(
        {"close": prices, "volume": np.random.randint(1000, 10000, 100)}, index=dates
    )


@pytest.fixture
def intraday_data():
    """Creates intraday data for tests"""
    dates = pd.date_range(start="2023-01-01 09:30:00", periods=100, freq="5min")
    prices = np.sin(np.linspace(0, 4 * np.pi, 100)) * 10 + 100
    return pd.DataFrame(
        {"close": prices, "volume": np.random.randint(1000, 10000, 100)}, index=dates
    )


@pytest.fixture
def crypto_data():
    """Load crypto datasets for tests"""
    btc_path = Path("data/test_BTC_daily.csv")
    eth_path = Path("data/test_ETH_daily.csv")
    if not (btc_path.exists() and eth_path.exists()):
        pytest.skip("Crypto data files are not available")
    btc = pd.read_csv(btc_path, index_col=0, parse_dates=True)
    eth = pd.read_csv(eth_path, index_col=0, parse_dates=True)
    return [btc, eth]


@pytest.fixture
def returns_data():
    """Creates a returns series for tests"""
    return pd.Series([0.01, -0.02, 0.03, -0.01, 0.02], name="returns")


@pytest.fixture
def positions_data():
    """Creates a positions series for tests"""
    return pd.Series([0, 1, 1, -1, -1, 0], name="position")


@pytest.fixture
def sample_positions(sample_data):
    """Create sample positions for testing"""
    return pd.DataFrame(
        {
            "position": np.random.choice([-1, 0, 1], size=len(sample_data)),
            "timestamp": sample_data.index,
        },
        index=sample_data.index,
    )


@pytest.fixture
def multi_sample_positions(multi_sample_data):
    """Create sample positions for multiple assets"""
    positions_dict = {}
    for i, data in enumerate(multi_sample_data):
        positions_dict[f"Asset{i+1}"] = pd.DataFrame(
            {
                "position": np.random.choice([-1, 0, 1], size=len(data)),
                "timestamp": data.index,
            },
            index=data.index,
        )
    return positions_dict


@pytest.fixture
def result_instance(sample_data, sample_positions):
    """Create a Result instance for testing"""
    from backtester_framework.backtester.result import Result

    return Result(
        positions=sample_positions,
        data=sample_data,
        initial_capital=10000,
        commission=0.001,
        slippage=0.001,
        frequency="D",
    )


@pytest.fixture
def multi_result_instance(multi_sample_data, multi_sample_positions):
    """Create a Result instance with multiple assets for testing"""
    from backtester_framework.backtester.result import Result

    data_dict = {f"Asset{i+1}": data for i, data in enumerate(multi_sample_data)}
    return Result(
        positions=multi_sample_positions,
        data=data_dict,
        initial_capital=10000,
        commission=0.001,
        slippage=0.001,
        frequency="D",
    )

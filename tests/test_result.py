import pytest  # type: ignore
import pandas as pd
import numpy as np
from backtester_framework.backtester.result import Result


def test_result_initialization(multi_result_instance):
    """Test Result instance initialization"""
    assert all(isinstance(nav, pd.Series) for nav in multi_result_instance.nav.values())
    assert len(multi_result_instance.asset_names) == 2
    assert all(
        nav.iloc[0] == multi_result_instance.initial_capital
        for nav in multi_result_instance.nav.values()
    )


def test_invalid_positions(sample_data):
    """Test invalid positions handling"""
    invalid_positions = {
        "Asset1": pd.DataFrame(
            {
                "position": np.random.uniform(-2, 2, size=len(sample_data)),
                "timestamp": sample_data.index,
            },
            index=sample_data.index,
        )
    }

    with pytest.raises(
        ValueError, match="Positions for Asset1 must belong to \[-1,1\]"
    ):
        Result(
            positions=invalid_positions,
            data={"Asset1": sample_data},
            initial_capital=10000,
            commission=0.001,
            slippage=0.001,
            frequency="D",
        )


def test_calculate_nav(multi_result_instance):
    """Test NAV calculation"""
    for asset_name in multi_result_instance.asset_names:
        nav = multi_result_instance.nav[asset_name]
        assert nav.iloc[0] == multi_result_instance.initial_capital
        assert not nav.isnull().any()


def test_get_metrics_per_asset(multi_result_instance):
    """Test metrics computation for individual assets"""
    for asset in multi_result_instance.asset_names:
        metrics = multi_result_instance.get_all_metrics(asset_name=asset)
        assert isinstance(metrics, dict)
        assert "Total Return (%)" in metrics
        assert "Sharpe Ratio" in metrics
        assert "Number of Trades" in metrics
        assert "Winning Trades (%)" in metrics


def test_plotting_multi_asset(multi_result_instance):
    """Test plotting functionality for multiple assets"""
    for asset in multi_result_instance.asset_names:
        for what in ["nav", "returns", "positions"]:
            multi_result_instance.plot(what=what, asset_name=asset, backend="plotly")

    for what in ["nav", "returns", "positions"]:
        multi_result_instance.plot(what=what, backend="plotly")

    with pytest.raises(ValueError):
        multi_result_instance.plot(backend="invalid")
    with pytest.raises(ValueError):
        multi_result_instance.plot(what="invalid")
    with pytest.raises(ValueError):
        multi_result_instance.plot(asset_name="invalid_asset")


def test_compare_results_multi_asset(multi_result_instance, multi_sample_data):
    """Test results comparison with multiple assets"""
    other_positions = {
        asset: pd.DataFrame(
            {
                "position": np.random.choice([-1, 0, 1], size=len(data)),
                "timestamp": data.index,
            },
            index=data.index,
        )
        for asset, data in zip(multi_result_instance.asset_names, multi_sample_data)
    }

    data_dict = {
        asset: data
        for asset, data in zip(multi_result_instance.asset_names, multi_sample_data)
    }

    other_result = Result(
        positions=other_positions,
        data=data_dict,
        initial_capital=10000,
        commission=0.001,
        slippage=0.001,
        frequency="D",
    )

    comparison = multi_result_instance.compare_results(other_result)
    assert isinstance(comparison, pd.DataFrame)

    for i in range(1, 3):
        for asset in multi_result_instance.asset_names:
            assert f"Strategy {i} - {asset}" in comparison.columns

    metrics = ["Total Return (%)", "Sharpe Ratio", "Maximum Drawdown (%)"]
    assert all(metric in comparison.index for metric in metrics)


def test_different_asset_comparison(multi_result_instance, sample_data):
    """Test comparison handling with different assets"""
    other_positions = {
        "Different_Asset": pd.DataFrame(
            {
                "position": np.random.choice([-1, 0, 1], size=len(sample_data)),
                "timestamp": sample_data.index,
            },
            index=sample_data.index,
        )
    }

    other_result = Result(
        positions=other_positions,
        data={"Different_Asset": sample_data},
        initial_capital=10000,
        commission=0.001,
        slippage=0.001,
        frequency="D",
    )

    comparison = multi_result_instance.compare_results(other_result)
    assert isinstance(comparison, pd.DataFrame)
    assert "Strategy 2 - Different_Asset" in comparison.columns

import pytest  # type: ignore
import pandas as pd
import numpy as np
from pathlib import Path


def test_data_loading_from_file(tmp_path):
    """Test data loading from files"""
    test_data = pd.DataFrame(
        {"close": [100, 101, 102], "volume": [1000, 1000, 1000]},
        index=pd.date_range("2023-01-01", periods=3),
    )
    csv_path = tmp_path / "test.csv"
    parquet_path = tmp_path / "test.parquet"

    test_data.to_csv(csv_path)
    test_data.to_parquet(parquet_path)

    csv_data = pd.read_csv(csv_path, index_col=0)
    parquet_data = pd.read_parquet(parquet_path)

    assert not csv_data.empty
    assert not parquet_data.empty
    assert "close" in csv_data.columns
    assert "close" in parquet_data.columns


def test_data_validation(multi_sample_data):
    """Test data validation"""
    for data in multi_sample_data:
        assert "close" in data.columns
        assert "volume" in data.columns

        assert data["close"].dtype in [np.float64, np.float32]
        assert data["volume"].dtype in [np.float64, np.float32, np.int64, np.int32]

        assert not data["close"].isnull().any()
        assert not data["volume"].isnull().any()

        assert isinstance(data.index, pd.DatetimeIndex)
        assert data.index.is_monotonic_increasing


def test_data_frequency_detection(multi_sample_data):
    """Test data frequency detection"""
    frequencies = ["1min", "5min", "15min", "30min", "1H", "4H", "D", "W", "M"]

    for freq in frequencies:
        data = pd.DataFrame(
            {
                "close": np.random.randn(10),
                "volume": np.random.randint(1000, 10000, 10),
            },
            index=pd.date_range("2023-01-01", periods=10, freq=freq),
        )

        detected_freq = pd.infer_freq(data.index)
        assert detected_freq is not None


def test_data_format_standardization(multi_sample_data):
    """Test data format standardization"""
    for data in multi_sample_data:
        data_mixed = data.copy()
        data_mixed.columns = ["Close", "Volume"]

        data_mixed.columns = [col.lower() for col in data_mixed.columns]
        assert all(col.islower() for col in data_mixed.columns)

        data_str = data.copy()
        data_str.index = data_str.index.strftime("%Y-%m-%d")
        data_str.index = pd.to_datetime(data_str.index)
        assert isinstance(data_str.index, pd.DatetimeIndex)


def test_multi_asset_data_alignment(multi_sample_data):
    """Test alignment of multiple asset data"""
    reference_index = multi_sample_data[0].index
    for data in multi_sample_data[1:]:
        assert data.index.equals(reference_index)

    reference_columns = multi_sample_data[0].columns
    for data in multi_sample_data[1:]:
        assert set(data.columns) == set(reference_columns)

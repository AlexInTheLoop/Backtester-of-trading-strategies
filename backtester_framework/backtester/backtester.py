from dataclasses import dataclass
from typing import Union, List, Dict
import pandas as pd
from pathlib import Path

from backtester_framework.strategies.strategy_constructor import Strategy
from backtester_framework.backtester.result import Result

# Dictionary to convert the rebalancy_frequency attribute in the correct format
FREQ_MAP = {
    "1min": "1min",
    "5min": "5min",
    "5T": "5min",
    "15min": "15min",
    "30min": "30min",
    "60min": "1h",
    "240min": "4h",
    "D": "D",
    "W": "W-MON",
    "M": "M",
}


@dataclass
class Backtester:
    """
    Class to backtest trading strategies on multiple assets independently
    """

    data: Union[List[pd.DataFrame], List[Union[str, Path]], pd.DataFrame, str, Path]
    names: Union[List[str], str, None] = None
    initial_capital: float = 10000.0
    commission: float = 0.001
    slippage: float = 0.0
    rebalancing_frequency: str = "D"

    def __post_init__(self):
        # Convert single input to list for uniform processing
        if isinstance(self.data, (pd.DataFrame, str, Path)):
            self.data = [self.data]

        # Process names
        if self.names is None:
            self.names = [f"asset_{i}" for i in range(len(self.data))]
        elif isinstance(self.names, str):
            self.names = [self.names]

        if len(self.names) != len(self.data):
            raise ValueError("Number of names must match number of assets")

        self.data_dict: Dict[str, pd.DataFrame] = {}

        # Process each data source
        for i, data_source in enumerate(self.data):
            if isinstance(data_source, (str, Path)):
                file_path = Path(data_source)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")

                if file_path.suffix == ".csv":
                    df = pd.read_csv(file_path, index_col=0)
                elif file_path.suffix == ".parquet":
                    df = pd.read_parquet(file_path)
                else:
                    raise ValueError(
                        "File format not supported. Use CSV or Parquet files."
                    )
            else:
                if not isinstance(data_source, pd.DataFrame):
                    raise ValueError(
                        "Data must be a pandas DataFrame or a path to a CSV/Parquet file"
                    )
                df = data_source

            asset_name = self.names[i]
            self.data_dict[asset_name] = df

        if self.rebalancing_frequency not in FREQ_MAP:
            raise ValueError(
                f"Frequency not available. Available frequencies: {', '.join(FREQ_MAP.keys())}"
            )

        # Process all dataframes
        for asset_name, df in self.data_dict.items():
            df.index = pd.to_datetime(df.index)
            df.columns = [col.lower() for col in df.columns]

            if "close" not in df.columns:
                raise ValueError(f"Missing 'close' column in {asset_name}")

        # Determine data frequency from the first dataset
        first_df = next(iter(self.data_dict.values()))
        if len(first_df) > 1:
            self.data_frequency = pd.infer_freq(first_df.index)
            if self.data_frequency is None:
                time_diff = (first_df.index[1] - first_df.index[0]).total_seconds()
                if time_diff < 86400:  # 86400 sec = 24h
                    self.data_frequency = f"{int(time_diff/60)}min"
                else:
                    self.data_frequency = "D"
        else:
            self.data_frequency = "D"

        data_minutes = self._freq_to_minutes(self.data_frequency)
        rebal_minutes = self._freq_to_minutes(FREQ_MAP[self.rebalancing_frequency])

        if rebal_minutes < data_minutes:
            raise ValueError(
                f"Rebalancing frequency ({self.rebalancing_frequency}) cannot be higher than data frequency ({self.data_frequency})"
            )

    @staticmethod
    def _freq_to_minutes(freq: str) -> int:
        """
        Frequency conversion to minutes

        Parameter
        ----------
        freq: str
            data frequency indicator

        Returns
        ----------
        int
            number of minutes
        """
        if "min" in freq:
            return int(freq.replace("min", ""))
        elif "h" in freq.lower():
            return int(freq[0]) * 60
        elif freq == "D":
            return 1440  # 1440 min = 24h
        elif freq.startswith("W"):
            return 10080  # 10080 min = 1 week
        elif freq.startswith("M"):
            return 43200  # 43200 sec = 1 month
        else:
            return 0

    @staticmethod
    def _get_period_start(timestamp: pd.Timestamp, freq: str) -> pd.Timestamp:
        """
        Identification of a period start for a given timestamp

        Parameters
        ----------
        timestamp: Timestamp
            time index
        freq: float
            data frequency

        Returns
        ----------
        timestamp
            new time index for position rebalancing
        """
        if freq == "M":
            return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif freq == "W-MON":
            return timestamp - pd.Timedelta(days=timestamp.dayofweek)
        else:
            return timestamp.floor(freq)

    def run(self, strategies: Union[Strategy, Dict[str, Strategy]]) -> Result:
        """
        Execute the backtest of given strategies on multiple assets independently

        Parameter
        ----------
        strategies: Union[Strategy, Dict[str, Strategy]]
            Either a single strategy to apply to all assets or a dictionary mapping asset names to strategies

        Returns
        ----------
        Result
            instance of the Result class containing backtest results for each asset
        """
        # Convert single strategy to dictionary for uniform processing
        if not isinstance(strategies, dict):
            strategies = {
                asset_name: strategies for asset_name in self.data_dict.keys()
            }

        positions_dict = {}
        resampled_data_dict = {}
        rebal_freq = FREQ_MAP[self.rebalancing_frequency]

        for asset_name in self.data_dict.keys():
            data = self.data_dict[asset_name]

            if self.rebalancing_frequency != self.data_frequency:
                resampled_data = (
                    data.resample(rebal_freq)
                    .agg({"close": "last", "volume": "sum"})
                    .ffill()
                )
                resampled_data = resampled_data.reindex(data.index, method="ffill")
            else:
                resampled_data = data.copy()

            resampled_data_dict[asset_name] = resampled_data

            positions = []
            current_position = 0.0
            last_rebalancing_time = None
            current_rebalancing_position = 0.0

            strategy = strategies[asset_name]
            has_fit = hasattr(strategy, "fit")
            min_data = 100 if has_fit else 1
            last_fit_index = 0

            for i, timestamp in enumerate(data.index):
                if i < min_data - 1:
                    positions.append(0.0)
                    continue

                if has_fit and i >= last_fit_index + 100:
                    strategy.fit(data.iloc[: i + 1])
                    last_fit_index = i

                if self.rebalancing_frequency != self.data_frequency:
                    period_start = self._get_period_start(timestamp, rebal_freq)

                    if last_rebalancing_time != period_start:
                        historical_data = resampled_data.loc[:timestamp]
                        current_rebalancing_position = strategy.get_position(
                            historical_data, current_position
                        )
                        last_rebalancing_time = period_start

                    positions.append(current_rebalancing_position)
                    current_position = current_rebalancing_position

                else:
                    historical_data = data.loc[:timestamp]
                    new_position = strategy.get_position(
                        historical_data, current_position
                    )
                    positions.append(new_position)
                    current_position = new_position

            positions_dict[asset_name] = pd.DataFrame(
                {"position": positions, "timestamp": data.index}, index=data.index
            )

        return Result(
            positions=positions_dict,
            data=self.data_dict,
            initial_capital=self.initial_capital,
            commission=self.commission,
            slippage=self.slippage,
            frequency=self.data_frequency,
        )

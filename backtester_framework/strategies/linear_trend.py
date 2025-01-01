from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional
from backtester_framework.strategies.strategy_constructor import Strategy


@dataclass
class LinearTrendStrategy(Strategy):
    """
    Strategy based on a linear regression to detect the trend
    """

    window_size: int = 20
    trend_threshold: float = 0.001

    def __post_init__(self):
        self.model = LinearRegression()
        self.is_fitted = False
        self.optimal_threshold: Optional[float] = None

    def _calculate_slope(self, prices: pd.Series) -> float:
        """
        Compute the linear slope over a window

        Parameters
        ----------
        data: pd.Series
            historical data series
        """
        n = len(prices)
        X = np.arange(n).reshape(-1, 1)
        y = prices.values.reshape(-1, 1)
        self.model.fit(X, y)
        return self.model.coef_[0][0]

    def fit(self, data: pd.DataFrame) -> None:
        """
        Optimize the slope threshold by testing different values and selecting the one
        that maximises the return

        Parameters
        ----------
        data: DataFrame
            historical data series
        """
        if len(data) < self.window_size * 2:
            return

        close = next((col for col in data.columns if col in ["Close", "close"]), None)

        prices = data[close].copy()
        slopes = []

        for i in range(len(data) - self.window_size + 1):
            window = prices.iloc[i : i + self.window_size]
            slope = self._calculate_slope(window)
            slopes.append(slope)

        slopes = pd.Series(slopes, index=prices.index[self.window_size - 1 :])
        returns = np.log(prices).diff().dropna()

        aligned_data = pd.concat([slopes, returns], axis=1, join="inner")
        slopes = aligned_data.iloc[:, 0]
        returns = aligned_data.iloc[:, 1]

        best_return = -np.inf
        best_threshold = self.trend_threshold

        for threshold in np.percentile(np.abs(slopes), [50, 60, 70, 80, 90]):
            positions = pd.Series(0, index=slopes.index)
            positions[slopes > threshold] = 1
            positions[slopes < -threshold] = -1

            strategy_return = (positions * returns).sum()

            if strategy_return > best_return:
                best_return = strategy_return
                best_threshold = threshold

        self.optimal_threshold = best_threshold
        self.is_fitted = True

    def get_position(
        self, historical_data: pd.DataFrame, current_position: float
    ) -> float:
        """
        Identify the position according to the trend slope

        Parameters
        ----------
        historical_data: DataFrame
            historical data series
        current_position: float
            current position (-1.0, 0 ou 1.0)

        Returns
        ----------
        current_position
            new position (-1.0, 0.0, ou 1.0)
        """
        if len(historical_data) < self.window_size:
            return current_position

        try:
            close = next(
                (col for col in historical_data.columns if col in ["Close", "close"]),
                None,
            )
            prices = historical_data[close].iloc[-self.window_size :]
            slope = self._calculate_slope(prices)

            threshold = (
                self.optimal_threshold if self.is_fitted else self.trend_threshold
            )

            if slope > threshold:
                return 1.0
            elif slope < -threshold:
                return -1.0
            else:
                return 0.0

        except Exception as e:
            print(f"Linear trend error: {e}")
            return current_position

from backtester_framework.strategies.strategy_constructor import Strategy
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from dataclasses import dataclass
from typing import Union
import itertools
import warnings

NumericArray = Union[np.ndarray, pd.DataFrame]


@dataclass
class ARIMAStrategy(Strategy):
    """
    Strategy based on the forecast of an ARIMA model
    """

    window_size: int = 252
    threshold: float = 0.05

    def __post_init__(self):
        self.model = None
        self.is_fitted = False
        self.best_order = None

    def select_order(self, data: NumericArray) -> set:
        """
        Select the best orders (p,d,q) according to the BIC criterion

        Parameters
        ----------
        data: NumericArray
            historical data series

        Returns
        ----------
        best_order: set
            best orders
        """
        best_bic = np.inf
        best_order = None

        p = range(0, 3)
        d = range(0, 2)
        q = range(0, 3)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            for order in itertools.product(p, d, q):
                try:
                    model = ARIMA(data, order=order)
                    results = model.fit()
                    if results.bic < best_bic:
                        best_bic = results.bic
                        best_order = order
                except:
                    continue

        return best_order if best_order is not None else (1, 0, 1)

    def fit(self, data: pd.DataFrame) -> None:
        """
        Estimate the ARIMA model on historical data
        Estime le modèle ARIMA sur les données historiques

        Parameters
        ----------
        data: DataFrame
            historical data series
        """

        close = next((col for col in data.columns if col in ["Close", "close"]), None)

        returns = np.log(data[close]).diff().dropna()

        if len(returns) >= self.window_size:
            try:
                training_data = returns[-self.window_size :]
                self.best_order = self.select_order(training_data)

                self.model = ARIMA(training_data, order=self.best_order).fit()
                self.is_fitted = True
            except Exception as e:
                print(f"ARIMA fitting error: {e}")
                self.is_fitted = False

    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        """
        Determine the position based on the ARIMA model forecast

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

        if not self.is_fitted or len(historical_data) < self.window_size:
            return current_position

        try:
            returns = np.log(historical_data["close"]).diff().dropna()

            self.model = ARIMA(
                returns[-self.window_size :], order=self.best_order
            ).fit()

            predicted_return = self.model.forecast(steps=1)[0]

            if predicted_return > self.threshold:
                return 1.0
            elif predicted_return < -self.threshold:
                return -1.0
            else:
                return 0.0

        except Exception as e:
            print(f"ARIMA forecasting error: {e}")
            return current_position

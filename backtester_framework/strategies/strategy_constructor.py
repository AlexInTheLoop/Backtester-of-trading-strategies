from abc import ABC, abstractmethod
from typing import Callable, Any
import inspect


class Strategy(ABC):
    """
    Abstract class used to set up the interface for trading strategies
    """

    @abstractmethod
    def get_position(self, historical_data: Any, current_position: float) -> float:
        """
        Determine the position to take according to historical data

        Parameters
        ----------
        historical_data: Any
            historical data used to determine the position
        current_position: float
            current position (-1.0, 0 or 1.0)

        Returns
        ----------
        float
            new position (-1.0, 0 or 1.0)
        """
        pass

    def fit(self, data: Any) -> None:
        """
        Optional method to optimize the strategy parameters

        Parameters
        ----------
        data: Any
            training data set
        """
        pass


def strategy(*, name: str) -> Callable:
    """
    Decorator to create a simple strategy from a function

    Parameters
    ----------
    name: str
        name of the strategy
    """

    def decorator(func: Callable) -> type:
        sig = inspect.signature(func)
        params = list(sig.parameters.items())
        strategy_params = {
            name: param.default
            for name, param in params[2:]
            if param.default is not param.empty
        }

        class SimpleStrategy(Strategy):
            def __init__(self, **kwargs):
                for param_name, default_value in strategy_params.items():
                    setattr(self, param_name, kwargs.get(param_name, default_value))

            def get_position(
                self, historical_data: Any, current_position: float
            ) -> float:
                params = {
                    param: getattr(self, param) for param in strategy_params.keys()
                }
                return func(historical_data, current_position, **params)

        SimpleStrategy.__name__ = name
        return SimpleStrategy

    return decorator

from backtester_framework.strategies.strategy_constructor import Strategy, strategy
from backtester_framework.strategies.moving_average import (
    ma_crossover as MovingAverageCrossover,
    trend_indicator as TrendIndicator,
)
from backtester_framework.strategies.RSI import rsi_strategy as RSIStrategy
from backtester_framework.strategies.arima import ARIMAStrategy
from backtester_framework.strategies.linear_trend import LinearTrendStrategy

__all__ = [
    "Strategy",
    "strategy",
    "MovingAverageCrossover",
    "TrendIndicator",
    "RSIStrategy",
    "ARIMAStrategy",
    "LinearTrendStrategy",
]

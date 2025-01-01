from backtester_framework.strategies.strategy_constructor import strategy
import pandas as pd
from typing import List


@strategy(name="MA Crossover")
def ma_crossover(
    historical_data: pd.DataFrame,
    current_position: float,
    short_window: int = 20,
    long_window: int = 50,
) -> float:
    """
    Strategy based on the crossover of two Simple Moving Averages (SMA)

    Parameters
    ----------
    historical_data: DataFrame
        historical data series
    current_position: float
        current position (-1.0, 0 or 1.0)
    short_window: int
        short SMA window (default: 20)
    long_window: int
        long SMA window (default: 50)

    Returns
    ----------
    float
        new position (-1.0, 0.0, or 1.0)
    """
    if len(historical_data) < long_window:
        return 0.0

    historical_data.columns = [col.lower() for col in historical_data.columns]
    prices = historical_data["close"]

    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()

    # Short MA > Long MA => BUY
    if short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]:
        return 1.0
    # Short MA < Long MA => SELL
    elif short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= long_ma.iloc[-2]:
        return -1.0

    return float(current_position)


@strategy(name="TrendIndicator")
def trend_indicator(
    historical_data: pd.DataFrame,
    current_position: float,
    window_size: int = 179,
    half_lives: List[float] = [1, 2.5, 5, 10, 20, 40],
    decay_factors: List[float] = [
                                    0.5,
                                    0.757858283,
                                    0.870550563,
                                    0.933032992,
                                    0.965936329,
                                    0.982820599,
                                ],
    normalization_factors: List[float] = [
                                            1.0000,
                                            1.0000,
                                            1.0000,
                                            1.0000,
                                            1.0020,
                                            1.0462,
                                        ],
    ) -> float:
    """
    Strategy based on multiple EMAs comparison to detect trends

    Parameters
    ----------
    historical_data: DataFrame
        historical data series
    current_position: float
        current position (-1.0, 0 or 1.0)
    window_size: int
        Size of the observation window (default: 179)
    half_lives: List[float]
        List of half-lives for EMA calculation
    decay_factors: List[float]
        List of decay factors for each half-life
    normalization_factors: List[float]
        List of normalization factors for each EMA

    Returns
    ----------
    float
        new position (-1.0, 0.0, or 1.0)
    """

    if not (len(half_lives) == len(decay_factors) == len(normalization_factors)):
        raise ValueError(
            "half_lives, decay_factors et normalization_factors lists should have the same size."
        )

    if len(historical_data) < window_size:
        return current_position

    try:
        close = next(
            (col for col in historical_data.columns if col in ["Close", "close"]), None
        )
        if close is None:
            raise ValueError("No 'close' column found in historical data")

        prices = historical_data[close].iloc[-window_size:]

        def calc_ema(
            prices: pd.Series, decay_factor: float, normalization_factor: float
        ) -> float:
            ema = 0
            prices_array = prices.values
            for i in range(len(prices_array)):
                weight = (1 - decay_factor) * (decay_factor**i)
                ema += weight * prices_array[-(i + 1)]
            return ema * normalization_factor

        # EMAs computations
        emas = []
        for i in range(len(half_lives)):
            ema = calc_ema(prices, decay_factors[i], normalization_factors[i])
            emas.append(ema)

        # EMAs comparisons (short term vs long term)
        comparisons = []
        for i in range(0, len(emas) - 1, 2):
            comparisons.append((emas[i], emas[i + 1]))

        # Trend Indicator components extraction
        trend_components = []
        for short_ema, long_ema in comparisons:
            component = 1 if short_ema >= long_ema else -1
            trend_components.append(component)

        # Components mean calculation
        trend_indicator = sum(trend_components) / len(trend_components)

        if trend_indicator == 1:
            return 1.0
        elif trend_indicator == -1:
            return 0.0
        else:
            return 0.5

    except Exception as e:
        print(f"Trend indicator error: {e}")
        return current_position

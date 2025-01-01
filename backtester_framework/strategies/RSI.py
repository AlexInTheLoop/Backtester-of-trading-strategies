from backtester_framework.strategies.strategy_constructor import strategy
import pandas as pd


@strategy(name="RSI")
def rsi_strategy(
    historical_data: pd.DataFrame,
    current_position: float,
    rsi_period: int = 14,
    overbought: float = 70,
    oversold: float = 30,
) -> float:
    """
    Strategy based on the Relative Strength Index:
        - Buy order in oversold zone
        - Sell order in overbought zone

    Parameters
    ----------
    historical_data: DataFrame
        historical data with price information
    current_position: float
        current position (-1.0, 0 or 1.0)
    rsi_period: int
        period to compute the RSI (default: 14)
    overbought: float
        lower bound of the overbought zone (default: 70)
    oversold: float
        lower bound of the oversold zone (default: 30)

    Returns
    ----------
    float
        new position (-1.0, 0.0, or 1.0)
    """
    if len(historical_data) < rsi_period:
        return 0.0

    historical_data.columns = [col.lower() for col in historical_data.columns]
    close_prices = historical_data["close"]

    abs_return = close_prices.diff()

    gains = (abs_return.where(abs_return > 0, 0)).rolling(window=rsi_period).mean()
    losses = (-abs_return.where(abs_return < 0, 0)).rolling(window=rsi_period).mean()

    rsi = 100 - (100 / (1 + gains / losses))

    if rsi.iloc[-1] < oversold:
        return 1.0
    elif rsi.iloc[-1] > overbought:
        return -1.0

    return float(current_position)

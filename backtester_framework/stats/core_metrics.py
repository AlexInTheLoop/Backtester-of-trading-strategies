import numpy as np
import pandas as pd


def calculate_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change()


def annualized_return(returns: pd.Series, N: int) -> float:
    return returns.mean() * N


def annualized_cagr(returns: pd.Series, N: int) -> float:
    total_return = np.prod(1 + returns)
    n_years = len(returns) / N
    return (total_return ** (1 / n_years)) - 1


def annualized_std(returns: pd.Series, N: int) -> float:
    return returns.std() * np.sqrt(N)


def downside_deviation(returns: pd.Series, target_return: float, N: int) -> float:
    downside_returns = returns[returns < target_return] - target_return
    return np.sqrt(np.sum(downside_returns**2) / len(returns)) * np.sqrt(N)


def upside_deviation(returns: pd.Series, target_return: float, N: int) -> float:
    upside_returns = returns[returns > target_return] - target_return
    return np.sqrt(np.sum(upside_returns**2) / len(returns)) * np.sqrt(N)


def covariance(returns1: pd.Series, returns2: pd.Series, N: int) -> float:
    return returns1.cov(returns2) * N


def correlation(returns1: pd.Series, returns2: pd.Series) -> float:
    return returns1.corr(returns2)


def count_trades(positions: pd.Series) -> int:
    return int((positions.diff() != 0).sum())


def winning_trades_percentage(positions: pd.Series, returns: pd.Series) -> float:
    trades = positions.diff() != 0

    trade_returns = returns[trades]
    trade_positions = positions[trades]

    winning_trades = (trade_returns * trade_positions) > 0

    return winning_trades.mean() if len(winning_trades) > 0 else 0.0

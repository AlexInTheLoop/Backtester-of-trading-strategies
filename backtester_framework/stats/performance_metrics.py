import numpy as np
import pandas as pd
from backtester_framework.stats.core_metrics import annualized_return, annualized_std
from backtester_framework.stats.tail_metrics import skewness, kurtosis, drawdown, max_drawdown


def sharpe_ratio(returns: pd.Series, risk_free_rate: float, N: int) -> float:
    excess_returns = returns - risk_free_rate / N
    return annualized_return(excess_returns, N) / annualized_std(returns, N)


def adjusted_sharpe_ratio(returns: pd.Series, risk_free_rate: float, N: int) -> float:
    sr = sharpe_ratio(returns, risk_free_rate, N)
    skew = skewness(returns)
    kurt = kurtosis(returns)

    return sr * (1 + skew * sr / 6 - (kurt - 3) * sr**2 / 24)


def sortino_ratio(returns: pd.Series, target_return: float, N: int) -> float:
    downside_returns = returns[returns < target_return] - target_return
    downside_std = np.sqrt(np.sum(downside_returns**2) / len(returns)) * np.sqrt(N)

    return (annualized_return(returns, N) - target_return) / downside_std


def calmar_ratio(
    returns: pd.Series, nav: pd.Series, N: int, window: int = 756
) -> float:
    return -annualized_return(returns, N) / max_drawdown(nav, window)


def pain_ratio(returns: pd.Series, nav: pd.Series, N: int) -> float:
    pain_index = -drawdown(nav).mean()
    return annualized_return(returns, N) / pain_index


def var_ratio(returns: pd.Series, N: int, confidence: float = 0.95) -> float:
    var = np.percentile(returns, (1 - confidence) * 100)
    return -annualized_return(returns, N) / (N * var)


def cvar_ratio(returns: pd.Series, N: int, confidence: float = 0.95) -> float:
    var = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var].mean()
    return -annualized_return(returns, N) / (N * cvar)


def hit_rate(returns: pd.Series) -> float:
    return (returns > 0).mean()


def gain_to_pain_ratio(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    pains = -returns[returns < 0].sum()
    return gains / pains if pains != 0 else np.inf


def calculate_alpha_beta(
    returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float, N: int
) -> tuple:
    excess_returns = returns - risk_free_rate / N
    excess_benchmark = benchmark_returns - risk_free_rate / N

    beta, alpha = np.polyfit(excess_benchmark, excess_returns, 1)

    alpha = alpha * N

    return alpha, beta

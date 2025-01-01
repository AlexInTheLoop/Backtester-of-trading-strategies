import numpy as np
import pandas as pd
from typing import Optional


def skewness(returns: pd.Series) -> float:
    T = len(returns)
    r_bar = returns.mean()
    s_r = returns.std()
    return (T / ((T - 1) * (T - 2))) * np.sum(((returns - r_bar) / s_r) ** 3)


def kurtosis(returns: pd.Series) -> float:
    T = len(returns)
    r_bar = returns.mean()
    s_r = returns.std()
    kurt = (T * (T + 1) / ((T - 1) * (T - 2) * (T - 3))) * np.sum(
        ((returns - r_bar) / s_r) ** 4
    )
    kurt -= 3 * (T - 1) ** 2 / ((T - 2) * (T - 3))
    return kurt


def coskewness(returns1: pd.Series, returns2: pd.Series) -> float:
    T = len(returns1)
    r1_bar = returns1.mean()
    r2_bar = returns2.mean()
    s1 = returns1.std()
    s2 = returns2.std()

    return (T / ((T - 1) * (T - 2))) * np.sum(
        ((returns1 - r1_bar) * (returns2 - r2_bar) ** 2) / (s1 * s2**2)
    )


def cokurtosis(returns1: pd.Series, returns2: pd.Series) -> float:
    T = len(returns1)
    r1_bar = returns1.mean()
    r2_bar = returns2.mean()
    s1 = returns1.std()
    s2 = returns2.std()

    ck = (T * (T + 1) / ((T - 1) * (T - 2) * (T - 3))) * np.sum(
        ((returns1 - r1_bar) * (returns2 - r2_bar) ** 3) / (s1 * s2**3)
    )
    ck -= 3 * (T - 1) ** 2 / ((T - 2) * (T - 3))
    return ck


def drawdown(nav: pd.Series) -> pd.Series:
    rolling_max = nav.expanding().max()
    drawdowns = nav / rolling_max - 1
    return drawdowns


def max_drawdown(nav: pd.Series, window: Optional[int] = None) -> float:
    if window is not None:
        nav = nav.iloc[-window:]
    return drawdown(nav).min()


def drawdown_duration(nav: pd.Series) -> float:
    rolling_max = nav.expanding().max()
    drawdowns = nav / rolling_max - 1

    is_drawdown = drawdowns < 0

    duration = 0
    for i in range(len(is_drawdown) - 1, -1, -1):
        if not is_drawdown.iloc[i]:
            break
        duration += 1

    return duration / 252

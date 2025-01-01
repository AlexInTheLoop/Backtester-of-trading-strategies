from dataclasses import dataclass
from typing import Dict, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
from backtester_framework.stats import core_metrics, tail_metrics, performance_metrics
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# Dictionary to convert the frequency attribute into the number of data points into a year
FREQ_MAP = {
    "1min": 525600,
    "T": 525600,
    "5min": 105120,
    "5T": 105120,
    "15min": 35040,
    "15T": 35040,
    "30min": 17520,
    "30T": 17520,
    "60min": 8760,
    "240min": 2190,
    "D": 365,
    "W": 52,
    "MS": 12,
}


@dataclass
class Result:
    """
    Class to store and analyse the results of a backtest for multiple assets
    """

    positions: Dict[str, pd.DataFrame]
    data: Dict[str, pd.DataFrame]
    initial_capital: float
    commission: float
    slippage: float
    frequency: str

    def __post_init__(self):
        """
        Initialize the basic computations after the instance creation
        """
        self.asset_names = list(self.positions.keys())

        for asset_name in self.asset_names:
            if (
                not isinstance(self.positions[asset_name], pd.DataFrame)
                or "position" not in self.positions[asset_name].columns
            ):
                raise ValueError(
                    f"positions for {asset_name} must be a DataFrame with a 'position' field"
                )

            if (
                not isinstance(self.data[asset_name], pd.DataFrame)
                or "close" not in self.data[asset_name].columns
            ):
                raise ValueError(
                    f"data for {asset_name} must be a DataFrame with a 'close' field"
                )

            if len(self.positions[asset_name]) != len(self.data[asset_name]):
                raise ValueError(
                    f"positions and data for {asset_name} must share the same length"
                )

            if (
                not (-1 <= self.positions[asset_name]["position"]).all()
                or not (self.positions[asset_name]["position"] <= 1).all()
            ):
                raise ValueError(f"Positions for {asset_name} must belong to [-1,1]")

            self.data[asset_name].index = pd.to_datetime(self.data[asset_name].index)
            self.positions[asset_name].index = pd.to_datetime(
                self.positions[asset_name].index
            )

        self.returns = {
            asset: self.data[asset]["close"].pct_change().fillna(0)
            for asset in self.asset_names
        }

        self.calculate_nav()
        self.N = FREQ_MAP[self.frequency]

    def calculate_nav(self):
        """
        Compute the Net Asset Value (NAV) according to the positions adjusted by the commissions
        and slippage % for each asset
        """
        self.nav = {}
        for asset_name in self.asset_names:
            nav_series = pd.Series(index=self.data[asset_name].index, dtype=float)
            nav_series.iloc[0] = self.initial_capital

            prev_position = 0

            for i in range(1, len(nav_series)):
                current_position = self.positions[asset_name]["position"].iloc[i - 1]
                ret = self.returns[asset_name].iloc[i] * prev_position

                if current_position != prev_position:
                    transaction_cost = abs(current_position - prev_position) * (
                        self.commission + self.slippage
                    )
                else:
                    transaction_cost = 0

                nav_series.iloc[i] = nav_series.iloc[i - 1] * (
                    1 + ret - transaction_cost
                )
                prev_position = current_position

            self.nav[asset_name] = nav_series

    def get_essential_metrics(
        self, asset_name: Optional[str] = None
    ) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Returns the essential metrics of the backtested strategy

        Parameters
        ----------
        asset_name: str, optional
            name of the asset to compute metrics for. If None, computes metrics for all assets.

        Returns
        ----------
        Union[Dict[str, float], Dict[str, Dict[str, float]]]
            If asset_name is provided: dictionary with metrics names in key and metrics in value
            If asset_name is None: dictionary with asset names in key and metrics dictionaries in value
        """
        if asset_name is not None:
            if asset_name not in self.asset_names:
                raise ValueError(f"Asset {asset_name} not found in backtested assets")

            nav = self.nav[asset_name]
            nav_returns = nav.pct_change().fillna(0)
            positions_series = pd.Series(
                self.positions[asset_name]["position"].values,
                index=self.data[asset_name].index,
            )

            return {
                "Total Return (%)": (nav.iloc[-1] / self.initial_capital - 1) * 100,
                "Annualized Return (%)": core_metrics.annualized_return(
                    nav_returns, self.N
                )
                * 100,
                "Number of Trades": core_metrics.count_trades(positions_series),
                "Winning Trades (%)": core_metrics.winning_trades_percentage(
                    positions_series, self.returns[asset_name]
                )
                * 100,
                "Volatility (%)": core_metrics.annualized_std(nav_returns, self.N)
                * 100,
                "Sharpe Ratio": performance_metrics.sharpe_ratio(
                    nav_returns, 0, self.N
                ),
                "Maximum Drawdown (%)": tail_metrics.max_drawdown(nav) * 100,
                "Sortino Ratio": performance_metrics.sortino_ratio(
                    nav_returns, 0, self.N
                ),
            }

        return {asset: self.get_essential_metrics(asset) for asset in self.asset_names}

    def get_all_metrics(
        self, asset_name: Optional[str] = None
    ) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Returns all metrics available for the backtested strategy

        Parameters
        ----------
        asset_name: str, optional
            name of the asset to compute metrics for. If None, computes metrics for all assets.

        Returns
        ----------
        Union[Dict[str, float], Dict[str, Dict[str, float]]]
            If asset_name is provided: dictionary with metrics names in key and metrics in value
            If asset_name is None: dictionary with asset names in key and metrics dictionaries in value
        """
        if asset_name is not None:
            if asset_name not in self.asset_names:
                raise ValueError(f"Asset {asset_name} not found in backtested assets")

            nav = self.nav[asset_name]
            nav_returns = nav.pct_change().fillna(0)

            metrics = self.get_essential_metrics(asset_name)

            additional_metrics = {
                "CAGR (%)": core_metrics.annualized_cagr(nav_returns, self.N) * 100,
                "Skewness": tail_metrics.skewness(nav_returns),
                "Kurtosis": tail_metrics.kurtosis(nav_returns),
                "Adjusted Sharpe Ratio": performance_metrics.adjusted_sharpe_ratio(
                    nav_returns, 0, self.N
                ),
                "Calmar Ratio": performance_metrics.calmar_ratio(
                    nav_returns, nav, self.N
                ),
                "Pain Ratio": performance_metrics.pain_ratio(nav_returns, nav, self.N),
                "VaR Ratio": performance_metrics.var_ratio(nav_returns, self.N),
                "CVaR Ratio": performance_metrics.cvar_ratio(nav_returns, self.N),
                "Gain to Pain Ratio": performance_metrics.gain_to_pain_ratio(
                    nav_returns
                ),
            }

            metrics.update(additional_metrics)
            return metrics

        return {asset: self.get_all_metrics(asset) for asset in self.asset_names}

    def plot(
        self, what: str = "nav", asset_name: str = None, backend: str = "matplotlib"
    ):
        """
        Help to visualize the backtest results

        Parameters
        ----------
        what: str
            name of what is supposed to be displayed (nav, positions or returns with by default nav)
        asset_name: str, optional
            name of the asset to plot. If None, plots all assets
        backend: str
            name of the chart style (matplotlib, seaborn or plotly with by default plotly)
        """
        if backend not in ["matplotlib", "seaborn", "plotly"]:
            raise ValueError(
                "Backend must be one of: 'matplotlib', 'seaborn', 'plotly'"
            )

        if what not in ["nav", "returns", "positions"]:
            raise ValueError("what must be one of: 'nav', 'returns', 'positions'")

        if asset_name is not None and asset_name not in self.asset_names:
            raise ValueError(f"Asset {asset_name} not found in backtested assets")

        assets_to_plot = [asset_name] if asset_name else self.asset_names

        if backend == "matplotlib":
            plt.figure(figsize=(12, 6))

            if what == "nav":
                for asset in assets_to_plot:
                    plt.plot(
                        self.nav[asset].index,
                        self.nav[asset].values,
                        linewidth=2,
                        label=asset,
                    )
                plt.title("Net Asset Value Evolution", fontsize=12)
                plt.ylabel("NAV")
                plt.legend()

            elif what == "returns":
                for asset in assets_to_plot:
                    nav_returns = self.nav[asset].pct_change().fillna(0)
                    plt.hist(nav_returns, bins=50, density=True, alpha=0.5, label=asset)
                plt.title("Returns Distribution", fontsize=12)
                plt.ylabel("Frequency")
                plt.legend()

            elif what == "positions":
                for asset in assets_to_plot:
                    plt.plot(
                        self.positions[asset].index,
                        self.positions[asset]["position"],
                        label=asset,
                    )
                plt.title("Position Evolution", fontsize=12)
                plt.ylabel("Position (-1 to 1)")
                plt.legend()

            plt.grid(True, alpha=0.3)
            plt.show()

        elif backend == "seaborn":
            plt.figure(figsize=(12, 6))

            if what == "nav":
                for asset in assets_to_plot:
                    sns.lineplot(data=self.nav[asset], label=asset)
                plt.title("Net Asset Value Evolution", fontsize=12)
                plt.ylabel("NAV")

            elif what == "returns":
                for asset in assets_to_plot:
                    nav_returns = self.nav[asset].pct_change().fillna(0)
                    sns.histplot(
                        data=nav_returns,
                        bins=50,
                        stat="density",
                        alpha=0.5,
                        label=asset,
                    )
                plt.title("Returns Distribution", fontsize=12)
                plt.ylabel("Frequency")

            elif what == "positions":
                for asset in assets_to_plot:
                    sns.lineplot(data=self.positions[asset]["position"], label=asset)
                plt.title("Position Evolution", fontsize=12)
                plt.ylabel("Position (-1 to 1)")

            sns.despine()
            plt.legend()
            plt.show()

        elif backend == "plotly":
            fig = make_subplots(rows=1, cols=1)

            if what == "nav":
                for asset in assets_to_plot:
                    fig.add_trace(
                        go.Scatter(
                            x=self.nav[asset].index,
                            y=self.nav[asset].values,
                            mode="lines",
                            name=asset,
                            line=dict(width=2),
                        )
                    )
                fig.update_layout(
                    title="Net Asset Value Evolution",
                    xaxis_title="Time",
                    yaxis_title="NAV",
                    template="ggplot2",
                )

            elif what == "returns":
                for asset in assets_to_plot:
                    nav_returns = self.nav[asset].pct_change().fillna(0)
                    fig.add_trace(
                        go.Histogram(
                            x=nav_returns,
                            nbinsx=50,
                            name=asset,
                            opacity=0.7,
                            histnorm="probability density",
                        )
                    )
                fig.update_layout(
                    title="Returns Distribution",
                    xaxis_title="Returns",
                    yaxis_title="Density",
                    template="ggplot2",
                    barmode="overlay",
                )

            elif what == "positions":
                for asset in assets_to_plot:
                    fig.add_trace(
                        go.Scatter(
                            x=self.positions[asset].index,
                            y=self.positions[asset]["position"],
                            mode="lines",
                            name=asset,
                            line=dict(width=2),
                        )
                    )
                fig.update_layout(
                    title="Position Evolution",
                    xaxis_title="Time",
                    yaxis_title="Position (-1 to 1)",
                    template="ggplot2",
                )

            fig.show()

    def compare_results(self, *other_results: "Result") -> pd.DataFrame:
        """
        Compare the results of the current instance with other Result instances

        Parameters
        ----------
        *other_results: Result
            other Result class instances to compare

        Returns
        ----------
        DataFrame
            metrics of different strategies for each asset
        """
        metrics = {}

        # Add current result metrics
        for asset_name in self.asset_names:
            metrics[f"Strategy 1 - {asset_name}"] = self.get_all_metrics(asset_name)

        # Add other results metrics
        for i, result in enumerate(other_results, 2):
            for asset_name in result.asset_names:
                metrics[f"Strategy {i} - {asset_name}"] = result.get_all_metrics(
                    asset_name
                )

        return pd.DataFrame(metrics)

import plotly.graph_objects as go  # type: ignore
import pandas as pd


def create_asset_price_chart(data: pd.DataFrame) -> go.Figure:
    """Creates asset price chart"""
    fig = go.Figure()

    data.columns = [col.lower() for col in data.columns]

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["close"],
            name="Price",
            line=dict(color="black", width=1),
        )
    )

    fig.update_layout(
        title="Asset price",
        xaxis_title="Time",
        yaxis_title="Price",
        height=400,
        showlegend=True,
        template="plotly_white",
    )

    return fig


def create_strategy_nav_chart(
    nav: pd.Series, initial_capital: float, strategy_name: str
) -> go.Figure:
    """Creates the chart for the NAV with a conditional color"""
    fig = go.Figure()

    mask = nav > initial_capital

    pos_nav = nav.copy()
    pos_nav[~mask] = None
    fig.add_trace(
        go.Scatter(
            x=nav.index,
            y=pos_nav,
            name="Profit",
            line=dict(color="green", width=1),
            mode="lines",
        )
    )

    neg_nav = nav.copy()
    neg_nav[mask] = None
    fig.add_trace(
        go.Scatter(
            x=nav.index,
            y=neg_nav,
            name="Loss",
            line=dict(color="red", width=1),
            mode="lines",
        )
    )

    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="black",
        annotation_text="Initial capital",
    )

    fig.update_layout(
        title=f"NAV evolution - {strategy_name}",
        xaxis_title="Time",
        yaxis_title="NAV",
        height=500,
        showlegend=True,
        template="plotly_white",
    )

    return fig


def create_strategy_signals_chart(
    data: pd.DataFrame, positions: pd.Series, strategy_name: str
) -> go.Figure:
    """Creates the chart of trading signals"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["close"],
            name="Prix",
            line=dict(color="black", width=1),
        )
    )

    long_entries = positions.diff() > 0
    if long_entries.any():
        fig.add_trace(
            go.Scatter(
                x=data[long_entries].index,
                y=data[long_entries]["close"],
                name="Buy",
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=15,
                    color="green",
                ),
                hovertemplate="Date: %{x}<br>Prix: %{y:.2f}<extra>Achat</extra>",
            )
        )

    short_entries = positions.diff() < 0
    if short_entries.any():
        fig.add_trace(
            go.Scatter(
                x=data[short_entries].index,
                y=data[short_entries]["close"],
                name="Sell",
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=15,
                    color="red",
                ),
                hovertemplate="Date: %{x}<br>Prix: %{y:.2f}<extra>Vente</extra>",
            )
        )

    fig.update_layout(
        title=f"Trading Signals - {strategy_name}",
        xaxis_title="Time",
        yaxis_title="Price",
        height=500,
        showlegend=True,
        template="plotly_white",
    )

    return fig


def create_strategy_returns_distribution(
    nav: pd.Series, strategy_name: str
) -> go.Figure:
    """Creates an histogram for the returns distribution"""
    returns = nav.pct_change() * 100

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=50,
            name="Returns",
            histnorm="probability density",
            marker_color="grey",
            opacity=0.75,
        )
    )

    fig.add_vline(
        x=0, line_dash="dash", line_color="black", annotation_text="Null return"
    )

    mean_return = returns.mean()
    std_return = returns.std()

    annotations = []
    annotations.append(
        dict(
            x=mean_return,
            y=1,
            xref="x",
            yref="paper",
            text=f"Mean: {mean_return:.2f}%",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
        )
    )

    fig.update_layout(
        title=f"Returns distribution - {strategy_name}",
        xaxis_title="Returns (%)",
        yaxis_title="Density",
        height=500,
        showlegend=True,
        template="plotly_white",
        annotations=annotations,
    )

    return fig


def create_strategy_summary(
    result, strategy_name: str, data: pd.DataFrame, graph_type: str
) -> go.Figure:
    """Creates the chart according to data asked"""
    if graph_type == "nav":
        return create_strategy_nav_chart(
            result.nav, result.initial_capital, strategy_name
        )
    elif graph_type == "signals":
        return create_strategy_signals_chart(
            data, result.positions["position"], strategy_name
        )
    else:
        return create_strategy_returns_distribution(result.nav, strategy_name)

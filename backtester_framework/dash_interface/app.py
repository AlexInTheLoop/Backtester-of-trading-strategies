from dash import Dash, html, dcc, Input, Output, State, callback, ALL, MATCH, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import base64
import io

from backtester_framework.dash_interface.visualization import (
    create_asset_price_chart,
    create_strategy_summary,
)
from backtester_framework.strategies.moving_average import ma_crossover, trend_indicator
from backtester_framework.strategies.RSI import rsi_strategy
from backtester_framework.strategies.arima import ARIMAStrategy
from backtester_framework.strategies.linear_trend import LinearTrendStrategy
from backtester_framework.backtester.backtester import Backtester
from backtester_framework.dash_interface.layout import create_layout

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "Backtesting Framework Interface"
app.layout = create_layout()

STRATEGIES = {
    "MA Crossover": ma_crossover,
    "RSI": rsi_strategy,
    "ARIMA": ARIMAStrategy,
    "Linear Trend": LinearTrendStrategy,
    "Trend Indicator": trend_indicator,
}


def parse_csv(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(
        io.StringIO(decoded.decode("utf-8")), index_col=0, parse_dates=True
    )
    return df


@callback(
    [
        Output("upload-status", "children"),
        Output("data-storage", "data"),
        Output("asset-price-graph", "figure"),
    ],
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def store_data(contents, filename):
    if contents is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Waiting for data",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
        )
        return "No file selected", None, empty_fig

    try:
        df = parse_csv(contents)
        price_fig = create_asset_price_chart(df)

        return [
            html.Div(
                [
                    html.I(
                        className="fas fa-check-circle",
                        style={"color": "green", "margin-right": "10px"},
                    ),
                    f"File successfully uploaded: {filename}",
                ]
            ),
            {
                "filename": filename,
                "data": df.to_json(date_format="iso", orient="split"),
            },
            price_fig,
        ]
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error during data loading",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
        )
        return f"Error during file loading: {str(e)}", None, empty_fig


@callback(
    [
        Output("rebalancing-frequency", "options"),
        Output("rebalancing-frequency", "value"),
    ],
    [Input("data-frequency", "value")],
)
def update_rebalancing_frequencies(data_freq):
    if not data_freq:
        return [], None

    freq_order = ["1min", "5min", "15min", "30min", "1H", "4H", "D", "W", "M"]
    freq_labels = {
        "1min": "1 minute",
        "5min": "5 minutes",
        "15min": "15 minutes",
        "30min": "30 minutes",
        "1H": "1 hour",
        "4H": "4 hours",
        "D": "Daily",
        "W": "Weekly",
        "M": "Monthly",
    }

    try:
        start_idx = freq_order.index(data_freq)
        available_freqs = freq_order[start_idx:]
        options = [
            {"label": freq_labels[freq], "value": freq} for freq in available_freqs
        ]
        return options, data_freq
    except ValueError:
        return [], None


@callback(
    [Output("strategy-params", "children"), Output("strategy-params-storage", "data")],
    Input("strategy-selector", "value"),
)
def update_strategy_params(selected_strategies):
    if not selected_strategies:
        return None, {}

    params_components = []
    params_storage = {}

    for strat_name in selected_strategies:
        if strat_name == "MA Crossover":
            params = {"short_window": 20, "long_window": 50}
            strategy_params = []
            for param_name, param_value in params.items():
                param_input = dbc.Row(
                    [
                        dbc.Label(f"{strat_name} - {param_name}:"),
                        dcc.Input(
                            id={
                                "type": "strategy-param",
                                "strategy": strat_name,
                                "param": param_name,
                            },
                            type="number",
                            value=param_value,
                            className="form-control",
                        ),
                    ],
                    className="mb-2",
                )
                strategy_params.append(param_input)
            params_components.extend(strategy_params)

        elif strat_name == "RSI":
            params = {"rsi_period": 14, "overbought": 70, "oversold": 30}
            strategy_params = []
            for param_name, param_value in params.items():
                param_input = dbc.Row(
                    [
                        dbc.Label(f"{strat_name} - {param_name}:"),
                        dcc.Input(
                            id={
                                "type": "strategy-param",
                                "strategy": strat_name,
                                "param": param_name,
                            },
                            type="number",
                            value=param_value,
                            className="form-control",
                        ),
                    ],
                    className="mb-2",
                )
                strategy_params.append(param_input)
            params_components.extend(strategy_params)

        elif strat_name == "ARIMA":
            params = {"window_size": 252, "prediction_threshold": 0.001}
            strategy_params = []
            for param_name, param_value in params.items():
                param_input = dbc.Row(
                    [
                        dbc.Label(f"{strat_name} - {param_name}:"),
                        dcc.Input(
                            id={
                                "type": "strategy-param",
                                "strategy": strat_name,
                                "param": param_name,
                            },
                            type="number",
                            value=param_value,
                            className="form-control",
                        ),
                    ],
                    className="mb-2",
                )
                strategy_params.append(param_input)
            params_components.extend(strategy_params)

        elif strat_name == "Linear Trend":
            params = {"window_size": 20, "trend_threshold": 0.001}
            strategy_params = []
            for param_name, param_value in params.items():
                param_input = dbc.Row(
                    [
                        dbc.Label(f"{strat_name} - {param_name}:"),
                        dcc.Input(
                            id={
                                "type": "strategy-param",
                                "strategy": strat_name,
                                "param": param_name,
                            },
                            type="number",
                            value=param_value,
                            className="form-control",
                        ),
                    ],
                    className="mb-2",
                )
                strategy_params.append(param_input)
            params_components.extend(strategy_params)

        elif strat_name == "Trend Indicator":
            params = {
                "window_size": 179,
                "half_lives": [1, 2.5, 5, 10, 20, 40],
                "decay_factors": [
                    0.5,
                    0.757858283,
                    0.870550563,
                    0.933032992,
                    0.965936329,
                    0.982820599,
                ],
                "normalization_factors": [
                    1.0000,
                    1.0000,
                    1.0000,
                    1.0000,
                    1.0020,
                    1.0462,
                ],
            }

            strategy_params = []
            param_input = dbc.Row(
                [
                    dbc.Label(f"{strat_name} - window_size:"),
                    dcc.Input(
                        id={
                            "type": "strategy-param",
                            "strategy": strat_name,
                            "param": "window_size",
                        },
                        type="number",
                        value=params["window_size"],
                        className="form-control",
                    ),
                ],
                className="mb-2",
            )
            strategy_params.append(param_input)

            for list_param in ["half_lives", "decay_factors", "normalization_factors"]:
                param_input = dbc.Row(
                    [
                        dbc.Label(f"{strat_name} - {list_param}:"),
                        dcc.Input(
                            id={
                                "type": "strategy-param",
                                "strategy": strat_name,
                                "param": list_param,
                            },
                            type="text",
                            value=str(params[list_param]),
                            className="form-control",
                        ),
                    ],
                    className="mb-2",
                )
                strategy_params.append(param_input)

            params_components.extend(strategy_params)

        params_storage[strat_name] = params

    return html.Div(params_components), params_storage


@callback(
    Output("strategy-tabs-container", "children"), [Input("strategy-selector", "value")]
)
def update_strategy_tabs(selected_strategies):
    if not selected_strategies:
        return html.Div("Select strategies to see the results")

    tabs = []
    for strat in selected_strategies:
        tabs.append(
            dbc.Tab(
                label=strat,
                children=[
                    html.Div(
                        [
                            dcc.RadioItems(
                                id={"type": "graph-type", "strategy": strat},
                                options=[
                                    {"label": " NAV", "value": "nav"},
                                    {"label": " Trading Signals", "value": "signals"},
                                    {
                                        "label": " Returns distribution",
                                        "value": "returns",
                                    },
                                ],
                                value="nav",
                                labelStyle={
                                    "display": "inline-block",
                                    "marginRight": "20px",
                                },
                                className="my-3",
                            ),
                            dbc.Spinner(
                                dcc.Graph(
                                    id={"type": "strategy-graph", "strategy": strat},
                                    figure={},
                                ),
                                color="primary",
                                type="border",
                            ),
                        ]
                    )
                ],
            )
        )

    return dbc.Tabs(tabs, className="mt-3", active_tab=selected_strategies[0])


def parse_list(s):
    """Helper function to parse string lists"""
    try:
        s = s.strip("[]").replace(" ", "")
        return [float(x) for x in s.split(",")]
    except Exception as e:
        print(f"Error parsing list: {e}")
        return None


@callback(
    Output({"type": "strategy-graph", "strategy": MATCH}, "figure"),
    [
        Input("run-backtest", "n_clicks"),
        Input({"type": "graph-type", "strategy": MATCH}, "value"),
    ],
    [
        State("data-storage", "data"),
        State("strategy-selector", "value"),
        State({"type": "strategy-param", "strategy": ALL, "param": ALL}, "value"),
        State("initial-capital", "value"),
        State("commission", "value"),
        State("slippage", "value"),
        State("rebalancing-frequency", "value"),
        State({"type": "strategy-graph", "strategy": MATCH}, "id"),
    ],
)
def update_strategy_graph(
    n_clicks,
    graph_type,
    stored_data,
    selected_strategies,
    param_values,
    initial_capital,
    commission,
    slippage,
    rebal_freq,
    graph_id,
):
    if not stored_data or not selected_strategies or param_values is None:
        return go.Figure()

    try:
        df = pd.read_json(io.StringIO(stored_data["data"]), orient="split")
        current_strategy = graph_id["strategy"]

        param_dict = {}
        param_idx = 0

        for strat in selected_strategies:
            strat_params = {}
            if strat == "MA Crossover":
                if param_idx + 1 >= len(param_values):
                    raise ValueError(f"Not enough parameters for {strat}")
                strat_params = {
                    "short_window": int(param_values[param_idx]),
                    "long_window": int(param_values[param_idx + 1]),
                }
                param_idx += 2
            elif strat == "RSI":
                if param_idx + 2 >= len(param_values):
                    raise ValueError(f"Not enough parameters for {strat}")
                strat_params = {
                    "rsi_period": int(param_values[param_idx]),
                    "overbought": float(param_values[param_idx + 1]),
                    "oversold": float(param_values[param_idx + 2]),
                }
                param_idx += 3
            elif strat == "ARIMA":
                if param_idx + 1 >= len(param_values):
                    raise ValueError(f"Not enough parameters for {strat}")
                strat_params = {
                    "window_size": int(param_values[param_idx]),
                    "prediction_threshold": float(param_values[param_idx + 1]),
                }
                param_idx += 2
            elif strat == "Linear Trend":
                if param_idx + 1 >= len(param_values):
                    raise ValueError(f"Not enough parameters for {strat}")
                strat_params = {
                    "window_size": int(param_values[param_idx]),
                    "trend_threshold": float(param_values[param_idx + 1]),
                }
                param_idx += 2
            elif strat == "Trend Indicator":
                if param_idx + 3 >= len(param_values):
                    raise ValueError(f"Not enough parameters for {strat}")
                strat_params = {
                    "window_size": int(param_values[param_idx]),
                    "half_lives": parse_list(param_values[param_idx + 1]),
                    "decay_factors": parse_list(param_values[param_idx + 2]),
                    "normalization_factors": parse_list(param_values[param_idx + 3]),
                }
                if None in [
                    strat_params["half_lives"],
                    strat_params["decay_factors"],
                    strat_params["normalization_factors"],
                ]:
                    raise ValueError(f"Invalid list parameters format for {strat}")
                param_idx += 4

            param_dict[strat] = strat_params

        if current_strategy not in param_dict:
            raise ValueError(f"Strategy {current_strategy} not found in parameters")

        strategy_params = param_dict[current_strategy]
        strategy = STRATEGIES[current_strategy](**strategy_params)

        backtester = Backtester(
            df,
            initial_capital=initial_capital,
            commission=commission / 100,
            slippage=slippage / 100,
            rebalancing_frequency=rebal_freq,
        )
        result = backtester.run(strategy)

        fig = create_strategy_summary(result, current_strategy, df, graph_type)
        fig.update_layout(template="plotly_white")
        return fig

    except Exception as e:
        print(f"Error in update_strategy_graph: {e}")
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Error: {str(e)}")
        return empty_fig


@callback(
    Output("metrics-table", "children"),
    [Input("run-backtest", "n_clicks")],
    [
        State("data-storage", "data"),
        State("stats-type", "value"),
        State("strategy-selector", "value"),
        State({"type": "strategy-param", "strategy": ALL, "param": ALL}, "value"),
        State("initial-capital", "value"),
        State("commission", "value"),
        State("slippage", "value"),
        State("rebalancing-frequency", "value"),
    ],
)
def update_metrics_table(
    n_clicks,
    stored_data,
    stats_type,
    selected_strategies,
    param_values,
    initial_capital,
    commission,
    slippage,
    rebal_freq,
):
    """Update metrics table"""
    if (
        not n_clicks
        or not stored_data
        or not selected_strategies
        or param_values is None
    ):
        return None

    try:
        df = pd.read_json(io.StringIO(stored_data["data"]), orient="split")
        results = []
        param_idx = 0

        for strat_name in selected_strategies:
            if strat_name == "MA Crossover":
                strategy_params = {
                    "short_window": int(param_values[param_idx]),
                    "long_window": int(param_values[param_idx + 1]),
                }
                param_idx += 2
            elif strat_name == "RSI":
                strategy_params = {
                    "rsi_period": int(param_values[param_idx]),
                    "overbought": float(param_values[param_idx + 1]),
                    "oversold": float(param_values[param_idx + 2]),
                }
                param_idx += 3
            elif strat_name == "ARIMA":
                strategy_params = {
                    "window_size": int(param_values[param_idx]),
                    "prediction_threshold": float(param_values[param_idx + 1]),
                }
                param_idx += 2
            elif strat_name == "Linear Trend":
                strategy_params = {
                    "window_size": int(param_values[param_idx]),
                    "trend_threshold": float(param_values[param_idx + 1]),
                }
                param_idx += 2
            elif strat_name == "Trend Indicator":
                strategy_params = {
                    "window_size": int(param_values[param_idx]),
                    "half_lives": parse_list(param_values[param_idx + 1]),
                    "decay_factors": parse_list(param_values[param_idx + 2]),
                    "normalization_factors": parse_list(param_values[param_idx + 3]),
                }
                if None in [
                    strategy_params["half_lives"],
                    strategy_params["decay_factors"],
                    strategy_params["normalization_factors"],
                ]:
                    print(f"Invalid parameters for Trend Indicator")
                    continue
                param_idx += 4

            strategy = STRATEGIES[strat_name](**strategy_params)
            backtester = Backtester(
                df,
                initial_capital=initial_capital,
                commission=commission / 100,
                slippage=slippage / 100,
                rebalancing_frequency=rebal_freq,
            )
            result = backtester.run(strategy)
            results.append((strat_name, result))

        metrics_data = {}
        for strat_name, result in results:
            if stats_type == "essential":
                metrics = result.get_essential_metrics()
            else:
                metrics = result.get_all_metrics()

            for metric, value in metrics.items():
                if metric not in metrics_data:
                    metrics_data[metric] = {}
                metrics_data[metric][strat_name] = value

        metrics_df = pd.DataFrame(metrics_data).T

        table = dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [html.Th("Metric", style={"width": "200px"})]
                        + [
                            html.Th(strat, style={"textAlign": "center"})
                            for strat in selected_strategies
                        ]
                    )
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(metric, style={"whiteSpace": "nowrap"}),
                                *[
                                    html.Td(
                                        (
                                            f"{metrics_df.loc[metric, strat]:.2f}"
                                            if isinstance(
                                                metrics_df.loc[metric, strat], float
                                            )
                                            else metrics_df.loc[metric, strat]
                                        ),
                                        style={"textAlign": "center"},
                                    )
                                    for strat in selected_strategies
                                ],
                            ]
                        )
                        for metric in metrics_df.index
                    ]
                ),
            ],
            striped=True,
            bordered=True,
            hover=True,
            className="mt-3",
        )

        return table

    except Exception as e:
        print(f"Error in update_metrics_table: {e}")
        return html.Div(f"Erreur: {str(e)}", style={"color": "red"})


if __name__ == "__main__":
    app.run_server(debug=True)

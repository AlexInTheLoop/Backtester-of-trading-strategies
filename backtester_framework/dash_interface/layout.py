from dash import html, dcc  # type: ignore
import dash_bootstrap_components as dbc  # type: ignore


def get_freq_options():
    """Retourne les options de fréquence"""
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
    return [{"label": label, "value": value} for value, label in freq_labels.items()]


def create_layout():
    """Creates interface layout"""
    freq_options = get_freq_options()

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.H1("BACKTESTING INTERFACE", className="text-center my-4")
                    )
                ]
            ),
            html.Div(
                [
                    dcc.Store(id="data-storage"),
                    dcc.Store(id="strategy-params-storage"),
                    dcc.Store(id="graphs-storage"),
                    dcc.Store(id="figures-store", data={}),
                    dcc.Store(
                        id="calculated-results-store", storage_type="memory", data={}
                    ),
                ]
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Importaton and data set up"),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Upload(
                                                id="upload-data",
                                                children=html.Div(
                                                    [
                                                        "Drag and Drop or ",
                                                        html.A("Select a .csv file"),
                                                    ]
                                                ),
                                                style={
                                                    "width": "100%",
                                                    "height": "60px",
                                                    "lineHeight": "60px",
                                                    "borderWidth": "1px",
                                                    "borderStyle": "dashed",
                                                    "borderRadius": "5px",
                                                    "textAlign": "center",
                                                    "margin": "10px",
                                                },
                                            ),
                                            html.Div(
                                                id="upload-status", className="mt-2"
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Data frequency"),
                                            dcc.Dropdown(
                                                id="data-frequency",
                                                options=freq_options,
                                                value="D",
                                                clearable=False,
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("Rebalancing frequency"),
                                            dcc.Dropdown(
                                                id="rebalancing-frequency",
                                                options=freq_options,
                                                value="D",
                                                clearable=False,
                                            ),
                                        ],
                                        width=6,
                                    ),
                                ],
                                className="mt-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Initial capital"),
                                            dbc.Input(
                                                id="initial-capital",
                                                type="number",
                                                value=10000,
                                                min=1,
                                                step=1000,
                                            ),
                                        ],
                                        width=4,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("Commission (%)"),
                                            dbc.Input(
                                                id="commission",
                                                type="number",
                                                value=0.1,
                                                min=0,
                                                max=100,
                                                step=0.01,
                                            ),
                                        ],
                                        width=4,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("Slippage (%)"),
                                            dbc.Input(
                                                id="slippage",
                                                type="number",
                                                value=0.1,
                                                min=0,
                                                max=100,
                                                step=0.01,
                                            ),
                                        ],
                                        width=4,
                                    ),
                                ],
                                className="mt-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Spinner(
                                                dcc.Graph(id="asset-price-graph"),
                                                color="primary",
                                                type="border",
                                            )
                                        ],
                                        width=12,
                                    )
                                ],
                                className="mt-3",
                            ),
                        ]
                    ),
                ],
                className="mb-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Backtesting configuration et execution"),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H5("Strategies selection"),
                                            dcc.Dropdown(
                                                id="strategy-selector",
                                                options=[
                                                    {
                                                        "label": "Moving Average Crossover",
                                                        "value": "MA Crossover",
                                                    },
                                                    {
                                                        "label": "Relavtive Strength Index",
                                                        "value": "RSI",
                                                    },
                                                    {
                                                        "label": "ARIMA",
                                                        "value": "ARIMA",
                                                    },
                                                    {
                                                        "label": "Linear Trend",
                                                        "value": "Linear Trend",
                                                    },
                                                    {
                                                        "label": "Trend Indicator",
                                                        "value": "Trend Indicator",
                                                    },
                                                ],
                                                multi=True,
                                                placeholder="Select one or several strategies",
                                            ),
                                        ],
                                        width=12,
                                    )
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H5(
                                                "Strategy parameters", className="mt-3"
                                            ),
                                            html.Div(id="strategy-params"),
                                        ]
                                    )
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H5("Metrics choice", className="mt-3"),
                                            dcc.RadioItems(
                                                id="stats-type",
                                                options=[
                                                    {
                                                        "label": " Basics",
                                                        "value": "essential",
                                                    },
                                                    {"label": " All", "value": "all"},
                                                ],
                                                value="essential",
                                                labelStyle={
                                                    "display": "inline-block",
                                                    "marginRight": "20px",
                                                },
                                            ),
                                        ],
                                        width=12,
                                    )
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Button(
                                                "Launch backtest",
                                                id="run-backtest",
                                                color="primary",
                                                size="lg",
                                                className="mt-4 w-100",
                                            )
                                        ],
                                        width=12,
                                    )
                                ]
                            ),
                        ]
                    ),
                ],
                className="mb-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Backtest results"),
                    dbc.CardBody(
                        [
                            html.Div(id="strategy-tabs-container"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Spinner(
                                                html.Div(id="metrics-table"),
                                                color="primary",
                                                type="border",
                                            )
                                        ]
                                    )
                                ],
                                className="mt-4",
                            ),
                        ]
                    ),
                ]
            ),
        ],
        fluid=True,
    )

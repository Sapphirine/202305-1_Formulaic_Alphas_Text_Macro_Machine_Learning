from dash import Dash, dcc, html


def returns_section(data, assets, benchmark_assets):
    return html.Div(
        className="section",
        children=[
            html.H1(
                children="Returns",
                className="section-title",
            ),
            html.P(
                children=("Analyze the final model returns over time"),
                className="section-description",
            ),
            html.Div(
                className="filter-group",
                children=[
                    html.Div(
                        className="filter-item",
                        children=[
                            html.Div(
                                children="Asset",
                                className="menu-title",
                            ),
                            dcc.Dropdown(
                                id="returns-asset-filter",
                                options=[
                                    {"label": asset, "value": asset} for asset in assets
                                ],
                                value="Overall",
                                clearable=False,
                                className="dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-item",
                        children=[
                            html.Div(
                                children="Date Range",
                                className="menu-title",
                            ),
                            dcc.DatePickerRange(
                                id="returns-date-range",
                                min_date_allowed=data["Date"].min().date(),
                                max_date_allowed=data["Date"].max().date(),
                                start_date=data["Date"].min().date(),
                                end_date=data["Date"].max().date(),
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-item",
                        children=[
                            html.Div(
                                children="Benchmark",
                                className="menu-title",
                            ),
                            dcc.Dropdown(
                                id="benchmark-asset-filter",
                                options=[
                                    {"label": asset, "value": asset}
                                    for asset in benchmark_assets
                                ],
                                value="Overall",
                                clearable=False,
                                className="dropdown",
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Graph(
                id="daily-returns-chart",
                className="card",
            ),
            dcc.Graph(
                id="cumulative-returns-chart",
                className="card",
            ),
            dcc.Graph(
                id="cumulative-log-returns-chart",
                className="card",
            ),
        ],
    )


def alpha_section(data, assets, benchmark_assets):
    assets = assets[1:]
    benchmark_assets = benchmark_assets[1:]

    return html.Div(
        className="section",
        children=[
            html.H1(
                children="Alpha",
                className="section-title",
            ),
            html.P(
                children=("Analyze the final model alpha over time vs a benchmark"),
                className="section-description",
            ),
            html.Div(
                className="filter-group",
                children=[
                    html.Div(
                        className="filter-item",
                        children=[
                            html.Div(
                                children="Asset",
                                className="menu-title",
                            ),
                            dcc.Dropdown(
                                id="alpha-asset-filter",
                                options=[
                                    {"label": asset, "value": asset} for asset in assets
                                ],
                                value=assets[0],
                                clearable=False,
                                className="dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-item",
                        children=[
                            html.Div(
                                children="Date Range",
                                className="menu-title",
                            ),
                            dcc.DatePickerRange(
                                id="alpha-date-range",
                                min_date_allowed=data["Date"].min().date(),
                                max_date_allowed=data["Date"].max().date(),
                                start_date=data["Date"].min().date(),
                                end_date=data["Date"].max().date(),
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-item",
                        children=[
                            html.Div(
                                children="Benchmark",
                                className="menu-title",
                            ),
                            dcc.Dropdown(
                                id="alpha-benchmark-asset-filter",
                                options=[
                                    {"label": asset, "value": asset}
                                    for asset in benchmark_assets
                                ],
                                value=benchmark_assets[0],
                                clearable=False,
                                className="dropdown",
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Graph(
                id="daily-alpha-chart",
                className="card",
            ),
            dcc.Graph(
                id="cumulative-alpha-chart",
                className="card",
            ),
        ],
    )


def lime_section(data, assets):
    default_asset = assets[0]
    default_date = data["Date"].max().date()
    explanation_data = data[
        (data["ticker"] == default_asset) & (data["Date"] == default_date)
    ]
    targets = list(explanation_data["target"])
    preds = list(explanation_data["LGBM_Prediction"])

    return html.Div(
        className="section",
        children=[
            html.H1(
                children="LIME",
                className="section-title",
            ),
            html.P(
                children=("LIME interpretations"),
                className="section-description",
            ),
            html.Div(
                className="filter-group",
                children=[
                    html.Div(
                        className="filter-item",
                        children=[
                            html.Div(
                                children="Asset",
                                className="menu-title",
                            ),
                            dcc.Dropdown(
                                id="lime-asset-filter",
                                options=[
                                    {"label": asset, "value": asset} for asset in assets
                                ],
                                value=default_asset,
                                clearable=False,
                                className="dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-item",
                        children=[
                            html.Div(
                                children="Date",
                                className="menu-title",
                            ),
                            dcc.Dropdown(
                                id="lime-date-filter",
                                options=[
                                    {"label": date, "value": date}
                                    for date in data["Date"]
                                    .map(lambda d: d.date())
                                    .unique()
                                ],
                                value=default_date,
                                clearable=False,
                                className="dropdown",
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Graph(
                id="lime-chart",
                className="card",
                figure={
                    "data": [
                        {
                            "x": explanation_data["variable"],
                            "y": explanation_data["score_contribution"],
                            "type": "bar",
                        },
                    ],
                    "layout": {"title": "LIME Explanation"},
                },
            ),
            html.Div(
                children=[
                    html.P(
                        id="lime-target",
                        children=(f"Target: {targets[0] if targets else None}"),
                        style={"font-size": "24px"},
                    ),
                    html.P(
                        id="lime-pred",
                        children=(f"Prediction: {preds[0] if preds else None}"),
                        style={"font-size": "24px"},
                    ),
                ]
            ),
        ],
    )

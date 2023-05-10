import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

from data.loader import get_returns_data, get_explanation_data, get_alpha_data
from view.sections import returns_section, alpha_section, lime_section


print("Loading data...")
returns_data, return_assets, benchmark_assets = get_returns_data()
explanation_data, explanation_assets = get_explanation_data()
alpha_data = get_alpha_data(returns_data, return_assets, benchmark_assets)
print("Data successfully loaded!")


external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?" "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
]
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.H1(
                                    children="Portfolio Insights",
                                    className="header-title",
                                ),
                                html.P(
                                    children=(
                                        "Analyze the portfolio performance against various benchmarks"
                                        " and view what model features drove returns"
                                    ),
                                    className="header-description",
                                ),
                            ],
                            className="header",
                        ),
                    ],
                ),
                returns_section(returns_data, return_assets, benchmark_assets),
                alpha_section(alpha_data, return_assets, benchmark_assets),
                lime_section(explanation_data, explanation_assets),
            ]
        ),
    ]
)


@app.callback(
    Output("daily-returns-chart", "figure"),
    Output("cumulative-returns-chart", "figure"),
    Output("cumulative-log-returns-chart", "figure"),
    Input("returns-asset-filter", "value"),
    Input("returns-date-range", "start_date"),
    Input("returns-date-range", "end_date"),
    Input("benchmark-asset-filter", "value"),
)
def update_returns_charts(asset, start_date, end_date, benchmark):
    is_overall = asset == "Overall"
    is_benchmark_overall = benchmark == "Overall"

    daily_returns_col = "Returns" if is_overall else asset
    benchmark_daily_returns_col = (
        ["SP500", "NASDAQ", "DJI"] if is_benchmark_overall else [benchmark]
    )
    complete_daily_returns_cols = [daily_returns_col] + benchmark_daily_returns_col

    cumulative_returns_col = "Cumulative Returns" if is_overall else f"{asset}_cumsum"
    benchmark_cumulative_returns_col = (
        ["SP500_cumsum", "NASDAQ_cumsum", "DJI_cumsum"]
        if is_benchmark_overall
        else [f"{benchmark}_cumsum"]
    )
    complete_cumulative_returns_cols = [
        cumulative_returns_col
    ] + benchmark_cumulative_returns_col

    cumulative_log_returns_col = (
        "Cumulative Log Returns" if is_overall else f"{asset}_cumlogsum"
    )
    benchmark_cumulative_log_returns_col = (
        ["SP500_cumlogsum", "NASDAQ_cumlogsum", "DJI_cumlogsum"]
        if is_benchmark_overall
        else [f"{benchmark}_cumlogsum"]
    )
    complete_cumulative_log_returns_cols = [
        cumulative_log_returns_col
    ] + benchmark_cumulative_log_returns_col

    filtered_data = returns_data[
        (returns_data["Date"] >= start_date) & (returns_data["Date"] <= end_date)
    ]

    daily_returns = px.line(
        filtered_data,
        x="Date",
        y=complete_daily_returns_cols,
        title="Daily Returns",
    )
    daily_returns.update_layout(title_x=0.5, yaxis_title="Pct Change")

    cumulative_returns = px.line(
        filtered_data,
        x="Date",
        y=complete_cumulative_returns_cols,
        title="Cumulative Returns",
    )
    cumulative_returns.update_layout(title_x=0.5, yaxis_title="Pct Change")
    cumulative_col_names = {
        col: col[:-7] if "cumsum" in col else col
        for col in complete_cumulative_returns_cols
    }
    cumulative_returns.for_each_trace(
        lambda t: t.update(
            name=cumulative_col_names[t.name],
            legendgroup=cumulative_col_names[t.name],
            hovertemplate=t.hovertemplate.replace(t.name, cumulative_col_names[t.name]),
        )
    )

    cumulative_log_returns = {
        "data": [
            {
                "x": filtered_data["Date"],
                "y": filtered_data[cumulative_log_returns_col],
                "type": "lines",
            },
        ],
        "layout": {"title": "Cumulative Log Returns"},
    }
    cumulative_log_returns = px.line(
        filtered_data,
        x="Date",
        y=complete_cumulative_log_returns_cols,
        title="Cumulative Log Returns",
    )
    cumulative_log_returns.update_layout(title_x=0.5, yaxis_title="Pct Change")
    cumulative_log_col_names = {
        col: col[:-10] if "cumlogsum" in col else col
        for col in complete_cumulative_log_returns_cols
    }
    cumulative_log_returns.for_each_trace(
        lambda t: t.update(
            name=cumulative_log_col_names[t.name],
            legendgroup=cumulative_log_col_names[t.name],
            hovertemplate=t.hovertemplate.replace(
                t.name, cumulative_log_col_names[t.name]
            ),
        )
    )

    return daily_returns, cumulative_returns, cumulative_log_returns


@app.callback(
    Output("daily-alpha-chart", "figure"),
    Output("cumulative-alpha-chart", "figure"),
    Input("alpha-asset-filter", "value"),
    Input("alpha-date-range", "start_date"),
    Input("alpha-date-range", "end_date"),
    Input("alpha-benchmark-asset-filter", "value"),
)
def update_alpha_charts(asset, start_date, end_date, benchmark):
    is_overall = asset == "Overall"
    returns_col = "Returns" if is_overall else asset
    daily_col = f"{returns_col}_{benchmark}_alpha"
    cumulative_col = f"{returns_col}_{benchmark}_cumulative_alpha"

    filtered_data = alpha_data[
        (alpha_data["Date"] >= start_date) & (alpha_data["Date"] <= end_date)
    ]

    daily_alpha = px.line(
        filtered_data,
        x="Date",
        y=daily_col,
        title=f"{asset} vs {benchmark} Daily Alpha",
    )
    daily_alpha.update_layout(title_x=0.5, yaxis_title="Pct Change")

    cumulative_alpha = px.line(
        filtered_data,
        x="Date",
        y=cumulative_col,
        title=f"{asset} vs {benchmark} Cumulative Alpha",
    )
    cumulative_alpha.update_layout(title_x=0.5, yaxis_title="Pct Change")

    return daily_alpha, cumulative_alpha


@app.callback(
    Output("lime-chart", "figure"),
    Output("lime-target", "children"),
    Output("lime-pred", "children"),
    Input("lime-asset-filter", "value"),
    Input("lime-date-filter", "value"),
)
def update_lime_charts(asset, date):
    filtered_data = explanation_data[
        (explanation_data["ticker"] == asset) & (explanation_data["Date"] == date)
    ]
    explanation_chart = {
        "data": [
            {
                "x": filtered_data["variable"],
                "y": filtered_data["score_contribution"],
                "type": "bar",
            },
        ],
        "layout": {"title": "LIME Explanation"},
    }

    targets = list(filtered_data["target"])
    preds = list(filtered_data["LGBM_Prediction"])

    target = f"Target: {targets[0] if targets else None}"
    pred = f"Prediction: {preds[0] if preds else None}"

    return explanation_chart, target, pred


if __name__ == "__main__":
    app.run_server(debug=True)

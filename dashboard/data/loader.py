import numpy as np
import pandas as pd
from pathlib import Path
import os

import yfinance as yf

from s3 import read_from_s3, list_s3_contents

ROOT_DIR = Path(__file__).parent.parent.parent


def get_returns_data():
    portfolio_returns = (
        read_from_s3(f"dashboard_data/final_model_returns.csv")
        .assign(Date=lambda data: pd.to_datetime(data["date"], format="%Y-%m-%d"))
        .sort_values(by="Date")
    )

    individual_returns = (
        read_from_s3(
            f"dashboard_data/individual_stock_returns.csv"
        )
        .assign(Date=lambda data: pd.to_datetime(data["date"], format="%Y-%m-%d"))
        .set_index("Date")
        .sort_values(by="Date")
    )

    holdings = (
        read_from_s3(
            f"dashboard_data/final_portfolio_holdings.csv"
        )
        .assign(Date=lambda data: pd.to_datetime(data["date"], format="%Y-%m-%d"))
        .set_index("Date")
        .sort_values(by="Date")
    )

    weights = (
        read_from_s3(
            f"dashboard_data/final_portfolio_weights.csv"
        )
        .assign(Date=lambda data: pd.to_datetime(data["date"], format="%Y-%m-%d"))
        .set_index("Date")
        .sort_values(by="Date")
    )

    portfolio_returns.drop("date", inplace=True, axis=1)
    individual_returns.drop("date", inplace=True, axis=1)
    holdings.drop("date", inplace=True, axis=1)
    weights.drop("date", inplace=True, axis=1)

    return_assets = ["Overall"] + list(individual_returns.columns)

    individual_returns = individual_returns * holdings * weights

    individual_cumsum_returns = (individual_returns + 1).cumprod()
    individual_cumlogsum_returns = np.log(individual_returns + 1).cumsum()

    individual_cumsum_returns.columns = [
        f"{c}_cumsum" for c in individual_cumsum_returns.columns
    ]
    individual_cumlogsum_returns.columns = [
        f"{c}_cumlogsum" for c in individual_cumlogsum_returns.columns
    ]

    individual_returns = individual_returns.reset_index()

    individual_returns = pd.merge(
        individual_returns, individual_cumsum_returns, on="Date"
    )
    individual_returns = pd.merge(
        individual_returns, individual_cumlogsum_returns, on="Date"
    )

    returns_data = pd.merge(portfolio_returns, individual_returns, on="Date")

    benchmark_data, benchmark_assets = get_benchmark_data(returns_data)
    returns_data = pd.merge(returns_data, benchmark_data, on="Date")

    return returns_data, return_assets, benchmark_assets


def get_explanation_data():
    lime_dir = f"dashboard_data/LIME"
    explanation_files = list_s3_contents(lime_dir)

    lime_data_dfs = []
    for explanation_file in explanation_files:
        lime_data = (
            read_from_s3(f"{lime_dir}/{explanation_file}")
            .assign(Date=lambda data: pd.to_datetime(data["date"], format="%Y-%m-%d"))
            .sort_values(by="Date")
        )
        lime_data.drop("date", inplace=True, axis=1)
        lime_data_dfs.append(lime_data)

    explanation_data = pd.concat(lime_data_dfs)
    explanation_assets = sorted(explanation_data.ticker.unique())

    return explanation_data, explanation_assets


def get_benchmark_data(returns_data):
    dates = returns_data["Date"]
    min_date = dates.min()
    max_date = dates.max()

    sp_data = yf.download("^GSPC", start=min_date, end=max_date)[["Adj Close"]]
    nasdaq_data = yf.download("^IXIC", start=min_date, end=max_date)[["Adj Close"]]
    dji_data = yf.download("^DJI", start=min_date, end=max_date)[["Adj Close"]]

    sp_returns = sp_data.pct_change().fillna(0)
    nasdaq_returns = nasdaq_data.pct_change().fillna(0)
    dji_returns = dji_data.pct_change().fillna(0)

    sp_returns["SP500"] = sp_returns["Adj Close"]
    nasdaq_returns["NASDAQ"] = nasdaq_returns["Adj Close"]
    dji_returns["DJI"] = dji_returns["Adj Close"]

    sp_returns.drop("Adj Close", inplace=True, axis=1)
    nasdaq_returns.drop("Adj Close", inplace=True, axis=1)
    dji_returns.drop("Adj Close", inplace=True, axis=1)

    benchmark_returns = pd.merge(sp_returns, nasdaq_returns, on="Date")
    benchmark_returns = pd.merge(benchmark_returns, dji_returns, on="Date")
    benchmark_assets = ["Overall"] + list(benchmark_returns.columns)

    benchmark_cumsum_returns = (benchmark_returns + 1).cumprod()
    benchmark_cumlogsum_returns = np.log(benchmark_returns + 1).cumsum()

    benchmark_cumsum_returns.columns = [
        f"{c}_cumsum" for c in benchmark_cumsum_returns.columns
    ]
    benchmark_cumlogsum_returns.columns = [
        f"{c}_cumlogsum" for c in benchmark_cumlogsum_returns.columns
    ]

    benchmark_returns = pd.merge(benchmark_returns, benchmark_cumsum_returns, on="Date")
    benchmark_returns = pd.merge(
        benchmark_returns, benchmark_cumlogsum_returns, on="Date"
    )

    benchmark_returns = benchmark_returns.fillna(0).replace([np.inf, -np.inf], 0)

    return benchmark_returns, benchmark_assets


def get_alpha_data(returns_data, return_assets, benchmark_assets):
    ret = returns_data[
        ["Date", "Returns"] + return_assets[1:] + benchmark_assets[1:]
    ].set_index("Date")

    sp_corr = ret.rolling(252).corr(ret["SP500"])
    nasdaq_corr = ret.rolling(252).corr(ret["NASDAQ"])
    dji_corr = ret.rolling(252).corr(ret["DJI"])
    vol = ret.rolling(252).std()

    sp_beta = (sp_corr * vol).divide(vol["SP500"], axis=0)
    nasdaq_beta = (nasdaq_corr * vol).divide(vol["NASDAQ"], axis=0)
    dji_beta = (dji_corr * vol).divide(vol["DJI"], axis=0)

    sp_resid = ret - sp_beta.multiply(ret["SP500"], 0)
    nasdaq_resid = ret - nasdaq_beta.multiply(ret["NASDAQ"], 0)
    dji_resid = ret - dji_beta.multiply(ret["DJI"], 0)

    sp_resid = sp_resid[["Returns"] + return_assets[1:]]
    nasdaq_resid = nasdaq_resid[["Returns"] + return_assets[1:]]
    dji_resid = dji_resid[["Returns"] + return_assets[1:]]

    cumulative_sp_resid = sp_resid[["Returns"] + return_assets[1:]].cumsum()
    cumulative_nasdaq_resid = nasdaq_resid[["Returns"] + return_assets[1:]].cumsum()
    cumulative_dji_resid = dji_resid[["Returns"] + return_assets[1:]].cumsum()

    sp_resid.columns = [f"{c}_SP500_alpha" for c in sp_resid.columns]
    nasdaq_resid.columns = [f"{c}_NASDAQ_alpha" for c in nasdaq_resid.columns]
    dji_resid.columns = [f"{c}_DJI_alpha" for c in dji_resid.columns]

    cumulative_sp_resid.columns = [
        f"{c}_SP500_cumulative_alpha" for c in cumulative_sp_resid.columns
    ]
    cumulative_nasdaq_resid.columns = [
        f"{c}_NASDAQ_cumulative_alpha" for c in cumulative_nasdaq_resid.columns
    ]
    cumulative_dji_resid.columns = [
        f"{c}_DJI_cumulative_alpha" for c in cumulative_dji_resid.columns
    ]

    alpha_data = (
        pd.concat(
            [
                sp_resid,
                nasdaq_resid,
                dji_resid,
                cumulative_sp_resid,
                cumulative_nasdaq_resid,
                cumulative_dji_resid,
            ],
            axis=1,
        )
        .dropna(axis=1, how="all")
        .reset_index()
    )

    return alpha_data

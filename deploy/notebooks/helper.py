import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay

import itertools
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from datetime import datetime, date, timedelta

from tqdm.notebook import tqdm
from numpy_ext import rolling_apply

from s3 import read_from_s3, write_to_s3, list_s3_contents, pickle_dump_to_s3

from tqdm.notebook import tqdm
from numpy_ext import rolling_apply

def triple_barrier(price_df, vertical_barrier = 21, factor = 0.7):
    
    """
    Triple barrier method to label returns.
    All barriers active
    - Upper barrier for positive returns
    - Verticle barrier to imply returns aren't significant enough (neutral)
    - Lower barrier for negative returns
    
    vertical_barrier: time in days until we classify as 0
    factor: Multiple of volatility to define upper and lower barrier. Higher factor means we consider only significant moves
    
    """
    
    def get_target(returns, barrier):
        barrier = barrier[0]
        # Does a cumulative sum. If any barrier is hit, return the respective label
        rtn = pd.Series(returns).cumsum()
        if np.isnan(barrier):
            return np.nan
        for r in rtn:
            if r > barrier:
                return 1
            elif r < -barrier:
                return -1

        return 0

    log_returns_df = np.log(price_df).diff().loc["2010-01-01":]

    volatility = log_returns_df.copy()
    # Obtain exponential moving average standard deviation
    volatility = volatility.ewm(span = vertical_barrier).std()
    # Scale to period, e.g * 260**0.5 to annualize
    volatility = volatility * (vertical_barrier**0.5)
    # Backfill accordingly in case too much data burned
    # We have 2009-01-01 data onwards
    barrier_thresholds = volatility.backfill(limit=vertical_barrier).loc["2010-01-01":] * factor
    barrier_thresholds[log_returns_df.isnull()] = np.nan

    triple_barrier_target_df = pd.DataFrame(index=barrier_thresholds.index)

    for ticker in tqdm(log_returns_df.columns):
        current_df = log_returns_df[[ticker]].merge(barrier_thresholds[[ticker]], on="Date")
        current_df.columns = ["Returns", "Barrier"]
        triple_barrier_target_df[ticker] = rolling_apply(get_target, vertical_barrier, current_df.Returns.values, current_df.Barrier.values)
        
    triple_barrier_target_df = triple_barrier_target_df.shift(-vertical_barrier)
    
    return triple_barrier_target_df

def get_start_end_dates(years):
    today = date.today()
    start_date = (today - timedelta(days = (365 * years) + 1))
    end_date = today
    return start_date, end_date

def get_price_data_yf(dates):
    price_df_yf = {}

    for date in dates:
        year = date.year
        month = date.month
        day = date.day

        path = f"market_data_yf/{year}/{month}/{day}/"
        contents = list_s3_contents(path)
        if contents is None:
            continue

        for content in contents:
            key = content["Key"]
            if "failed_tickers" in key:
                continue

            ticker = key.split('/')[-1].replace(".csv", "")
            curr_df = read_from_s3(key)

            if "Date" not in curr_df.columns:
                curr_df["Date"] = date.date()

            curr_df[ticker] = curr_df["Adj Close"]
            curr_df = curr_df.set_index("Date")[[ticker]]
            
            ticker_dfs = price_df_yf.get(ticker, [])
            ticker_dfs.append(curr_df)
            price_df_yf[ticker] = ticker_dfs
    
    price_df_yf_data = []
    for k, v in price_df_yf.items():
        ticker_df = pd.concat(v)
        price_df_yf_data.append(ticker_df)

    price_df_yf = pd.concat(price_df_yf_data, axis=1)
    price_df_yf = price_df_yf.fillna(0)
    price_df_yf.index = pd.to_datetime(price_df_yf.index)

    return price_df_yf

price_df_investingdotcom = []

def get_price_investingdotcom_data():
    price_df_investingdotcom = []

    contents = list_s3_contents("market_data_investingdotcom/")
    for content in contents:
        key = content["Key"]
        parts = key.split("/")

        if parts[1] == "":
            continue
        
        ticker = parts[1].replace(" Historical Data.csv", "")
        curr_df = read_from_s3(key)
        curr_df.Date = pd.to_datetime(curr_df.Date)
        curr_df[ticker] = curr_df["Price"]
        curr_df = curr_df.set_index("Date")[[ticker]]

        price_df_investingdotcom.append(curr_df)

    return pd.concat(price_df_investingdotcom, axis=1)

def read_alphas_nyt():
    nyt_sentiment_chatgpt_final = read_from_s3("nlp_processed/nyt_sentiment_chatgpt_final.csv")
    nyt_sentiment_chatgpt_final = nyt_sentiment_chatgpt_final.rename(columns={"pub_date":"date"}).drop(columns="main_headline")
    nyt_sentiment_chatgpt_final["date"] = pd.to_datetime(nyt_sentiment_chatgpt_final["date"])
    return nyt_sentiment_chatgpt_final

def read_alphas_analyst():
    analyst_ratings_processed = read_from_s3("nlp_processed/analyst_ratings_processed.csv")
    analyst_ratings_processed["date"] = analyst_ratings_processed["date"].apply(lambda x: x.split(" ")[0])
    analyst_ratings_processed["date"] = pd.to_datetime(analyst_ratings_processed["date"])
    return analyst_ratings_processed

def read_alphas_usnews():
    us_equities_news_dataset_processed = read_from_s3("nlp_processed/us_equities_news_dataset_processed.csv").drop(columns="Unnamed: 0")
    us_equities_news_dataset_processed["date"] = pd.to_datetime(us_equities_news_dataset_processed["date"])
    return us_equities_news_dataset_processed

def get_alpha_data_yf(dates, start_date, end_date):
    get_columns = ["Date", 'alpha001', 'alpha002', 'alpha003', 'alpha004',
       'alpha005', 'alpha006', 'alpha007', 'alpha008', 'alpha009', 'alpha010',
       'alpha011', 'alpha012', 'alpha013', 'alpha014', 'alpha015', 'alpha016',
       'alpha017', 'alpha018', 'alpha019', 'alpha020', 'alpha021', 'alpha022',
       'alpha023', 'alpha024', 'alpha025', 'alpha026', 'alpha027', 'alpha028',
       'alpha029', 'alpha030', 'alpha032', 'alpha033', 'alpha034', 'alpha035',
       'alpha036', 'alpha037', 'alpha038', 'alpha040', 'alpha041', 'alpha042',
       'alpha043', 'alpha044', 'alpha045', 'alpha046', 'alpha047', 'alpha049',
       'alpha050', 'alpha051', 'alpha052', 'alpha053', 'alpha054', 'alpha055',
       'alpha060', 'alpha061', 'alpha062', 'alpha064', 'alpha065', 'alpha068',
       'alpha074', 'alpha075', 'alpha078', 'alpha081', 'alpha083', 'alpha084',
       'alpha085', 'alpha086', 'alpha094', 'alpha095', 'alpha099', 'alpha101']

    alpha_df = {}

    for date in dates:
        year = date.year
        month = date.month
        day = date.day

        path = f"alpha_augmented_market_data_yf/{year}/{month}/{day}/"
        contents = list_s3_contents(path)
        if contents is None:
            continue

        print(f"Fetching {date}...")
        for content in contents:
            key = content["Key"]
            if "failed_tickers" in key:
                continue

            ticker = key.split('/')[-1].replace(".csv", "")
            curr_df = read_from_s3(key)

            if "Date" not in curr_df.columns:
                curr_df["Date"] = date.date()

            curr_df = curr_df[get_columns].rename(columns={"Date":"date"})
            curr_df["ticker"] = ticker
            curr_df["date"] = pd.to_datetime(curr_df["date"])  + BDay(1)
            curr_df = curr_df.set_index("date").loc[str(start_date):str(end_date)]
            curr_df.reset_index(inplace=True)
            
            ticker_dfs = alpha_df.get(ticker, [])
            ticker_dfs.append(curr_df)
            alpha_df[ticker] = ticker_dfs
    
    alpha_df_data = []
    for k, v in alpha_df.items():
        ticker_df = pd.concat(v)
        alpha_df_data.append(ticker_df)

    return pd.concat(alpha_df_data)

def get_macroeconomic_data_pmi(backtest_dates):
    PMI = read_from_s3("macroeconomic/PMI.xlsx", format="excel")
    PMI["date"] = PMI["Release Date"].apply(lambda x:x[:12])
    PMI["date"] = pd.to_datetime(PMI["date"], format="%b %d, %Y")
    PMI.sort_values("date", inplace=True)
    PMI["PMI_Surprise"] = PMI["Actual"] - PMI["Forecast"]
    PMI["PMI_Actual"] = PMI["Actual"]
    PMI["PMI_1Q_Forecast"] = PMI["Forecast"].shift(1)
    PMI["PMI_Surprise_Mom"] = PMI["PMI_Surprise"].rolling(3).sum()
    PMI["PMI_1Q_Forecast_Change"] = PMI["PMI_1Q_Forecast"].diff().rolling(3).sum()
    PMI = backtest_dates.merge(PMI, how="left")
    PMI = PMI.ffill().dropna()
    PMI["date"] = pd.to_datetime(PMI["date"]) + BDay(1)
    return PMI[["date", "PMI_Surprise_Mom", "PMI_1Q_Forecast_Change"]].copy()

def get_macroeconomic_data_real_gdp(backtest_dates):
    REAL_GDP = read_from_s3("macroeconomic/REAL_GDP.csv")
    REAL_GDP.DATE = pd.to_datetime(REAL_GDP.DATE, format="%Y-%m-%d")
    REAL_GDP.columns = ["date", "Real_GDP_Level"]
    REAL_GDP["GDP_Growth_2Q"] = REAL_GDP.Real_GDP_Level.pct_change().rolling(2).sum()
    REAL_GDP = backtest_dates.merge(REAL_GDP, how="left")
    REAL_GDP["date"] = pd.to_datetime(REAL_GDP["date"])  + BDay(63) # Three months lag
    REAL_GDP = REAL_GDP.ffill()
    REAL_GDP.GDP_Growth_2Q = REAL_GDP.GDP_Growth_2Q * 100
    return REAL_GDP[["date", "GDP_Growth_2Q"]].copy()

def get_macroeconomic_data_t10y_inflation_breakeven(backtest_dates):
    T10Y_Inflation_Breakeven = read_from_s3("macroeconomic/T10Y_Inflation_Breakeven.csv")
    T10Y_Inflation_Breakeven.DATE = pd.to_datetime(T10Y_Inflation_Breakeven.DATE, format="%Y-%m-%d")
    T10Y_Inflation_Breakeven.columns = ["date", "Inflation_Breakeven_Level"]
    T10Y_Inflation_Breakeven = backtest_dates.merge(T10Y_Inflation_Breakeven, how="left")
    T10Y_Inflation_Breakeven = T10Y_Inflation_Breakeven.bfill(limit=54)
    T10Y_Inflation_Breakeven.Inflation_Breakeven_Level = T10Y_Inflation_Breakeven.Inflation_Breakeven_Level.replace(".", np.nan)
    T10Y_Inflation_Breakeven.Inflation_Breakeven_Level = T10Y_Inflation_Breakeven.Inflation_Breakeven_Level.ffill()
    T10Y_Inflation_Breakeven.date = T10Y_Inflation_Breakeven.date  + BDay(1)
    T10Y_Inflation_Breakeven["Inflation_Breakeven_Level"] = T10Y_Inflation_Breakeven["Inflation_Breakeven_Level"].astype(float)
    T10Y_Inflation_Breakeven["Inflation_Breakeven_1M_Change"] = T10Y_Inflation_Breakeven["Inflation_Breakeven_Level"] - T10Y_Inflation_Breakeven["Inflation_Breakeven_Level"].shift(21)
    return T10Y_Inflation_Breakeven.dropna()

def get_macroeconomic_data_t10y_minus_2y(backtest_dates):
    T10Y_minus_2Y = read_from_s3("macroeconomic/T10Y_minus_2Y.csv")
    T10Y_minus_2Y.DATE = pd.to_datetime(T10Y_minus_2Y.DATE, format="%Y-%m-%d")
    T10Y_minus_2Y.columns = ["date", "T10Y_minus_2Y"]
    T10Y_minus_2Y.date = T10Y_minus_2Y.date  + BDay(1)
    T10Y_minus_2Y = backtest_dates.merge(T10Y_minus_2Y, how="left")
    T10Y_minus_2Y.T10Y_minus_2Y = T10Y_minus_2Y.T10Y_minus_2Y.replace(".", np.nan)
    T10Y_minus_2Y.T10Y_minus_2Y = T10Y_minus_2Y.T10Y_minus_2Y.ffill()
    T10Y_minus_2Y.T10Y_minus_2Y = T10Y_minus_2Y.T10Y_minus_2Y.astype(float)
    T10Y_minus_2Y["T10Y_minus_2Y_1M_Change"] = T10Y_minus_2Y.T10Y_minus_2Y - T10Y_minus_2Y.T10Y_minus_2Y.shift(21)
    return T10Y_minus_2Y.set_index("date")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
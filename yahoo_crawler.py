from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
import pandas as pd

yf.pdr_override()

# ticker_list = ['TSLA']
start_date = "2020-09-18"
end_date = "2021-03-17"

files = []


def get_delta(ticker):
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    close = data['Close']
    percent_change = round((close.iloc[-1] - close.iloc[0]) / data['Close'].iloc[0], 3) * 100
    return percent_change


# def save_data(df, filename):
#     df.to_csv('./data/' + filename + '.csv')
#
#     for ticker in ticker_list:
#         delta = get_delta(ticker)
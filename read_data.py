import csv
from yahoo_finance import Share
import datetime
import threading
import queue
from enum import Enum
import random
import numpy as np

START_DATE = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d')
END_DATE = datetime.datetime.strptime('2017-04-18', '%Y-%m-%d')
ONE_DAY = datetime.timedelta(days=1)

tickers_to_idx = {}
idx = 0

with open('nasdaq_tickers.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = True
    for row in reader:
        if not header:
            ticker = row[0]
            ticker = ticker.replace(" ", "")
            tickers_to_idx[ticker] = idx
            idx += 1
        else:
            header = False

num_tickers = len(tickers_to_idx)
days = (END_DATE - START_DATE).days
data_points = days + 1


raw_data = np.zeros((num_tickers, data_points, 5))
raw_dt = np.zeros((data_points))
for idx in range(data_points):
    dt = START_DATE + datetime.timedelta(days=idx)
    raw_dt[idx] = dt.timestamp()

with open('nasdaq_history.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    # idx = 0
    for row in reader:
        ticker = row[0]
        ticker = ticker.replace("%20", "")
        ticker_idx = tickers_to_idx[ticker]
        dt = datetime.datetime.strptime(row[1], '%Y-%m-%d')
        dt_idx = (dt - START_DATE).days
        o = float(row[2])
        c = float(row[3])
        h = float(row[4])
        l = float(row[5])
        v = float(row[6])
        raw_data[ticker_idx, dt_idx, 0] = o
        raw_data[ticker_idx, dt_idx, 1] = h
        raw_data[ticker_idx, dt_idx, 2] = l
        raw_data[ticker_idx, dt_idx, 3] = c
        raw_data[ticker_idx, dt_idx, 4] = v
        # idx += 1
        # if idx == 10:
        #     break

np.savez('nasdaq_raw_data.npz', raw_dt=raw_dt, raw_data=raw_data)

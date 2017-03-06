import numpy as np
import os, fnmatch
import datetime
import math
import csv
import shutil

from rl_fin.config import get_config
from rl_fin.data_reader import register_csv_dialect

# col_raw_open
# col_raw_close
# col_raw_high
# col_raw_low
# col_raw_volume
# col_raw_last_close

l = 1068481
px = 50.0
sin_amplitude = 1.0
std_factor = 0.001

def generate_flat():
    return np.ones((l, 6)) * px


def generate_trend():
    data = np.ones((l, 6))
    p = px
    for idx in range(l):
        data[idx, 5] = p
        p = p + 0.01
        data[idx, 0] = p
        data[idx, 1] = p
        data[idx, 2] = p
        data[idx, 3] = p
    return data


def generate_sin():
    data = np.ones((l, 6))
    p = px
    for idx in range(l):
        data[idx, 5] = p
        min_w = get_config().ww * get_config().bar_min
        i = idx % min_w
        p = px + sin_amplitude * math.sin(2. * math.pi * i / (min_w - 1))
        data[idx, 0] = p
        data[idx, 1] = p
        data[idx, 2] = p
        data[idx, 3] = p
    return data


def generate_random_walk():
    data = np.ones((l, 6))
    p = px
    for idx in range(l):
        data[idx, 5] = p
        r = np.random.normal(0, std_factor)
        p = p * math.exp(r)
        data[idx, 0] = p
        data[idx, 1] = p
        data[idx, 2] = p
        data[idx, 3] = p
    return data

def generate_random_sin():
    data = np.ones((l, 6))
    p = p_combined = px
    for idx in range(l):
        data[idx, 5] = p_combined

        min_w = get_config().ww * get_config().bar_min
        i = idx % min_w
        # easier to play when you link sin amplitude to current price, may be some EMA
        sin_amplitude = p * 0.1
        # if i == 0:
        #     sin_amplitude = p * 0.1
        r = np.random.normal(0, std_factor)
        p = p * math.exp(r)
        sin_component = sin_amplitude * math.sin(2. * math.pi * i / (min_w - 1))
        p_combined = p + sin_component
        data[idx, 0] = p_combined
        data[idx, 1] = p_combined
        data[idx, 2] = p_combined
        data[idx, 3] = p_combined
    return data

shutil.rmtree('./data/preprocessed/RSIN_30m', ignore_errors=True)

register_csv_dialect()
data = generate_random_sin()

with open('./data/RSIN_20141116_000000.csv', 'w') as f:
    writer = csv.writer(f, dialect='data')
    for idx in range(0, data.shape[0]):
        row = data[idx, :]
        writer.writerow(row)

from play import main as play_game

play_game()
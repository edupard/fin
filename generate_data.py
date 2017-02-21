import numpy as np
import os, fnmatch
import datetime
import math
import csv

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

def generate_flat():
    global l, px
    return np.ones((l, 6)) * px

def generate_trend():
    global l, px
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
    global l, px
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

register_csv_dialect()
data = generate_flat()

with open('data.csv', 'w') as f:
    writer = csv.writer(f, dialect='data')
    for idx in range(0, data.shape[0]):
        row = data[idx, :]
        writer.writerow(row)
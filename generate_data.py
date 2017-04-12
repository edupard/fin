import numpy as np
import os, fnmatch
import datetime
import math
import csv
import shutil

from config import get_config
from data_source.data_reader import get_paths
from config import get_config
from data_source.data_reader import register_csv_dialect

data_len = 2 * 365 * 24 * 60
start_px = 50.0

# rolling factor explained:
# if factor close to zero -> we don't tend to change sin amplitude -> more enthropy, hard to play
# if factor close to one -> we link sin amplitude to current px -> less enthropy, easy to play

# + 150K
# easy : plain sin
# expectation = 0.0
# volatility = 0.0
# sin_amplitude_pct = 0.1
# rolling_factor = 0.0

# + 150K
# easy : modern vol, sin amplitude correlated to current price
# expectation = 0.0
# volatility = 35.0
# sin_amplitude_pct = 0.1
# rolling_factor = 1.0

# +- 150K - flat, but sometimes switch to long or short
# easy: flat is solution if comission cost is non zero - modern volatility, no sin component
# expectation = 0.0
# volatility = 35.0
# sin_amplitude_pct = 0.0
# rolling_factor = 1.0

# - looks like it tend to long strategy, especially if vol is low or expectation is super high, but convergence is not stable and very slow
# hard (theoretically solution exists, but pl is not super stable) : big trend, modern vol, no sin component
# expectation = 0.0
# volatility = 35.0
# sin_amplitude_pct = 0.00
# rolling_factor = 1.0

# ~48
# + 600K
# easy : low vol, sin amplitude correlated to current price, but low
# expectation = 0.0
# volatility = 5.0
# sin_amplitude_pct = 0.01
# rolling_factor = 1.0

# ~234
# + 150K
# easy : low vol, sin amplitude correlated to current price, but low
# expectation = 0.0
# volatility = 35.0
# sin_amplitude_pct = 0.05
# rolling_factor = 1.0

# ~20
# + 2M reward ~50
# hard : modern vol, sin amplitude correlated to current price slowly and low
expectation = 0.0
volatility = 35.0
sin_amplitude_pct = 0.01
rolling_factor = 0.01

# log normal distribution parameters calculation
mu = 1.0 + expectation / 100.0
std = volatility / 100.0

mu_square = math.pow(mu, 2)
var = math.pow(std, 2)
ln_var = math.log(1 + var / mu_square)
ln_mu = math.log(mu) - ln_var / 2
ln_std = math.sqrt(ln_var)

ln_mu /= (data_len - 1)
ln_std /= math.sqrt(data_len - 1)


def generate_data():
    data = np.ones((data_len, 6))
    p = p_combined = start_px
    sin_amplitude = start_px * sin_amplitude_pct
    for idx in range(data_len):
        data[idx, 5] = p_combined
        sin_amplitude += (p * sin_amplitude_pct - sin_amplitude) * rolling_factor
        r = np.random.lognormal(ln_mu, ln_std)
        p *= r
        min_w = get_config().ww * get_config().bar_min
        i = idx % min_w
        sin_component = sin_amplitude * math.sin(2. * math.pi * i / (min_w - 1))
        p_combined = p + sin_component
        data[idx, 0] = p_combined
        data[idx, 1] = p_combined
        data[idx, 2] = p_combined
        data[idx, 3] = p_combined
    return data

DATA_FOLDER_PATH, DATA_FILE_PATH = get_paths()

print('Folder {} removed'.format(DATA_FOLDER_PATH) )
shutil.rmtree(DATA_FOLDER_PATH, ignore_errors=True)

print('Generating data...')
data = generate_data()

data_file_path = './data/{}_20141116_000000.csv'.format(get_config().ticker)
print('Writing data to {}...'.format(data_file_path))
register_csv_dialect()
with open(data_file_path, 'w', newline='') as f:
    writer = csv.writer(f, dialect='data')
    for idx in range(0, data.shape[0]):
        row = data[idx, :]
        writer.writerow(row)

from play import main as play_game
play_game()
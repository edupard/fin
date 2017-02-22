import math
from enum import Enum

class EnvType(Enum):
    PONG = 0
    FIN = 1


class Config(object):
    env_type = EnvType.PONG
    name = "pong"

    # dataset options
    train_set_size = 0.8
    cv_set_size = 0.1
    test_set_size = 1 - train_set_size - cv_set_size

    # fin environment
    # ticker = 'QO'
    ticker = 'TREND'
    # ticker = 'FLAT'
    # ticker = 'SIN'
    bar_min = 30
    episode_days = 90
    episode_length = episode_days * 24 * 60 // bar_min
    switch_off_zero_bars = True

    # rendering parameters
    # screen resolution
    window_px_width = 160
    window_px_height = 160
    # window width in bars
    ww = 100
    # bars per second
    bps = 24.
    # frames per second
    fps = 24.
    # bars per frame
    bpf = bps / fps
    # avoid vertical window collapsing
    min_px_window_height_pct = 0.01
    # window height as px std dev
    px_std_deviations = 3.0
    # exponential moving averages to smoth rendering
    rolling_px_factor = 0.2
    rolling_px_std_factor = 0.9
    # grid
    grid_on = False
    grid_px_delta = 0.20

    # RL parameters
    # slippage + commission
    costs = 0.03

    # NN & learning params
    # 1 to render, None to use all availiable cpu's, N to custom number
    num_workers = None
    conv_layers = 4
    state_buffer_size = 20
    max_grad_norm = 40.
    # discount rate for advantage estimation and reward discounting
    gamma = .99

_config = Config()

def get_config() -> Config:
    return _config
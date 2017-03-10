import math
from enum import Enum
from data_source.config import get_config as get_data_config


class RenderingBackend(Enum):
    SOFTWARE = 1
    HARDWARE = 0


class ThreadingModel(Enum):
    ST = 0
    MT = 1

class RewardType(Enum):
    RPL = 0
    URPL = 1
    TPL = 2

class RewardAlgo(Enum):
    PCT = 0
    CCY = 1


class Config(object):
    # rendering parameters
    rendering_backend = RenderingBackend.SOFTWARE
    threading_model = ThreadingModel.ST
    # screen resolution
    window_px_width = 160#42
    window_px_height = 160#42
    # episode length
    episode_length = round(3.8 * 3066)
    # episode_length = 24 * 60 * 365 * 1.4 //  get_data_config().bar_min
    # window width in bars
    ww = 4 * 100
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
    rolling_px_range_factor = 0.9

    # RL parameters
    reward_type = RewardType.URPL
    reward_algo = RewardAlgo.PCT
    # slippage + commission
    costs = 0.00

    rand_start = False
    start_seed = 0
    play_length = None # set it to some value >= episode length

    draw_training_line = False

_config = Config()

def get_config() -> Config:
    return _config
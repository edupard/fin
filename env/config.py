import math
from enum import Enum

class RenderingBackend(Enum):
    SOFTWARE = 1
    HARDWARE = 0


class Config(object):
    # rendering parameters
    rendering_backend = RenderingBackend.SOFTWARE
    # screen resolution
    window_px_width = 42
    window_px_height = 42
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
    rolling_px_range_factor = 0.9

    # RL parameters
    # slippage + commission
    costs = 0.03

    rand_start = True

_config = Config()

def get_config() -> Config:
    return _config
import math
from enum import Enum


class Config(object):
    yahoo = False
    # fin environment
    # Coca cola
    # yahoo = True
    # start = '2000-01-01'
    # end = '2009-12-31'
    # ticker = 'KO'
    # bar_min = 24 * 60
    # Brent
    ticker = 'QO'
    bar_min = 60
    # Experiments
    # yahoo = False
    # ticker = 'EXP'
    # bar_min = 30

    switch_off_zero_bars = True


_config = Config()


def get_config() -> Config:
    return _config

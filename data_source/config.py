import math
from enum import Enum

class Config(object):
    # fin environment
    ticker = 'QO'
    # ticker = 'EXP'
    bar_min = 4 * 60
    # bar_min = 30
    switch_off_zero_bars = True

_config = Config()

def get_config() -> Config:
    return _config
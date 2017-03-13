import numpy as np
from enum import Enum


class EnvironmentType(Enum):
    PONG = 1
    FIN = 0


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
    # env factory config
    environment = EnvironmentType.FIN
    model = 'qo_1h'
    log_dir = './models/' + model
    # log_dir = './models/exp'

    # Data config
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

    # Environment parameters
    rendering_backend = RenderingBackend.SOFTWARE
    threading_model = ThreadingModel.ST
    # screen resolution
    window_px_width = 160  # 42
    window_px_height = 160  # 42
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
    # Set to true if you want to draw line during training
    draw_training_line = False

    # Reward algo
    reward_type = RewardType.URPL
    reward_algo = RewardAlgo.PCT
    # slippage + commission
    costs = 0.00
    # NB: PCT reward do not converge due to floating point arithmetic precision
    # so we just scale reward to achieve convergence
    reward_scale_multiplier = 100.0

    # Episode parameters
    episode_length = 12000
    # episode_length = 24 * 60 * 365 * 1.4 //  get_data_config().bar_min
    rand_start = False
    start_seed = 0
    play_length = None  # set it to some value >= episode length

    # Learning parameters
    num_global_steps = 20e6
    algo_modification = True
    _lambda = 1.0
    gamma = 1.0
    buffer_length = 20
    fwd_buffer_length = 20 if algo_modification else 0
    learning_rate = 0.0001
    enthropy_weight = 0.01
    num_conv_layers = 6
    max_grad_norm = 40.0

    def __init__(self):
        self.b_gamma = np.zeros((self.fwd_buffer_length))
        self.b_gamma_lambda = np.zeros((self.fwd_buffer_length))
        acc_gamma = 1
        acc_gamma_lambda = 1
        for i in range(self.fwd_buffer_length):
            self.b_gamma[i] = acc_gamma
            self.b_gamma_lambda[i] = acc_gamma_lambda
            acc_gamma *= self.gamma
            acc_gamma_lambda *= self.gamma * self._lambda


_config = Config()

def get_config() -> Config:
    return _config
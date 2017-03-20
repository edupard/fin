import numpy as np
from enum import Enum
import configparser
import os
from distutils.util import strtobool


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


class RewardAlgo(Enum):
    PCT = 0
    CCY = 1
    LR = 2


class Config(object):
    # env factory config
    environment = EnvironmentType.FIN
    model = 'qo_1h'
    base_log_dir = os.path.join('./models/', model)

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
    costs_ccy = 0.03
    costs_on = False
    costs = 0.0 if not costs_on else costs_ccy
    # NB: PCT reward do not converge due to floating point arithmetic precision
    # so we just scale reward to converge
    reward_scale_multiplier = 100.0

    # Episode parameters
    # train_length = 24 * 60 * 7 // bar_min
    train_length = 10000
    train_episode_length = train_length // 13
    rand_start = True
    retrain_interval = 5000
    evaluation = False
    retrain_seed = 0

    # Learning parameters
    num_global_steps = 20e8
    algo_modification = True
    _lambda = 1.0
    gamma = 1.0
    buffer_length = 20
    fwd_buffer_length = 20 if algo_modification else 0
    learning_rate = 0.0001
    enthropy_weight = 0.01
    num_conv_layers = 6
    max_grad_norm = 40.0

    def get_model_path(self, retrain_seed, costs_on):
        return os.path.join(self.base_log_dir, str(retrain_seed), 'costs' if costs_on else 'no_costs')

    def __init__(self):
        # overwrite some values if nn.ini found
        config = configparser.ConfigParser()
        try:
            config.read('nn.ini')
            configSection = config['DEFAULT']

            self.evaluation = bool(strtobool(configSection['evaluation']))
            self.retrain_seed = int(configSection['retrain_seed'])
            self.costs_on = bool(strtobool(configSection['costs_on']))
            self.costs = 0.0 if not self.costs_on else self.costs_ccy
        except:
            pass
        self.log_dir = self.get_model_path(self.retrain_seed, self.costs_on)

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

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


class StateMode(Enum):
    ONE_D = 0
    TWO_D = 1


class Mode(Enum):
    TRAIN = 0
    CV = 1
    LOG = 2


def parse_mode(s_mode):
    s_m = s_mode.lower()
    if s_m == "train":
        return Mode.TRAIN
    elif s_m == "cv":
        return Mode.CV
    elif s_m == "log":
        return Mode.LOG


class Config(object):
    # env factory config
    environment = EnvironmentType.FIN
    # model = 'qo_5min'
    model = 'qo_1h'
    # model = 'qo_15min'
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
    bar_min = 5  # 60  # 15
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
    draw_training_line = True

    # Reward algo
    reward_type = RewardType.URPL
    reward_algo = RewardAlgo.PCT
    # slippage + commission
    costs_on = False
    costs_adv = True
    costs = 0.03
    # NB: PCT reward do not converge due to floating point arithmetic precision
    # so we just scale reward to converge
    reward_scale_multiplier = 100.0

    render = False
    # Episode parameters
    mode = Mode.TRAIN
    train_length = 12 * 6000  # 6000  # 12 * 6000  # 3000  # 6000 * 4
    train_episode_length = train_length
    retrain_interval = train_episode_length  # 2100  # train_episode_length
    train_seed = 0

    # Learning parameters
    num_global_steps = 20e8
    algo_modification = True
    _lambda = 0.96  # 1.0
    gamma = 0.98  # 0.95
    buffer_length = 100
    fwd_buffer_length = buffer_length if algo_modification else 0
    keep_prob = 0.5

    learning_rate = 0.0001
    enthropy_weight = 0.01
    state_mode = StateMode.ONE_D
    # conv_layers_2d = [(3, 2, 32), (3, 2, 32), (3, 2, 16), (3, 2, 16), (3, 2, 8), (3, 2, 4), (3, 2, 2)]
    # rnn_2d_size = 8
    # conv_layers_2d = [(3, 2, 16), (3, 2, 16), (3, 2, 8), (3, 2, 8), (3, 2, 4), (3, 2, 4), (3, 2, 2)]
    # conv_layers_2d = [(3, 2, 8), (3, 2, 8), (3, 2, 8), (3, 2, 8), (3, 2, 8), (3, 2, 8), (3, 3, 8)]
    # rnn_2d_size = 8
    # conv_layers_2d = [(3, 2, 32), (3, 2, 32), (3, 2, 32), (3, 2, 32), (3, 2, 32), (3, 2, 32), (3, 3, 32)]
    # rnn_2d_size = 32

    # conv_layers_2d = [(3, 8, 2, 4, 32), (3, 8, 2, 4, 16), (3, 6, 2, 3, 8), (3, 4, 2, 4, 4)]
    # rnn_2d_size = 32
    conv_layers_2d = [(3, 3, 2, 2, 32), (3, 3, 2, 2, 32), (3, 3, 2, 2, 32), (3, 3, 2, 2, 32), (3, 3, 2, 2, 32), (3, 3, 2, 2, 32), (3, 3, 3, 3, 32)]
    rnn_2d_size = 32


    # conv_layers_1d = [(3, 2, 5), (3, 2, 5), (3, 2, 5), (3, 2, 5), (3, 2, 5)]
    # conv_layers_1d = [(3, 2, 4), (3, 2, 4), (3, 2, 4), (3, 2, 4), (3, 2, 3), (3, 2, 2)]
    # conv_layers_1d = [(3, 2, 4), (3, 2, 4), (3, 2, 4), (3, 2, 4), (3, 2, 3), (3, 2, 2)]
    # fit perfectly
    # conv_layers_1d = [(3, 2, 200), (3, 2, 150), (3, 2, 150), (3, 2, 100), (3, 2, 100), (3, 2, 50)]
    # rnn_1d_size = 64  # 4  # 255
    # conv_layers_1d = [(3, 2, 32), (3, 2, 32), (3, 2, 32), (3, 2, 32)]
    # rnn_1d_size = 4
    # conv_layers_1d = [(3, 2, 4), (3, 2, 4)]
    # rnn_1d_size = 2
    # conv_layers_1d = [(3, 2, 64), (3, 2, 64), (3, 2, 64), (3, 2, 64), (3, 2, 64), (4, 4, 64)]
    # rnn_1d_size = 64
    # leha model - 1h 6000 bars
    conv_layers_1d = [(3, 2, 32), (3, 2, 16), (3, 2, 1)]
    rnn_1d_size = 13

    max_grad_norm = 40.0
    propogate_position_to_rnn = False

    def get_model_path(self, train_seed, costs, mode):
        model_dir = 'costs' if costs else 'no_costs'
        if mode == Mode.CV:
            model_dir = 'cv_costs' if costs else 'cv_no_costs'
        return os.path.join(self.base_log_dir, str(train_seed), model_dir)

    def is_evaluation(self):
        return self.mode == Mode.LOG or self.mode == Mode.CV

    def is_log_mode(self):
        return self.mode == Mode.LOG

    def is_cv_mode(self):
        return self.mode == Mode.CV

    def turn_on_costs(self):
        self.costs_on = True
        self.reset_log_dir()

    def set_mode(self, s_mode):
        self.mode = parse_mode(s_mode)
        self.reset_log_dir()

    def set_train_seed(self, train_seed):
        self.train_seed = train_seed
        self.reset_log_dir()

    def reset_log_dir(self):
        self.log_dir = self.get_model_path(self.train_seed, self.costs_on, self.mode)

    def turn_on_render(self):
        self.render = True

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
        self.reset_log_dir()


_config = Config()


def get_config() -> Config:
    return _config

import numpy as np

class Config(object):
    _lambda = 1.0
    gamma = 1.0 #0.85
    buffer_length = 20
    fwd_buffer_length = 20
    learning_rate = 0.0001
    enthropy_weight = 0.01#0.01
    num_conv_layers = 6

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
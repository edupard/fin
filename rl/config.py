class Config(object):
    _lambda = 1.0
    gamma = 0.99
    buffer_length = 100
    learning_rate = 0.00025
    enthropy_weight = 0.01

_config = Config()

def get_config() -> Config:
    return _config
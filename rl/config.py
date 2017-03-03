class Config(object):
    _lambda = 1.0
    gamma = 0.8
    buffer_length = 100
    learning_rate = 0.0003
    enthropy_weight = 0.01

_config = Config()

def get_config() -> Config:
    return _config
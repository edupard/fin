class Config(object):
    _lambda = 1.0
    gamma = 0.0
    buffer_length = 100
    learning_rate = 0.001
    enthropy_weight = 1.0

_config = Config()

def get_config() -> Config:
    return _config
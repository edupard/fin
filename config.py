from enum import Enum

class EnvironmentType(Enum):
    PONG = 1
    FIN = 0


class Config(object):
    environment = EnvironmentType.FIN
    log_dir = './models/sin'
    num_workers = 8

_config = Config()

def get_config() -> Config:
    return _config
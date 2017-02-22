import gym

from rl_fin.config import get_config, EnvType
from rl_fin.env import Environment, Mode

def create_env(dr):
    # Environment setup
    if get_config().env_type == EnvType.PONG:
        env = gym.make("Pong-v0")
    elif get_config().env_type == EnvType.FIN:
        env = Environment(dr, Mode.STATE_2D)
    return env
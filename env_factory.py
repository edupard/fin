import gym

from config import get_config, EnvironmentType

def startup():
    if get_config().environment == EnvironmentType.FIN:
        from data_source.data_source import get_datasource
        from env.env import get_ui_thread

        get_datasource()
        get_ui_thread().start()

def shutdown():
    if get_config().environment == EnvironmentType.FIN:
        from env.env import get_ui_thread

        get_ui_thread().stop()

def create_env():
    # Environment setup
    if get_config().environment == EnvironmentType.PONG:
        env = gym.make("Pong-v0")
    elif get_config().environment == EnvironmentType.FIN:
        from env.env import Environment

        env = Environment()
    return env

def stop_env(env):
    if get_config().environment == EnvironmentType.FIN:
        env.stop()
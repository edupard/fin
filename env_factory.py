import cv2
from gym.spaces.box import Box
import numpy as np
import gym
from gym import ObservationWrapper

from config import get_config, EnvironmentType

def _process_frame42(frame):
    frame = frame[34:34+160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame


class AtariRescale42x42(ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [42, 42, 1])

    def _observation(self, observation_n):
        return _process_frame42(observation_n)


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
        env = AtariRescale42x42(env)
    elif get_config().environment == EnvironmentType.FIN:
        from env.env import Environment

        env = Environment()
    return env

def stop_env(env):
    if get_config().environment == EnvironmentType.FIN:
        env.stop()
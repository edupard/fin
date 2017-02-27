from enum import Enum
import threading
import numpy as np
from PIL import Image
import queue
import pygame

from env.config import get_config
from env.env import DrawData, Line
from env.buttons import get_buttons


def key_callback(window, key, scan_code, action, mods):
    # TODO: implement
    return
    # if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
    #     get_buttons().on_press_esc()
    #
    # if key == glfw.KEY_UP and action == glfw.PRESS:
    #     get_buttons().on_press_up()
    # if key == glfw.KEY_UP and action == glfw.RELEASE:
    #     get_buttons().on_release_up()
    #
    # if key == glfw.KEY_DOWN and action == glfw.PRESS:
    #     get_buttons().on_press_down()
    # if key == glfw.KEY_DOWN and action == glfw.RELEASE:
    #     get_buttons().on_release_down()

class EnvInfo:

    def __init__(self, env, fbo):
        self._env = env
        self._fbo = fbo

    @property
    def env(self):
        return self._env

    @property
    def fbo(self):
        return self._fbo


class CommandType(Enum):
    START_ENV = 0
    DRAW = 1
    RENDER = 2
    STOP_ENV = 3
    STOP = 4
    GRAB_DATA = 5
    REGISTER_KEY_CALLBACK = 6


class Command:
    def __init__(self, command_type: CommandType, payload):
        self._command_type = command_type
        self._payload = payload

    @property
    def command_type(self) -> CommandType:
        return self._command_type

    @property
    def payload(self):
        return self._payload

def recalc_x(x: float) ->float:
    x * get_config().window_px_width

class UiThread:
    def __init__(self):
        thread_func = lambda: self.gui_thread()
        self._t = threading.Thread(target=(thread_func))
        self._started = False
        # Task queue
        self._q = queue.Queue()
        # Registered environments
        self._envs = {}

    def start_env(self, env):
        self._q.put(Command(CommandType.START_ENV, env))

    def stop_env(self, env):
        self._q.put(Command(CommandType.STOP_ENV, env))

    def register_key_callback(self, callback):
        self._q.put(Command(CommandType.REGISTER_KEY_CALLBACK, callback))

    def _on_register_key_callback(self, callback):
        return
        # TODO: implement
        glfw.set_key_callback(self._window, cbfun=callback)

    def gui_thread(self):
        # Initialize pygame
        # pygame.display.init()
        # pygame.display.set_mode((1, 1))

        while True:
            # pygame.event.poll()
            if self._process_queue():
                break

        # pygame.quit()

    def render(self, env):
        self._q.put(Command(CommandType.RENDER, env))

    def draw(self, dd: DrawData):
        self._q.put(Command(CommandType.DRAW, dd))

    def grab_data(self, env):
        self._q.put(Command(CommandType.GRAB_DATA, env))

    def start(self):
        if not self._started:
            self._started = True
            self._t.start()

        self.register_key_callback(key_callback)

    def stop(self):
        self._q.put(Command(CommandType.STOP, None))
        self._t.join()

    def _process_queue(self) -> bool:
        try:
            c = self._q.get(timeout=0.001)
            if c.command_type == CommandType.START_ENV:
                self._on_start_env(c.payload)
            elif c.command_type == CommandType.DRAW:
                self._on_draw(c.payload)
            elif c.command_type == CommandType.RENDER:
                self._on_render(c.payload)
            elif c.command_type == CommandType.GRAB_DATA:
                self._on_grab_data(c.payload)
            elif c.command_type == CommandType.STOP_ENV:
                self._on_stop_env(c.payload)
            elif c.command_type == CommandType.REGISTER_KEY_CALLBACK:
                self._on_register_key_callback(c.payload)
            elif c.command_type == CommandType.STOP:
                return True
        except queue.Empty:
            pass
        return False

    def _on_stop_env(self, env):
        self._envs.pop(env.id)

    def _on_start_env(self, env):
        # create framebuffer object to render into
        fbo = pygame.Surface((get_config().window_px_width, get_config().window_px_height), pygame.SRCALPHA, 32)

        env_info = EnvInfo(env, fbo=fbo)
        self._envs[env.id] = env_info

    def _on_render(self, env):
        env_info = self._envs[env.id]
        # TODO: implement - flip buffers

    def _on_draw(self, dd: DrawData):
        env_info = self._envs[dd.env.id]

        color = (0, 0, 0)
        env_info.fbo.fill(color)

        # draw prices
        quads = dd.quads
        count = quads.shape[0]

        for i in range(count):
            color = (153, 50, 204)

            x_l = quads[i][0]
            y_l = quads[i][1]
            x_r = quads[i][2]
            y_h = quads[i][3]

            w = x_r - x_l
            h = y_h - y_l

            x_l_px = x_l * get_config().window_px_width
            y_l_px = y_l * get_config().window_px_height
            w_px = w * get_config().window_px_width
            h_px = h * get_config().window_px_height

            pygame.draw.rect(env_info.fbo, color, (x_l_px, y_l_px, w_px, h_px), 0)

        # draw line
        if dd.line is not None:
            if dd.line.pl_positive:

                color = (50, 205, 50)
            else:
                color = (128, 0, 0)

            x0 = dd.line.x0 * get_config().window_px_width
            y0 = dd.line.y0 * get_config().window_px_height
            x1 = dd.line.x1 * get_config().window_px_width
            y1 = dd.line.y1 * get_config().window_px_height
            pygame.draw.line(env_info.fbo, color, (x0, y0), (x1, y1), 1)

    def _on_grab_data(self, env):
        env_info = self._envs[env.id]
        arr = pygame.surfarray.array3d(env_info.fbo)
        arr = np.transpose(arr, [1, 0, 2])
        arr = arr[::-1, :, :]
        # debug check
        # image = Image.fromarray(arr)
        # image.save('state.png')
        env.post_data(arr.reshape((-1,get_config().window_px_height, get_config().window_px_width, 3)))

_renderer = UiThread()


def get_ui_thread() -> UiThread:
    return _renderer
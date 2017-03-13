from enum import Enum
import threading
import numpy as np
from PIL import Image
import queue
import pygame

from config import get_config, ThreadingModel
from env.env import DrawData, Line
from env.buttons import get_buttons


def _process_event(event):
    if event.type == pygame.QUIT:
        get_buttons().on_press_esc()

    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        get_buttons().on_press_esc()

    if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
        get_buttons().on_press_up()
    if event.type == pygame.KEYUP and event.key == pygame.K_UP:
        get_buttons().on_release_up()

    if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
        get_buttons().on_press_down()
    if event.type == pygame.KEYUP and event.key == pygame.K_DOWN:
        get_buttons().on_release_down()


class EnvInfo:
    def __init__(self, env, fbo):
        self._env = env
        self._fbo = fbo
        self.dd = None

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


def recalc_x(x: float) -> float:
    x * get_config().window_px_width


def render_line_to_surface(surf, dd: DrawData):
    if dd.line is not None:
        if dd.line.pl_positive:
            color = (50, 205, 50)
        else:
            color = (128, 0, 0)

        x0 = dd.line.x0 * get_config().window_px_width
        y0 = get_config().window_px_height - dd.line.y0 * get_config().window_px_height
        x1 = dd.line.x1 * get_config().window_px_width
        y1 = get_config().window_px_height - dd.line.y1 * get_config().window_px_height
        pygame.draw.line(surf, color, (x0, y0), (x1, y1), 1)


tls = threading.local()


class UiThread:
    def __init__(self):
        thread_func = lambda: self.gui_thread()
        self._t = threading.Thread(target=(thread_func))
        self._started = False
        # Task queue
        self._q = queue.Queue()
        # Registered environments
        self._envs = {}
        # Flag indicating if window is hidden
        self._screen = None

    def start_env(self, env):
        if get_config().threading_model == ThreadingModel.ST:
            self._on_start_env(env)
        elif get_config().threading_model == ThreadingModel.MT:
            self._q.put(Command(CommandType.START_ENV, env))

    def stop_env(self, env):
        if get_config().threading_model == ThreadingModel.ST:
            self._on_stop_env(env)
        elif get_config().threading_model == ThreadingModel.MT:
            self._q.put(Command(CommandType.STOP_ENV, env))

    def gui_thread(self):
        while True:
            if self._screen is not None:
                event = pygame.event.poll()
                _process_event(event)
            if self._process_queue():
                break
        pygame.quit()

    def render(self, env):
        if get_config().threading_model == ThreadingModel.ST:
            env_info = self._get_env_info(env)
            surface = env_info.fbo.copy()
            self._q.put(Command(CommandType.RENDER, (surface, env_info.dd)))
        elif get_config().threading_model == ThreadingModel.MT:
            self._q.put(Command(CommandType.RENDER, env))

    def draw(self, dd: DrawData):
        if get_config().threading_model == ThreadingModel.ST:
            self._on_draw(dd)
        elif get_config().threading_model == ThreadingModel.MT:
            self._q.put(Command(CommandType.DRAW, dd))

    def grab_data(self, env):
        if get_config().threading_model == ThreadingModel.ST:
            self._on_grab_data(env)
        elif get_config().threading_model == ThreadingModel.MT:
            self._q.put(Command(CommandType.GRAB_DATA, env))

    def start(self):
        if not self._started:
            self._started = True
            self._t.start()

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
            elif c.command_type == CommandType.STOP:
                return True
        except queue.Empty:
            pass
        return False

    def _put_env_info(self, env_info):
        if get_config().threading_model == ThreadingModel.ST:
            tls.env_info = env_info
        elif get_config().threading_model == ThreadingModel.MT:
            self._envs[env_info.env.id] = env_info

    def _get_env_info(self, env):
        if get_config().threading_model == ThreadingModel.ST:
            return tls.env_info
        elif get_config().threading_model == ThreadingModel.MT:
            return self._envs[env.id]

    def _remove_env_info(self, env):
        if get_config().threading_model == ThreadingModel.ST:
            tls.env_info = None
        elif get_config().threading_model == ThreadingModel.MT:
            self._envs.pop(env.id)

    def _on_stop_env(self, env):
        self._remove_env_info(env)

    def _on_start_env(self, env):
        # create framebuffer object to render into
        fbo = pygame.Surface((get_config().window_px_width, get_config().window_px_height), pygame.SRCALPHA, 32)

        env_info = EnvInfo(env, fbo=fbo)
        self._put_env_info(env_info)

    def _on_render(self, payload):
        if self._screen is None:
            pygame.display.init()
            self._screen = pygame.display.set_mode((get_config().window_px_width, get_config().window_px_height))

        if get_config().threading_model == ThreadingModel.ST:
            surface, dd = payload
            self._screen.blit(surface, (0, 0))
        elif get_config().threading_model == ThreadingModel.MT:
            env = payload
            env_info = self._get_env_info(env)
            dd = env_info.dd
            self._screen.blit(env_info.fbo, (0, 0))
        if not get_config().draw_training_line:
            render_line_to_surface(self._screen, dd)

        pygame.display.flip()

    def _on_draw(self, dd: DrawData):
        env_info = self._get_env_info(dd.env)

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

            # pygame coordinate system origin is at the top-left corner
            x_l_px = x_l * get_config().window_px_width
            y_h_px = get_config().window_px_height - y_h * get_config().window_px_height
            w_px = w * get_config().window_px_width
            h_px = h * get_config().window_px_height

            pygame.draw.rect(env_info.fbo, color, (x_l_px, y_h_px, w_px, h_px), 0)

        # draw line
        if get_config().draw_training_line:
            render_line_to_surface(env_info.fbo, dd)

        env_info.dd = dd

    def _on_grab_data(self, env):
        env_info = self._get_env_info(env)
        arr = pygame.surfarray.array3d(env_info.fbo)
        arr = np.transpose(arr, [1, 0, 2])
        # arr = arr[::-1, :, :]
        # debug check
        # image = Image.fromarray(arr)
        # image.save('state.png')
        env.post_data(arr.reshape((-1, get_config().window_px_height, get_config().window_px_width, 3)))


_renderer = UiThread()


def get_ui_thread() -> UiThread:
    return _renderer

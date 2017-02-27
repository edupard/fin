from enum import Enum
import numpy as np
import math
import multiprocessing
import queue
import uuid
import time
from gym import spaces
from gym.spaces.box import Box
import cv2

from data_source.data_source import get_datasource
from env.config import get_config, RenderingBackend
from env.action import Action, convert_to_action

def _process_frame42(frame):
    frame = frame.reshape((42,42,3))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame

class Line:
    def __init__(self, pl_positive: bool, x0: float, y0: float, x1: float, y1: float):
        self._pl_positive = pl_positive
        self._x0 = x0
        self._y0 = y0
        self._x1 = x1
        self._y1 = y1

    @property
    def pl_positive(self) -> bool:
        return self._pl_positive

    @property
    def x0(self) -> float:
        return self._x0

    @property
    def y0(self) -> float:
        return self._y0

    @property
    def x1(self) -> float:
        return self._x1

    @property
    def y1(self) -> float:
        return self._y1

class DrawData:
    def __init__(self, env, quads, line: Line):
        self._env = env
        self._quads = quads
        self._line = line

    @property
    def env(self):
        return self._env

    @property
    def quads(self):
        return self._quads

    @property
    def line(self) -> Line:
        return self._line

if get_config().rendering_backend == RenderingBackend.HARDWARE:
    from env.hw_ui_thread import get_ui_thread as get_backend_ui_thread
    ui_thread = get_backend_ui_thread()
elif get_config().rendering_backend == RenderingBackend.SOFTWARE:
    from env.sw_ui_thread import get_ui_thread as get_backend_ui_thread
    ui_thread = get_backend_ui_thread()

def get_ui_thread():
    return ui_thread

class State(Enum):
    LONG = 1
    FLAT = 0
    SHORT = -1

class StateItem:
    def __init__(self, d: bool, s, r: float):
        self._d = d
        self._s = s
        self._r = r

    @property
    def done(self) -> bool:
        return self._d

    @property
    def state(self):
        return self._s

    @property
    def reward(self):
        return self._r


class Environment:
    def __init__(self):
        self._id = uuid.uuid1()

        self._dd = None

        self._data = get_datasource()
        self._data_length = self._data.shape[0]
        self._ep_start_idx = None
        self._pos_array = None

        self._ent_px = None
        self._ent_time = None
        self._state = State.FLAT
        self._pl_positive = False

        self._data_queue = queue.Queue()

        get_ui_thread().start_env(self)

        self._action_space = spaces.Discrete(3)
        self._observation_space = Box(0.0, 1.0, [42, 42, 1])

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def post_data(self, arr):
        self._data_queue.put(arr)

    @property
    def id(self) -> uuid.UUID:
        return self._id

    def render(self):
        while True:
            tp = time.perf_counter() - self._perf_counter_start
            frames_rendered = self._f
            frames_to_render_at_moment = tp * get_config().fps
            if frames_to_render_at_moment >= frames_rendered:
                break
            time.sleep(0.001)
        get_ui_thread().render(self)

    def _draw(self):
        self._f += 1

        t_max = self._current_time
        t_min = t_max - get_config().ww
        last_data_idx = math.floor(t_max)
        if last_data_idx > self._data_length - 1:
            last_data_idx = self._data_length - 1

        last_px = self._data[last_data_idx][1]

        # http://stackoverflow.com/questions/488670/calculate-exponential-moving-average-in-python
        self._px_rolling_mean += (last_px - self._px_rolling_mean) * get_config().rolling_px_factor

        w_slice = self._data[last_data_idx - get_config().ww:last_data_idx + 1, 1]
        std = np.std(w_slice)
        self._px_range_rolling_mean += (std - self._px_range_rolling_mean) * get_config().rolling_px_range_factor
        # alternative algo
        # min_px = np.min(w_slice)
        # max_px = np.max(w_slice)
        # px_h = max_px - min_px
        # self._px_range_rolling_mean += (px_h - self._px_range_rolling_mean) * get_config().rolling_px_range_factor

        wh = max(self._px_range_rolling_mean, self._px_rolling_mean * get_config().min_px_window_height_pct)

        px_min = self._px_rolling_mean - get_config().px_std_deviations * wh
        px_max = self._px_rolling_mean + get_config().px_std_deviations * wh

        def calc_scaled_x(t: float) -> float:
            return (t - t_min) / (t_max - t_min)

        def calc_scaled_y(px: float) -> float:
            return (px - px_min) / (px_max - px_min)

        quads = np.zeros((get_config().ww + 1, 4), dtype=np.float)
        # First point
        x_r = calc_scaled_x(t_max)

        for i in range(get_config().ww + 1):
            data_idx = last_data_idx - i
            px = self._data[data_idx][1]
            t = float(data_idx)
            x_l = calc_scaled_x(t)
            y_h = calc_scaled_y(px)
            y_l = 0.0

            quads[i][0] = x_l
            quads[i][1] = y_l
            quads[i][2] = x_r
            quads[i][3] = y_h

            x_r = x_l

        line = None
        if self._ent_time is not None:
            line = Line(self._pl_positive, calc_scaled_x(self._ent_time), calc_scaled_y(self._ent_px),
                        calc_scaled_x(self._current_time), calc_scaled_y(last_px))

        self._dd = DrawData(self, quads, line)

        get_ui_thread().draw(self._dd)

    def reset(self):
        self._ep_start_idx = get_config().ww
        if get_config().rand_start:
            self._ep_start_idx = np.random.randint(get_config().ww,
                                                   high=self._data_length - 1)
        return self._setup()

    def _setup(self):
        self._dd = None
        self._f = 0
        self._perf_counter_start = time.perf_counter()

        self._start_time = float(self._ep_start_idx)
        self._current_time = self._start_time

        s_idx = self._ep_start_idx

        self._px_rolling_mean = self._data[s_idx][1]

        w_slice = self._data[s_idx - get_config().ww:s_idx + 1, 1]
        self._px_range_rolling_mean = np.std(w_slice)
        # alternative algo
        # min_px = np.min(w_slice)
        # max_px = np.max(w_slice)
        # px_h = max_px - min_px
        # self._px_range_rolling_mean = px_h

        self._state = State.FLAT
        self._ent_time = None
        self._ent_px = None

        self._draw()
        return self._get_state()

    def rollback(self):
        return self._setup()

    def _get_state(self):
        get_ui_thread().grab_data(self)
        arr = self._data_queue.get()

        return _process_frame42(arr)

    def step(self, action: int):
        d = False
        r = 0.0
        self._current_time += get_config().bpf
        last_data_idx = math.floor(self._current_time)
        if last_data_idx >= self._data_length - 1:
            last_data_idx = self._data_length - 1
            d = True

        a = convert_to_action(action)

        last_px = self._data[last_data_idx][1]

        # handle reward
        if self._state == State.LONG and (a == Action.FLAT or a == Action.SELL):
            ex_px = last_px
            pl = ex_px - self._ent_px - 2 * get_config().costs
            r = pl / self._ent_px
        elif self._state == State.SHORT and (a == Action.FLAT or a == Action.BUY):
            ex_px = last_px + get_config().costs
            pl = self._ent_px - ex_px - 2 * get_config().costs
            r = pl / self._ent_px
        # handle position
        if a == Action.BUY and self._state != State.LONG:
            self._ent_px = last_px
            self._ent_time = self._current_time
            self._state = State.LONG
        elif a == Action.SELL and self._state != State.SHORT:
            self._ent_px = last_px
            self._ent_time = self._current_time
            self._state = State.SHORT
        elif a == Action.FLAT:
            self._ent_px = None
            self._ent_time = None
            self._state = State.FLAT
            self._pl_positive = False

        if self._state == State.LONG:
            self._pl_positive = (last_px > self._ent_px)
        elif self._state == State.SHORT:
            self._pl_positive = (last_px < self._ent_px)

        self._draw()
        if d:
            if self._state == State.LONG:
                ex_px = last_px
                pl = ex_px - self._ent_px - 2 * get_config().costs
                r = pl / self._ent_px
            elif self._state == State.SHORT:
                ex_px = last_px + get_config().costs
                pl = self._ent_px - ex_px - 2 * get_config().costs
                r = pl / self._ent_px

        return self._get_state(), r, d, None

    def stop(self):
        get_ui_thread().stop_env(self)

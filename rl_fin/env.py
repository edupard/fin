from enum import Enum
import numpy as np
import math
import multiprocessing
import queue
import uuid
import time
from gym import spaces

from rl_fin.data_reader import DataReader
from rl_fin.config import get_config
from rl_fin.ui_thread import get_ui_thread, DrawData, Line


class Mode(Enum):
    STATE_1D = 0
    STATE_2D = 1


class State(Enum):
    LONG = 1
    FLAT = 0
    SHORT = -1


class Action(Enum):
    BUY = 1
    FLAT = 0
    SELL = -1


def convert_to_action(a: int) -> Action:
    return {
        0: Action.FLAT,
        1: Action.BUY,
        2: Action.SELL
    }[a]


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
    def __init__(self, dr: DataReader, mode: Mode):
        self._id = uuid.uuid1()

        self._mode = mode

        self._dd = None

        self._buf_1d = None

        self._dr = dr
        self._ep_start_idx = None
        self._data = None
        self._pos_array = None

        self._ent_px = None
        self._ent_time = None
        self._state = State.FLAT
        self._pl_positive = False

        self._data_queue = queue.Queue()

        get_ui_thread().start_env(self)

        self._action_space = spaces.Discrete(3)

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
        if self._mode == Mode.STATE_1D:
            get_ui_thread().draw(self._dd)
            get_ui_thread().render(self)
        else:
            get_ui_thread().render(self)
        # get_renderer().render(self)

    def _draw(self):
        self._f += 1

        t_max = self._current_time
        t_min = t_max - get_config().ww
        last_data_idx = math.floor(t_max)

        last_px = self._data[last_data_idx][0]

        # http://stackoverflow.com/questions/488670/calculate-exponential-moving-average-in-python
        self._px_rolling_mean += (last_px - self._px_rolling_mean) * get_config().rolling_px_factor

        # w_slice = self._data[last_data_idx - get_config().bars_window_width:last_data_idx + 1, 0]
        # min_px = np.min(w_slice)
        # max_px = np.max(w_slice)
        # px_h = max_px - min_px
        # px_h = max(px_h, self._px_rolling_mean * get_config().min_px_range_pct)
        # self._px_range_rolling_mean += (px_h - self._px_range_rolling_mean) * get_config().rolling_px_range_factor
        # px_min = self._px_rolling_mean + self._px_range_rolling_mean;
        # px_max = self._px_rolling_mean - self._px_range_rolling_mean;

        # px_min = self._px_rolling_mean * (1 - get_config().half_price_window_height_pct)
        # px_max = self._px_rolling_mean * (1 + get_config().half_price_window_height_pct)

        w_slice = self._data[last_data_idx - get_config().ww:last_data_idx + 1, 0]
        std = np.std(w_slice)

        self._rolling_px_std += (std - self._rolling_px_std) * get_config().rolling_px_std_factor

        wh = max(self._rolling_px_std, self._px_rolling_mean * get_config().min_px_window_height_pct)

        px_min = self._px_rolling_mean - get_config().px_std_deviations * wh
        px_max = self._px_rolling_mean + get_config().px_std_deviations * wh

        def calc_scaled_x(t: float) -> float:
            return (t - t_min) / (t_max - t_min)

        def calc_scaled_y(px: float) -> float:
            return (px - px_min) / (px_max - px_min)

        grid_lines = None
        if get_config().grid_on:
            px_v = 0.0
            lines_count = math.floor((px_max - px_min) // get_config().grid_px_delta + 1)
            grid_lines = np.zeros((lines_count), dtype=np.float)
            idx = 0
            while True:
                if px_v > px_max:
                    break
                if px_v >= px_min:
                    grid_lines[idx] = calc_scaled_y(px_v)
                    idx += 1
                px_v += get_config().grid_px_delta

            grid_lines = grid_lines[:idx]

        if self._mode == Mode.STATE_1D:
            w = get_config().window_px_width
            self._buf_1d = np.zeros((1, w, 1, 3), dtype=float)
            buf_blue = self._buf_1d[:, :, :, 0].reshape((w))
            buf_green = self._buf_1d[:, :, :, 1].reshape((w))
            buf_red = self._buf_1d[:, :, :, 2].reshape((w))
            buf_pl = buf_red
            if self._pl_positive:
                buf_pl = buf_green
            for idx in range(w):
                # find time t on [t_min;t_max] interval which correspond to pixel idx in resulting 1d image vector
                t = t_min + (t_max - t_min) * float(idx) / float(w - 1)
                # find corresponding price
                data_idx = math.floor(t)
                px = self._data[data_idx][0]
                # project price as in 2d case
                y = calc_scaled_y(px)
                # write px to image vector: truncation possible here
                buf_blue[idx] = y
                # if agent in position and t >= [position enter time]
                # fill corresponding pl channel
                if self._ent_time is not None and t >= self._ent_time:
                    # check if t_max > [position enter time] to avoid division by zero
                    # and find price on the line
                    if t_max > self._ent_time:
                        px = last_px + (self._ent_px - last_px) * (t_max - t) / (t_max - self._ent_time)
                    else:
                        px = last_px
                    # project price as in 2d case
                    y = calc_scaled_y(px)
                    # write px to pl image vector: truncation possible here
                    buf_pl[idx] = y

        quads = np.zeros((get_config().ww + 1, 4, 2), dtype=np.float)
        # First point
        x_r = calc_scaled_x(t_max)

        for i in range(get_config().ww + 1):
            data_idx = last_data_idx - i
            idx = self._data[data_idx][0]
            t = float(data_idx)
            x_l = calc_scaled_x(t)
            y = calc_scaled_y(idx)

            quads[i][0][0] = x_l
            quads[i][0][1] = y

            quads[i][1][0] = x_r
            quads[i][1][1] = y

            quads[i][2][0] = x_r
            quads[i][2][1] = 0

            quads[i][3][0] = x_l
            quads[i][3][1] = 0

            x_r = x_l

        line = None
        if self._ent_time is not None:
            line = Line(self._pl_positive, calc_scaled_x(self._ent_time), calc_scaled_y(self._ent_px), calc_scaled_x(self._current_time), calc_scaled_y(last_px))

        self._dd = DrawData(self, quads, line, grid_lines)
        if self._mode == Mode.STATE_2D:
            get_ui_thread().draw(self._dd)
        # get_renderer().draw(self._dd)

    def reset(self):
        self._data = self._dr.train_data.reshape((-1, 2))

        data_length = self._data.shape[0]

        if get_config().ww > data_length - get_config().episode_length:
            raise "game length is too long"

        self._ep_start_idx = np.random.randint(get_config().ww,
                                               high=data_length - get_config().episode_length + 1)
        # self._ep_start_idx = 1000
        self._ep_end_idx = self._ep_start_idx + get_config().episode_length

        return self._setup()

    def _setup(self):
        self._dd = None
        self._f = 0
        self._perf_counter_start = time.perf_counter()

        self._start_time = float(self._ep_start_idx)
        self._end_time = float(self._ep_end_idx)

        self._current_time = self._start_time

        s_idx = self._ep_start_idx

        self._px_rolling_mean = self._data[s_idx][0]
        # surprisingly it works worst than fixed % of current price
        # w_slice = self._data[s_idx - get_config().bars_window_width:s_idx + 1, 0]
        # min_px = np.min(w_slice)
        # max_px = np.max(w_slice)
        # px_h = max_px - min_px
        # px_h = max(px_h, self._px_rolling_mean * get_config().min_px_range_pct)
        # self._px_range_rolling_mean = px_h

        w_slice = self._data[s_idx - get_config().ww:s_idx + 1, 0]
        self._rolling_px_std = np.std(w_slice)

        self._state = State.FLAT
        self._ent_time = None
        self._ent_px = None

        self._draw()
        return self._get_state()

    def rollback(self):
        return self._setup()


    def _get_state(self):
        if self._mode == Mode.STATE_2D:
            get_ui_thread().grab_data(self)
            arr = self._data_queue.get()
            return arr
        elif self._mode == Mode.STATE_1D:
            return self._buf_1d

    def step(self, action: int):
        d = False
        r = 0.0
        self._current_time += get_config().bpf
        if self._current_time > self._end_time:
            self._current_time = self._end_time
            d = True

        a = convert_to_action(action)
        last_data_idx = math.floor(self._current_time)
        last_px = self._data[last_data_idx][0]

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

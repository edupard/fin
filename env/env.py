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
from PIL import Image
import copy

from data_source.data_source import get_datasource
from config import get_config, RenderingBackend, RewardType, RewardAlgo, StateMode
from env.action import Action, convert_to_action


# counter = 0


def _process_frame(frame):
    frame = frame.reshape((get_config().window_px_width, get_config().window_px_height, 3))

    # global counter
    # counter += 1
    # # save rgb image
    # if counter == 50:
    #     image = Image.fromarray(frame.reshape([get_config().window_px_width, get_config().window_px_height, 3]))
    #     image.save('rgb.png')

    frame = frame.mean(2)
    # if you want human to percieve this image use weighted average instead
    # frame = np.average(frame, axis=2, weights=[0.299,0.587,0.114])

    # # save gray image
    # if counter == 50:
    #     gray = frame.astype(np.uint8)
    #     gray = gray.reshape([get_config().window_px_width, get_config().window_px_height])
    #     image = Image.fromarray(gray, mode='L')
    #     image.save('gray.png')

    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [get_config().window_px_width, get_config().window_px_height, 1])
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


class Info:
    def __init__(self):
        self.long = 0
        self.short = 0
        self.long_length = 0
        self.short_length = 0

        self.price = None
        self.time = None

        self.next_price = None
        self.next_time = None

        self.ccy = 0.0
        self.ccy_cost = 0.0
        self.ccy_c = 0.0
        self.ccy_c_cost = 0.0
        self.pct = 0.0
        self.pct_cost = 0.0
        self.pct_c = 0.0
        self.pct_c_cost = 0.0
        self.lr = 0.0
        self.lr_cost = 0.0
        self.lr_c = 0.0
        self.lr_c_cost = 0.0

        self.state = State.FLAT

        self.length = 0
        self.r = 0.0
        self.rwc = 0.0

    def subtract(self, info):
        ret = copy.deepcopy(self)

        ret.long = ret.long - info.long
        ret.short = ret.short - info.short
        ret.long_length = ret.long_length - info.long_length
        ret.short_length = ret.short_length - info.short_length
        ret.length = ret.length - info.length
        ret.r = ret.r - info.r
        ret.rwc = ret.rwc - info.rwc

        return ret


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
        self._initialized = False

        self._action_space = spaces.Discrete(3)
        if get_config().state_mode == StateMode.TWO_D:
            self._observation_space = Box(0.0, 1.0, [get_config().window_px_width, get_config().window_px_height, 1])
        elif get_config().state_mode == StateMode.ONE_D:
            # Lower and upper bound is incorrect but it doesnt matter
            # o, c, h, l, v, h_ma,l_ma,h_u_ma,h_l_ma,l_u_ma,l_l_ma
            self._observation_space = Box(0.0, 1.0, [get_config().ww + 1, 1, 11])

        self._info = Info()

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

        wh = max(self._px_range_rolling_mean, self._px_rolling_mean * get_config().min_px_window_height_pct)

        px_min = self._px_rolling_mean - get_config().px_std_deviations * wh
        px_max = self._px_rolling_mean + get_config().px_std_deviations * wh

        def calc_scaled_x(t: float) -> float:
            return (t - t_min) / (t_max - t_min)

        def calc_scaled_y(px: float) -> float:
            return (px - px_min) / (px_max - px_min)

        if get_config().state_mode == StateMode.ONE_D:
            self.one_d_state = np.zeros((get_config().ww + 1, 1, 11), dtype=np.float)
            for i in range(get_config().ww + 1):
                data_idx = last_data_idx - i
                # px = self._data[data_idx][1]
                # y = calc_scaled_y(px)
                # self.one_d_state[get_config().ww - i][0][0] = y

                self.one_d_state[get_config().ww - i][0][0] = calc_scaled_y(self._data[data_idx][3])
                self.one_d_state[get_config().ww - i][0][1] = calc_scaled_y(self._data[data_idx][4])
                self.one_d_state[get_config().ww - i][0][2] = calc_scaled_y(self._data[data_idx][5])
                self.one_d_state[get_config().ww - i][0][3] = calc_scaled_y(self._data[data_idx][6])
                self.one_d_state[get_config().ww - i][0][4] = calc_scaled_y(self._data[data_idx][2])
                self.one_d_state[get_config().ww - i][0][5] = calc_scaled_y(self._data[data_idx][7])
                self.one_d_state[get_config().ww - i][0][6] = calc_scaled_y(self._data[data_idx][8])
                self.one_d_state[get_config().ww - i][0][7] = calc_scaled_y(
                    self._data[data_idx][7] + 2 * self._data[data_idx][9])
                self.one_d_state[get_config().ww - i][0][8] = calc_scaled_y(
                    self._data[data_idx][7] - 2 * self._data[data_idx][9])
                self.one_d_state[get_config().ww - i][0][9] = calc_scaled_y(
                    self._data[data_idx][8] + 2 * self._data[data_idx][10])
                self.one_d_state[get_config().ww - i][0][10] = calc_scaled_y(
                    self._data[data_idx][8] - 2 * self._data[data_idx][10])

        if get_config().state_mode == StateMode.TWO_D or get_config().render:
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
        if not self._initialized:
            get_ui_thread().start_env(self)
            self._initialized = True

        start = get_config().ww + get_config().retrain_interval * get_config().train_seed
        # For train loop we must lie inside data
        if not get_config().is_test_mode() and start + get_config().train_episode_length > self._data_length:
            raise "train interval lie outside of availiable data"

        self._ep_start_idx = np.random.randint(start,
                                               high=start + get_config().train_length - get_config().train_episode_length + 1)
        self._ep_end_idx = self._ep_start_idx + get_config().train_episode_length
        # overwrite start and end for test mode
        if get_config().is_test_mode():
            self._ep_start_idx = start
            self._ep_end_idx = min(start + get_config().train_episode_length + get_config().retrain_interval,
                                   self._data_length - 1)

        self._info = Info()
        self._dd = None
        self._prev_px = None
        self._f = 0
        self._perf_counter_start = time.perf_counter()

        self._start_time = float(self._ep_start_idx)
        self._end_time = float(self._ep_end_idx)
        self._current_time = self._start_time

        s_idx = self._ep_start_idx

        self._px_rolling_mean = self._data[s_idx][1]

        w_slice = self._data[s_idx - get_config().ww:s_idx + 1, 1]
        self._px_range_rolling_mean = np.std(w_slice)

        self._state = State.FLAT
        self._ent_time = None
        self._ent_px = None

        self._draw()
        return self._get_state()

    def _get_state(self):
        if get_config().state_mode == StateMode.TWO_D or get_config().render:
            get_ui_thread().grab_data(self)
            arr = self._data_queue.get()
            if get_config().state_mode == StateMode.TWO_D:
                return _process_frame(arr)
        if get_config().state_mode == StateMode.ONE_D:
            return self.one_d_state

    def _fill_info(self, data_idx, next_data_idx):
        self._info.price = self._data[data_idx][1]
        self._info.time = self._data[data_idx][0]
        self._info.next_price = self._data[next_data_idx][1]
        self._info.next_time = self._data[next_data_idx][0]

    def step(self, action: int):
        d = False

        data_idx = math.floor(self._current_time)

        next_time = self._current_time + get_config().bpf
        if next_time >= self._end_time:
            next_time = self._end_time
            d = True
        next_data_idx = math.floor(next_time)
        # Fill info prices and time
        self._fill_info(data_idx, next_data_idx)

        action = convert_to_action(action)

        px = self._data[data_idx][1]
        next_px = self._data[next_data_idx][1]

        # Calculate reward
        ccy = 0.0
        ccy_cost = 0.0
        ccy_c = 0.0
        ccy_c_cost = 0.0
        pct = 0.0
        pct_cost = 0.0
        pct_c = 0.0
        pct_c_cost = 0.0
        lr = 0.0
        lr_cost = 0.0
        lr_c = 0.0
        lr_c_cost = 0.0

        # zero cost - to make code more readable
        zc = 0.0
        c = get_config().costs

        # Check if we liquidate position
        if (self._state == State.LONG and action != Action.BUY) or (
                        self._state == State.SHORT and action != Action.SELL):
            ccy, pct, lr = Environment.calc_reward(ccy, pct, lr, self._state, self._ent_px, zc, px, zc)
            ccy_cost, pct_cost, lr_cost = Environment.calc_reward(ccy_cost, pct_cost, lr_cost, self._state,
                                                                  self._ent_px, c, px, c)

            ccy_c, pct_c, lr_c = Environment.calc_reward(ccy_c, pct_c, lr_c, self._state, self._prev_px, 0,
                                                         px, zc)
            ccy_c_cost, pct_c_cost, lr_c_cost = Environment.calc_reward(ccy_c_cost, pct_c_cost, lr_c_cost, self._state,
                                                                        self._prev_px, 0,
                                                                        px, c)
            self._state = State.FLAT
            self._ent_px = None
            self._ent_time = None
        # Check if we still in position
        if self._state != State.FLAT:
            if self._state == State.LONG:
                self._info.long_length += 1
            elif self._state == State.SHORT:
                self._info.short_length += 1
            ccy_c, pct_c, lr_c = Environment.calc_reward(ccy_c, pct_c, lr_c, self._state, self._prev_px, 0,
                                                         px, 0)
            ccy_c_cost, pct_c_cost, lr_c_cost = Environment.calc_reward(ccy_c_cost, pct_c_cost, lr_c_cost, self._state,
                                                                        self._prev_px, 0,
                                                                        px, 0)
        # Check if we open new position
        if self._state == State.FLAT and action != Action.FLAT:
            if action == Action.BUY:
                self._info.long += 1
                self._info.long_length += 1
            elif action == Action.SELL:
                self._info.short += 1
                self._info.short_length += 1
            self._ent_px = px
            self._ent_time = self._current_time
            self._state = State.LONG if action == Action.BUY else State.SHORT
            ccy_c, pct_c, lr_c = Environment.calc_reward(ccy_c, pct_c, lr_c, self._state, px, zc, px, 0)
            ccy_c_cost, pct_c_cost, lr_c_cost = Environment.calc_reward(ccy_c_cost, pct_c_cost, lr_c_cost, self._state,
                                                                        px, c, px, 0)
        self._current_time = next_time
        self._prev_px = px

        # handle terminal state
        if d and self._state != State.FLAT:
            ccy, pct, lr = Environment.calc_reward(ccy, pct, lr, self._state, self._ent_px, zc, next_px, zc)
            ccy_cost, pct_cost, lr_cost = Environment.calc_reward(ccy_cost, pct_cost, lr_cost, self._state,
                                                                  self._ent_px, c, next_px, c)
            ccy_c, pct_c, lr_c = Environment.calc_reward(ccy_c, pct_c, lr_c, self._state, px, 0, next_px, zc)
            ccy_c_cost, pct_c_cost, lr_c_cost = Environment.calc_reward(ccy_c_cost, pct_c_cost, lr_c_cost, self._state,
                                                                        px, 0, next_px, c)
            self._state = State.FLAT
            self._ent_px = None
            self._ent_time = None
        # Calculate if pl positive
        self._pl_positive = False
        if self._state == State.LONG:
            self._pl_positive = (next_px > self._ent_px)
        elif self._state == State.SHORT:
            self._pl_positive = (next_px < self._ent_px)

        self._draw()

        # Emit reward
        r = 0.0
        rs = get_config().reward_scale_multiplier
        if get_config().reward_type == RewardType.RPL:
            switcher = {
                RewardAlgo.CCY: (ccy, ccy_cost),
                RewardAlgo.PCT: (pct * rs, pct_cost * rs),
                RewardAlgo.LR: (lr * rs, lr_cost * rs),
            }
            r = switcher.get(get_config().reward_algo)
        elif get_config().reward_type == RewardType.URPL:
            switcher = {
                RewardAlgo.CCY: (ccy_c, ccy_c_cost),
                RewardAlgo.PCT: (pct_c * rs, pct_c_cost * rs),
                RewardAlgo.LR: (lr_c * rs, lr_c_cost * rs),
            }
            r = switcher.get(get_config().reward_algo)
        # Update info
        self._info.ccy = ccy
        self._info.ccy_cost = ccy_cost
        self._info.ccy_c = ccy_c
        self._info.ccy_c_cost = ccy_c_cost
        self._info.pct = pct * 100.0
        self._info.pct_cost = pct_cost * 100.0
        self._info.pct_c = pct_c * 100.0
        self._info.pct_c_cost = pct_c_cost * 100.0
        self._info.lr = lr * 100.0
        self._info.lr_cost = lr_cost * 100.0
        self._info.lr_c = lr_c * 100.0
        self._info.lr_c_cost = lr_c_cost * 100.0
        self._info.state = self._state

        self._info.length += 1
        self._info.r += r[0]
        self._info.rwc += r[1]

        return self._get_state(), r, d, self._info

    @staticmethod
    def calc_reward(ccy, pct, lr, state, p1, p1_cost, p2, p2_cost):
        if state == State.FLAT:
            raise "Illegal state argument"
        pos_mult = 1.0 if state == State.LONG else -1.0
        p1_eff = p1 + pos_mult * p1_cost
        p2_eff = p2 - pos_mult * p2_cost
        return ccy + pos_mult * (p2_eff - p1_eff), pct + pos_mult * (
            p2_eff - p1_eff) / p1_eff, lr + pos_mult * math.log(p2_eff / p1_eff)

    def stop(self):
        get_ui_thread().stop_env(self)

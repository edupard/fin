import glfw
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math

from rl_fin.data_reader import DataReader
from rl_fin.data_reader import register_csv_dialect
from rl_fin.env import Environment, Mode, get_ui_thread
from rl_fin.config import get_config

a = 0
long = False
short = False
enter_hit = False
esc_hit = False

def key_callback(window, key, scan_code, action, mods):
    global long, short, a, enter_hit, esc_hit

    if key == glfw.KEY_ENTER and action == glfw.PRESS:
        enter_hit = True

    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        esc_hit = True

    if key == glfw.KEY_UP and action == glfw.PRESS:
        long = True
    if key == glfw.KEY_UP and action == glfw.RELEASE:
        long = False

    if key == glfw.KEY_DOWN and action == glfw.PRESS:
        short = True
    if key == glfw.KEY_DOWN and action == glfw.RELEASE:
        short = False

    if long and not short:
         a = 1
    elif short and not long:
        a = 2
    else:
        a = 0

def main():
    global a, enter_hit, esc_hit
    get_ui_thread().start()

    get_ui_thread().register_key_callback(key_callback)

    register_csv_dialect()
    dr = DataReader()
    dr.read_training_data()

    env = Environment(dr, Mode.STATE_2D)

    def get_file_name():
        now = datetime.now()
        s_now = now.strftime("%H%M%S")
        return 'eq/eq_bm_{}_sl_{:.3f}_w_{}_h_{}_bars_{}_hpct_{:.2f}_ed_{}_bps_{:.0f}_fps_{:.0f}_t_{}.png'.format(
            get_config().bar_min,
            get_config().costs,
            get_config().window_px_width,
            get_config().window_px_height,
            get_config().ww,
            get_config().min_px_window_height_pct,
            get_config().episode_days,
            get_config().bps,
            get_config().fps,
            s_now
        )

    def play_round():
        episode_frames = math.floor(get_config().episode_days * 24 * 60 / get_config().bar_min / get_config().bpf + 1)
        eq = np.zeros((episode_frames), dtype=np.float)

        d = False
        s = env.reset()
        idx = 0

        t_r = 0
        days_passed = 0
        frames_passed = 0
        while not d:
            env.render()
            s, r, d, _ = env.step(a)
            frames_passed += 1
            d_p = (frames_passed / get_config().fps) * get_config().bps * get_config().bar_min // (24 * 60)
            if d_p != days_passed:
                days_passed = d_p
                print('{} days gone'.format(days_passed))

            t_r += r
            eq[idx] += t_r
            if r != 0:
                print('Cum reward: {:.3f} reward: {:.3f}'.format(t_r, r))
            idx += 1
        round_length = idx
        eq = eq[:round_length]
        plt.plot(eq)
        plt.savefig(get_file_name())
        plt.clf()

    while True:
        game_duration_min = get_config().episode_days * 24 * 60 / get_config().bar_min / get_config().bps / 60
        print('Game duration: {:.1f}'.format(game_duration_min))
        play_round()
        while not enter_hit and not esc_hit:
            time.sleep(0.25)
        if esc_hit:
            break
        if enter_hit:
            enter_hit = False

    env.stop()

    get_ui_thread().stop()

if __name__ == '__main__':
    main()

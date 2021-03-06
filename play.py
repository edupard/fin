import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from env_factory import startup, shutdown, create_env, stop_env
from config import get_config
from data_source.data_source import get_datasource
from env.buttons import get_buttons


def main():
    get_config().render = True
    get_config().set_train_seed(0)
    get_config().turn_on_costs()

    startup()

    env = create_env()

    def get_file_name():
        if not os.path.exists('eq'):
            os.makedirs('eq')
        now = datetime.now()
        s_now = now.strftime("%H%M%S")
        return 'eq/eq_bm_{}_sl_{:.3f}_w_{}_h_{}_bars_{}_hpct_{:.2f}_bps_{:.0f}_fps_{:.0f}_t_{}.png'.format(
            get_config().bar_min,
            get_config().costs,
            get_config().window_px_width,
            get_config().window_px_height,
            get_config().ww,
            get_config().min_px_window_height_pct,
            get_config().bps,
            get_config().fps,
            s_now
        )

    def play_round():
        data_len = get_datasource().data.shape[0]
        eq = np.zeros((data_len), dtype=np.float)

        d = False
        s = env.reset()
        idx = 0

        t_r = 0
        t_r_c = 0
        days_passed = 0
        frames_passed = 0
        while not d and not get_buttons().esc_hit:
            env.render()
            s, (r, r_c), d, i = env.step(get_buttons().action)
            frames_passed += 1
            d_p = (frames_passed / get_config().fps) * get_config().bps * get_config().bar_min // (24 * 60)
            if d_p != days_passed:
                days_passed = d_p
                print('{} days gone'.format(days_passed))

            t_r += r
            t_r_c += r_c
            eq[idx] += t_r
            if r != 0:
                print('Cum reward: {:.3f} reward: {:.3f}'.format(t_r, r))
                # print('Cum reward: {:.3f} with cost: {:.3f} reward: {:.3f}'.format(t_r, t_r_c, r))
            idx += 1
        print('long deals: {} length: {} short deals: {} length: {}'.format(i.long, i.long_length, i.short,
                                                                            i.short_length))
        round_length = idx
        eq = eq[:round_length]
        plt.plot(eq)
        plt.savefig(get_file_name())
        plt.clf()

    play_round()
    stop_env(env)

    shutdown()


if __name__ == '__main__':
    main()

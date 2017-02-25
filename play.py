import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from env_factory import startup, shutdown, create_env, stop_env
from config import get_config as get_env_config, EnvironmentType
from data_source.config import get_config as get_data_config
from data_source.data_source import get_datasource
from env.config import get_config
from env.buttons import get_buttons

def main():
    get_env_config().environment = EnvironmentType.FIN

    startup()

    env = create_env()

    def get_file_name():
        if not os.path.exists('eq'):
            os.makedirs('eq')
        now = datetime.now()
        s_now = now.strftime("%H%M%S")
        return 'eq/eq_bm_{}_sl_{:.3f}_w_{}_h_{}_bars_{}_hpct_{:.2f}_bps_{:.0f}_fps_{:.0f}_t_{}.png'.format(
            get_data_config().bar_min,
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
        days_passed = 0
        frames_passed = 0
        while not d and not get_buttons().esc_hit:
            env.render()
            s, r, d, _ = env.step(get_buttons().action)
            frames_passed += 1
            d_p = (frames_passed / get_config().fps) * get_config().bps * get_data_config().bar_min // (24 * 60)
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

    play_round()
    stop_env(env)

    shutdown()

if __name__ == '__main__':
    main()

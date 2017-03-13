from config import get_config
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from env.env import State
from env.action import Action, convert_to_action

subplots = 4
time_ftm = matplotlib.dates.DateFormatter('%y %b %d')


def hide_time_labels(ax):
    plt.setp(ax.get_xticklabels(), visible=False)


def create_axis(fig, shared_ax, id, y_fmt_str):
    ax = fig.add_subplot(subplots, 1, id, sharex=shared_ax)
    ax.grid(True, linestyle='-', color='0.75')

    ax.xaxis.set_major_formatter(time_ftm)
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)

    ax.yaxis.set_major_formatter(FormatStrFormatter(y_fmt_str))

    return ax


def main():
    results = np.genfromtxt('results/{}.csv'.format(get_config().model), delimiter=',', dtype=np.float32)
    t = results[:, 0].reshape((-1))
    p = results[:, 1].reshape((-1))
    r = results[:, 2].reshape((-1))
    a = results[:, 3].reshape((-1))
    v = results[:, 4].reshape((-1))
    p_f = results[:, 5].reshape((-1))
    p_l = results[:, 6].reshape((-1))
    p_s = results[:, 7].reshape((-1))
    data_len = len(t)

    def reduce_time():
        for idx in range(data_len):
            dt = datetime.datetime.fromtimestamp(t[idx])
            yield matplotlib.dates.date2num(dt)

    def reduce_reward():
        tr = 0.0
        for idx in range(data_len):
            tr += r[idx]
            yield tr / get_config().reward_scale_multiplier * 100.0

    mpl_t = np.fromiter(reduce_time(), dtype=np.float64)
    tr = np.fromiter(reduce_reward(), dtype=np.float64)

    def reduce_price():
        px = None
        for idx in range(data_len):
            new_px = p[idx]
            new_t = mpl_t[idx]
            if px is not None:
                yield px
                yield new_t
            px = new_px
            yield px
            yield new_t

    def generate_deals():
        state = State.FLAT
        ent_t = None
        ent_px = None
        for idx in range(data_len):
            action = convert_to_action(a[idx])
            curr_px = p[idx]
            curr_t = mpl_t[idx]
            # process position liquidation
            if state == State.LONG and action != Action.BUY:
                yield (ent_t, ent_px, curr_t, curr_px, curr_px > ent_px)
                state = State.FLAT
            if state == State.SHORT and action != Action.SELL:
                yield (ent_t, ent_px, curr_t, curr_px, curr_px < ent_px)
                state = State.FLAT
            # process position entry
            if state == State.FLAT and (action == Action.BUY or action == Action.SELL):
                ent_t = curr_t
                ent_px = curr_px
                state = State.LONG if action == Action.BUY else State.SHORT

    px_and_t = np.fromiter(reduce_price(), dtype=np.float64).reshape((-1, 2))
    px = px_and_t[:, 0].reshape((-1))
    px_t = px_and_t[:, 1].reshape((-1))

    fig = plt.figure()
    # Plot prices
    p_ax = create_axis(fig, None, 1, '%.2f')
    p_ax.plot_date(px_t, px, color='b', fmt='-')
    # Plot deals
    for (ent_t, ent_px, exit_t, exit_px, pl_positive) in generate_deals():
        c = 'g' if pl_positive else 'r'
        p_ax.plot_date([ent_t, exit_t], [ent_px, exit_px], color=c, fmt='-')
    # Plot returns
    tr_ax = create_axis(fig, p_ax, 2, '%.3f')
    tr_ax.plot_date(mpl_t, tr, color='b', fmt='-')
    # Plot value estimate
    v_ax = create_axis(fig, p_ax, 3, '%.3f')
    v_ax.plot_date(mpl_t, v, color='r', fmt='-')
    # Plot probabilities estimate
    prob_ax = create_axis(fig, p_ax, 4, '%.3f')
    prob_ax.plot_date(mpl_t, p_f, color='b', fmt='-')
    prob_ax.plot_date(mpl_t, p_l, color='g', fmt='-')
    prob_ax.plot_date(mpl_t, p_s, color='r', fmt='-')

    hide_time_labels(p_ax)
    hide_time_labels(tr_ax)
    hide_time_labels(v_ax)

    plt.show(True)


if __name__ == '__main__':
    main()

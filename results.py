from config import get_config
import datetime
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from env.env import State
from env.action import Action, convert_to_action
import numpy as np
import math
import os
import fnmatch

draw_deals = False
# PL
draw_ccy = True
draw_ccy_c = False
draw_pct = True
draw_pct_c = False
draw_lr = False
draw_lr_c = False
# NN
draw_value = False
draw_probabilities = False

subplots = 0


def count_subpots(l):
    global subplots
    for d in l:
        if d:
            subplots += 1


count_subpots((True, draw_ccy, draw_ccy_c, draw_pct, draw_pct_c, draw_lr, draw_lr_c, draw_value, draw_probabilities))

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
    folder_path = os.path.join('results', get_config().model)
    files = fnmatch.filter(os.listdir(folder_path), '*.csv')
    files.sort()
    t_a = []
    p_a = []
    n_t_a = []
    n_p_a = []
    ccy_a = []
    ccy_c_a = []
    pct_a = []
    pct_c_a = []
    lr_a = []
    lr_c_a = []
    a_a = []
    v_a = []
    p_f_a = []
    p_l_a = []
    p_s_a = []
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        results = np.genfromtxt(file_path, delimiter=',', dtype=np.float64)
        col_idx = 0
        t_a.append(results[:, col_idx].reshape((-1)))
        col_idx += 1
        p_a.append(results[:, col_idx].reshape((-1)))
        col_idx += 1
        n_t_a.append(results[:, col_idx].reshape((-1)))
        col_idx += 1
        n_p_a.append(results[:, col_idx].reshape((-1)))
        col_idx += 1
        ccy_a.append(results[:, col_idx].reshape((-1)))
        col_idx += 1
        ccy_c_a.append(results[:, col_idx].reshape((-1)))
        col_idx += 1
        pct_a.append(results[:, col_idx].reshape((-1)))
        col_idx += 1
        pct_c_a.append(results[:, col_idx].reshape((-1)))
        col_idx += 1
        lr_a.append(results[:, col_idx].reshape((-1)))
        col_idx += 1
        lr_c_a.append(results[:, col_idx].reshape((-1)))
        col_idx += 1
        a_a.append(results[:, col_idx].reshape((-1)))
        col_idx += 1
        v_a.append(results[:, col_idx].reshape((-1)))
        col_idx += 1
        p_f_a.append(results[:, col_idx].reshape((-1)))
        col_idx += 1
        p_l_a.append(results[:, col_idx].reshape((-1)))
        col_idx += 1
        p_s_a.append(results[:, col_idx].reshape((-1)))
        col_idx += 1

    t = np.concatenate(t_a)
    p = np.concatenate(p_a)
    n_t = np.concatenate(n_t_a)
    n_p = np.concatenate(n_p_a)
    ccy = np.concatenate(ccy_a)
    ccy_c = np.concatenate(ccy_c_a)
    pct = np.concatenate(pct_a)
    pct_c = np.concatenate(pct_c_a)
    lr = np.concatenate(lr_a)
    lr_c = np.concatenate(lr_c_a)
    a = np.concatenate(a_a)
    v = np.concatenate(v_a)
    p_f = np.concatenate(p_f_a)
    p_l = np.concatenate(p_l_a)
    p_s = np.concatenate(p_s_a)
    data_len = len(t)

    sharp_ratio = math.sqrt(pct_c.shape[0]) * np.mean(pct_c) / np.std(pct_c)

    def convert_to_cum_reward(r):
        tr = 0.0
        for idx in range(r.shape[0]):
            tr += r[idx]
            r[idx] = tr

    convert_to_cum_reward(ccy)
    convert_to_cum_reward(ccy_c)
    convert_to_cum_reward(pct)
    convert_to_cum_reward(pct_c)
    convert_to_cum_reward(lr)
    convert_to_cum_reward(lr_c)

    def reduce_time(ta):
        for idx in range(data_len):
            dt = datetime.datetime.fromtimestamp(ta[idx])
            yield matplotlib.dates.date2num(dt)

    mpl_t = np.fromiter(reduce_time(t), dtype=np.float64)
    mpl_n_t = np.fromiter(reduce_time(n_t), dtype=np.float64)

    def make_step_line(ax, ay):
        if len(ax) != len(ay):
            raise "Inconsistent length"
        y = None
        for idx in range(len(ax)):
            new_y = ay[idx]
            new_x = ax[idx]
            if y is not None:
                yield y
                yield new_x
            y = new_y
            yield y
            yield new_x

    def extract_data_axes(generator):
        arr = np.fromiter(generator, dtype=np.float64).reshape((-1, 2))
        return arr[:, 0].reshape((-1)), arr[:, 1].reshape((-1))

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
                yield (ent_t, ent_px, curr_t, curr_px, curr_px > ent_px, state)
                state = State.FLAT
            if state == State.SHORT and action != Action.SELL:
                yield (ent_t, ent_px, curr_t, curr_px, curr_px < ent_px, state)
                state = State.FLAT
            # process position entry
            if state == State.FLAT and (action == Action.BUY or action == Action.SELL):
                ent_t = curr_t
                ent_px = curr_px
                state = State.LONG if action == Action.BUY else State.SHORT
        if state != State.FLAT:
            curr_px = n_p[data_len - 1]
            curr_t = mpl_n_t[data_len - 1]
            yield (ent_t, ent_px, curr_t, curr_px, curr_px > ent_px if state == State.LONG else curr_px < ent_px, state)

    last_axes = None
    fig = plt.figure()
    subplot_idx = 0
    # axes
    p_ax = None
    ccy_ax = None
    ccy_c_ax = None
    pct_ax = None
    pct_c_ax = None
    lr_ax = None
    lr_c_ax = None
    v_ax = None
    prob_ax = None

    # Plot prices
    subplot_idx += 1
    p_ax = create_axis(fig, None, subplot_idx, '%.2f')
    last_axes = p_ax
    p_ax.set_title("Sharp ratio: %.3f" % sharp_ratio)
    px, px_t = extract_data_axes(make_step_line(mpl_t, p))
    p_ax.plot_date(px_t, px, color='b', fmt='-')

    deals = 0
    for (ent_t, ent_px, exit_t, exit_px, pl_positive, state) in generate_deals():
        deals += 1

        # Plot deals
        if draw_deals:
            c = 'g' if pl_positive else 'r'
            p_ax.plot_date([ent_t, exit_t], [ent_px, exit_px], color=c, fmt='-')

    def calc_dd(r):
        def generate_previous_max():
            max = 0.0
            for idx in range(len(r)):
                # update max
                if r[idx] > max:
                    max = r[idx]
                yield max

        prev_max = np.fromiter(generate_previous_max(), dtype=np.float64)
        dd_a = r - prev_max
        return np.min(dd_a)

    print('Deals count %d' % deals)

    def plot_reward(caption, r, subplot_idx):
        ax = create_axis(fig, p_ax, subplot_idx, '%.3f')
        tr = r[-1:][0]
        dd = calc_dd(r)
        ax.set_title('{}: {:.3f} Max drop down: {:.3f}'.format(caption, tr, dd))
        a_r, a_t = extract_data_axes(make_step_line(mpl_t, r))
        ax.plot_date(a_t, a_r, color='b', fmt='-')
        return ax

    # Plot returns
    if draw_ccy:
        subplot_idx += 1
        ccy_ax = plot_reward('Usd reward per fixed lot', ccy, subplot_idx)
        last_axes = ccy_ax
    if draw_ccy_c:
        subplot_idx += 1
        ccy_c_ax = plot_reward('Usd continous reward per fixed lot', ccy_c, subplot_idx)
        last_axes = ccy_c_ax
    if draw_pct:
        subplot_idx += 1
        pct_ax = plot_reward('Pct reward per fixed nominal', pct, subplot_idx)
        last_axes = pct_ax
    if draw_pct_c:
        subplot_idx += 1
        pct_c_ax = plot_reward('Pct continous reward per fixed nominal', pct_c, subplot_idx)
        last_axes = pct_c_ax
    if draw_lr:
        subplot_idx += 1
        lr_ax = plot_reward('Log reward per fixed nominal', lr, subplot_idx)
        last_axes = lr_ax
    if draw_lr_c:
        subplot_idx += 1
        lr_ax_c = plot_reward('Log continous reward per fixed nominal', lr_c, subplot_idx)
        last_axes = lr_ax_c

    # Plot value estimate
    if draw_value:
        subplot_idx += 1
        v_ax = create_axis(fig, p_ax, subplot_idx, '%.3f')
        last_axes = v_ax
        v_ax.set_title("Neural network value estimate")
        a_v, a_v_t = extract_data_axes(make_step_line(mpl_t, v))
        v_ax.plot_date(a_v_t, a_v, color='r', fmt='-')
    # Plot probabilities estimate
    if draw_probabilities:
        subplot_idx += 1
        prob_ax = create_axis(fig, p_ax, subplot_idx, '%.3f')
        last_axes = prob_ax
        prob_ax.set_title("Policy actions probabilites")
        a_p_f, a_p_f_t = extract_data_axes(make_step_line(mpl_t, p_f))
        a_p_l, a_p_l_t = extract_data_axes(make_step_line(mpl_t, p_l))
        a_p_s, a_p_s_t = extract_data_axes(make_step_line(mpl_t, p_s))
        prob_ax.plot_date(a_p_f_t, a_p_f, color='b', fmt='-')
        prob_ax.plot_date(a_p_l_t, a_p_l, color='g', fmt='-')
        prob_ax.plot_date(a_p_s_t, a_p_s, color='r', fmt='-')

    def check_and_hide_time(ax):
        if ax is not None and ax is not last_axes: hide_time_labels(ax)

    check_and_hide_time(p_ax)
    check_and_hide_time(ccy_ax)
    check_and_hide_time(ccy_c_ax)
    check_and_hide_time(pct_ax)
    check_and_hide_time(pct_c_ax)
    check_and_hide_time(lr_ax)
    check_and_hide_time(lr_c_ax)
    check_and_hide_time(v_ax)
    check_and_hide_time(prob_ax)

    plt.show(True)


if __name__ == '__main__':
    main()

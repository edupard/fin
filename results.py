from config import get_config
import datetime
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from env.env import State
from env.action import Action, convert_to_action
import numpy as np
import math

draw_deals = False
draw_pct_reward = True
draw_pct_reward_check = True
draw_usd_reward = True
draw_value = False
draw_probabilities = False

subplots = 1
if draw_pct_reward:
    subplots += 1
if draw_pct_reward_check:
    subplots += 1
if draw_usd_reward:
    subplots += 1
if draw_value:
    subplots += 1
if draw_probabilities:
    subplots += 1

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
    results = np.genfromtxt('results/{}.csv'.format(get_config().model), delimiter=',', dtype=np.float64)
    t = results[:, 0].reshape((-1))
    p = results[:, 1].reshape((-1))
    n_t = results[:, 2].reshape((-1))
    n_p = results[:, 3].reshape((-1))
    r = results[:, 4].reshape((-1))
    a = results[:, 5].reshape((-1))
    v = results[:, 6].reshape((-1))
    p_f = results[:, 7].reshape((-1))
    p_l = results[:, 8].reshape((-1))
    p_s = results[:, 9].reshape((-1))
    data_len = len(t)

    def reduce_time(ta):
        for idx in range(data_len):
            dt = datetime.datetime.fromtimestamp(ta[idx])
            yield matplotlib.dates.date2num(dt)

    def reduce_reward():
        tr = 0.0
        for idx in range(data_len):
            tr += r[idx]
            yield tr / get_config().reward_scale_multiplier * 100.0

    mpl_t = np.fromiter(reduce_time(t), dtype=np.float64)
    mpl_n_t = np.fromiter(reduce_time(n_t), dtype=np.float64)
    tr = np.fromiter(reduce_reward(), dtype=np.float64)

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
            # TODO: correct is p[idx] as well as env calc is incorrect
            curr_px = n_p[idx]
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
    p_ax = None
    tr_ax = None
    pct_ax = None
    usd_ax = None
    v_ax = None
    prob_ax = None

    # Plot prices
    subplot_idx += 1
    p_ax = create_axis(fig, None, subplot_idx, '%.2f')
    last_axes = p_ax
    p_ax.set_title("Price")
    px, px_t = extract_data_axes(make_step_line(mpl_t, p))
    p_ax.plot_date(px_t, px, color='b', fmt='-')

    t_a = np.array([mpl_t[0]], dtype=np.float64)
    dc_a = np.array([0])
    usd_pl_a = np.array([0.0], dtype=np.float64)
    pct_pl_a = np.array([0.0], dtype=np.float64)
    r_a = np.array([], dtype=np.float64)

    usd_pl = 0.0
    pct_pl = 0.0
    deals = 0
    for (ent_t, ent_px, exit_t, exit_px, pl_positive, state) in generate_deals():
        deals += 1

        # Plot deals
        if draw_deals:
            c = 'g' if pl_positive else 'r'
            p_ax.plot_date([ent_t, exit_t], [ent_px, exit_px], color=c, fmt='-')

        pos_mult = 1.0 if state == State.LONG else -1.0
        act_ent_px = ent_px + pos_mult * get_config().costs
        act_exit_px = exit_px - pos_mult * get_config().costs
        lot_usd_pl = pos_mult * (act_exit_px - act_ent_px)
        nominal_pct_pl = lot_usd_pl / act_ent_px * 100.0

        usd_pl += lot_usd_pl
        pct_pl += nominal_pct_pl

        t_a = np.append(t_a, exit_t)
        dc_a = np.append(dc_a, deals)
        usd_pl_a = np.append(usd_pl_a, usd_pl)
        pct_pl_a = np.append(pct_pl_a, pct_pl)
        r_a = np.append(r_a, nominal_pct_pl)

    def generate_previous_max_pl(tr):
        max = 0.0
        for idx in range(len(tr)):
            # update max
            if tr[idx] > max:
                max = tr[idx]
            yield max

    pct_prev_max = np.fromiter(generate_previous_max_pl(tr), dtype=np.float64)
    pct_dd_a = tr - pct_prev_max
    pct_dd = np.min(pct_dd_a)

    usd_pl_prev_max = np.fromiter(generate_previous_max_pl(usd_pl_a), dtype=np.float64)
    usd_pl_dd_a = usd_pl_a - usd_pl_prev_max
    usd_dd = np.min(usd_pl_dd_a)

    sharp_ratio = math.sqrt(r_a.shape[0]) * np.mean(r_a) / np.std(r_a)

    pct_final_reward = tr[-1:][0]
    pct_check_final_reward = pct_pl_a[-1:][0]
    usd_final_reward = usd_pl_a[-1:][0]

    print('Deals count %d' % deals)
    print('Pct final reward %.3f' % pct_final_reward)
    print('Pct check final reward %.3f' % pct_check_final_reward)
    print('Usd final reward %.3f' % usd_final_reward)
    print('Max pct drop down %.3f' % pct_dd)
    print('Max usd drop down %.3f' % usd_dd)
    print('Normalized sharp ratio: %.3f' % sharp_ratio)

    # Plot returns
    if draw_pct_reward:
        subplot_idx += 1
        tr_ax = create_axis(fig, p_ax, subplot_idx, '%.3f')
        last_axes = tr_ax
        tr_ax.set_title("Pct reward per fixed nominal: %.3f%% Max drop down: %.3f%% Sharp ratio: %.2f" % (
            pct_final_reward, pct_dd, sharp_ratio))
        tr_ax.plot_date(mpl_t, tr, color='b', fmt='-')

    if draw_pct_reward_check:
        subplot_idx += 1
        pct_ax = create_axis(fig, p_ax, subplot_idx, '%.3f')
        last_axes = pct_ax
        pct_ax.set_title("Check pct reward per fixed nominal: %.3f%%" % pct_check_final_reward)
        pct_ax.plot_date(t_a, pct_pl_a, color='b', fmt='-')
    if draw_usd_reward:
        subplot_idx += 1
        usd_ax = create_axis(fig, p_ax, subplot_idx, '%.3f')
        last_axes = usd_ax
        usd_ax.set_title("Usd reward per lot: %.3f usd, Max drop down: %.3f usd" % (usd_final_reward, usd_dd))
        usd_ax.plot_date(t_a, usd_pl_a, color='b', fmt='-')
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
    check_and_hide_time(tr_ax)
    check_and_hide_time(pct_ax)
    check_and_hide_time(usd_ax)
    check_and_hide_time(v_ax)
    check_and_hide_time(prob_ax)

    plt.show(True)


if __name__ == '__main__':
    main()

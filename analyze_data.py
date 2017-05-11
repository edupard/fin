import csv
from yahoo_finance import Share
import datetime
import threading
import queue
from enum import Enum
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import tensorflow as tf
import scipy.stats as stats
import math

from rbm import RBM
from au import AutoEncoder
from ffnn import FFNN

NUM_WEEKS = 12
NUM_DAYS = 5

TRAIN_RBM = False
RBM_EPOCH_TO_TRAIN = 50
RBM_BATCH_SIZE = 10

TRAIN_AU = False
LOAD_RBM_WEIGHTS = True
AU_EPOCH_TO_TRAIN = 30
AU_BATCH_SIZE = 10

TRAIN_FFNN = False
LOAD_AU_WEIGHTS = True
FFNN_EPOCH_TO_TRAIN = 1000
FFNN_BATCH_SIZE = 10

PERCENTILE = 10

TRAIN_UP_TO_DATE = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d')

DDMMMYY_FMT = matplotlib.dates.DateFormatter('%y %b %d')

YYYY_FMT = matplotlib.dates.DateFormatter('%Y')

input = np.load('nasdaq_raw_data.npz')
raw_dt = input['raw_dt']
raw_data = input['raw_data']

STOCKS = raw_data.shape[0]


def reduce_time(arr):
    for idx in range(arr.shape[0]):
        dt = datetime.datetime.fromtimestamp(raw_dt[idx])
        yield matplotlib.dates.date2num(dt)


raw_mpl_dt = np.fromiter(reduce_time(raw_dt), dtype=np.float64)

g_a = raw_data[:, :, 4] * raw_data[:, :, 3]
g_a_a = np.average(g_a, axis=0)
# mask = (g_a[:, :] > (g_a_a[:] / 2.))
mask = g_a[:, :] > 10000000
# mask = (g_a[:, :] > 100000) & (raw_data[:, :, 3] > 5.)
# mask = raw_data[:,:,4] != 0


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.grid(True, linestyle='-', color='0.75')
# ax.xaxis.set_major_formatter(time_ftm)
# for label in ax.xaxis.get_ticklabels():
#     label.set_rotation(45)

# for i in range(20):
#     idx = random.randrange(0, raw_data.shape[0])
#     px = raw_data[i, :, 3]
#     ax.plot_date(raw_mpl_dt, px, fmt='-')

# ax.plot_date(raw_mpl_dt, g_a_a, color='b', fmt='o')

traded_stocks = mask[:, :].sum(0)
# ax.plot_date(raw_mpl_dt, traded_stocks, color='b', fmt='o')

# for i in range(20):
#     idx = random.randrange(0, raw_data.shape[0])
#     traded_stocks = mask[idx,:].astype(int) * (i + 1)
#     ax.plot_date(raw_mpl_dt, traded_stocks, fmt='o')

# for i in range(20):
#     idx = random.randrange(0, raw_data.shape[0])
#     g = g_a[idx, :]
#     ax.plot_date(raw_mpl_dt, g, fmt='o')


#

start_date = datetime.datetime.fromtimestamp(raw_dt[0])
end_date = datetime.datetime.fromtimestamp(raw_dt[len(raw_dt) - 1])
sunday = start_date + datetime.timedelta(days=7 - start_date.isoweekday())


def get_data_idx(dt):
    if dt < start_date or dt > end_date:
        return None
    return (dt - start_date).days


def get_dates_for_weekly_return(sunday, n_w):
    dates = []
    t_d = sunday
    populated = 0
    while populated < n_w + 1:
        data_idx = get_data_idx(t_d)
        if data_idx is None:
            return None
        for j in range(7):
            if traded_stocks[data_idx] > 0:
                dates.append(data_idx)
                populated += 1
                break
            data_idx -= 1
            if data_idx < 0:
                return None
        t_d = t_d - datetime.timedelta(days=7)
    return dates[::-1]


def get_dates_for_daily_return(sunday, n_d):
    dates = []
    data_idx = get_data_idx(sunday)
    if data_idx is None:
        return None
    populated = 0
    while populated < n_d + 1:
        if traded_stocks[data_idx] > 0:
            dates.append(data_idx)
            populated += 1
        data_idx -= 1
        if data_idx < 0:
            return None
    return dates[::-1]


train_records = 0
train_weeks = 0
total_weeks = 0
data_set_records = 0

dr = None
wr = None
hpr = None
c_l = None
c_s = None
stocks = None
w_data_index = None
w_num_stocks = None
w_enter_index = None
w_exit_index = None


def append_data(data, _data):
    if data is None:
        return _data
    else:
        return np.concatenate([data, _data], axis=0)


def make_array(value):
    return np.array([value]).astype(np.int32)


while True:
    # iterate over weeks
    sunday = sunday + datetime.timedelta(days=7)
    # break when all availiable data processed
    if sunday > end_date:
        break
    w_r_i = get_dates_for_weekly_return(sunday, NUM_WEEKS + 1)
    # continue if all data not availiable yet
    if w_r_i is None:
        continue
    # continue if all data not availiable yet
    d_r_i = get_dates_for_daily_return(sunday - datetime.timedelta(days=7), NUM_DAYS)
    if d_r_i is None:
        continue

    # stocks slice on days used to calculate returns
    s_s = mask[:, w_r_i + d_r_i]
    # tradable stocks slice
    t_s = np.all(s_s, axis=1)
    # get tradable stocks indices
    t_s_i = np.where(t_s)[0]
    stocks = append_data(stocks, t_s_i)

    # sample size
    num_stocks = t_s_i.shape[0]

    # daily closes
    # numpy can not slice on indices in 2 dimensions
    # so slice in one dimension followed by slice in another dimension
    d_c = raw_data[:, d_r_i, :]
    d_c = d_c[t_s_i, :, :]
    d_c = d_c[:, :, 3]

    # calc daily returns
    d_r = (d_c[:, 1:] - d_c[:, :-1]) / d_c[:, :-1]
    # accumulate daily returns
    d_c_r = np.cumsum(d_r, axis=1)
    # calculate accumulated return mean over all weeks
    d_r_m = np.average(d_c_r, axis=0)
    # calculate accumulated return std over all weeks
    d_r_std = np.std(d_c_r, axis=0)
    # calc z score
    d_n_r = (d_c_r - d_r_m) / d_r_std

    dr = append_data(dr, d_n_r)

    # weekly closes
    # numpy can not slice on indices in 2 dimensions
    # so slice in one dimension followed by slice in another dimension
    w_c = raw_data[:, w_r_i, :]
    w_c = w_c[t_s_i, :, :]
    w_c = w_c[:, :, 3]

    # calc weekly returns
    w_r = (w_c[:, 1:-1] - w_c[:, :-2]) / w_c[:, :-2]
    # accumulate weekly returns
    w_c_r = np.cumsum(w_r, axis=1)
    # calculate accumulated return mean over all weeks
    w_r_m = np.average(w_c_r, axis=0)
    # calculate accumulated return std over all weeks
    w_r_std = np.std(w_c_r, axis=0)
    # calc z score
    w_n_r = (w_c_r - w_r_m) / w_r_std

    wr = append_data(wr, w_n_r)

    _hpr = (w_c[:, NUM_WEEKS + 1] - w_c[:, NUM_WEEKS]) / w_c[:, NUM_WEEKS]
    hpr = append_data(hpr, _hpr)

    hpr_med = np.median(_hpr)
    _c_l = _hpr >= hpr_med
    _c_s = ~_c_l
    c_l = append_data(c_l, _c_l)
    c_s = append_data(c_s, _c_s)

    enter_date_idx = w_r_i[NUM_WEEKS]
    exit_date_idx = w_r_i[NUM_WEEKS + 1]

    w_data_index = append_data(w_data_index, make_array(data_set_records))
    w_num_stocks = append_data(w_num_stocks, make_array(num_stocks))
    w_enter_index = append_data(w_enter_index, make_array(enter_date_idx))
    w_exit_index = append_data(w_exit_index, make_array(exit_date_idx))

    # record counts
    data_set_records += num_stocks
    total_weeks += 1
    if sunday <= TRAIN_UP_TO_DATE:
        train_records += num_stocks
        train_weeks += 1


def calc_classes_and_decisions(data_set_records, total_weeks, data):
    c_l = np.zeros((data_set_records), dtype=np.bool)
    c_s = np.zeros((data_set_records), dtype=np.bool)

    s_l = np.zeros((data_set_records), dtype=np.bool)
    s_s = np.zeros((data_set_records), dtype=np.bool)

    top_hpr = np.zeros((total_weeks))
    bottom_hpr = np.zeros((total_weeks))
    top_stocks_num = np.zeros((total_weeks))
    bottom_stocks_num = np.zeros((total_weeks))

    for i in range(total_weeks):
        w_i = i
        beg = w_data_index[w_i]
        end = beg + w_num_stocks[w_i]

        _data = data[beg: end]

        median = np.median(_data)
        _s_c_l = c_l[beg: end]
        _s_c_s = c_s[beg: end]
        pred_long_cond = _data >= median
        _s_c_l |= pred_long_cond
        _s_c_s |= ~pred_long_cond

        top_bound = np.percentile(_data, 100 - PERCENTILE)
        bottom_bound = np.percentile(_data, PERCENTILE)
        _s_s_l = s_l[beg: end]
        _s_s_s = s_s[beg: end]
        long_cond = _data >= top_bound
        short_cond = _data <= bottom_bound
        _s_s_l |= long_cond
        _s_s_s |= short_cond
        _hpr = hpr[beg: end]
        l_hpr = _hpr[_s_s_l]
        s_hpr = _hpr[_s_s_s]
        top_hpr[w_i] = np.mean(l_hpr)
        bottom_hpr[w_i] = np.mean(s_hpr)
        top_stocks_num[w_i] = l_hpr.shape[0]
        bottom_stocks_num[w_i] = s_hpr.shape[0]
    return c_l, c_s, top_hpr, bottom_hpr, top_stocks_num, bottom_stocks_num


# naming convention: s_c_l mean Simple strategy Class Long e_c_l mean Enhanced strategy Class Long
# naming convention: s_s_l mean Simple strategy Stock(selected) Long e_c_l mean Enhanced strategy Stock(selected) Long

def confusion_matrix(a_l, a_s, p_l, p_s):
    p_l_a_l = (p_l & a_l).sum(0)
    p_l_a_s = (p_l & a_s).sum(0)
    p_s_a_s = (p_s & a_s).sum(0)
    p_s_a_l = (p_s & a_l).sum(0)
    total = p_l_a_l + p_l_a_s + p_s_a_s + p_s_a_l
    print('L +: {:.2f} -: {:.2f} accuracy: {:.2f}'
        .format(
        100. * p_l_a_l / total,
        100. * p_l_a_s / total,
        100. * p_l_a_l / (p_l_a_l + p_l_a_s)
    ))
    print('S +: {:.2f} -: {:.2f} accuracy: {:.2f}'
        .format(
        100. * p_s_a_s / total,
        100. * p_s_a_l / total,
        100. * p_s_a_s / (p_s_a_s + p_s_a_l)
    ))
    print('Total accuracy: {:.2f}'
        .format(
        100. * (p_l_a_l + p_s_a_s) / total
    ))


def hpr_analysis(t_hpr, b_hpr):
    d_hpr = t_hpr - b_hpr
    t_hpr_mean = np.mean(t_hpr)
    b_hpr_mean = np.mean(b_hpr)
    d_hpr_mean = np.mean(d_hpr)

    t_t, p_t = stats.ttest_1samp(t_hpr, t_hpr_mean)
    t_b, p_b = stats.ttest_1samp(b_hpr, b_hpr_mean)
    t_d, p_d = stats.ttest_1samp(d_hpr, d_hpr_mean)
    print(
        "T: {:.4f} t-stat: {:.2f} p: {:.2f} B: {:.4f} t-stat: {:.2f} p: {:.2f} D: {:.4f} t-stat: {:.2f} p: {:.2f}".format(
            t_hpr_mean,
            t_t,
            p_t,
            b_hpr_mean,
            t_b,
            p_b,
            d_hpr_mean,
            t_d,
            p_d
        ))


def wealth_graph(t_hpr, b_hpr, t_stocks, b_stocks, w_exit_index):
    def format_time_labels(ax, fmt):
        ax.xaxis.set_major_formatter(fmt)
        for label in ax.xaxis.get_ticklabels():
            label.set_rotation(45)

    def draw_grid(ax):
        ax.grid(True, linestyle='-', color='0.75')

    def hide_time_labels(ax):
        plt.setp(ax.get_xticklabels(), visible=False)

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

    def calc_sharp(r):
        return math.sqrt(r.shape[0]) * np.mean(r) / np.std(r)

    fig = plt.figure()

    diff = t_hpr - b_hpr

    progress = diff
    wealth = np.cumsum(progress)
    dd = calc_dd(wealth)
    sharp = calc_sharp(diff)

    ax = fig.add_subplot(1, 1, 1)
    draw_grid(ax)
    format_time_labels(ax, fmt=DDMMMYY_FMT)
    ax.set_title("1 USD PL Sharp: %.3f Drop down: %.3f" % (sharp, dd))
    ax.plot_date(raw_mpl_dt[w_exit_index], wealth, color='b', fmt='-')

    rc_progress = (diff) + 1.00
    rc_wealth = np.cumprod(rc_progress) - 1.
    rc_dd = calc_dd(rc_wealth)

    rc_base = np.cumprod(rc_progress)
    rc_r = (rc_base[1:] - rc_base[:-1])/rc_base[:-1]
    rc_r = np.concatenate([np.array([rc_base[0] - 1.]),rc_r])
    rc_sharp = calc_sharp(rc_r)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, sharex=ax)
    draw_grid(ax)
    # hide_time_labels(ax)
    ax.set_title("1 USD PL RECAP Drop down: %.3f" % (rc_dd))
    ax.plot_date(raw_mpl_dt[w_exit_index], rc_wealth, color='b', fmt='-')
    format_time_labels(ax, fmt=DDMMMYY_FMT)
    # ax = fig.add_subplot(3, 1, 3, sharex=ax)
    # draw_grid(ax)
    # format_time_labels(ax)
    # ax.plot_date(raw_mpl_dt[w_exit_index], t_stocks, color='g', fmt='o')
    # ax.plot_date(raw_mpl_dt[w_exit_index], b_stocks, color='r', fmt='o')

    fig = plt.figure()

    yr = []
    yr_alt = []
    yts = []
    c_y = datetime.datetime.fromtimestamp(raw_dt[w_exit_index[0]]).year


    c_yr = 0.
    c_y_w = 0

    c_yr_beg_w = 1.
    c_yr_end_w = wealth[0] + 1.

    w_i = []

    weeks_to_append_year = 365//7*0.8
    for w in range(t_hpr.shape[0]):
        y = datetime.datetime.fromtimestamp(raw_dt[w_exit_index[w]]).year

        c_yr_end_w = wealth[w] + 1.

        if y != c_y:
            w_i.append(w)
            if c_y_w > weeks_to_append_year:
                yr_alt.append((c_yr_end_w - c_yr_beg_w) / c_yr_beg_w * 100.0)
                yr.append(c_yr * 100.0)
                dt =  datetime.date(year=c_y,day=1,month=1)
                yts.append(matplotlib.dates.date2num(dt))
            c_yr_beg_w = c_yr_end_w
            c_y = y
            c_yr = 0.

            c_y_w = 0
        c_yr += diff[w]
        c_y_w += 1


    if c_y_w > weeks_to_append_year:
        yr_alt.append((c_yr_end_w - c_yr_beg_w) / c_yr_beg_w * 100.0)
        yr.append(c_yr)
        dt = datetime.date(year=c_y, day=1, month=1)
        yts.append(matplotlib.dates.date2num(dt))

    ax = fig.add_subplot(1, 1, 1)
    draw_grid(ax)
    format_time_labels(ax, fmt=YYYY_FMT)
    ax.bar(yts, yr, color='b', width=100)
    ax.xaxis_date()
    ax.set_title("Year pct return")

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # draw_grid(ax)
    # format_time_labels(ax, fmt=YYYY_FMT)
    # ax.bar(yts, yr_alt, color='b', width=100)
    # ax.xaxis_date()
    # ax.set_title("Year pct return alt")

    with open('pl.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for w in range(wealth.shape[0]):
            dt = datetime.datetime.fromtimestamp(raw_dt[w_exit_index[w]])
            _wealth = wealth[w]
            _rc_wealth = rc_wealth[w]
            writer.writerow((dt.strftime('%Y-%m-%d'), _wealth, _rc_wealth))



prob_l = np.zeros((data_set_records), dtype=np.float)

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b
    # return tf.nn.relu(tf.matmul(x, w) + b)


rbmobject1 = RBM(17, 40, ['rbmw1', 'rbvb1', 'rbmhb1'], 0.001)

rbmobject2 = RBM(40, 4, ['rbmw2', 'rbvb2', 'rbmhb2'], 0.001)

autoencoder = AutoEncoder(17, [40, 4], [['rbmw1', 'rbmhb1'],
                                        ['rbmw2', 'rbmhb2']],
                          tied_weights=False)

ffnn = FFNN(17, [40, 4], [['rbmw1', 'rbmhb1'],
                          ['rbmw2', 'rbmhb2']], transfer_function=tf.nn.sigmoid)
data_indices = np.arange(train_records)

if TRAIN_RBM:
    print("Training RBM layer 1")
    batches_per_epoch = train_records // RBM_BATCH_SIZE

    for i in range(RBM_EPOCH_TO_TRAIN):
        np.random.shuffle(data_indices)
        epoch_cost = 0.
        curr_progress = 0

        for b in range(batches_per_epoch):
            # get data indices for slice
            d_i_s = data_indices[b * RBM_BATCH_SIZE: (b + 1) * RBM_BATCH_SIZE]

            _wr = wr[d_i_s, :]
            _dr = dr[d_i_s, :]
            input = np.concatenate([_wr, _dr], axis=1)

            cost = rbmobject1.partial_fit(input)
            epoch_cost += cost
            progress = b // (batches_per_epoch // 10)
            if progress != curr_progress:
                print('.', sep=' ', end='', flush=True)
                curr_progress = progress
        print("Epoch cost: {:.3f}".format(epoch_cost / batches_per_epoch))

    rbmobject1.save_weights('./rbm/rbmw1.chp')

if TRAIN_RBM:
    print("Training RBM layer 2")
    for i in range(RBM_EPOCH_TO_TRAIN):
        np.random.shuffle(data_indices)
        epoch_cost = 0.
        curr_progress = 0

        for b in range(batches_per_epoch):
            # get data indices for slice
            d_i_s = data_indices[b * RBM_BATCH_SIZE: (b + 1) * RBM_BATCH_SIZE]

            _wr = wr[d_i_s, :]
            _dr = dr[d_i_s, :]
            input = np.concatenate([_wr, _dr], axis=1)

            input = rbmobject1.transform(input)
            cost = rbmobject2.partial_fit(input)
            epoch_cost += cost
            progress = b // (batches_per_epoch // 10)
            if progress != curr_progress:
                print('.', sep=' ', end='', flush=True)
                curr_progress = progress
        print("Epoch cost: {:.3f}".format(epoch_cost / batches_per_epoch))

    rbmobject2.save_weights('./rbm/rbmw2.chp')

if TRAIN_AU:

    print("Training Autoencoder")

    if LOAD_RBM_WEIGHTS:
        autoencoder.load_rbm_weights('./rbm/rbmw1.chp', ['rbmw1', 'rbmhb1'], 0)
        autoencoder.load_rbm_weights('./rbm/rbmw2.chp', ['rbmw2', 'rbmhb2'], 1)

    batches_per_epoch = train_records // AU_BATCH_SIZE
    for i in range(AU_EPOCH_TO_TRAIN):
        np.random.shuffle(data_indices)
        epoch_cost = 0.
        curr_progress = 0

        for b in range(batches_per_epoch):
            # get data indices for slice
            d_i_s = data_indices[b * AU_BATCH_SIZE: (b + 1) * AU_BATCH_SIZE]

            _wr = wr[d_i_s, :]
            _dr = dr[d_i_s, :]
            input = np.concatenate([_wr, _dr], axis=1)

            cost = autoencoder.partial_fit(input)
            # print("Batch cost: {:.3f}".format(cost))
            epoch_cost += cost
            progress = b // (batches_per_epoch // 10)
            if progress != curr_progress:
                print('.', sep=' ', end='', flush=True)
                curr_progress = progress
        print("Epoch cost: {:.3f}".format(epoch_cost / batches_per_epoch))

    autoencoder.save_weights('./rbm/au.chp')

if TRAIN_FFNN:
    print("Training FFNN")

    if LOAD_AU_WEIGHTS:
        ffnn.load_au_weights('./rbm/au.chp', ['rbmw1', 'rbmhb1'], 0)
        ffnn.load_au_weights('./rbm/au.chp', ['rbmw2', 'rbmhb2'], 1)

    batches_per_epoch = train_records // FFNN_BATCH_SIZE
    for i in range(FFNN_EPOCH_TO_TRAIN):
        np.random.shuffle(data_indices)
        epoch_cost = 0.
        curr_progress = 0

        for b in range(batches_per_epoch):
            # get data indices for slice
            d_i_s = data_indices[b * FFNN_BATCH_SIZE: (b + 1) * FFNN_BATCH_SIZE]

            _wr = wr[d_i_s, :]
            _dr = dr[d_i_s, :]
            input = np.concatenate([_wr, _dr], axis=1)

            _cl = c_l[d_i_s].reshape((-1, 1))
            _cs = c_s[d_i_s].reshape((-1, 1))
            observation = np.concatenate([_cl, _cs], axis=1).astype(np.float32)

            cost = ffnn.partial_fit(input, observation)
            # print("Batch cost: {:.3f}".format(cost))
            epoch_cost += cost
            progress = b // (batches_per_epoch // 10)
            if progress != curr_progress:
                print('.', sep=' ', end='', flush=True)
                curr_progress = progress
        print("Epoch cost: {:.3f}".format(epoch_cost / batches_per_epoch))
        if i % 10 == 0:
            ffnn.save_weights('./rbm/ffnn.chp')

    ffnn.save_weights('./rbm/ffnn.chp')
else:
    ffnn.load_weights('./rbm/ffnn.chp')

print("Evaluating")
b = 0
curr_progress = 0
batches_per_epoch = data_set_records // FFNN_BATCH_SIZE
while True:
    start_idx = b * FFNN_BATCH_SIZE
    end_idx = (b + 1) * FFNN_BATCH_SIZE
    d_i_s = np.arange(start_idx, min(end_idx, data_set_records))
    _wr = wr[d_i_s, :]
    _dr = dr[d_i_s, :]
    input = np.concatenate([_wr, _dr], axis=1)
    p_dist = ffnn.predict(input)
    for idx in d_i_s:
        prob_l[idx] = p_dist[idx - start_idx, 0]
    if end_idx >= data_set_records:
        break
    progress = b // (batches_per_epoch // 10)
    if progress != curr_progress:
        print('.', sep=' ', end='', flush=True)
        curr_progress = progress
    b += 1

# s_c_l, s_c_s, t_hpr, b_hpr, t_stocks, b_stocks = calc_classes_and_decisions(
#     data_set_records, total_weeks, wr[:, NUM_WEEKS - 1]
# )

# confusion_matrix(c_l, c_s, s_c_l, s_c_s)
# hpr_analysis(t_hpr, b_hpr)
# wealth_graph(t_hpr, b_hpr, t_stocks, b_stocks, w_exit_index)

e_c_l, e_c_s, t_e_hpr, b_e_hpr, t_e_stocks, b_e_stocks = calc_classes_and_decisions(
    data_set_records, total_weeks, prob_l
)

confusion_matrix(c_l[train_records:], c_s[train_records:], e_c_l[train_records:], e_c_s[train_records:])
hpr_analysis(t_e_hpr[train_weeks:], b_e_hpr[train_weeks:])
wealth_graph(t_e_hpr[train_weeks:],
             b_e_hpr[train_weeks:],
             t_e_stocks[train_weeks:],
             b_e_stocks[train_weeks:],
             w_exit_index[train_weeks:])

plt.show(True)

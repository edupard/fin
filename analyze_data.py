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

NUM_WEEKS = 12
NUM_DAYS = 5
EPOCHS_TO_TRAIN = 0
BATCH_SIZE = 500
PERCENTILE = 10

TRAIN_UP_TO_DATE = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d')

time_ftm = matplotlib.dates.DateFormatter('%y %b %d')

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
    d_r_i = get_dates_for_daily_return(sunday, NUM_DAYS)
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

# naming convention: s_c_l mean Simple strategy Class Long e_c_l mean Enhanced strategy Class Long
# naming convention: s_s_l mean Simple strategy Stock(selected) Long e_c_l mean Enhanced strategy Stock(selected) Long

s_c_l = np.zeros((data_set_records), dtype=np.bool)
s_c_s = np.zeros((data_set_records), dtype=np.bool)

s_s_l = np.zeros((data_set_records), dtype=np.bool)
s_s_s = np.zeros((data_set_records), dtype=np.bool)

t_hpr = np.zeros((total_weeks))
b_hpr = np.zeros((total_weeks))
t_stocks = np.zeros((total_weeks))
b_stocks = np.zeros((total_weeks))

for i in range(total_weeks):
    w_i = i
    beg = w_data_index[w_i]
    end = beg + w_num_stocks[w_i]

    l_w_r = wr[beg: end, NUM_WEEKS - 1]

    median = np.median(l_w_r)
    _s_c_l = s_c_l[beg: end]
    _s_c_s = s_c_s[beg: end]
    pred_long_cond = l_w_r >= median
    _s_c_l |= pred_long_cond
    _s_c_s |= ~pred_long_cond

    top_bound = np.percentile(l_w_r, 100 - PERCENTILE)
    bottom_bound = np.percentile(l_w_r, PERCENTILE)
    _s_s_l = s_s_l[beg: end]
    _s_s_s = s_s_s[beg: end]
    long_cond = l_w_r >= top_bound
    short_cond = l_w_r <= bottom_bound
    _s_s_l |= long_cond
    _s_s_s |= short_cond
    _hpr = hpr[beg: end]
    l_hpr = _hpr[_s_s_l]
    s_hpr = _hpr[_s_s_s]
    top_hpr = np.mean(l_hpr)
    bottom_hpr = np.mean(s_hpr)
    t_hpr[w_i] = top_hpr
    b_hpr[w_i] = bottom_hpr
    t_stocks[w_i] = l_hpr.shape[0]
    b_stocks[w_i] = s_hpr.shape[0]


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
    t_t, p_t = stats.ttest_1samp(t_hpr, 0)
    t_b, p_b = stats.ttest_1samp(b_hpr, 0)
    t_d, p_d = stats.ttest_1samp(d_hpr, 0)
    print(
        "T: {:.2f} t-stat: {:.2f} p: {:.2f} B: {:.2f} t-stat: {:.2f} p: {:.2f} D: {:.2f} t-stat: {:.2f} p: {:.2f}".format(
            np.mean(t_hpr),
            t_t,
            p_t,
            np.mean(b_hpr),
            t_b,
            p_b,
            np.mean(d_hpr),
            t_d,
            p_d
        ))


def wealth_graph(t_hpr, b_hpr, t_stocks, b_stocks, w_exit_index):
    def format_time_labels(ax):
        ax.xaxis.set_major_formatter(time_ftm)
        for label in ax.xaxis.get_ticklabels():
            label.set_rotation(45)

    def draw_grid(ax):
        ax.grid(True, linestyle='-', color='0.75')

    def hide_time_labels(ax):
        plt.setp(ax.get_xticklabels(), visible=False)

    fig = plt.figure()
    ax = fig.add_subplot(3, 1, 1)
    draw_grid(ax)
    hide_time_labels(ax)

    # progress = (t_hpr - b_hpr) + 1.00
    # wealth = np.cumprod(progress)

    progress = t_hpr - b_hpr
    wealth = np.cumsum(progress)


    ax.plot_date(raw_mpl_dt[w_exit_index], wealth, color='b', fmt='-')

    ax = fig.add_subplot(3, 1, 2, sharex=ax)
    ax.grid(True, linestyle='-', color='0.75')
    draw_grid(ax)
    hide_time_labels(ax)
    ax.plot_date(raw_mpl_dt[w_exit_index], t_stocks, color='g', fmt='o')

    ax = fig.add_subplot(3, 1, 3, sharex=ax)
    ax.grid(True, linestyle='-', color='0.75')
    draw_grid(ax)
    format_time_labels(ax)
    ax.plot_date(raw_mpl_dt[w_exit_index], b_stocks, color='r', fmt='o')


confusion_matrix(c_l, c_s, s_c_l, s_c_s)
hpr_analysis(t_hpr, b_hpr)
wealth_graph(t_hpr, b_hpr, t_stocks, b_stocks, w_exit_index)

# train_data = preprocessed[:train_records, :]
# test_data = preprocessed[train_records:, :]

prob_l = np.zeros((data_set_records), dtype=np.float)


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b
    # return tf.nn.relu(tf.matmul(x, w) + b)


g = tf.Graph()
with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    input = tf.placeholder(tf.float32, shape=(None, NUM_WEEKS + NUM_DAYS))
    l2 = linear(input, 40, 'l2', initializer=tf.contrib.layers.xavier_initializer())
    l3 = linear(l2, 4, 'l3', initializer=tf.contrib.layers.xavier_initializer())
    l4 = linear(l3, 50, 'l4', initializer=tf.contrib.layers.xavier_initializer())
    l5 = linear(l4, 2, 'l5', initializer=tf.contrib.layers.xavier_initializer())
    predictions = tf.nn.softmax(l5)
    observations = tf.placeholder(tf.float32, shape=(None, 2))
    loss = tf.reduce_mean(-tf.reduce_sum(observations * tf.log(predictions), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate=0.00005).minimize(loss)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    epoch_number = 0
    try:
        ckpt = tf.train.get_checkpoint_state('stocks')
        load_path = ckpt.model_checkpoint_path
        saver.restore(sess, load_path)
        epoch_number = int(load_path.split('-')[-1])
        print("loaded model saved after {} epochs".format(epoch_number))
    except:
        pass

    batches_per_epoch = train_records // BATCH_SIZE
    data_indices = np.arange(train_records)
    for train_iteration in range(EPOCHS_TO_TRAIN):
        epoch = epoch_number + train_iteration + 1
        total_loss = 0
        curr_progress = 0
        # shuffle data
        np.random.shuffle(data_indices)
        for b in range(batches_per_epoch):
            # get data indices for slice
            d_i_s = data_indices[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]

            _wr = wr[d_i_s, :]
            _dr = dr[d_i_s, :]
            _cl = c_l[d_i_s].reshape((-1, 1))
            _cs = c_s[d_i_s].reshape((-1, 1))
            i = np.concatenate([_wr, _dr], axis=1)
            o = np.concatenate([_cl, _cs], axis=1).astype(np.float32)
            feed_dict = {
                input: i,
                observations: o
            }
            # l = sess.run(loss, feed_dict)
            l, p, _ = sess.run([loss, predictions, train_step], feed_dict)
            total_loss += l
            progress = b // (batches_per_epoch // 10)
            if progress != curr_progress:
                print('.', sep=' ', end='', flush=True)
                curr_progress = progress

        print("epoch: {} loss: {}".format(epoch, total_loss / batches_per_epoch))
        saver.save(sess, 'stocks/stocks.ckpt', global_step=epoch)
    for idx in range(data_set_records):
        _wr = wr[[idx], :]
        _dr = dr[[idx], :]
        i = np.concatenate([_wr, _dr], axis=1)
        feed_dict = {
            input: i
        }
        p_dist = sess.run(predictions, feed_dict)
        prob_l[idx] = p_dist[0, 0]

# naming convention: s_c_l mean Simple strategy Class Long e_c_l mean Enhanced strategy Class Long
# naming convention: s_s_l mean Simple strategy Stock(selected) Long e_c_l mean Enhanced strategy Stock(selected) Long

e_c_l = np.zeros((data_set_records), dtype=np.bool)
e_c_s = np.zeros((data_set_records), dtype=np.bool)

e_s_l = np.zeros((data_set_records), dtype=np.bool)
e_s_s = np.zeros((data_set_records), dtype=np.bool)

t_e_hpr = np.zeros((total_weeks))
b_e_hpr = np.zeros((total_weeks))
t_e_stocks = np.zeros((total_weeks))
b_e_stocks = np.zeros((total_weeks))

for i in range(total_weeks):
    w_i = i
    beg = w_data_index[w_i]
    end = beg + w_num_stocks[w_i]

    _prob_l = prob_l[beg: end]

    median = np.median(_prob_l)
    _e_c_l = e_c_l[beg: end]
    _e_c_s = e_c_s[beg: end]
    pred_long_cond = _prob_l >= median
    _e_c_l |= pred_long_cond
    _e_c_s |= ~pred_long_cond

    top_bound = np.percentile(_prob_l, 100 - PERCENTILE)
    bottom_bound = np.percentile(_prob_l, PERCENTILE)
    _e_s_l = e_s_l[beg: end]
    _e_s_s = e_s_s[beg: end]
    long_cond = _prob_l >= top_bound
    short_cond = _prob_l <= bottom_bound
    _e_s_l |= long_cond
    _e_s_s |= short_cond
    _hpr = hpr[beg: end]
    l_hpr = _hpr[_e_s_l]
    s_hpr = _hpr[_e_s_s]
    top_hpr = np.mean(l_hpr)
    bottom_hpr = np.mean(s_hpr)
    t_e_hpr[w_i] = top_hpr
    b_e_hpr[w_i] = bottom_hpr
    t_e_stocks[w_i] = l_hpr.shape[0]
    b_e_stocks[w_i] = s_hpr.shape[0]

confusion_matrix(c_l[train_records:], c_s[train_records:], e_c_l[train_records:], e_c_s[train_records:])
hpr_analysis(t_e_hpr[train_weeks:], b_e_hpr[train_weeks:])
wealth_graph(t_e_hpr[train_weeks:],
             b_e_hpr[train_weeks:],
             t_e_stocks[train_weeks:],
             b_e_stocks[train_weeks:],
             w_exit_index[train_weeks:])

plt.show(True)

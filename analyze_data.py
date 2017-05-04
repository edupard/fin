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

NUM_WEEKS = 12
NUM_DAYS = 5
EPOCHS_TO_TRAIN = 1000
BATCH_SIZE = 500

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


# plt.show(True)

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


preprocessed = None
train_records = 0

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
    # sample size
    num_stocks = t_s_i.shape[0]
    # record train set size
    if sunday <= TRAIN_UP_TO_DATE:
        train_records += num_stocks

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
    # calculate 12 week z score median
    w_r_med = np.median(w_n_r[:, NUM_WEEKS - 1])
    w_r_med = np.full(num_stocks, w_r_med)
    # simple strategy class
    s_s_c_l = (w_n_r[:, NUM_WEEKS - 1] >= w_r_med).astype(np.float)
    s_s_c_s = (w_n_r[:, NUM_WEEKS - 1] < w_r_med).astype(np.float)

    # calc hpr
    hpr = (w_c[:, NUM_WEEKS + 1] - w_c[:, NUM_WEEKS]) / w_c[:, NUM_WEEKS]
    hpr_med = np.median(hpr)
    hpr_med = np.full(num_stocks, hpr_med)
    c_l = (hpr >= hpr_med).astype(np.float)
    c_s = (hpr < hpr_med).astype(np.float)

    # enter - exit price
    enter_px = w_c[:, NUM_WEEKS]
    exit_px = w_c[:, NUM_WEEKS + 1]
    # enter - exit date idx
    enter_date_idx = w_r_i[NUM_WEEKS]
    exit_date_idx = w_r_i[NUM_WEEKS + 1]
    enter_date = start_date + datetime.timedelta(days=enter_date_idx)
    exit_date = start_date + datetime.timedelta(days=exit_date_idx)
    enter_date_idx = np.full(num_stocks, enter_date_idx)
    exit_date_idx = np.full(num_stocks, exit_date_idx)
    enter_date = np.full(num_stocks, enter_date.timestamp())
    exit_date = np.full(num_stocks, exit_date.timestamp())

    week_data = np.concatenate([
        w_n_r,
        d_n_r,
        c_l[:, np.newaxis],
        c_s[:, np.newaxis],
        t_s_i[:, np.newaxis],
        hpr[:, np.newaxis],
        hpr_med[:, np.newaxis],
        w_r_med[:, np.newaxis],
        s_s_c_l[:, np.newaxis],
        s_s_c_s[:, np.newaxis],
        enter_px[:, np.newaxis],
        exit_px[:, np.newaxis],
        enter_date_idx[:, np.newaxis],
        exit_date_idx[:, np.newaxis],
        enter_date[:, np.newaxis],
        exit_date[:, np.newaxis]
    ], axis=1)

    if preprocessed is None:
        preprocessed = week_data
    else:
        preprocessed = np.concatenate([preprocessed, week_data], axis=0)

train_data = preprocessed[:train_records, :]
test_data = preprocessed[train_records:, :]


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


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
    train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

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

    batches_per_epoch = train_data.shape[0] // BATCH_SIZE
    data_indices = np.arange(train_data.shape[0])
    for train_iteration in range(EPOCHS_TO_TRAIN):
        epoch = epoch_number + train_iteration + 1
        total_loss = 0
        curr_progress = 0
        # shuffle data
        np.random.shuffle(data_indices)
        for b in range(batches_per_epoch):
            # get data indices for slice
            d_i_s = data_indices[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]
            batch_data = preprocessed[d_i_s, :]

            INPUT_DIM = NUM_WEEKS + NUM_DAYS
            i = batch_data[:, :INPUT_DIM]
            o = batch_data[:, INPUT_DIM: INPUT_DIM + 2]
            feed_dict = {
                input: i,
                observations: o
            }
            # l = sess.run(loss, feed_dict)
            l, _ = sess.run([loss, train_step], feed_dict)
            total_loss += l
            progress = b // (batches_per_epoch // 10)
            if progress != curr_progress:
                print('.', sep=' ', end='', flush=True)
                curr_progress = progress

        print("epoch: {} loss: {}".format(epoch, total_loss / batches_per_epoch))
        saver.save(sess, 'stocks/stocks.ckpt', global_step=epoch)

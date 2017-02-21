from rl_fin.config import Config
import numpy as np
import os, fnmatch
import datetime
import math
import csv
from rl_fin.config import get_config

def register_csv_dialect():
    csv.register_dialect(
        'data',
        delimiter=',',
        quotechar='"',
        doublequote=True,
        skipinitialspace=True,
        lineterminator='\r\n',
        quoting=csv.QUOTE_MINIMAL)


class DataReader(object):
    def __init__(self):
        self._data = np.array([], dtype=np.float32)
        self._train_data = None
        self._test_data = None
        self._cv_data = None

        self._train_px = None
        self._cv_px = None
        self._test_px = None
        self._start_time = None

        PREPROCESSED_FOLDER_PATH = './data/preprocessed'
        if not os.path.exists(PREPROCESSED_FOLDER_PATH):
            os.makedirs(PREPROCESSED_FOLDER_PATH)
        self._DATA_FOLDER_PATH = PREPROCESSED_FOLDER_PATH + '/{}_{}m'.format(get_config().ticker, get_config().bar_min)
        self._TRAIN_FILE_PATH = self._DATA_FOLDER_PATH + '/train.csv'
        self._TEST_FILE_PATH = self._DATA_FOLDER_PATH + '/test.csv'
        self._CV_FILE_PATH = self._DATA_FOLDER_PATH + '/cv.csv'

    def _calc_start_time(self):
        files = fnmatch.filter(os.listdir('./data'), '*.csv')
        for file_name in files:
            sdt = file_name.split('.')[0]
            self._start_time = datetime.datetime.strptime(sdt, '%Y%m%d_%H%M%S')
            break

    def _preprocess_data(self) -> None:
        files = fnmatch.filter(os.listdir('./data'), '*.csv')
        for file_name in files:
            file_path = os.path.join('./data', file_name)
            if file_name.startswith(get_config().ticker):
                raw_data = np.genfromtxt(file_path, delimiter=',', dtype=np.float32)
                break

        col_raw_open = raw_data[:, 0]
        col_raw_close = raw_data[:, 1]
        col_raw_high = raw_data[:, 2]
        col_raw_low = raw_data[:, 3]
        col_raw_volume = raw_data[:, 4]
        col_raw_last_close = raw_data[:, 5]

        raw_data_len = col_raw_volume.shape[0]

        def reduce_close(minutes: int):
            close = np.float32('nan')
            for idx in range(raw_data_len):
                c = col_raw_close[idx]
                if not np.isnan(c):
                    close = c
                if (idx + 1) % minutes == 0:
                    yield close
                    close = np.float32('nan')

        def reduce_high(minutes: int):
            high = np.float32('nan')
            for idx in range(raw_data_len):
                h = col_raw_high[idx]
                if np.isnan(high) or high < h:
                    high = h
                if (idx + 1) % minutes == 0:
                    yield high
                    high = np.float32('nan')

        def reduce_low(minutes: int):
            low = np.float32('nan')
            for idx in range(raw_data_len):
                l = col_raw_low[idx]
                if np.isnan(low) or low > l:
                    low = l
                if (idx + 1) % minutes == 0:
                    yield low
                    low = np.float32('nan')

        def reduce_open(minutes: int):
            open = np.float32('nan')
            for idx in range(col_raw_volume.shape[0]):
                o = col_raw_open[idx]
                if np.isnan(open):
                    open = o
                if (idx + 1) % minutes == 0:
                    yield open
                    open = np.float32('nan')

        def reduce_volume(minutes: int):
            volume = 0.
            for idx in range(col_raw_volume.shape[0]):
                v = col_raw_volume[idx]
                volume += v
                if (idx + 1) % minutes == 0:
                    yield volume
                    volume = 0.

        def reduce_last_close(minutes: int):
            last_close = np.float32('nan')
            for idx in range(col_raw_volume.shape[0]):
                lc = col_raw_last_close[idx]
                if np.isnan(last_close):
                    last_close = lc
                if (idx + 1) % minutes == 0:
                    yield last_close
                    last_close = np.float32('nan')

        data_len = raw_data_len // get_config().bar_min

        col_open = np.fromiter(reduce_open(get_config().bar_min), dtype=np.float32, count=data_len)
        col_close = np.fromiter(reduce_close(get_config().bar_min), dtype=np.float32, count=data_len)
        col_high = np.fromiter(reduce_high(get_config().bar_min), dtype=np.float32, count=data_len)
        col_low = np.fromiter(reduce_low(get_config().bar_min), dtype=np.float32, count=data_len)
        col_volume = np.fromiter(reduce_volume(get_config().bar_min), dtype=np.float32, count=data_len)
        col_last_close = np.fromiter(reduce_last_close(get_config().bar_min), dtype=np.float32, count=data_len)
        col_ones = np.ones(col_volume.shape, dtype=np.float32)
        # col_sess = np.ones(col_volume.shape, dtype=np.float32)

        def reduce_last_px():
            for idx in range(col_volume.shape[0]):
                lc = col_last_close[idx]
                c = col_close[idx]
                if not np.isnan(c):
                    yield c
                else:
                    yield lc

        col_last_px = np.fromiter(reduce_last_px(), dtype=np.float32, count=data_len)

        def periods_without_trades():
            zeros = 0
            start_idx = 0
            idx = 0
            for vol in np.nditer(col_volume):
                if vol <= 0:
                    if zeros == 0:
                        start_idx = idx
                    zeros += 1
                else:
                    if zeros != 0:
                        yield (start_idx, zeros)
                        zeros = 0
                idx += 1

        # gap_bars = get_config().gap_min // get_config().bar_min + 1
        # zero_periods = list(filter(lambda x: x[1] > gap_bars, periods_without_trades()))
        # for start_idx, zeros in zero_periods:
        #     for idx in range(start_idx - 1, start_idx + zeros - 1):
        #         col_sess[idx] = 0

        vol_mean = np.mean(col_volume)
        vol_std = np.std(col_volume)
        vol_max = np.max(col_volume)
        vol_min = 0.

        col_vol = col_volume
        # first variant
        # col_vol = (col_volume - vol_min) / (vol_max - vol_min)
        # second variant
        # col_vol = (col_volume - vol_mean) / vol_std
        # third variant
        # col_vol = (col_volume - vol_min) / vol_std

        col_ret = np.log(np.divide(col_close, col_last_close, out=col_ones, where=col_volume > 0))
        columns = list()
        # columns.append(col_ret)
        # columns.append(col_sess)
        columns.append(col_last_px)
        columns.append(col_vol)
        data_dim = len(columns)
        self._data = np.append(self._data, columns)
        self._data = self._data.reshape(data_dim, data_len)
        self._data = np.transpose(self._data, [1, 0])

        train_data_len = math.floor(data_len * get_config().train_set_size)
        cv_data_len = math.floor(data_len * get_config().cv_set_size)
        test_data_len = data_len - train_data_len - cv_data_len

        self._train_data = self._data[0:train_data_len, :]
        self._cv_data = self._data[train_data_len:train_data_len + cv_data_len, :]
        self._test_data = self._data[train_data_len + cv_data_len:, :]

    def _save_preprocessed_data(self):
        if not os.path.exists(self._DATA_FOLDER_PATH):
            os.makedirs(self._DATA_FOLDER_PATH)

        with open(self._TRAIN_FILE_PATH, 'w') as f:
            writer = csv.writer(f, dialect='data')
            for idx in range(0, self._train_data.shape[0]):
                row = self._train_data[idx, :]
                writer.writerow(row)

        with open(self._CV_FILE_PATH, 'w') as f:
            writer = csv.writer(f, dialect='data')
            for idx in range(0, self._cv_data.shape[0]):
                row = self._cv_data[idx, :]
                writer.writerow(row)

        with open(self._TEST_FILE_PATH, 'w') as f:
            writer = csv.writer(f, dialect='data')
            for idx in range(0, self._test_data.shape[0]):
                row = self._test_data[idx, :]
                writer.writerow(row)

    def _restore_preprocessed_data(self):
        self._train_data = np.genfromtxt(self._TRAIN_FILE_PATH, delimiter=',', dtype=np.float32)
        self._cv_data = np.genfromtxt(self._CV_FILE_PATH, delimiter=',', dtype=np.float32)
        self._test_data = np.genfromtxt(self._TEST_FILE_PATH, delimiter=',', dtype=np.float32)

    def read_training_data(self) -> None:

        if not os.path.exists(self._DATA_FOLDER_PATH):
            self._preprocess_data()
            self._save_preprocessed_data()
        else:
            self._restore_preprocessed_data()
        self._postprocess_data()


    def _postprocess_data(self):
        if get_config().switch_off_zero_bars:
            cond = (self._train_data[:,1] > 0)
            self._train_data = self._train_data[cond]

            cond = (self._test_data[:, 1] > 0)
            self._test_data = self._test_data[cond]

            cond = (self._cv_data[:, 1] > 0)
            self._cv_data = self._cv_data[cond]


    def read_analysis_data(self):
        self._calc_start_time()
        self._restore_preprocessed_data()
        self._prepare_prices()

    def _prepare_prices(self):
        data_dim = self._train_data.shape[1]
        net_data_dim = data_dim - 1

        self._train_px = self._train_data[:, net_data_dim:].reshape((-1))
        self._cv_px = self._cv_data[:, net_data_dim:].reshape((-1))
        self._test_px = self._test_data[:, net_data_dim:].reshape((-1))

    @property
    def start_time(self):
        return self._start_time

    @property
    def train_px(self):
        return self._train_px

    @property
    def cv_px(self):
        return self._cv_px

    @property
    def test_px(self):
        return self._test_px

    @property
    def train_data(self):
        return self._train_data

    @property
    def cv_data(self):
        return self._cv_data

    @property
    def test_data(self):
        return self._test_data

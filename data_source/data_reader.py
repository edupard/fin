import numpy as np
import os, fnmatch
import datetime
import math
import csv
from yahoo_finance import Share

from config import get_config


def get_paths():
    PREPROCESSED_FOLDER_PATH = './data/preprocessed'
    if get_config().yahoo:
        DATA_FOLDER_PATH = PREPROCESSED_FOLDER_PATH + '/{}_{}_{}'.format(get_config().ticker, get_config().start, get_config().end)
    else:
        DATA_FOLDER_PATH = PREPROCESSED_FOLDER_PATH + '/{}_{}m'.format(get_config().ticker, get_config().bar_min)

    DATA_FILE_PATH = DATA_FOLDER_PATH + '/data.csv'
    return DATA_FOLDER_PATH, DATA_FILE_PATH


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
        self._data = np.array([], dtype=np.float64)
        self._start_time = None

        PREPROCESSED_FOLDER_PATH = './data/preprocessed'
        if not os.path.exists(PREPROCESSED_FOLDER_PATH):
            os.makedirs(PREPROCESSED_FOLDER_PATH)
        self._DATA_FOLDER_PATH, self._DATA_FILE_PATH = get_paths()

    def _preprocess_data(self) -> None:
        print('Preprocessing data...')
        if get_config().yahoo:
            print('Retrieving data for {} from YAHOO...'.format(get_config().ticker))
            yahoo = Share(get_config().ticker)
            data = yahoo.get_historical(get_config().start, get_config().end)
            print('Data retrieved')
            data_len = len(data)
            def reduce_time():
                for idx in range(data_len):
                    yield datetime.datetime.strptime(data[idx]['Date'], '%Y-%m-%d').timestamp()
            def reduce_volume():
                for idx in range(data_len):
                    yield float(data[idx]['Volume'])
            def reduce_px():
                for idx in range(data_len):
                    yield float(data[idx]['Close'])
            col_time = np.fromiter(reduce_time(), dtype=np.float64)
            col_vol = np.fromiter(reduce_volume(), dtype=np.float64)
            col_px = np.fromiter(reduce_px(), dtype=np.float64)

        else:
            files = fnmatch.filter(os.listdir('./data'), '*.csv')
            raw_data = None
            start_time = datetime.datetime.utcnow()
            for file_name in files:
                file_path = os.path.join('./data', file_name)
                if file_name.startswith(get_config().ticker):
                    sdt = file_name[len(get_config().ticker) + 1:].split('.')[0]
                    start_time = datetime.datetime.strptime(sdt, '%Y%m%d_%H%M%S').replace(tzinfo=datetime.timezone.utc)
                    raw_data = np.genfromtxt(file_path, delimiter=',', dtype=np.float64)
                    break

            if raw_data is None:
                raise "data not found"

            col_raw_open = raw_data[:, 0]
            col_raw_close = raw_data[:, 1]
            col_raw_high = raw_data[:, 2]
            col_raw_low = raw_data[:, 3]
            col_raw_volume = raw_data[:, 4]
            col_raw_last_close = raw_data[:, 5]

            raw_data_len = col_raw_volume.shape[0]

            def time_generator():
                time = start_time
                for idx in range(raw_data_len):
                    yield time.timestamp()
                    time += datetime.timedelta(minutes=1)

            col_raw_time = np.fromiter(time_generator(), dtype=np.float64, count=raw_data_len)

            def reduce_time(minutes: int):
                for idx in range(raw_data_len):
                    t = col_raw_time[idx]
                    if (idx + 1) % minutes == 0:
                        yield t

            def reduce_close(minutes: int):
                close = np.float64('nan')
                for idx in range(raw_data_len):
                    c = col_raw_close[idx]
                    if not np.isnan(c):
                        close = c
                    if (idx + 1) % minutes == 0:
                        yield close
                        close = np.float64('nan')

            def reduce_high(minutes: int):
                high = np.float64('nan')
                for idx in range(raw_data_len):
                    h = col_raw_high[idx]
                    if np.isnan(high) or high < h:
                        high = h
                    if (idx + 1) % minutes == 0:
                        yield high
                        high = np.float64('nan')

            def reduce_low(minutes: int):
                low = np.float64('nan')
                for idx in range(raw_data_len):
                    l = col_raw_low[idx]
                    if np.isnan(low) or low > l:
                        low = l
                    if (idx + 1) % minutes == 0:
                        yield low
                        low = np.float64('nan')

            def reduce_open(minutes: int):
                open = np.float64('nan')
                for idx in range(col_raw_volume.shape[0]):
                    o = col_raw_open[idx]
                    if np.isnan(open):
                        open = o
                    if (idx + 1) % minutes == 0:
                        yield open
                        open = np.float64('nan')

            def reduce_volume(minutes: int):
                volume = 0.
                for idx in range(col_raw_volume.shape[0]):
                    v = col_raw_volume[idx]
                    volume += v
                    if (idx + 1) % minutes == 0:
                        yield volume
                        volume = 0.

            def reduce_last_close(minutes: int):
                last_close = np.float64('nan')
                for idx in range(col_raw_volume.shape[0]):
                    lc = col_raw_last_close[idx]
                    if np.isnan(last_close):
                        last_close = lc
                    if (idx + 1) % minutes == 0:
                        yield last_close
                        last_close = np.float64('nan')

            data_len = raw_data_len // get_config().bar_min

            col_time = np.fromiter(reduce_time(get_config().bar_min), dtype=np.float64, count=data_len)
            col_open = np.fromiter(reduce_open(get_config().bar_min), dtype=np.float64, count=data_len)
            col_close = np.fromiter(reduce_close(get_config().bar_min), dtype=np.float64, count=data_len)
            col_high = np.fromiter(reduce_high(get_config().bar_min), dtype=np.float64, count=data_len)
            col_low = np.fromiter(reduce_low(get_config().bar_min), dtype=np.float64, count=data_len)
            col_volume = np.fromiter(reduce_volume(get_config().bar_min), dtype=np.float64, count=data_len)
            col_last_close = np.fromiter(reduce_last_close(get_config().bar_min), dtype=np.float64, count=data_len)
            col_ones = np.ones(col_volume.shape, dtype=np.float64)

            def reduce_px():
                for idx in range(col_volume.shape[0]):
                    lc = col_last_close[idx]
                    c = col_close[idx]
                    if not np.isnan(c):
                        yield c
                    else:
                        yield lc

            col_px = np.fromiter(reduce_px(), dtype=np.float64, count=data_len)
            col_vol = col_volume
            # possible normalization
            # vol_mean = np.mean(col_volume)
            # vol_std = np.std(col_volume)
            # vol_max = np.max(col_volume)
            # vol_min = 0.
            # first variant
            # col_vol = (col_volume - vol_min) / (vol_max - vol_min)
            # second variant
            # col_vol = (col_volume - vol_mean) / vol_std
            # third variant
            # col_vol = (col_volume - vol_min) / vol_std

            col_ret = np.log(np.divide(col_close, col_last_close, out=col_ones, where=col_volume > 0))



        columns = list()
        columns.append(col_time)
        columns.append(col_px)
        columns.append(col_vol)
        data_dim = len(columns)
        self._data = np.append(self._data, columns)
        self._data = self._data.reshape(data_dim, data_len)
        self._data = np.transpose(self._data, [1, 0])

    def _save_preprocessed_data(self):
        print('Saving preprocessed data...')
        if not os.path.exists(self._DATA_FOLDER_PATH):
            os.makedirs(self._DATA_FOLDER_PATH)

        with open(self._DATA_FILE_PATH, 'w', newline='') as f:
            writer = csv.writer(f, dialect='data')
            for idx in range(0, self._data.shape[0]):
                row = self._data[idx, :]
                writer.writerow(row)

    def _restore_preprocessed_data(self):
        print('Restoring preprocessed data...')
        self._data = np.genfromtxt(self._DATA_FILE_PATH, delimiter=',', dtype=np.float64)

    def read_data(self) -> None:

        if not os.path.exists(self._DATA_FILE_PATH):
            self._preprocess_data()
            self._save_preprocessed_data()
        else:
            self._restore_preprocessed_data()
        self._postprocess_data()


    def _postprocess_data(self):
        print('Postprocessing data...')
        if get_config().switch_off_zero_bars:
            cond = (self._data[:,2] > 0)
            self._data = self._data[cond]

    @property
    def start_time(self):
        return self._start_time

    @property
    def data(self):
        return self._data

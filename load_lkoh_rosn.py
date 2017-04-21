import csv
import os, fnmatch
import datetime
import numpy as np
import math

LKOH_DIR = './LKOH'
ROSN_DIR = './ROSN'

dt_beg = datetime.datetime.strptime('20140101 00:00:00', '%Y%m%d %H:%M:%S')
dt_end = datetime.datetime.strptime('20170420 00:00:00', '%Y%m%d %H:%M:%S')

def get_total_minutes(td):
    return td.days * 24 * 60 + td.seconds // 60


time_frame = dt_end - dt_beg
time_frame_min = get_total_minutes(time_frame)

data = np.empty((2, time_frame_min + 1, 5), dtype=np.float32)


def get_data_idx(dt):
    if dt < dt_beg or dt > dt_end:
        return None
    td = dt - dt_beg
    return get_total_minutes(td)


def process_contract(data_dir, contract_idx):
    files = fnmatch.filter(os.listdir(data_dir), '*.csv')
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = True
            for row in reader:
                if not header:
                    dt = datetime.datetime.strptime('{} {}'.format(row[2], row[3]), '%Y%m%d %H%M%S')
                    data_idx = get_data_idx(dt)
                    if data_idx is None:
                        continue
                    o = float(row[4])
                    h = float(row[5])
                    l = float(row[6])
                    c = float(row[7])
                    v = float(row[8])
                    data[contract_idx, data_idx, 0] = o
                    data[contract_idx, data_idx, 1] = h
                    data[contract_idx, data_idx, 2] = l
                    data[contract_idx, data_idx, 3] = c
                    data[contract_idx, data_idx, 4] = v
                else:
                    header = False

process_contract(LKOH_DIR, 0)
process_contract(ROSN_DIR, 1)

with open('lr.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    prices = [None] * 2
    for i in range(data.shape[1]):
        l_r = None
        #LKOH
        v = data[0, i, 4]
        if v != 0:
            n_l_p = data[0, i, 3]
            l_p = prices[0]
            if l_p is not None:
                l_r = math.log(n_l_p / l_p)
            prices[0] = n_l_p
        r_r = None
        # ROSN
        v = data[1, i, 4]
        if v != 0:
            n_r_p = data[1, i, 3]
            r_p = prices[1]
            if r_p is not None:
                r_r = math.log(n_r_p / r_p)
            prices[1] = n_r_p

        v_l = data[0, i, 4]
        v_r = data[1, i, 4]
        v = 0
        lr = 0.
        if v_l != 0 and v_r != 0 and r_r is not None and l_r is not None:
            lr = r_r - l_r
            v = v_l + v_r
        dt = dt_beg + datetime.timedelta(minutes=i)
        writer.writerow([dt.timestamp(), lr, v])

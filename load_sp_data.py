import csv
import os, fnmatch
import datetime
import numpy as np

DATA_DIR = './data/F.US.QO.BARS'
SWITCH_DAYS_BEFORE_EXPIRATION = 7
CONTRACT_DAYS = 30

files = fnmatch.filter(os.listdir(DATA_DIR), '*.csv')
contracts = len(files)

def calc_last_dt(file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = True
        for row in reader:
            if not header:
                dt = datetime.datetime.strptime('{} {}'.format(row[2], row[3]), '%Y%m%d %H:%M:%S')
            else:
                header = False
    return dt


# first contract
dt_beg = calc_last_dt(files[0]) - datetime.timedelta(days=CONTRACT_DAYS)
dt_end = calc_last_dt(files[contracts - 1]) - datetime.timedelta(days=SWITCH_DAYS_BEFORE_EXPIRATION)

def get_total_minutes(td):
    return td.days * 24 * 60 + td.seconds // 60

time_frame = dt_end - dt_beg
time_frame_min = get_total_minutes(time_frame)

data = np.empty((contracts, time_frame_min + 1, 5), dtype=np.float32)
switch_idxs = []

def get_data_idx(dt):
    if dt < dt_beg or dt > dt_end:
        return None
    td = dt - dt_beg
    return get_total_minutes(td)


contract_idx = 0
for file_name in files:
    file_path = os.path.join(DATA_DIR, file_name)
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = True
        for row in reader:
            if not header:
                dt = datetime.datetime.strptime('{} {}'.format(row[2], row[3]), '%Y%m%d %H:%M:%S')
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
        switch_idx = get_data_idx(dt - datetime.timedelta(days=SWITCH_DAYS_BEFORE_EXPIRATION))
        switch_idxs.append(switch_idx)
    contract_idx += 1

with open('sp.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    prices = [None] * contracts
    lead_contract_idx = 0

    s_p = None
    for i in range(data.shape[1]):
        if i > switch_idxs[lead_contract_idx]:
            lead_contract_idx += 1
            # abandon cycle
            if lead_contract_idx == contracts - 1:
                break
        for ci in range(contracts):
            v = data[ci, i, 4]
            if v != 0:
                prices[ci] = data[ci, i, 3]

        v_n = data[lead_contract_idx, i, 4]
        v_f = data[lead_contract_idx + 1, i, 4]
        p_f = prices[lead_contract_idx + 1]
        p_n = prices[lead_contract_idx]
        v = 0
        if v_n != 0 and v_f != 0:
            s_p = p_f - p_n
            v = v_n + v_f
        dt = dt_beg + datetime.timedelta(minutes=i)
        writer.writerow([dt.timestamp(), s_p, v])

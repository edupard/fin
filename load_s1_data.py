import csv
import os, fnmatch
import datetime
import numpy as np

def calc_dt_range():
    dt_beg = None
    dt_end = None
    with open('data.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dt_beg = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            if dt_end is None:
                dt_end = dt_beg

    return dt_beg, dt_end


# first contract
dt_beg, dt_end = calc_dt_range()

def get_total_minutes(td):
    return td.days * 24 * 60 + td.seconds // 60

time_frame = dt_end - dt_beg
time_frame_min = get_total_minutes(time_frame)

data = np.empty((time_frame_min + 1, 5), dtype=np.float32)

def get_data_idx(dt):
    if dt < dt_beg or dt > dt_end:
        return None
    td = dt - dt_beg
    return get_total_minutes(td)


with open('data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        dt = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
        data_idx = get_data_idx(dt)
        if data_idx is None:
            continue
        o = float(row[1])
        h = float(row[2])
        l = float(row[3])
        c = float(row[4])
        v = float(row[5])
        data[data_idx, 0] = o
        data[data_idx, 1] = h
        data[data_idx, 2] = l
        data[data_idx, 3] = c
        data[data_idx, 4] = v

with open('s1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for data_idx in range(data.shape[0]):
        o = data[data_idx, 0]
        h = data[data_idx, 1]
        l = data[data_idx, 2]
        c = data[data_idx, 3]
        v = data[data_idx, 4]
        dt = dt_beg + datetime.timedelta(minutes=data_idx)
        writer.writerow([dt.timestamp(), o, h, l, c, v])

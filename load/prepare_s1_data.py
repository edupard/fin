import csv
import os, fnmatch
import datetime
import numpy as np

bar_minutes = 15


with open('s1_{}min.csv'.format(bar_minutes), 'w', newline='') as f, open('s1.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    writer = csv.writer(f)
    bar_dt = None
    bar_px = None
    bar_volume = 0
    idx = 0
    for row in reader:
        if idx % bar_minutes == 0:
            if bar_volume != 0:
                #emit merged non-empty bar
                writer.writerow([bar_dt, bar_px, bar_volume])

        dt = row[0]
        px = row[4]
        v = float(row[5])
        #reset bar info
        if idx % bar_minutes == 0:
            bar_dt = dt
            bar_volume = 0
        #accumulate bar info
        bar_volume += v
        if v != 0:
            bar_px = px
        #increment idx
        idx += 1
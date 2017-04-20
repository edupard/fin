import csv
from yahoo_finance import Share
import datetime
import threading
import queue
from enum import Enum
import random

tickers = []

ONE_DAY = datetime.timedelta(days=1)
BATCH_DOWNLOAD_DURATION = datetime.timedelta(days=30)
START_DATE = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date()
END_DATE = datetime.datetime.strptime('2017-04-18', '%Y-%m-%d').date()
# END_DATE = datetime.datetime.strptime('2000-01-20', '%Y-%m-%d').date()

WORKERS = 300

class PayloadType(Enum):
    DATA = 0
    TASK_COMPLETED = 1


class Payload:

    def __init__(self, payload_type, payload):
        self.payload_type = payload_type
        self.payload = payload


class Writer:
    def __init__(self):
        thread_func = lambda: self.task()
        self.thread = threading.Thread(target=(thread_func))
        self.queue = queue.Queue()
        self.tasks_completed = 0

    def task(self):
        with open('nasdaq_history.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            while True:
                p = self.queue.get()
                if p.payload_type == PayloadType.TASK_COMPLETED:
                    self.tasks_completed += 1
                    if self.tasks_completed == WORKERS:
                        break
                elif p.payload_type == PayloadType.DATA:
                    try:
                        for d in p.payload:
                            o = d['Open']
                            c = d['Close']
                            h = d['High']
                            l = d['Low']
                            v = d['Volume']
                            dt = d['Date']
                            t = d['Symbol']
                            writer.writerow((t, dt, o, c, h, l, v))
                    except:
                        pass
        print('write task completed')

    def start(self):
        self.thread.start()

    def write_row(self, row):
        self.queue.put(Payload(PayloadType.DATA, row))

    def task_completed(self):
        self.queue.put(Payload(PayloadType.TASK_COMPLETED, None))


class Worker:

    def __init__(self, writer, tickers, idx):
        self.idx = idx
        self.writer = writer
        self.tickers = tickers
        thread_func = lambda: self.task()
        self.thread = threading.Thread(target=(thread_func))

    def start(self):
        self.thread.start()

    def task(self):
        shares = {}
        batch_beg = START_DATE
        last_year = 0
        while True:
            if batch_beg.year != last_year:
                last_year = batch_beg.year
                print('{} task - processing {} year'.format(self.idx, last_year))
            # break when all data gathered
            if batch_beg > END_DATE:
                break
            # calc batch end
            batch_end = batch_beg + BATCH_DOWNLOAD_DURATION - ONE_DAY
            # set batch end greater than today
            if batch_end > END_DATE:
                batch_end = END_DATE
            for ticker in self.tickers:
                share = shares.get(ticker)
                if share is None:
                    try:
                        share = Share(ticker)
                        shares[ticker] = share
                    except:
                        print('Can not handle {}'.format(tickers))
                        pass
                s_beg = batch_beg.strftime('%Y-%m-%d')
                s_end = batch_end.strftime('%Y-%m-%d')
                try:
                    data = share.get_historical(s_beg, s_end)
                    # print('Data for %s for period: %s to %s retrieved' % (ticker, s_beg, s_end))
                    self.writer.write_row(data)
                except:
                    # print('Data for %s for period: %s to %s absent' % (ticker, s_beg, s_end))
                    pass

            batch_beg = batch_end + ONE_DAY
        print('{} task completed'.format(self.idx))
        writer.task_completed()

with open('nasdaq_tickers.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = True
    for row in reader:
        if not header:
            ticker = row[0]
            tickers.append(ticker)
        else:
            header = False
print(tickers)
random.shuffle(tickers)

writer = Writer()
writer.start()

tickers_per_worker = len(tickers) // WORKERS
for idx in range(WORKERS):
    tickers_slice = tickers[idx * tickers_per_worker: min((idx + 1) * tickers_per_worker, len(tickers))]
    worker = Worker(writer, tickers_slice, idx)
    worker.start()





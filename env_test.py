import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from random import choice
from time import sleep
from time import time
import os
from timeit import default_timer as timer

from rl_fin.data_reader import DataReader
from rl_fin.data_reader import register_csv_dialect
from rl_fin.env import Environment, Mode, get_ui_thread


def env_thread(dr, wroker_idx):
    env = Environment(dr, Mode.STATE_2D)
    s = env.reset()
    i = 0
    start = timer()
    a = 0
    while True:
        if i % 100 == 0 and i != 0:
            end = timer()
            print(end - start)
            start = end
        if wroker_idx == 0:
            env.render()
        if i % 50 == 0 and i != 0:
            a = 1
        if i % 100 == 0 and i != 0:
            a = 0

        s, r, d, _ = env.step(a)
        if d:
            break
        i += 1
    env.stop()


def _main():
    register_csv_dialect()
    dr = DataReader()
    dr.read_training_data()

    threads = []
    for i in range(1):
        thread_func = lambda: env_thread(dr, i)
        t = threading.Thread(target=(thread_func))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    get_ui_thread().stop()


if __name__ == '__main__':
    _main()

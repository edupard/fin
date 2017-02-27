import threading
import numpy as np
import itertools
import multiprocessing
from timeit import default_timer as timer

from env_factory import startup, shutdown, create_env, stop_env
from config import get_config, EnvironmentType

num_threads = 1
# num_threads = multiprocessing.cpu_count()
render = False

def env_thread(wroker_idx):
    global render
    env = create_env()
    s = env.reset()
    start = timer()
    for i in itertools.count():
        if i % 100 == 0 and i != 0:
            end = timer()
            print('Worker: {} {} fps'.format(wroker_idx, 1000 / (end - start)))
            start = end
        if wroker_idx == 0 and render:
            env.render()
        s, r, d, _ = env.step(1)
        # s, r, d, _ = env.step(np.random.randint(0, 3))
        if d:
            break
    stop_env(env)

def _main():
    startup()

    threads = []
    for i in range(num_threads):
        thread_func = lambda: env_thread(i)
        t = threading.Thread(target=(thread_func))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    shutdown()


if __name__ == '__main__':
    _main()

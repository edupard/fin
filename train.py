import tensorflow as tf
import os
import threading
import multiprocessing

from rl_fin.data_reader import DataReader
from rl_fin.data_reader import register_csv_dialect
from rl_fin.worker import Worker
from rl_fin.ac_network import AC_Network
from rl_fin.config import get_config, EnvType
from rl_fin.env import get_ui_thread

get_ui_thread().start()

load_model = True

dr = None
if get_config().env_type == EnvType.FIN:
    register_csv_dialect()
    dr = DataReader()
    dr.read_training_data()

tf.reset_default_graph()

if not os.path.exists('./models'):
    os.makedirs('./models')

model_path ='./models/{}'.format(get_config().name)
if not os.path.exists(model_path):
    os.makedirs(model_path)
    load_model = False

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network('global', None)  # Generate global network
    num_workers = multiprocessing.cpu_count()  # Set workers ot number of available CPU threads
    workers = []
    # Create worker classes
    if get_config().num_workers is not None:
        num_workers = get_config().num_workers
    print('Creating {} workers'.format(num_workers))
    for i in range(num_workers):
        workers.append(Worker(name=i, dr=dr, trainer=trainer,global_episodes=global_episodes))

    # var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    # var_list.append(global_episodes)
    # saver = tf.train.Saver(max_to_keep=5, var_list = var_list)
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()

    # sess.run(tf.global_variables_initializer())

    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model loaded...')
        else:
            sess.run(tf.global_variables_initializer())
            print('No model to load...')
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)

get_ui_thread().stop()
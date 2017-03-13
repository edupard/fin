#!/usr/bin/env python
import cv2
import tensorflow as tf
import argparse
import logging
import time
import os

from rl.a3c import A3C
from env_factory import create_env, startup, shutdown, stop_env
from config import get_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)


def run(args, server):
    startup()
    env = create_env()

    logdir = os.path.join(get_config().log_dir, 'train')
    summary_writer = tf.summary.FileWriter(logdir + "_%d" % args.task)

    trainer = A3C(env, args.task, summary_writer, args.visualise, args.track)

    # Variable names that start with "local" are not saved in checkpoints.
    variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()
    saver = FastSaver(variables_to_save)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    logger.info('Trainable vars:')
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])

    logger.info("Events directory: %s_%s", logdir, args.task)
    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=trainer.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        sess.run(trainer.sync)
        global_step = sess.run(trainer.global_step)
        if args.track:
            logger.info("Starting evaluation at step=%d", global_step)
        else:
            logger.info("Starting training at step=%d", global_step)
        if args.track:
            trainer.evaluate(sess)
        else:
            while not sv.should_stop() and global_step < get_config().num_global_steps:
                trainer.process(sess)
                global_step = sess.run(trainer.global_step)

    # Ask for all the services to stop.
    sv.stop()
    if args.track:
        logger.info('evaluation complete. worker stopped.')
    else:
        logger.info('reached %s steps. worker stopped.', global_step)
    stop_env(env)
    shutdown()


def cluster_spec(num_workers, num_ps):
    """
More tensorflow setup for data parallelism
"""
    cluster = {}
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster


def main(_):
    """
Setting up Tensorflow for data parallel work
"""

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--visualise', action='store_true', help="Visualise")
    parser.add_argument('--track', action='store_true', help="Track")

    args = parser.parse_args()
    spec = cluster_spec(args.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run(args, server)
    else:
        if not os.path.exists(get_config().log_dir):
            os.makedirs(get_config().log_dir)
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        while True:
            time.sleep(1000)


if __name__ == "__main__":
    tf.app.run()

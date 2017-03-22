from collections import namedtuple
import numpy as np
import tensorflow as tf
from rl.model import LSTMPolicy
import scipy.signal
import csv
import os

from config import StateMode, EnvironmentType, get_config as get_config


def discount_gamma(x):
    return scipy.signal.lfilter(get_config().b_gamma, [1], x[::-1], axis=0)[::-1]


def discount_gamma_lambda(x):
    return scipy.signal.lfilter(get_config().b_gamma_lambda, [1], x[::-1], axis=0)[::-1]


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def process_rollout(rollout):
    """
given a rollout, compute its returns and the advantage
"""
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    if get_config().algo_modification:
        batch_r = discount_gamma(rewards_plus_v)[:-1]
    else:
        batch_r = discount(rewards_plus_v, get_config().gamma)[:-1]
    delta_t = rewards + get_config().gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    if get_config().algo_modification:
        batch_adv = discount_gamma_lambda(delta_t)
    else:
        batch_adv = discount(delta_t, get_config().gamma * get_config()._lambda)

    features = rollout.features[0]

    batch_si = batch_si[:get_config().buffer_length, ]
    batch_a = batch_a[:get_config().buffer_length, ]
    batch_adv = batch_adv[:get_config().buffer_length, ]
    batch_r = batch_r[:get_config().buffer_length, ]
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)


Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])


class Rollout(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []
        self.len = 0

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        self.len += 1

    def is_ready(self):
        return self.len == (get_config().buffer_length + get_config().fwd_buffer_length)


class A3C(object):
    def __init__(self, env, task, summary_writer):
        """
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""

        self.env = env
        self.task = task
        self.summary_writer = summary_writer

        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = LSTMPolicy(env.observation_space.shape, env.action_space.n)
                self.global_step = tf.get_variable("global_step", [], tf.int32,
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = LSTMPolicy(env.observation_space.shape, env.action_space.n)
                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            bs = tf.to_float(tf.shape(pi.x)[0])
            self.loss = pi_loss + 0.5 * vf_loss - entropy * get_config().enthropy_weight

            self.rg = self.rollout_generator()

            grads = tf.gradients(self.loss, pi.var_list)

            tf.summary.scalar("model/policy_loss", pi_loss / bs)
            tf.summary.scalar("model/value_loss", vf_loss / bs)
            tf.summary.scalar("model/entropy", entropy / bs)
            tf.summary.scalar("model/loss", self.loss / bs)
            if get_config().state_mode == StateMode.TWO_D:
                tf.summary.image("model/state", pi.x)
            tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
            tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
            self.summary_op = tf.summary.merge_all()

            grads, _ = tf.clip_by_global_norm(grads, get_config().max_grad_norm)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(get_config().learning_rate)
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.local_steps = 0

    def rollout_generator(self):
        """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
        last_state = self.env.reset()
        last_features = self.local_network.get_initial_features()
        length = 0
        rewards = 0
        pending_rollouts = []

        while True:
            pending_rollouts.append(Rollout())

            for _ in range(get_config().buffer_length):
                fetched = self.local_network.act(last_state, *last_features)
                action, action_distribution, value_, features = fetched[0], fetched[1], fetched[2], fetched[3:]
                # argmax to convert from one-hot
                a = action.argmax()
                state, reward, terminal, info = self.env.step(a)
                if get_config().render:
                    self.env.render()

                # collect the experience
                for rollout in pending_rollouts:
                    rollout.add(last_state, action, reward, value_, terminal, last_features)
                length += 1
                rewards += reward

                last_state = state
                last_features = features

                if pending_rollouts[0].is_ready():
                    rollout = pending_rollouts.pop(0)
                    if not terminal:
                        rollout.r = self.local_network.value(last_state, *last_features)
                        yield rollout

                if terminal:
                    last_state = self.env.reset()
                    last_features = self.local_network.get_initial_features()

                    print("Episode finished. Sum of rewards: %.3f Length: %d" % (rewards, length))
                    if get_config().environment == EnvironmentType.FIN:
                        print('Positions: {} long deals: {} length: {} short deals: {} length: {}'.format(
                            info.long + info.short,
                            info.long,
                            info.long_length,
                            info.short,
                            info.short_length))
                    summary = tf.Summary()
                    summary.value.add(tag='Total reward', simple_value=float(rewards))
                    summary.value.add(tag='Round length', simple_value=float(length))
                    if get_config().environment == EnvironmentType.FIN:
                        summary.value.add(tag='Long deals', simple_value=float(info.long))
                        summary.value.add(tag='Long length', simple_value=float(info.long_length))
                        summary.value.add(tag='Short deals', simple_value=float(info.short))
                        summary.value.add(tag='Short length', simple_value=float(info.short_length))
                        summary.value.add(tag='Positions', simple_value=float(info.long + info.short))
                    self.summary_writer.add_summary(summary, self.local_network.global_step.eval())
                    self.summary_writer.flush()
                    length = 0
                    rewards = 0
                    pending_rollouts.clear()
                    break
                    # once we have enough experience, yield it, and have the ThreadRunner place it on a queue

    def cross_validate(self, sess):
        sess.run(self.sync)
        cv_folder_path = os.path.join('results', get_config().model, 'cv')
        train_folder_path = os.path.join('results', get_config().model, 'train')
        if not os.path.exists(cv_folder_path):
            os.makedirs(cv_folder_path)
        if not os.path.exists(train_folder_path):
            os.makedirs(train_folder_path)
        cv_file_path = os.path.join(cv_folder_path, '{}.csv'.format(get_config().train_seed))
        train_file_path = os.path.join(train_folder_path, '{}.csv'.format(get_config().train_seed))
        with open(cv_file_path, 'w', newline='') as cv_f, open(train_file_path, 'w', newline='') as train_f:
            cv_writer = csv.writer(cv_f, dialect='data')
            train_writer = csv.writer(train_f, dialect='data')

            last_state = self.env.reset()
            last_features = self.local_network.get_initial_features()

            train_rewards = 0.0
            cv_rewards = 0.0
            t_l = 0
            cv_l = 0
            train_length = get_config().train_length

            while True:
                fetched = self.local_network.act(last_state, *last_features)
                action, action_distribution, value_, features = fetched[0], fetched[1], fetched[2], fetched[3:]

                a = action.argmax()
                state, reward, terminal, info = self.env.step(a)
                if get_config().render:
                    self.env.render()

                train_step = t_l <= train_length
                if train_step:
                    t_l += 1
                    train_rewards += reward
                else:
                    cv_l += 1
                    cv_rewards += reward

                row = [info.time,
                       info.price,
                       info.next_time,
                       info.next_price,
                       info.ccy,
                       info.ccy_c,
                       info.pct,
                       info.pct_c,
                       info.lr,
                       info.lr_c,
                       a]
                row.extend(value_)
                row.extend(action_distribution.reshape((-1)))
                if train_step:
                    train_writer.writerow(row)
                else:
                    cv_writer.writerow(row)

                last_state = state
                last_features = features

                if terminal:
                    break
            print("Episode finished. Train rewards: %.3f. Cv rewards: %.3f Train length: %d Cv length: %d" % (
                train_rewards, cv_rewards, t_l, cv_l))

    def process(self, sess):
        sess.run(self.sync)  # copy weights from shared to local

        rollout = next(self.rg)

        batch = process_rollout(rollout)

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            self.local_network.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            self.local_network.state_in[0]: batch.features[0],
            self.local_network.state_in[1]: batch.features[1],
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1

import tensorflow as tf
import numpy as np
import gym
from PIL import Image
import cv2

from rl_fin.utils import update_target_graph
from rl_fin.ac_network import AC_Network
from rl_fin.utils import discount
from rl_fin.config import get_config, EnvType
from rl_fin.env import Environment, Mode
from rl_fin.data_reader import DataReader
from rl_fin.env_factory import create_env

def _preprocess_pong_frame(frame):
    frame = frame[34:34+160, :160]
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [-1, 42, 42, 1])
    return frame


# def _preprocess_pong_frame(s):
#     # crop
#     s = s[34:194]
#     # debug check
#     # image = Image.fromarray(s)
#     # image.save('state.png')
#     # normalize
#     s = s.astype(np.float) / 255.0
#     s = s.reshape((-1, 160, 160, 3))
#     return s


class Worker():
    def __init__(self, name, dr: DataReader, trainer, global_episodes):
        self._dr = dr

        self.name = "worker_" + str(name)
        self.number = name
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_deals = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.model_path = './models/{}'.format(get_config().name)
        summary_path = self.model_path + '/train_' + str(self.number)
        self.summary_writer = tf.summary.FileWriter(summary_path)

        # Environment setup
        self._env = create_env(dr)

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(self.name, trainer, self._env)
        self.update_local_ops = update_target_graph('global', self.name)

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n



    def work(self, sess, coord, saver):


        episode_count = sess.run(self.global_episodes)
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

                episode_buffer = []
                episode_values = []
                episode_reward = 0
                a_prev = 1
                ep_deals = 0
                d = False
                s = env.reset()
                if get_config().env_type == EnvType.PONG:
                    s = _preprocess_pong_frame(s)
                rnn_state = self.local_AC.state_init

                while not d:
                    if get_config().num_workers == 1:
                        env.render()
                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs: s,
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    btn = a
                    if get_config().env_type == EnvType.PONG:
                        btn += 1
                    if a != a_prev:
                        ep_deals += 1
                        a_prev = a

                    s1, r, d, _ = env.step(btn)
                    if get_config().env_type == EnvType.PONG:
                        s1 = _preprocess_pong_frame(s1)

                    episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1

                    if len(episode_buffer) == get_config().state_buffer_size:
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.inputs: s,
                                                 self.local_AC.state_in[0]: rnn_state[0],
                                                 self.local_AC.state_in[1]: rnn_state[1]})[0, 0]
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, get_config().gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)

                # Since we don't know what the true final return is, we "bootstrap" from our current
                # value estimation.
                if len(episode_buffer) != 0:
                    v1 = 0.
                    # v1 = sess.run(self.local_AC.value,
                    #               feed_dict={self.local_AC.inputs: s,
                    #                          self.local_AC.state_in[0]: rnn_state[0],
                    #                          self.local_AC.state_in[1]: rnn_state[1]})[0, 0]

                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, get_config().gamma, v1)

                self.episode_deals.append(ep_deals)
                self.episode_rewards.append(episode_reward)
                self.episode_mean_values.append(np.mean(episode_values))

                print('worker {}: reward: {}'.format(self.number, episode_reward))

                if episode_count % 20 == 0 and self.name == 'worker_0':
                    saver.save(sess, self.model_path + '/model.ckpt', global_step=self.global_episodes)
                    print("Saved Model")

                if episode_count % 5 == 0:
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    mean_deals = np.mean(self.episode_deals[-5:])

                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    if get_config().env_type == EnvType.FIN:
                        summary.value.add(tag='Perf/Deals', simple_value=float(mean_deals))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()

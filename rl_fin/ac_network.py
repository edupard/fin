import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as d

import numpy as np

from rl_fin.utils import normalized_columns_initializer
from rl_fin.config import get_config, EnvType


class AC_Network():
    def __init__(self, scope, trainer):
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            if get_config().env_type == EnvType.PONG:
                # self.inputs = tf.placeholder(
                #     shape=[None, 160, 160, 3],
                #     dtype=tf.float32)
                self.inputs = tf.placeholder(
                    shape=[None, 42, 42, 1],
                    dtype=tf.float32)
            elif get_config().env_type == EnvType.FIN:
                self.inputs = tf.placeholder(
                    shape=[None, get_config().window_px_width, get_config().window_px_height, 3],
                    dtype=tf.float32)
            self.x = x = self.inputs
            # self.x = x = tf.reshape(self.inputs,
            #                         shape=[-1, get_config().window_px_width, get_config().window_px_height, 3])
            for i in range(get_config().conv_layers):
                x = slim.conv2d(activation_fn=tf.nn.elu,
                                inputs=x, num_outputs=32,
                                kernel_size=[3, 3], stride=[2, 2])

            # self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
            #                          inputs=self.x, num_outputs=32,
            #                          kernel_size=[8, 1], stride=[4, 4], padding='VALID')
            # self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
            #                          inputs=self.conv1, num_outputs=32,
            #                          kernel_size=[3, 1], stride=[3, 3], padding='VALID')
            # self.conv3 = slim.conv2d(activation_fn=tf.nn.elu,
            #                          inputs=self.conv2, num_outputs=32,
            #                          kernel_size=[3, 1], stride=[2, 2], padding='VALID')
            # self.conv4 = slim.conv2d(activation_fn=tf.nn.elu,
            #                          inputs=self.conv3, num_outputs=32,
            #                          kernel_size=[3, 1], stride=[2, 2], padding='VALID')
            # hidden = slim.fully_connected(slim.flatten(x), 256, activation_fn=tf.nn.elu)
            hidden = slim.flatten(x)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.x)[:1]
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            # Output layers for policy and value estimations
            n_actions = get_config().n_actions
            if get_config().env_type == EnvType.PONG:
                n_actions = 3
            self.policy = slim.fully_connected(rnn_out, n_actions,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)

            c = d.Categorical(self.policy)


            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, n_actions, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.1

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, get_config().max_grad_norm)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

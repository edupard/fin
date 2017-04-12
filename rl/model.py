import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from config import get_config, EnvironmentType, StateMode


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


def max_sample(logits, d):
    logits_normalized = logits - tf.reduce_max(logits, [1], keep_dims=True)
    value = tf.argmax(logits_normalized, 1)
    return tf.one_hot(value, d)


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        if get_config().environment == EnvironmentType.PONG:
            for i in range(4):
                x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))

        if get_config().environment == EnvironmentType.FIN:
            if get_config().state_mode == StateMode.TWO_D:
                i = 0
                for k_w, k_h, s_w, s_h, f in get_config().conv_layers_2d:
                    x = tf.nn.elu(conv2d(x, f, "l{}".format(i + 1), [k_w, k_h], [s_w, s_h]))
                    i += 1
            elif get_config().state_mode == StateMode.ONE_D or get_config().state_mode == StateMode.RETURNS:
                i = 0
                for k, s, f in get_config().conv_layers_1d:
                    x = tf.nn.elu(conv2d(x, f, "l{}".format(i + 1), [k, 1], [s, 1]))
                    i += 1
        x = flatten(x)
        self.pos = tf.placeholder(tf.float32, [None, 1])
        if get_config().propogate_position_to_rnn:
            x = tf.concat(1, [self.pos, x])

        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(x, [0])

        size = 255
        if get_config().environment == EnvironmentType.FIN:
            if get_config().state_mode == StateMode.TWO_D:
                size = get_config().rnn_2d_size
            if get_config().state_mode == StateMode.ONE_D or get_config().state_mode == StateMode.RETURNS:
                size = get_config().rnn_1d_size

        lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        if not get_config().is_test_mode():
            keep_prob = get_config().keep_prob
            lstm = tf.nn.rnn_cell.DropoutWrapper(
                lstm, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        if get_config().is_test_mode():
            self.sample = max_sample(self.logits, ac_space)[0, :]
        else:
            self.sample = categorical_sample(self.logits, ac_space)[0, :]
        # self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.sample_distribution = tf.nn.softmax(self.logits)
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, pos, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.sample_distribution, self.vf] + self.state_out,
                        {self.x: [ob], self.pos: [np.array([pos]).reshape((1))], self.state_in[0]: c,
                         self.state_in[1]: h})

    def value(self, ob, pos, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.pos: [np.array([pos])], self.state_in[0]: c, self.state_in[1]: h})[
            0]

import tensorflow as tf
from utilsnn import xavier_init


class FFNN(object):
    def __init__(self, input_size, layer_sizes, layer_names, optimizer=tf.train.AdamOptimizer(),
                 transfer_function=tf.nn.sigmoid):

        self.layer_names = layer_names

        # Build the encoding layers
        self.x = tf.placeholder("float", [None, input_size])
        next_layer_input = self.x

        assert len(layer_sizes) == len(layer_names)

        self.encoding_matrices = []
        self.encoding_biases = []
        for i in range(len(layer_sizes)):
            dim = layer_sizes[i]
            input_dim = int(next_layer_input.get_shape()[1])

            # Initialize W using xavier initialization
            # W = tf.get_variable(name=layer_names[i][0],shape=(input_dim, dim),dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            W = tf.Variable(xavier_init(input_dim, dim, transfer_function), name=layer_names[i][0])

            # Initialize b to zero
            b = tf.Variable(tf.zeros([dim]), name=layer_names[i][1])

            # We are going to use tied-weights so store the W matrix for later reference.
            self.encoding_matrices.append(W)
            self.encoding_biases.append(b)

            output = transfer_function(tf.matmul(next_layer_input, W) + b)

            # the input into the next layer is the output of this layer
            next_layer_input = output

        # The fully encoded x value is now stored in the next_layer_input
        self.encoded_x = next_layer_input

        # Feed forward net
        self.ff_matrices = []
        self.ff_biases = []
        # W = tf.get_variable(name="ffw1", shape=(4, 50), dtype=tf.float32,
        #                     initializer=tf.contrib.layers.xavier_initializer())
        W = tf.Variable(xavier_init(4, 50, transfer_function), name="ffw1")
        b = tf.Variable(tf.zeros([50]), name="ffb1")
        self.ff_matrices.append(W)
        self.ff_biases.append(b)
        output = transfer_function(tf.matmul(next_layer_input, W) + b)
        next_layer_input = output

        # W = tf.get_variable(name="ffw2", shape=(50, 2), dtype=tf.float32,
        #                     initializer=tf.contrib.layers.xavier_initializer())
        W = tf.Variable(xavier_init(50, 2, transfer_function), name="ffw2")
        b = tf.Variable(tf.zeros([2]), name="ffb2")
        self.ff_matrices.append(W)
        self.ff_biases.append(b)
        self.output = tf.nn.softmax(tf.matmul(next_layer_input, W) + b)

        self.y = tf.placeholder(tf.float32, shape=(None, 2))

        # compute cost
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.output), reduction_indices=[1]))
        self.optimizer = optimizer.minimize(self.cost)

        # initalize variables
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def transform(self, X):
        return self.sess.run(self.output, {self.x: X})

    def load_au_weights(self, path, layer_names, layer):
        saver = tf.train.Saver({layer_names[0]: self.encoding_matrices[layer]},
                               {layer_names[1]: self.encoding_biases[layer]})
        saver.restore(self.sess, path)

    def load_weights(self, path):
        dict_w = self.get_dict_layer_names()
        saver = tf.train.Saver(dict_w)
        saver.restore(self.sess, path)

    def save_weights(self, path):
        dict_w = self.get_dict_layer_names()
        saver = tf.train.Saver(dict_w)
        save_path = saver.save(self.sess, path)

    def get_dict_layer_names(self):
        dict_w = {}
        for i in range(len(self.layer_names)):
            dict_w[self.layer_names[i][0]] = self.encoding_matrices[i]
            dict_w[self.layer_names[i][1]] = self.encoding_biases[i]
        dict_w["ffw1"] = self.ff_matrices[0]
        dict_w["ffb1"] = self.ff_biases[0]
        dict_w["ffw2"] = self.ff_matrices[1]
        dict_w["ffb2"] = self.ff_biases[1]

        return dict_w

    def partial_fit(self, X, Y):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.y: Y})
        return cost

    def predict(self, X):
        return self.sess.run((self.output), feed_dict={self.x: X})

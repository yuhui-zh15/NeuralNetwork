import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class BasicRNNCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        # inputs: [batch_size, embed_units]
        # state: [batch_size, num_units]
        with tf.variable_scope(scope or "basic_rnn_cell", reuse=self._reuse):
            #todo: implement the new_state calculation given inputs and state
            W = tf.get_variable('W', [FLAGS.embed_units + self._num_units, self._num_units])
            b = tf.get_variable('b', [self._num_units], initializer=tf.constant_initializer(0.0))    
            new_state = self._activation(tf.matmul(tf.concat([inputs, state], axis=1), W) + b)
        return new_state, new_state

class GRUCell(tf.contrib.rnn.RNNCell):
    '''Gated Recurrent Unit cell (http://arxiv.org/abs/1406.1078).'''

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        # inputs: [batch_size, embed_units]
        # state: [batch_size, num_units]
        with tf.variable_scope(scope or "gru_cell", reuse=self._reuse):
            #We start with bias of 1.0 to not reset and not update.
            #todo: implement the new_h calculation given inputs and state
            W_zr = tf.get_variable('W_zr', [FLAGS.embed_units + self._num_units, self._num_units * 2])
            b_zr = tf.get_variable('b_zr', [self._num_units * 2], initializer=tf.constant_initializer(1.0))
            z, r = tf.split(value=tf.sigmoid(tf.matmul(tf.concat([inputs, state], axis=1), W_zr) + b_zr), num_or_size_splits=2, axis=1)
            W = tf.get_variable('W', [FLAGS.embed_units + self._num_units, self._num_units])
            b = tf.get_variable('b', [self._num_units], initializer=tf.constant_initializer(0.0))
            h = self._activation(tf.matmul(tf.concat([inputs, r * state], axis=1), W) + b)
            new_h = z * state + (1 - z) * h
        return new_h, new_h

class BasicLSTMCell(tf.contrib.rnn.RNNCell):
    '''Basic LSTM cell (http://arxiv.org/abs/1409.2329).'''

    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        # inputs: [batch_size, embed_units]
        # state: [[batch_size, num_units], [batch_size, num_units]]
        with tf.variable_scope(scope or "basic_lstm_cell", reuse=self._reuse):
            c, h = state
            #For forget_gate, we add forget_bias of 1.0 to not forget in order to reduce the scale of forgetting in the beginning of the training.
            #todo: implement the new_c, new_h calculation given inputs and state (c, h)
            W = tf.get_variable('W', [FLAGS.embed_units + self._num_units, self._num_units * 4])
            b = tf.get_variable('b', [self._num_units * 4], initializer=tf.constant_initializer(0.0))
            f, i, o, c_tilde = tf.split(value=tf.matmul(tf.concat([inputs, h], axis=1), W) + b, num_or_size_splits=4, axis=1)
            new_c = tf.sigmoid(f + self._forget_bias) * c + tf.sigmoid(i) * self._activation(c_tilde)
            new_h = tf.sigmoid(o) * self._activation(new_c)
        return new_h, (new_c, new_h)

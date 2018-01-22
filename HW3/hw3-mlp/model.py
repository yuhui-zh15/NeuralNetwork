# -*- coding: utf-8 -*-

import tensorflow as tf


class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 28*28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        # TODO:  implement input -- Linear -- BN -- ReLU -- Linear -- loss
        #        the 10-class prediction output is named as "logits"
        logits = self.x_ # Input
        logits = linear_layer(logits, 784, 400) # Linear
        logits = batch_normalization_layer(logits, is_train) # BN
        logits = relu_layer(logits) # Relu
        # logits = dropout_layer(logits, self.keep_prob) # Dropout
        logits = linear_layer(logits, 400, 10) # Linear

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.y_)
        self.pred = tf.argmax(logits, 1)  # Calculate the prediction result
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)  # Use Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_normalization_layer(inputs, isTrain=True):
    # TODO: implemented the batch normalization func and applied it on fully-connected layers
    mean_total, variance_total = tf.Variable(tf.zeros(inputs.get_shape()[-1]), trainable=False), tf.Variable(tf.zeros(inputs.get_shape()[-1]), trainable=False)
    mean, variance = tf.nn.moments(inputs, [0])
    decay = tf.constant(0.9)
    gamma = tf.Variable(tf.ones(inputs.get_shape()[-1]))
    beta = tf.Variable(tf.zeros(inputs.get_shape()[-1]))
    epsilon = tf.constant(1e-3)
    if isTrain:
        assign_mean = tf.assign(mean_total, mean_total * decay + mean * (1 - decay))
        assign_variance = tf.assign(variance_total, variance_total * decay + variance * (1 - decay))
        with tf.control_dependencies([assign_mean, assign_variance]):
            norm = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
    else: norm = tf.nn.batch_normalization(inputs, mean_total, variance_total, beta, gamma, epsilon)
    return norm

def linear_layer(inputs, input_size, output_size, isTrain=True):
    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    return tf.matmul(inputs, W) + b

def relu_layer(inputs, isTrain=True):
    return tf.nn.relu(inputs)
    
def dropout_layer(inputs, keep_prob, isTrain=True):
    return tf.nn.dropout(inputs, keep_prob)

# -*- coding: utf-8 -*-

import math
import tensorflow as tf


class MLPClassifier(object):
    """
    Multi-layer perceptron classifier.

    :param hidden_units: Number of hidden units at each layer.
    :param activation: Activation function to use at each hidden layer.
                       Possible values are: relu, sigmoid, tanh.  (Default relu)
    :param dropout: Probability to drop out units.  (Default None)
    :param optimizer: Optimizer to train the model.
                      Possible values are: SGD, Adam, Adagrad.  (Default None)
    :param learning_rate: Learning rate at each training step.  (Default 0.1).
    :param momentum: Momentum at each training step.  (Default 0.9).
    :param batch_size: Batch size at each training step.  (Default 32).
    """

    def __init__(self,
                 hidden_units,
                 activation='relu',
                 dropout=None,
                 optimizer='SGD',
                 learning_rate=0.1,
                 momentum=0.9,
                 batch_size=32):
        if activation.lower() not in ('relu', 'sigmoid', 'tanh'):
            raise ValueError('Unknown activation function %s.' % repr(activation))
        if optimizer.lower() not in ('sgd', 'adam', 'adagrad'):
            raise ValueError('Unknown optimizer %s.' % repr(optimizer))

        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size

    def _build_model(self, input_dim, target_num):
        prev_dim = input_dim

        # build all hidden layers.
        hidden_params = []
        for idx, size in enumerate(self.hidden_units, 1):
            with tf.name_scope('hidden%d' % idx):
                weights = tf.get_variable(
                    'weights',
                    [prev_dim, size],
                    initializer=tf.random_normal_initializer(
                        mean=0.0,
                        stddev=1.0 / math.sqrt(prev_dim)
                    )
                )
                biases = tf.get_variable(
                    'biases',
                    [prev_dim],
                    initializer=tf.constant_initializer(0.0)
                )
            hidden_params.append((weights, biases))
            prev_dim = size

            # build softmax layer.

    def __getstate__(self):
        states = self.__dict__.copy()
        for k in ('_model',):
            del states[k]
        return states

    def __setstate__(self, states):
        self.__dict__.update(states)

    def fit(self, X, Y):
        """
        Train the model with :param:`X` and :param:`Y`.

        :param X: 2D real-number array as the training input.
        :param Y: 2D integer label as the training label.
        :return: self
        """
        return self

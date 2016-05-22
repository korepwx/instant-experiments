# -*- coding: utf-8 -*-

import math
import time
from itertools import chain

import numpy as np
import tensorflow as tf

from ipwxlearn.utility import FilteredPickleSupport, safe_reduce, MiniBatchIterator


class BaseLayer(object):
    """
    Base layer for MLP.

    :param name: Name of this layer.
    :param inputs: Input tensor.
    :param n_in: Input dimension.
    :param n_out: Output dimension.
    """

    def __init__(self, name, inputs, n_in, n_out):
        self.name = name
        self.inputs = inputs
        self.n_in = n_in
        self.n_out = n_out

        with tf.name_scope(name):
            self.W = tf.get_variable(
                'weights',
                shape=[n_in, n_out],
                initializer=tf.truncated_normal_initializer(
                    mean=0.0,
                    stddev=1.0 / math.sqrt(n_in)
                )
            )
            self.b = tf.get_variable('biases', shape=[n_in], initializer=tf.constant_initializer(0.0))

    def get_params(self):
        return [self.W, self.b]

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, repr(self.name))


class HiddenLayer(BaseLayer):

    def __init__(self, activation, name, inputs, n_in, n_out):
        super(HiddenLayer, self).__init__(name, inputs, n_in, n_out)
        self.activation = activation
        with tf.name_scope(self.name):
            self.outputs = activation(tf.matmul(inputs, self.W) + self.b, name='outputs')


class SoftmaxLayer(BaseLayer):

    def __init__(self, name, inputs, n_in, n_out):
        super(SoftmaxLayer, self).__init__(name, inputs, n_in, n_out)
        self.logits = tf.matmul(inputs, self.W) + self.b
        with tf.name_scope(self.name):
            self.proba = tf.nn.softmax(self.logits, name='proba')
            self.log_proba = tf.nn.log_softmax(self.logits, name='log_proba')
            self.predicts = tf.arg_max(self.logits, dimension=1, name='predicts')

    def loss(self, labels):
        """
        Compute the loss for the given :param:`labels`.

        :param labels: Labels tensor, int32 - [batch_size].
        :return: Loss tensor of type float.
        """
        with tf.name_scope(self.name):
            labels = tf.to_int64(labels)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, labels, name='cross_entropy')
            loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        return loss

    def error(self, labels):
        """
        Compute the error for the given :param:`labels`.

        :param labels: Labels tensor, int32 - [batch_size].
        :return: A scalar int32 tensor with the number of examples (out of batch_size) that were predicted correctly.
        """
        correct = tf.nn.in_top_k(self.logits, labels, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))


class MLPClassifier(FilteredPickleSupport):
    """
    Multi-layer perceptron classifier.

    :param hidden_units: Number of hidden units at each layer.
    :param activation: Activation function to use at each hidden layer.
                       Possible values are: relu, sigmoid, tanh.  (Default relu)
    :param dropout: Probability to drop out units.  (Default None)
    :param l1_reg: L1 regularization factor.
    :param l2_reg: L2 regularization factor.
    :param optimizer: Optimizer to train the model.
                      Possible values are: SGD, Adam, Adagrad.  (Default None)
    :param learning_rate: Learning rate at each training step.  (Default 0.1).
    :param momentum: Momentum at each training step.  (Default 0.9).
    :param batch_size: Batch size at each training step.  (Default 32).
    :param max_steps: Maximum number of steps to run trainer.
    """

    _UNPICKABLE_FIELDS_ = ('_graph', '_inputs_ph', '_labels_ph', '_hidden', '_softmax', '_loss', '_error',
                           '_optimizer')
    ACTIVATIONS = {
        'relu': tf.nn.relu,
        'sigmoid': tf.nn.sigmoid,
        'tanh': tf.nn.tanh
    }

    def __init__(self,
                 hidden_units,
                 activation='relu',
                 dropout=None,
                 l1_reg=0.0,
                 l2_reg=0.001,
                 optimizer='SGD',
                 learning_rate=0.1,
                 momentum=0.9,
                 batch_size=32,
                 max_steps=2000,
                 validation_steps=1000):
        if activation.lower() not in self.ACTIVATIONS:
            raise ValueError('Unknown activation function %s.' % repr(activation))
        if optimizer.lower() not in ('sgd', 'adam', 'adagrad'):
            raise ValueError('Unknown optimizer %s.' % repr(optimizer))

        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.max_steps = max_steps

        self._input_dim = self._target_num = None
        self._graph = self._inputs_ph = self._labels_ph = self._hidden = self._softmax = self._loss = self._error = \
            self._optimizer = None

    def dispose(self):
        """Dispose all the TensorFlow objects created by this class."""
        if self._graph is not None:
            self._graph.finalize()
        for k in self._UNPICKABLE_FIELDS_:
            setattr(self, k, None)

    def _build_model(self):
        self.dispose()

        self._graph = tf.Graph()
        with self._graph.as_default():
            # make placeholders for inputs and labels.
            self._inputs_ph = inputs_ph = \
                tf.placeholder(tf.float32, shape=(self.batch_size, self._input_dim), name='inputs')
            self._labels_ph = labels_ph = \
                tf.placeholder(tf.int32, shape=(self.batch_size,), name='labels')

            # label building parameters.
            prev_v = inputs_ph
            prev_n = self._input_dim

            # build all hidden layers.
            hidden = []
            for idx, n_out in enumerate(self.hidden_units, 1):
                layer = HiddenLayer(self.ACTIVATIONS[self.activation], 'hidden%d' % idx, prev_v, prev_n, n_out)
                hidden.append(layer)
                prev_v = layer.outputs
                prev_n = n_out
            self._hidden = hidden

            # build softmax layer.
            self._softmax = softmax = SoftmaxLayer('softmax', prev_v, prev_n, self._target_num)

            # build loss operation.
            loss = softmax.loss(labels_ph)
            params = list(chain(*[l.get_params() for l in chain(softmax, *hidden)]))
            if params:
                if not np.isclose([self.l1_reg], [0.0]):
                    loss += self.l1_reg * safe_reduce(lambda x, y: x + y, (tf.reduce_sum(tf.abs(p)) for p in params))
                if not np.isclose([self.l2_reg], [0.0]):
                    loss += self.l2_reg * safe_reduce(lambda x, y: x + y, (tf.nn.l2_loss(p) for p in params))
            self._loss = loss

            # build the evaluation operation.
            self._error = softmax.error(labels_ph)

    def __setstate__(self, states):
        super(MLPClassifier, self).__setstate__(states)
        self._build_model()

    def fit(self, x, y):
        """
        Train the model with :param:`x` and :param:`y`.

        :param x: 2D real-number array as the training input.
        :param y: 2D integer label as the training label.
        :return: self
        """
        # check the data.
        assert(len(x.shape) == 2 and len(y.shape) == 1 and y.dtype == np.int32)
        assert(np.min(y) >= 0)
        self._input_dim = x.shape[1]
        self._target_num = np.max(y) + 1

        # build the model.
        self._build_model()

        # split train-validation set.
        indices = np.arange(x.shape[0])

        # create the mini-batch iterator.
        mini_batch = MiniBatchIterator(x, y)

        # construct the trainer.
        with self._graph.as_default():
            # training operation.
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(self._loss, global_step=global_step)

            # initialization operation.
            init_op = tf.initialize_all_variables()

            # start the session to train the model.
            with tf.Session() as sess:
                sess.run(init_op)
                for step in range(self.max_steps):
                    start_time = time.time()

                    # do gradient descent.
                    inputs, labels = mini_batch.next_batch(self.batch_size)
                    _, loss_value = sess.run([train_op, self._softmax.loss], feed_dict={
                        self._inputs_ph: inputs,
                        self._labels_ph: labels,
                    })

                    # do evaluation if necessary.
                    if (step + 1) % self.validation_steps == 0 or (step + 1) == self.max_steps:
                        pass

                    duration = time.time() - start_time

        return self

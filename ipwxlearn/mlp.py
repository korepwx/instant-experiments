# -*- coding: utf-8 -*-

import time
from itertools import chain

import numpy as np
import tensorflow as tf

from ipwxlearn.utility import FilteredPickleSupport, safe_reduce, TrainingBatchIterator, TestingBatchIterator


class BaseLayer(object):
    """
    Base layer for MLP.

    :param name: Name of this layer.
    :param inputs: Input tensor.
    :param n_in: Input dimension.
    :param n_out: Output dimension.
    """

    def __init__(self, name, inputs, n_in, n_out, weight_initializer=None, bias_initializer=None):
        weight_initializer = weight_initializer or tf.contrib.layers.xavier_initializer()
        bias_initializer = bias_initializer or tf.constant_initializer(0.0)

        self.name = name
        self.inputs = inputs
        self.n_in = n_in
        self.n_out = n_out

        with tf.variable_scope(name):
            self.W = tf.get_variable('weights', shape=[n_in, n_out], initializer=weight_initializer)
            self.b = tf.get_variable('biases', shape=[n_out], initializer=bias_initializer)

    def get_params(self):
        return [self.W, self.b]

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, repr(self.name))


class HiddenLayer(BaseLayer):

    def __init__(self, activation, name, inputs, n_in, n_out):
        super(HiddenLayer, self).__init__(name, inputs, n_in, n_out)
        self.activation = activation
        with tf.variable_scope(self.name):
            self.outputs = activation(tf.matmul(inputs, self.W) + self.b, name='outputs')


class SoftmaxLayer(BaseLayer):

    def __init__(self, name, inputs, n_in, n_out):
        super(SoftmaxLayer, self).__init__(name, inputs, n_in, n_out)
        self.logits = tf.matmul(inputs, self.W) + self.b
        with tf.variable_scope(self.name):
            self.proba = tf.nn.softmax(self.logits, name='proba')
            self.log_proba = tf.nn.log_softmax(self.logits, name='log_proba')
            self.predicts = tf.cast(tf.arg_max(self.logits, dimension=1, name='predicts'), tf.int32)

    def loss(self, labels):
        """
        Compute the loss for the given :param:`labels`.

        :param labels: Labels tensor, int32 - [batch_size].
        :return: Loss tensor of type float.
        """
        with tf.variable_scope(self.name):
            labels = tf.to_int64(labels)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, labels, name='xentropy')
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return loss

    def correct(self, labels):
        """
        Compute the correctness for the given :param:`labels`.

        :param labels: Labels tensor, int32 - [batch_size].
        :return: A scalar int32 tensor with the number of examples (out of batch_size) that were predicted correctly.
        """
        correct = tf.equal(self.predicts, labels)
        return tf.reduce_sum(tf.cast(correct, tf.int32))


class MLPClassifier(FilteredPickleSupport):
    """
    Multi-layer perceptron classifier.

    :param hidden_units: Number of hidden units at each layer.
    :param activation: Activation function to use at each hidden layer.
                       Possible values are: relu, sigmoid, tanh.  (Default relu)
    :param dropout: Probability to drop out units at each hidden layer.  (Default None)
    :param l1_reg: L1 regularization factor.
    :param l2_reg: L2 regularization factor.
    :param optimizer: Optimizer to train the model.
                      Possible values are: SGD, Adam, Adagrad.  (Default None)
    :param learning_rate: Learning rate at each training step.  (Default 0.1).
    :param momentum: Momentum at each training step.  (Default 0.9).
    :param batch_size: Batch size at each training step.  (Default 32).
    :param max_steps: Maximum number of steps to run trainer.
    :param validation_steps: Run evaluation every this number of steps.
                             If not given, will be set to int(n_examples / batch_size).
    :param validation_split: Split the training set to training/validation set by the portion of
                             (1-validation_split)/validation_split.
    """

    _UNPICKABLE_FIELDS_ = ('_graph', '_inputs_ph', '_labels_ph', '_hidden', '_softmax', '_loss', '_correct',
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
                 l2_reg=0.0001,
                 optimizer='SGD',
                 learning_rate=0.01,
                 momentum=0.9,
                 batch_size=64,
                 max_steps=2000,
                 validation_steps=None,
                 validation_split=0.1):
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
        self.validation_steps = validation_steps
        self.validation_split = validation_split

        self._input_dim = self._target_num = self._param_values = None
        for k in self._UNPICKABLE_FIELDS_:
            setattr(self, k, None)

    def dispose(self):
        """Dispose all the TensorFlow objects created by this class."""
        if self._graph is not None:
            self._graph.finalize()
        for k in self._UNPICKABLE_FIELDS_:
            setattr(self, k, None)

    def get_params(self):
        return {v.name: v for v in chain(*(l.get_params() for l in [self._softmax] + self._hidden))}

    def _get_param_values(self, sess):
        kp = list(self.get_params().items())
        values = sess.run([v for k, v in kp])
        return {k: v for (k, _), v in zip(kp, values)}

    def _set_param_values(self, values, sess):
        kp = list(self.get_params().items())
        op = [p.assign(values[k]) for (k, p), v in zip(kp, values)]
        sess.run(op)

    def _build_model(self):
        self.dispose()

        self._graph = tf.Graph()
        with self._graph.as_default():
            # make placeholders for inputs and labels, and n_samples variable.
            self._inputs_ph = inputs_ph = \
                tf.placeholder(tf.float32, shape=(None, self._input_dim), name='inputs')
            self._labels_ph = labels_ph = \
                tf.placeholder(tf.int32, shape=(None,), name='labels')

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
            params = list(chain(*[l.get_params() for l in [softmax] + hidden]))
            if params:
                if not np.isclose([self.l1_reg], [0.0]):
                    loss += self.l1_reg * safe_reduce(lambda x, y: x + y, [tf.reduce_sum(tf.abs(p)) for p in params])
                if not np.isclose([self.l2_reg], [0.0]):
                    loss += self.l2_reg * safe_reduce(lambda x, y: x + y, [tf.nn.l2_loss(p) for p in params])
            self._loss = loss

            # build the evaluation operation.
            self._correct = softmax.correct(labels_ph)

    def __setstate__(self, states):
        super(MLPClassifier, self).__setstate__(states)
        self._build_model()

    def _compute_error(self, sess, x, y):
        correct = 0
        for x_batch, y_batch in TestingBatchIterator(x, y).iter_batches(self.batch_size):
            correct += sess.run(self._correct, feed_dict={
                self._inputs_ph: x_batch,
                self._labels_ph: y_batch,
            })
        return 1.0 - float(correct) / x.shape[0]

    def fit(self, x, y, warm_start=False):
        """
        Train the model with :param:`x` and :param:`y`.

        :param x: 2D real-number array as the training input.
        :param y: 2D integer label as the training label.
        :param warm_start: Whether or not to warm start with the previous parameters.

        :return: self
        """
        # check the data.
        assert(len(x.shape) == 2)
        assert(len(y.shape) == 1)
        assert(x.shape[0] == y.shape[0])
        assert(str(y.dtype).find('int') >= 0)
        assert(np.min(y) >= 0)

        self._input_dim = x.shape[1]
        self._target_num = np.max(y) + 1
        x = x.astype(np.float32)
        y = y.astype(np.int32)

        # build the model.
        self._build_model()

        # split training/validation set.
        n_train = int(np.rint(self.validation_split * x.shape[0]))
        n_valid = x.shape[0] - n_train
        assert(n_train > 0)
        assert(n_valid > 0)

        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[: n_train], indices[n_train: ]
        train_x, train_y = x[train_idx], y[train_idx]
        valid_x, valid_y = x[valid_idx], y[valid_idx]

        # initialize mini-batch related controllers.
        mini_batch = TrainingBatchIterator(train_x, train_y)
        validation_steps = self.validation_steps or max(int(n_train / self.batch_size), 1)

        # do early-stopping.
        best_param_values = None
        best_valid_error = 1.0

        # construct the trainer.
        try:
            with self._graph.as_default():
                # training operation.
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum)
                global_step = tf.Variable(0, name='global_step', trainable=False)
                train_op = optimizer.minimize(self._loss, global_step=global_step)

                # start the session to train the model.
                with tf.Session() as sess:
                    # initialize parameters.
                    if warm_start and self._param_values is not None:
                        self._set_param_values(self._param_values, sess)
                    else:
                        sess.run(tf.initialize_all_variables())

                    # do mini-batches.
                    for step in range(1, self.max_steps + 1):
                        start_time = time.time()

                        # do gradient descent.
                        inputs, labels = mini_batch.next_batch(self.batch_size)
                        _, loss_value = sess.run([train_op, self._loss], feed_dict={
                            self._inputs_ph: inputs,
                            self._labels_ph: labels,
                        })

                        # do evaluation if necessary.
                        if step % validation_steps == 0 or step == self.max_steps:
                            train_error = self._compute_error(sess, train_x, train_y)
                            valid_error = self._compute_error(sess, valid_x, valid_y)
                            is_best = valid_error < best_valid_error
                            if is_best:
                                best_param_values = self._get_param_values(sess)
                                best_valid_error = valid_error

                            print('Step %d: loss %s, train_error %s, valid_error %s.%s' %
                                  (step, loss_value, train_error, valid_error, ' (*)' if is_best else ''))

                        duration = time.time() - start_time
        finally:
            if best_param_values is not None:
                self._param_values = best_param_values
        return self

    def predict(self, x):
        """
        Predict :param:`x` with trained model.

        :param x: 2D real-number array as the training input.
        :return: Predicted label for the x.
        """
        with self._graph.as_default():
            with tf.Session() as sess:
                self._set_param_values(self._param_values, sess)
                predicts = []
                for x_batch in TestingBatchIterator(x).iter_batches(self.batch_size):
                    predicts.append(
                        sess.run(self._softmax.predicts, feed_dict={
                            self._inputs_ph: x_batch
                        })
                    )
                return np.concatenate(predicts, axis=0).astype(np.int32)

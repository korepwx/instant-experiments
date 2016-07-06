# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import sys

import numpy as np

from ipwxlearn import glue, training
from ipwxlearn.datasets.utils import split_train_valid
from ipwxlearn.glue import G
from ipwxlearn.models.optimizers import AdamOptimizer
from .activations import ACTIVATIONS
from .base import BaseEstimator, ClassifierMixin, RegressorMixin

__all__ = [
    'MLPClassifier',
    'MLPRegressor'
]


class MLPEstimator(BaseEstimator):
    """
    Multi-layer perceptron estimator.

    This is the most basic form of a neural network estimator.  Input data should pass through several
    fully-connected hidden layers, and finally get to the output layer.

    :param layers: A tuple/list of integers, representing the number of units at each hidden layer.
                   Note that the MLP classifier would be equivalent to Logistic Regression if no layer
                   is specified.
    :param activation: Activation function for each hidden layer, one of {"tanh", "sigmoid", "relu"}.
                       (Default 'relu')
    :param dropout: A float number as the dropout probability at each hidden layer,
                    or a couple of float numbers as the dropout probability at the input layer,
                    as well as each hidden layer.  If None, will disable dropout.
    :param l1_reg: L1 regularization factor for this estimator. (Default None)
    :param l2_reg: L2 regularization factor for this estimator. (Default None)
    :param optimizer: Optimizer to train the network. (Default :class:`~ipwxlearn.estimators.optimizers.AdamOptimizer`).
                      See :module:`~ipwxlearn.estimators.optimizers` for more optimizers.
    :param batch_size: Training batch size. (Default 64)
    :param max_epoch: Maximum epoch to run for training the network. (Default 10)
    :param validation_split: Validation set split portion. (Default 0.1)
    :param verbose: Whether or not to print the training logs. (Default True)
    """

    #: Data type for the labels
    LABEL_DTYPE = None
    #: Initializer for the hidden weights
    WEIGHTS_INITIALIZER = G.init.XavierNormal()
    #: Initializer for the hidden biases
    BIASES_INITIALIZER = G.init.Constant(0.0)

    def __init__(self, layers, activation='relu', dropout=0.5, l1_reg=None, l2_reg=None, optimizer=AdamOptimizer(),
                 batch_size=64, max_epoch=100, validation_split=0.1, verbose=True):
        assert(activation in ACTIVATIONS)
        self.layers = layers
        self.activation = activation
        self.dropout = dropout
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.validation_split = validation_split
        self.verbose = verbose

    def fit(self, X, y, validation_steps=None, summary_steps=100, monitors=None, summary_dir=None):
        """
        Fit the MLP classifier with given X and y.

        :param X: N-d tensor as input data.
        :param y: 1-d integer tensor as labels.
        :param validation_steps: Perform validation every this number of steps.
                                 If not specified, will automatically select the steps.
        :param summary_steps: Perform summary every this number of steps.
        :param monitors: Monitors for this training.
        :param summary_dir: If specified, will write the variable summaries to this directory.
                            This directory must not exist, otherwise an IOError will be thrown.

        :return: self
        """
        # check the training data.
        assert(len(X.shape) >= 2)
        assert(len(X) == len(y))
        self.input_shape_ = X.shape[1:]
        self.output_shape_ = y.shape[1:]

        # check the summary directory.
        if summary_dir is not None:
            if os.path.exists(summary_dir):
                raise IOError('Summary directory %r already exists.' % summary_dir)

        # build the computation graph.
        self._build_graph()

        # split train/valid data
        (train_X, train_y), (valid_X, valid_y) = split_train_valid((X, y), validation_split=self.validation_split)

        # now train the model.
        with G.Session(self.graph):
            log_file = sys.stdout if self.verbose else None
            max_steps = self.max_epoch * len(train_X) // self.batch_size

            monitors = monitors or []
            if summary_dir is not None:
                summary_writer = G.summary.SummaryWriter(summary_dir)
                monitors.append(training.SummaryMonitor(summary_writer, self._var_summary, steps=summary_steps))
            else:
                summary_writer = None
            monitors.append(training.ValidationMonitor(
                self._valid_fn, (valid_X, valid_y), params=self._trainable_params, steps=validation_steps,
                log_file=log_file, validation_batch=256, summary_writer=summary_writer
            ))

            training.run_steps(G, self._train_fn, (train_X, train_y), monitor=monitors, batch_size=self.batch_size,
                               max_steps=max_steps, summary_writer=summary_writer)

    def _build_graph(self):
        train_input_shape = (self.batch_size,) + self.input_shape_
        train_output_shape = (self.batch_size,) + self.output_shape_
        test_input_shape = (None,) + self.input_shape_
        test_output_shape = (None,) + self.output_shape_

        dropout = self.dropout if isinstance(self.dropout, (tuple, list)) else (None, self.dropout)
        activation = ACTIVATIONS[self.activation]

        graph = self._graph = G.Graph()
        with graph.as_default():
            # input placeholders
            train_input = G.make_placeholder('trainX', shape=train_input_shape, dtype=glue.config.floatX)
            train_label = G.make_placeholder('trainY', shape=train_output_shape, dtype=self.LABEL_DTYPE)
            test_input = G.make_placeholder('testX', shape=test_input_shape, dtype=glue.config.floatX)
            test_label = G.make_placeholder('testY', shape=test_output_shape, dtype=self.LABEL_DTYPE)

            # compose the network
            network = input_layer = G.layers.InputLayer(train_input, shape=train_input_shape)
            if dropout[0]:
                network = G.layers.DropoutLayer('dropout0', network, p=dropout[0])
            for i, layer in enumerate(self.layers, 1):
                network = G.layers.DenseLayer('hidden%d' % i, network, num_units=layer, nonlinearity=activation,
                                              W=self.WEIGHTS_INITIALIZER, b=self.BIASES_INITIALIZER)
                if dropout[1]:
                    network = G.layers.DropoutLayer('dropout%d' % i, network, p=dropout[1])

            network, (train_output, train_loss), (test_output, test_loss) = \
                self._build_output_layer(input_layer, network, train_input, train_label, test_input, test_label)

            # gather summaries
            self._var_summary = G.summary.merge_summary(G.summary.collect_variable_summaries())
            train_loss_summary = G.summary.scalar_summary('training_loss', train_loss)

            # Create update expressions for training.
            self._trainable_params = G.layers.get_all_params(network, trainable=True)
            self._regularizable_params = G.layers.get_all_params(network, regularizable=True)
            loss_with_reg = self._add_regularization_term(train_loss, self._regularizable_params)
            updates = self.optimizer.minimize(loss_with_reg, self._trainable_params)

            self._train_fn = G.make_function(
                inputs=[train_input, train_label],
                outputs=[train_loss, train_loss_summary],
                updates=updates
            )
            self._valid_fn = G.make_function(inputs=[test_input, test_label], outputs=test_loss)
            self._predict_fn = G.make_function(inputs=[test_input], outputs=test_output)

    def _build_output_layer(self, input_layer, network, train_input, train_label, test_input, test_label):
        """
        Build the output layer for MLP estimator, then derivate the output & loss.

        :param input_layer: Input layer of the whole network.
        :param network: Previous layer of the output layer.
        :param train_input: Placeholder for training input.
        :param train_label: Placeholder for training label.
        :param test_input: Placeholder for testing input.
        :param test_label: Placeholder for testing label.

        :return: output_layer, (train_output, train_loss), (test_output, test_loss)
        """
        raise NotImplementedError()

    def _add_regularization_term(self, loss, params):
        """Add regularization term of specified parameters to the loss."""
        if self.l1_reg is not None:
            loss += self.l1_reg * G.op.l1_reg(params)
        if self.l2_reg is not None:
            loss += self.l2_reg * G.op.l2_reg(params)
        return loss


class MLPClassifier(MLPEstimator, ClassifierMixin):
    """Multi-layer perceptron classifier."""

    LABEL_DTYPE = np.int32

    def __init__(self, layers, activation='relu', dropout=0.5, l1_reg=None, l2_reg=None, optimizer=AdamOptimizer(),
                 batch_size=64, max_epoch=100, validation_split=0.1, verbose=True):
        super(MLPClassifier, self).__init__(
            layers=layers,
            activation=activation,
            dropout=dropout,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            optimizer=optimizer,
            batch_size=batch_size,
            max_epoch=max_epoch,
            validation_split=validation_split,
            verbose=verbose
        )

    def fit(self, X, y, validation_steps=None, summary_steps=100, monitors=None, summary_dir=None):
        assert(len(y.shape) == 1)
        assert(np.min(y) >= 0)
        self.target_num_ = np.max(y) + 1

        # ensure the data type matches our model.
        X = X.astype(glue.config.floatX)
        y = y.astype(np.int32)

        return super(MLPClassifier, self).fit(X, y, validation_steps, summary_steps, monitors, summary_dir)

    def _build_output_layer(self, input_layer, network, train_input, train_label, test_input, test_label):
        network = G.layers.SoftmaxLayer('softmax', network, num_units=self.target_num_)

        # derive the prediction and loss
        train_output, train_loss = G.layers.get_output_with_sparse_softmax_crossentropy(network, train_label)
        train_loss = G.op.mean(train_loss)

        test_output, test_loss = G.layers.get_output_with_sparse_softmax_crossentropy(
            network,
            test_label,
            inputs={input_layer: test_input},  # We use this to override the training input.
            deterministic=True,  # Disable dropout on testing.
        )
        test_loss = G.op.mean(test_loss)

        return network, (train_output, train_loss), (test_output, test_loss)


class MLPRegressor(MLPEstimator, RegressorMixin):
    """Multi-layer perceptron regressor."""

    LABEL_DTYPE = glue.config.floatX

    def __init__(self, layers, activation='relu', dropout=None, l1_reg=None, l2_reg=0.0001, optimizer=AdamOptimizer(),
                 batch_size=64, max_epoch=100, validation_split=0.1, verbose=True):
        super(MLPRegressor, self).__init__(
            layers=layers,
            activation=activation,
            dropout=dropout,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            optimizer=optimizer,
            batch_size=batch_size,
            max_epoch=max_epoch,
            validation_split=validation_split,
            verbose=verbose
        )

    def fit(self, X, y, validation_steps=None, summary_steps=100, monitors=None, summary_dir=None):
        X = X.astype(glue.config.floatX)
        y = y.astype(glue.config.floatX)

        # normalize the target to achieve better performance
        y_min = self.y_min_ = np.min(y, axis=0, keepdims=True)
        y_max = np.max(y, axis=0, keepdims=True)
        self.y_scale_ = y_max - y_min

        return super(MLPRegressor, self).fit(X, y, validation_steps, summary_steps, monitors, summary_dir)

    def _build_output_layer(self, input_layer, network, train_input, train_label, test_input, test_label):
        num_units = np.prod(self.output_shape_) if self.output_shape_ else 1
        network = G.layers.DenseLayer('output', network, num_units=num_units, nonlinearity=None,
                                      W=self.WEIGHTS_INITIALIZER, b=self.BIASES_INITIALIZER)

        train_output = G.layers.get_output(network)
        test_output = G.layers.get_output(
            network,
            inputs={input_layer: test_input},  # We use this to override the training input.
            deterministic=True,  # Disable dropout on testing.
        )

        # fix the output shape, if specified y is 1-D scalars
        if not self.output_shape_:
            if G.__backend__ == 'tensorflow':
                import tensorflow as tf
                f = lambda v: tf.squeeze(v, [1])
            elif G.__backend__ == 'theano':
                f = lambda v: v.reshape((-1,))
            else:
                raise RuntimeError('Backend %r not supported.' % G.__backend__)
            train_output = f(train_output)
            test_output = f(test_output)

        # Scale the y so that the network seems to be fitting something within range [0.0, 1.0].
        train_output = train_output * self.y_scale_ + self.y_min_
        test_output = test_output * self.y_scale_ + self.y_min_

        # We use MSE as the loss function.
        train_loss = G.op.mean((train_output - train_label) ** 2)
        test_loss = G.op.mean((test_output - test_label) ** 2)

        return network, (train_output, train_loss), (test_output, test_loss)

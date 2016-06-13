# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import sys

import numpy as np

from ipwxlearn import glue
from ipwxlearn.datasets.utils import split_train_valid
from ipwxlearn.glue import G
from ipwxlearn.utils import training
from .activations import ACTIVATIONS
from .base import BaseClassifier
from .optimizers import AdamOptimizer

__all__ = [
    'MLPClassifier'
]


class MLPClassifier(BaseClassifier):
    """
    Multi-layer perceptron classifier.

    This is the most basic form of a neural network classifier.  Input data should pass through several
    fully-connected hidden layers, and finally get classified at a softmax layer.

    :param layers: A tuple/list of integers, representing the number of units at each hidden layer.
                   Note that the MLP classifier would be equivalent to Logistic Regression if no layer
                   is specified.
    :param activation: Activation function for each hidden layer, one of {"tanh", "sigmoid", "relu"}.
                       (Default 'relu')
    :param dropout: A float number as the dropout probability at each hidden layer,
                    or a couple of float numbers as the dropout probability at the input layer,
                    as well as each hidden layer.  If None, will disable dropout.
    :param optimizer: Optimizer to train the network. (Default :class:`~ipwxlearn.estimators.optimizers.AdamOptimizer`).
                      See :module:`~ipwxlearn.estimators.optimizers` for more optimizers.
    :param batch_size: Training batch size. (Default 64)
    :param max_epoch: Maximum epoch to run for training the network. (Default 10)
    :param valid_portion: Validation set split portion. (Default 0.1)
    :param verbose: Whether or not to print the training logs. (Default True)
    """

    def __init__(self, layers, activation='relu', dropout=0.5, optimizer=AdamOptimizer(), batch_size=64,
                 max_epoch=10, valid_portion=0.1, verbose=True):
        assert(activation in ACTIVATIONS)
        self.layers = layers
        self.activation = activation
        self.dropout = dropout
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.valid_portion = valid_portion
        self.verbose = verbose

    def fit(self, X, y, monitors=None, summary_dir=None):
        """
        Fit the MLP classifier with given X and y.

        :param X: N-d tensor as input data.
        :param y: 1-d integer tensor as labels.
        :param monitors: Monitors for this training.
        :param summary_dir: If specified, will write the variable summaries to this directory.
                            This directory must not exist, otherwise an IOError will be thrown.

        :return: self
        """
        # check the training data.
        assert(len(X) == len(y))
        assert(len(y.shape) == 1)
        assert(np.min(y) >= 0)
        self.input_shape_ = X.shape[1:]
        self.target_num_ = np.max(y) + 1

        # ensure the data type matches our model.
        X = X.astype(glue.config.floatX)
        y = y.astype(np.int32)

        # check the summary directory.
        if summary_dir is not None:
            if os.path.exists(summary_dir):
                raise IOError('Summary directory %r already exists.' % summary_dir)

        # build the computation graph.
        self._build_graph()

        # split train/valid data
        (train_X, train_y), (valid_X, valid_y) = split_train_valid((X, y), valid_portion=0.1)

        # now train the model.
        with G.Session(self.graph):
            log_file = sys.stdout if self.verbose else None
            max_steps = self.max_epoch * len(train_X) // self.batch_size

            monitors = monitors or []
            if summary_dir is not None:
                monitors.append(training.SummaryMonitor(summary_dir, self._var_summary, steps=100))
                summary_writer = G.summary.SummaryWriter(summary_dir)
            else:
                summary_writer = None
            monitors.append(training.ValidationMonitor(
                self._valid_fn, (valid_X, valid_y), params=self._trainable_params, log_file=log_file,
                validation_batch=256, summary_writer=summary_writer
            ))

            training.run_steps(G, self._train_fn, (train_X, train_y), monitor=monitors, batch_size=self.batch_size,
                               max_steps=max_steps, summary_writer=summary_writer)

    def _build_graph(self):
        dropout = self.dropout if isinstance(self.dropout, (tuple, list)) else (None, self.dropout)
        train_input_shape = (self.batch_size,) + self.input_shape_
        test_input_shape = (None,) + self.input_shape_

        graph = self._graph = G.Graph()
        with graph.as_default():
            # input placeholders
            train_input = G.make_placeholder('trainX', shape=train_input_shape, dtype=glue.config.floatX)
            train_label = G.make_placeholder('trainY', shape=(self.batch_size,), dtype=np.int32)
            test_input = G.make_placeholder('testX', shape=test_input_shape, dtype=glue.config.floatX)
            test_label = G.make_placeholder('testY', shape=(None,), dtype=np.int32)

            # compose the network
            network = input_layer = G.layers.InputLayer(train_input, shape=train_input_shape)
            if dropout[0]:
                network = G.layers.DropoutLayer('dropout0', network, p=dropout[0])
            for i, layer in enumerate(self.layers, 1):
                network = G.layers.DenseLayer('hidden%d' % i, network, num_units=layer)
                if dropout[1]:
                    network = G.layers.DropoutLayer('dropout%d' % i, network, p=dropout[1])
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

            # gather summaries
            self._var_summary = G.summary.merge_summary(G.summary.collect_variable_summaries())
            train_loss_summary = G.summary.scalar_summary('training_loss', train_loss)

            # Create update expressions for training.
            params = self._trainable_params = G.layers.get_all_params(network, trainable=True)
            updates = G.updates.adam(train_loss, params)

            self._train_fn = G.make_function(
                inputs=[train_input, train_label],
                outputs=[train_loss, train_loss_summary],
                updates=updates
            )
            self._valid_fn = G.make_function(inputs=[test_input, test_label], outputs=test_loss)
            self._predict_fn = G.make_function(inputs=[test_input], outputs=test_output)

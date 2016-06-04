# -*- coding: utf-8 -*-
import unittest

import numpy as np

import ipwxlearn.glue.theano.layers.input
import ipwxlearn.glue.theano.layers.noise
from ipwxlearn import glue
from ipwxlearn.glue import G


class NoiseTestCase(unittest.TestCase):

    def _do_test_dropout(self, p, deterministic, rescale, expect_mean,
                         mean_epsilon=1e-2, std_epsilon=1e-1, input_dim=100, n_samples=2000):
        X = np.ones((n_samples, input_dim), dtype=np.dtype(glue.config.floatX))

        graph = G.Graph()
        with graph.as_default():
            input_var = G.make_placeholder('inputs', shape=(None, input_dim), dtype=glue.config.floatX)
            input_layer = ipwxlearn.glue.theano.layers.input.InputLayer(input_var, shape=(None, input_dim))
            dropout_layer = ipwxlearn.glue.theano.layers.noise.DropoutLayer('dropout', input_layer, p=p, rescale=rescale)
            output = G.layers.get_output(dropout_layer, deterministic=deterministic)
            predict_fn = G.make_function(inputs=input_var, outputs=output)

        with G.Session(graph):
            X2 = predict_fn(X)
            portion = np.sum(X2, axis=1) / np.sum(X, axis=1)
            mean, std = np.mean(portion), np.std(portion)
            self.assertLess(abs(mean - expect_mean), mean_epsilon)
            self.assertLess(abs(std), std_epsilon)

    def test_dropout_non_deterministic_rescaled(self):
        """Test the output of dropout layer when deterministic is False and rescaled is True."""
        self._do_test_dropout(0.31, deterministic=False, rescale=True, expect_mean=1.0)

    def test_dropout_non_deterministic_not_rescaled(self):
        """Test the output of dropout layer when deterministic is False and rescaled is False."""
        self._do_test_dropout(0.31, deterministic=False, rescale=False, expect_mean=0.69)

    def test_dropout_deterministic_rescaled(self):
        """Test the output of dropout layer when deterministic is True and rescaled is True."""
        self._do_test_dropout(0.31, deterministic=True, rescale=True, expect_mean=1.0, mean_epsilon=1e-7,
                              std_epsilon=1e-7)

    def test_dropout_deterministic_not_rescaled(self):
        """Test the output of dropout layer when deterministic is True and rescaled is False."""
        self._do_test_dropout(0.31, deterministic=True, rescale=False, expect_mean=1.0, mean_epsilon=1e-7,
                              std_epsilon=1e-7)

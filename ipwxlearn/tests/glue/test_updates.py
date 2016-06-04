# -*- coding: utf-8 -*-
import unittest

import numpy as np

from ipwxlearn import glue
from ipwxlearn.glue import G


class UpdatesTestCase(unittest.TestCase):

    def _do_test_update(self, optimizer, n_dim=256, *args, **kwargs):
        graph = G.Graph()
        with graph.as_default():
            # okay, compose the quadratic function.
            x = G.make_variable('x', shape=[n_dim], init=G.init.Uniform([-1, 1]), dtype=glue.config.floatX)

            # finally, create the training function.
            loss = G.op.dot(x, x)
            train_fn = G.make_function(updates=optimizer(loss, [x], *args, **kwargs), outputs=loss)

        with G.Session(graph):
            best_x = G.get_variable_values(x)
            best_loss = np.dot(best_x, best_x)
            self.assertGreater(np.mean((best_x - np.zeros_like(best_x)) ** 2), 1e-2)

            for i in range(700):
                train_loss = train_fn()
                if train_loss < best_loss:
                    best_x = G.get_variable_values(x)
                    best_loss = train_loss
            self.assertLess(np.mean((best_x - np.zeros_like(best_x)) ** 2), 1e-7)

    def test_sgd(self):
        """Test training with SGD."""
        self._do_test_update(G.updates.sgd, learning_rate=0.01)

    def test_momentum(self):
        """Test training with momentum."""
        self._do_test_update(G.updates.momentum, learning_rate=0.001)

    @unittest.skipIf(glue.config.backend == 'tensorflow', 'TensorFlow has not supported Nesterov momentum yet.')
    def test_nesterov_momentum(self):
        """Test training with nesterov momentum."""
        self._do_test_update(G.updates.nesterov_momentum, learning_rate=0.001)

    def test_adagrad(self):
        """Test training with AdaGrad."""
        self._do_test_update(G.updates.adagrad, learning_rate=1.0)

    def test_rmsprop(self):
        """Test training with RMSProp."""
        self._do_test_update(G.updates.rmsprop, learning_rate=10.0, rho=0.999)

    def test_adam(self):
        """Test training with Adam."""
        self._do_test_update(G.updates.adam, learning_rate=0.01)


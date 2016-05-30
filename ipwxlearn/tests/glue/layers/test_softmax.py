# -*- coding: utf-8 -*-
import unittest

import numpy as np

from ipwxlearn import glue
from ipwxlearn.glue import G


class SoftmaxUnitTest(unittest.TestCase):

    @staticmethod
    def make_softmax_data(n=10000, dim=10, target_num=2, dtype=np.float64):
        W = (np.random.random([dim, target_num]) - 0.5).astype(dtype)
        b = (np.random.random([target_num]) - 0.5).astype(dtype)
        X = ((np.random.random([n, dim]) - 0.5) * 10.0).astype(dtype)
        y = np.argmax(np.dot(X, W) + b, axis=1)
        return (W, b), (X, y)

    def test_binary_predicting(self):
        """Test binary softmax classifier."""
        (W, b), (X, y) = self.make_softmax_data(target_num=2, dtype=glue.config.floatX)

        graph = G.Graph()
        with graph.as_default():
            input_var = G.make_placeholder('inputs', shape=(None, W.shape[0]), dtype=glue.config.floatX)
            input_layer = G.layers.InputLayer(input_var, shape=(None, W.shape[0]))
            softmax_layer = G.layers.SoftmaxLayer('softmax', input_layer, num_units=b.shape[0], W=W, b=b)
            predict_prob = G.layers.get_output(softmax_layer)
            predict_label = G.op.argmax(predict_prob, axis=1)
            predict_fn = G.make_function(inputs=[input_var], outputs=[predict_prob, predict_label])

        with G.Session(graph):
            prob, predict = predict_fn(X)
            self.assertTrue(np.alltrue(predict == y))

    def test_categorical_predicting(self):
        """Test categorical softmax classifier."""
        (W, b), (X, y) = self.make_softmax_data(target_num=5, dtype=glue.config.floatX)

        graph = G.Graph()
        with graph.as_default():
            input_var = G.make_placeholder('inputs', shape=(None, W.shape[0]), dtype=glue.config.floatX)
            input_layer = G.layers.InputLayer(input_var, shape=(None, W.shape[0]))
            softmax_layer = G.layers.SoftmaxLayer('softmax', input_layer, num_units=b.shape[0], W=W, b=b)
            predict_prob = G.layers.get_output(softmax_layer)
            predict_label = G.op.argmax(predict_prob, axis=1)
            predict_fn = G.make_function(inputs=[input_var], outputs=[predict_prob, predict_label])

        with G.Session(graph):
            prob, predict = predict_fn(X)
            self.assertTrue(np.alltrue(predict == y))

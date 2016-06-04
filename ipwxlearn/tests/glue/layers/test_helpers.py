# -*- coding: utf-8 -*-
import unittest

from ipwxlearn import glue
from ipwxlearn.glue import G


class HelpersTestCase(unittest.TestCase):

    def test_get_all_params(self):
        """Test get_all_params on layers."""
        graph = G.Graph()
        with graph.as_default():
            input_shape = (None, 784)
            target_num = 10
            input_var = G.make_placeholder('X', shape=input_shape, dtype=glue.config.floatX)

            # compose the network
            input = G.layers.InputLayer(input_var, shape=input_shape)
            hidden1 = G.layers.DenseLayer('hidden1', input, num_units=128)
            hidden2 = G.layers.DenseLayer('hidden2', hidden1, num_units=32)
            softmax = G.layers.SoftmaxLayer('softmax', hidden2, num_units=target_num)

            # attach another network
            hidden3 = G.layers.DenseLayer('hidden3', input, num_units=800)
            softmax2 = G.layers.SoftmaxLayer('softmax2', hidden3, num_units=target_num)

            # Create update expressions for training.
            self.assertEqual(G.layers.get_all_params(softmax, trainable=True),
                             [hidden1.W, hidden1.b, hidden2.W, hidden2.b, softmax.W, softmax.b])
            self.assertEqual(G.layers.get_all_params(softmax, regularizable=True),
                             [hidden1.W, hidden2.W, softmax.W])
            self.assertEqual(G.layers.get_all_params(softmax2),
                             [hidden3.W, hidden3.b, softmax2.W, softmax2.b])
            self.assertEqual(graph.get_variables(),
                             [hidden1.W, hidden1.b, hidden2.W, hidden2.b, softmax.W, softmax.b,
                              hidden3.W, hidden3.b, softmax2.W, softmax2.b])

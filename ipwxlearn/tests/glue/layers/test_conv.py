# -*- coding: utf-8 -*-
import unittest

import numpy as np

from ipwxlearn import glue
from ipwxlearn.glue import G


class ConvTestCase(unittest.TestCase):

    def _do_test_input_output(self, ConvInputLayer, ConvOutputLayer, data_dims, n_channels=16, n_samples=100):
        input_shape = (n_samples, ) + data_dims + (n_channels, )
        X = np.random.randint(low=0, high=2147483647, size=input_shape, dtype=np.int32)

        graph = G.Graph()
        with graph.as_default():
            input_var = G.make_placeholder('X', shape=input_shape, dtype=np.int32)
            input_layer = G.layers.InputLayer(input_var, shape=input_shape)
            conv_input_layer = ConvInputLayer(input_layer)
            conv_output_layer = ConvOutputLayer(conv_input_layer)
            output = G.layers.get_output(conv_output_layer)
            get_output = G.make_function(inputs=input_var, outputs=output)

        with G.Session(graph):
            self.assertTrue(np.alltrue(get_output(X) == X))

    def test_2d_input_output(self):
        """Test input and output layer for 2D conv."""
        self._do_test_input_output(G.layers.Conv2DInputLayer, G.layers.Conv2DOutputLayer, (37, 41))

    def _do_test_conv_layer(self, ConvInputLayer, ConvOutputLayer, ConvLayer, data_dims, n_channels=5,
                            n_samples=10, n_filters=7, filter_size=5, stride=1, padding='valid',
                            untie_biases=False):
        n_dim = len(data_dims)
        input_shape = (n_samples, ) + data_dims + (n_channels, )
        X = np.random.random(input_shape).astype(glue.config.floatX)

        graph = G.Graph()
        with graph.as_default():
            input_var = G.make_placeholder('X1', shape=input_shape, dtype=glue.config.floatX)
            input_layer = G.layers.InputLayer(input_var, shape=input_shape)

            conv_input_layer = ConvInputLayer(input_layer)
            conv_layer = ConvLayer('conv', conv_input_layer, num_filters=n_filters, filter_size=(filter_size,) * n_dim,
                                   stride=(stride,) * n_dim, padding=padding, untie_biases=untie_biases)
            conv_output_layer = ConvOutputLayer(conv_layer)

            input_var2 = G.make_placeholder('X2', shape=(None,) + input_shape[1:], dtype=glue.config.floatX)
            output = G.layers.get_output(conv_output_layer)
            output2 = G.layers.get_output(conv_output_layer, inputs={input_layer: input_var2})

            get_output = G.make_function(inputs=input_var, outputs=output)
            get_output2 = G.make_function(inputs=input_var2, outputs=output2)

        with G.Session(graph):
            y1, y2 = get_output(X), get_output2(X)
            # data count should not change.
            self.assertEquals(len(y1), len(X))
            self.assertEquals(len(y2), len(X))

            # data dimension shrinks only if the padding is valid.
            if padding == 'valid':
                output_size_off = 1 - filter_size
            else:
                output_size_off = 0
            expect_data_shape = tuple(
                (v + output_size_off + stride - 1) // stride
                for v in X.shape[1: -1]
            )
            self.assertEquals(y1.shape, X.shape[:1] + expect_data_shape + (n_filters,))
            self.assertEquals(y2.shape, X.shape[:1] + expect_data_shape + (n_filters,))

    def test_2d_conv(self):
        """Test 2D conv."""
        for stride in (1, 2):
            for padding in ('same', 'valid'):
                for untie_biases in (True, False):
                    self._do_test_conv_layer(
                        G.layers.Conv2DInputLayer,
                        G.layers.Conv2DOutputLayer,
                        G.layers.Conv2DLayer,
                        (37, 41),
                        stride=stride,
                        padding=padding,
                        untie_biases=untie_biases
                    )

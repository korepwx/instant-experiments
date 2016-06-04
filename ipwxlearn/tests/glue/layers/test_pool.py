# -*- coding: utf-8 -*-
import unittest

import numpy as np

from ipwxlearn import glue
from ipwxlearn.glue import G


class PoolTestCase(unittest.TestCase):

    def _do_test_conv_layer(self, PoolLayer, data_dims, n_channels=5, n_samples=10, pool_size=2, stride=2,
                            pad=None):
        input_shape = (n_samples, ) + data_dims + (n_channels, )
        X = np.random.random(input_shape).astype(glue.config.floatX)

        graph = G.Graph()
        with graph.as_default():
            input_var = G.make_placeholder('X', shape=input_shape, dtype=glue.config.floatX)
            input_layer = G.layers.InputLayer(input_var, shape=input_shape)
            pool_layer = PoolLayer('pool', input_layer, pool_size=pool_size, stride=stride, pad=pad)
            output = G.layers.get_output(pool_layer)
            get_output = G.make_function(inputs=input_var, outputs=output)

        with G.Session(graph):
            y = get_output(X)
            # TODO: validate result.

    def test_2d_conv(self):
        """Test 2D conv."""
        for pool_size in (2, 3):
            for stride in range(1, pool_size+1):
                for pad in [(0, 0), (1, 1)]:
                    self._do_test_conv_layer(
                        G.layers.AvgPool2DLayer,
                        (37, 41),
                        pool_size=pool_size,
                        stride=stride,
                        pad=pad
                    )

# -*- coding: utf-8 -*-
import unittest

import numpy as np

from ipwxlearn import glue
from ipwxlearn.glue import G


class PoolTestCase(unittest.TestCase):

    def _do_test_conv_layer(self, PoolLayer, data_dims, n_channels=5, n_samples=10, pool_size=2, stride=2,
                            padding='none'):
        input_shape = (n_samples, ) + data_dims + (n_channels, )
        X = np.random.random(input_shape).astype(glue.config.floatX)

        graph = G.Graph()
        with graph.as_default():
            input_var = G.make_placeholder('X', shape=input_shape, dtype=glue.config.floatX)
            input_layer = G.layers.Conv2DInputLayer(G.layers.InputLayer(input_var, shape=input_shape))
            pool_layer = PoolLayer('pool', input_layer, pool_size=pool_size, stride=stride, padding=padding)
            output_layer = G.layers.Conv2DOutputLayer(pool_layer)
            output = G.layers.get_output(output_layer)
            get_output = G.make_function(inputs=input_var, outputs=output)

        with G.Session(graph):
            y = get_output(X)
            self.assertEquals(X.shape[0], y.shape[0], 'output batch size should be same to input.')
            self.assertEquals(X.shape[-1], y.shape[-1], 'output channel size should be same to input.')

            if padding == 'none':
                expect_data_shape = tuple((v - pool_size + stride) // stride for v in X.shape[1: -1])
                self.assertEquals(y.shape[1:-1], expect_data_shape)

            elif padding == 'backend':
                # There should be only two ways of padding in the backend.
                # 1. pad to exactly the same as the input size.
                # 2. pad with the same number of units at both sides, which would cause the output
                #    to have one extra unit than same padding.
                expect_data_shape_1 = tuple((v + stride - 1) // stride for v in X.shape[1: -1])
                if pool_size % 2 == 0:
                    expect_data_shape_2 = tuple(v + 1 for v in expect_data_shape_1)
                else:
                    expect_data_shape_2 = expect_data_shape_1
                self.assertIn(y.shape[1:-1], (expect_data_shape_1, expect_data_shape_2))

            elif padding == 'same':
                expect_data_shape = tuple((v + stride - 1) // stride for v in X.shape[1: -1])
                expect_shape = (X.shape[0],) + expect_data_shape + (X.shape[-1],)
                self.assertEquals(y.shape, expect_shape)

            else:
                raise ValueError('Unsupported padding type %r.' % padding)

    def test_2d_conv(self):
        """Test 2D conv."""
        for padding in ('none', 'backend', 'same'):
            for pool_size in (1, 2, 3, 4):
                strides = list(range(1, pool_size+1))
                if glue.config.backend != 'tensorflow':
                    # TensorFlow only supports stride <= pool_size, but Theano supports more.
                    strides += list(range(pool_size+1, 6))

                for stride in strides:
                    self._do_test_conv_layer(
                        G.layers.AvgPool2DLayer,
                        (37, 41),
                        pool_size=pool_size,
                        stride=stride,
                        padding=padding
                    )

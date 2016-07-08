# -*- coding: utf-8 -*-
from __future__ import absolute_import

import unittest

import numpy as np

from ipwxlearn.glue import G


class AddLayer(G.layers.MergeLayer):

    def get_output_for(self, inputs, **kwargs):
        return sum(inputs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]


class AmplifyLayer(G.layers.Layer):

    def __init__(self, incoming, amplify=1, name=None):
        super(AmplifyLayer, self).__init__(incoming=incoming, name=name)
        self.amplify = amplify

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return input * self.amplify


class CompoundTestCase(unittest.TestCase):

    def test_compound_layer(self):
        """Test CompoundLayer."""
        a = np.arange(18, dtype=np.int32).reshape([3, 6])
        b = a + 18
        c = a + 36

        # Test the basic layer relationship, as well as the input -> output chain.
        graph = G.Graph()
        with graph.as_default():
            a_input, a_var = G.layers.make_input('a', a, dtype=np.int32)
            b_input, b_var = G.layers.make_input('b', b, dtype=np.int32)
            c_input, c_var = G.layers.make_input('c', c, dtype=np.int32)

            left = AmplifyLayer(a_input, amplify=2, name='left')
            middle = AmplifyLayer(b_input, amplify=3, name='middle')
            right = AmplifyLayer(c_input, amplify=4, name='right')
            network = AddLayer([left, middle, right])

            # include the whole middle path (in this case the middle path is not leaked)
            compound1 = G.layers.ChainLayer([b_input, middle, network], name='compound1')
            self.assertEquals(compound1.input_layers, [left, right])
            self.assertEquals(G.layers.get_all_layers(compound1),
                              [a_input, left, c_input, right, compound1])
            self.assertEquals(compound1.output_shape, (None, 6))
            compute1 = G.make_function(inputs=[a_var, b_var, c_var], outputs=G.layers.get_output(compound1))

            # include partial of middle path (in this case the middle path is leaked)
            compound2 = G.layers.ChainLayer([b_input, network], name='compound2')
            self.assertEquals(compound2.input_layers, [left, middle, right])
            self.assertEquals(G.layers.get_all_layers(compound2),
                              [a_input, left, b_input, middle, c_input, right, compound2])
            self.assertEquals(compound2.output_shape, (None, 6))
            compute2 = G.make_function(inputs=[a_var, b_var, c_var], outputs=G.layers.get_output(compound2))

            compound3 = G.layers.ChainLayer([middle, network], name='compound3')
            self.assertEquals(compound3.input_layers, [b_input, left, right])
            self.assertEquals(G.layers.get_all_layers(compound3),
                              [b_input, a_input, left, c_input, right, compound3])
            self.assertEquals(compound3.output_shape, (None, 6))
            compute3 = G.make_function(inputs=[a_var, b_var, c_var], outputs=G.layers.get_output(compound3))

            with G.Session(graph):
                result = (a * 2) + (b * 3) + (c * 4)
                self.assertTrue(np.alltrue(result == compute1(a, b, c)))
                self.assertTrue(np.alltrue(result == compute2(a, b, c)))
                self.assertTrue(np.alltrue(result == compute3(a, b, c)))

        # Test collecting parameters.
        graph = G.Graph()
        with graph.as_default():
            a_input, a_var = G.layers.make_input('a', a, dtype=np.int32)
            b_input, b_var = G.layers.make_input('b', b, dtype=np.int32)
            c_input, c_var = G.layers.make_input('c', c, dtype=np.int32)

            left = G.layers.DenseLayer('left', a_input, num_units=2)
            middle = G.layers.DenseLayer('middle', b_input, num_units=2)
            middle2 = G.layers.DenseLayer('middle2', middle, num_units=3)
            right = G.layers.DenseLayer('right', c_input, num_units=2)
            network = AddLayer([left, middle2, right])

            # include the whole middle path (in this case the middle path is not leaked)
            compound1 = G.layers.ChainLayer([b_input, middle, middle2, network], name='compound1')
            self.assertEquals(compound1.get_params(), [middle.W, middle.b, middle2.W, middle2.b])
            self.assertEquals(G.layers.get_all_params(compound1),
                              [left.W, left.b, right.W, right.b, middle.W, middle.b, middle2.W, middle2.b])

            # include partial of middle path (in this case the middle path is leaked)
            compound2 = G.layers.ChainLayer([b_input, middle2, network], name='compound2')
            self.assertEquals(compound2.get_params(), [middle2.W, middle2.b])
            self.assertEquals(G.layers.get_all_params(compound2),
                              [middle.W, middle.b, left.W, left.b, right.W, right.b, middle2.W, middle2.b])

            compound3 = G.layers.ChainLayer([middle, middle2, network], name='compound3')
            self.assertEquals(compound3.get_params(), [middle.W, middle.b, middle2.W, middle2.b])
            self.assertEquals(G.layers.get_all_params(compound3),
                              [left.W, left.b, right.W, right.b, middle.W, middle.b, middle2.W, middle2.b])

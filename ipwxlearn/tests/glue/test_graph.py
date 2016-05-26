# -*- coding: utf-8 -*-
import re
import unittest

import numpy as np

from ipwxlearn.glue import G


class GraphTestCase(unittest.TestCase):

    def test_layer_variables(self):
        """Test catching layer variables during construction."""
        graph = G.Graph()

        # Layer creation without name should raise error.
        with graph.as_default():
            input_var = G.make_placeholder('X', shape=(None, 784), dtype=np.int32)
            inputs = G.layers.imports.InputLayer(input_var=input_var, shape=(None, 784))

        # Layer creation without a graph should raise error.
        with self.assertRaises(ValueError) as cm:
            _ = G.layers.imports.DenseLayer(incoming=inputs, num_units=128, name='hidden1')
        self.assertTrue(re.search(r'No name scope is activated.*', str(cm.exception)))

        # Check the catched variables during layer creation.
        with graph.as_default():
            hidden1 = G.layers.imports.DenseLayer(incoming=inputs, num_units=128, name='hidden1')
            with G.name_scope('nested'):
                nested_hidden1 = G.layers.imports.DenseLayer(incoming=inputs, num_units=128, name='hidden1')
                nested_hidden2 = G.layers.imports.DenseLayer(incoming=nested_hidden1, num_units=32, name='hidden2')

        self.assertEquals(list(graph.iter_variables()),
                          [hidden1.W, hidden1.b, nested_hidden1.W, nested_hidden1.b, nested_hidden2.W,
                           nested_hidden2.b])

        # Check that duplicated names would raise errors.
        with graph.as_default():
            with self.assertRaises(KeyError) as cm:
                _ = G.layers.imports.DenseLayer(incoming=inputs, num_units=128, name='hidden1')
            self.assertTrue(re.search(r'Full name hidden1/W is already used by.*', str(cm.exception)))

        # Check that trainable & persistent variables include all the parameters.
        for tag in (G.VariableTags.TRAINABLE, G.VariableTags.PERSISTENT, G.VariableTags.RESUMABLE):
            self.assertEquals(graph.get_variables(tags=[tag]),
                              [hidden1.W, hidden1.b, nested_hidden1.W, nested_hidden1.b, nested_hidden2.W,
                               nested_hidden2.b])

        # Check that filtering by regularizable would only result in W.
        self.assertEquals(graph.get_variables(tags=[G.VariableTags.REGULARIZABLE]),
                          [hidden1.W, nested_hidden1.W, nested_hidden2.W])

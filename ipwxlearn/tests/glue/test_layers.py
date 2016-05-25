# -*- coding: utf-8 -*-
import re
import unittest

from ipwxlearn.glue import G


class LayersTestCase(unittest.TestCase):

    def test_layer_construction(self):
        """Test constructing layers."""
        graph = G.Graph()

        # Layer creation without name should raise error.
        with graph.as_default():
            inputs = G.layers.InputLayer(shape=(None, 784), name='inputs')
            with self.assertRaises(ValueError, msg='No name specified for the layer.'):
                _ = G.layers.DenseLayer(inputs, num_units=128)

        # Layer creation without a graph should raise error.
        with self.assertRaises(ValueError) as cm:
            hidden1 = G.layers.DenseLayer(inputs, num_units=128, name='hidden1')
        self.assertTrue(re.search(r'No name scope is activated.*', str(cm.exception)))

        # Check the catched variables during layer creation.
        with graph.as_default():
            hidden1 = G.layers.DenseLayer(inputs, num_units=128, name='hidden1')
            with G.name_scope('nested'):
                nested_hidden1 = G.layers.DenseLayer(inputs, num_units=128, name='hidden1')
                nested_hidden2 = G.layers.DenseLayer(nested_hidden1, num_units=32, name='hidden2')

        self.assertEquals(list(graph.iter_variables()),
                          [hidden1.W, hidden1.b, nested_hidden1.W, nested_hidden1.b, nested_hidden2.W,
                           nested_hidden2.b])

        # Check that duplicated names would raise errors.
        with graph.as_default():
            with self.assertRaises(KeyError) as cm:
                _ = G.layers.DenseLayer(inputs, num_units=128, name='hidden1')
            self.assertTrue(re.search(r'Full name hidden1/W is already used by.*', str(cm.exception)))

        # Check that filtering by regularizable would only result in W.
        with graph.as_default():
            self.assertEquals(list(graph.iter_variables(tags=[G.VariableTags.REGULARIZABLE])),
                              [hidden1.W, nested_hidden1.W, nested_hidden2.W])

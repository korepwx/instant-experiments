# -*- coding: utf-8 -*-
import re
import unittest

from ipwxlearn.glue import G


class LayersTestCase(unittest.TestCase):

    def test_layer_creation(self):
        graph = G.Graph()

        # Layer creation without name should raise error.
        with graph.as_default():
            input_layer = G.layers.InputLayer(shape=(None, 784), name='inputs')
            with self.assertRaises(ValueError, msg='No name specified for the layer.'):
                _ = G.layers.DenseLayer(input_layer, num_units=128)

        # Layer creation without a graph should raise error.
        with self.assertRaises(ValueError) as cm:
            _ = G.layers.DenseLayer(input_layer, num_units=128, name='hidden1')
        self.assertTrue(re.match(r'No name scope is activated.*', str(cm.exception)))

        # Check the catched variables during layer creation.
        with graph.as_default():
            _ = G.layers.DenseLayer(input_layer, num_units=128, name='hidden1')
            with G.name_scope('nested'):
                h1 = G.layers.DenseLayer(input_layer, num_units=128, name='hidden1')
                _ = G.layers.DenseLayer(h1, num_units=32, name='hidden2')

        def expected_repr(name, regularizable=True):
            tags = ['persistent', 'regularizable', 'resumable', 'trainable']
            if not regularizable:
                tags.remove('regularizable')
            return 'Variable(%s, %s)' % (name, ', '.join(tags))
        self.assertEquals([repr(v) for v in graph.iter_variables()],
                          [expected_repr('hidden1/W', True),
                           expected_repr('hidden1/b', False),
                           expected_repr('nested/hidden1/W', True),
                           expected_repr('nested/hidden1/b', False),
                           expected_repr('nested/hidden2/W', True),
                           expected_repr('nested/hidden2/b', False)]
                          )

        # Check that duplicated names would raise errors.
        with graph.as_default():
            expected_err_msg = 'Full name hidden1/W is already used by %s.' % expected_repr('hidden1/W', True)
            with self.assertRaises(KeyError, msg=expected_err_msg):
                _ = G.layers.DenseLayer(input_layer, num_units=128, name='hidden1')

        # Check that filtering by regularizable would only result in W.
        with graph.as_default():
            self.assertEquals([repr(v) for v in graph.iter_variables(tags=[G.VariableTags.REGULARIZABLE])],
                              [expected_repr('hidden1/W', True),
                               expected_repr('nested/hidden1/W', True),
                               expected_repr('nested/hidden2/W', True)]
                              )

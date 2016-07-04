# -*- coding: utf-8 -*-
import os
import re
import unittest

import numpy as np

from ipwxlearn.glue import G
from ipwxlearn.utils import tempdir


class GraphTestCase(unittest.TestCase):

    def test_layer_variables(self):
        """Test catching layer variables during construction."""
        graph = G.Graph()

        # Layer creation without name should raise error.
        with graph.as_default():
            input_var = G.make_placeholder('X', shape=(None, 784), dtype=np.int32)
            inputs = G.layers.InputLayer(input_var=input_var, shape=(None, 784))

        # Layer creation without a graph should raise error.
        with self.assertRaises(ValueError) as cm:
            _ = G.layers.DenseLayer(incoming=inputs, num_units=128, name='hidden1')
        self.assertTrue(not not re.search(r'No graph is activated.*', str(cm.exception)))

        # Check the catched variables during layer creation.
        with graph.as_default():
            hidden1 = G.layers.DenseLayer(incoming=inputs, num_units=128, name='hidden1')
            with G.name_scope('nested'):
                nested_hidden1 = G.layers.DenseLayer(incoming=inputs, num_units=128, name='hidden1')
                nested_hidden2 = G.layers.DenseLayer(incoming=nested_hidden1, num_units=32, name='hidden2')

        self.assertEquals(list(graph.iter_variables()),
                          [hidden1.W, hidden1.b, nested_hidden1.W, nested_hidden1.b, nested_hidden2.W,
                           nested_hidden2.b])

        for v, n in zip((hidden1.W, hidden1.b, nested_hidden1.W, nested_hidden1.b),
                        ('hidden1/W', 'hidden1/b', 'nested/hidden1/W', 'nested/hidden1/b')):
            self.assertEqual(G.get_variable_name(v), n)
            self.assertEqual(graph.get_variable_info(v).full_name, n)

        # Check that duplicated names would raise errors.
        with graph.as_default():
            with self.assertRaises(KeyError) as cm:
                _ = G.layers.DenseLayer(incoming=inputs, num_units=128, name='hidden1')
            self.assertTrue(re.search(r'Full name hidden1/W is already used by.*', str(cm.exception)))

        # Check that trainable & persistent variables include all the parameters.
        for tag in (G.VariableTags.TRAINABLE, G.VariableTags.PERSISTENT, G.VariableTags.RESUMABLE):
            self.assertEquals(graph.get_variables(**{tag: True}),
                              [hidden1.W, hidden1.b, nested_hidden1.W, nested_hidden1.b, nested_hidden2.W,
                               nested_hidden2.b])

        # Check that filtering by regularizable would only result in W.
        self.assertEquals(graph.get_variables(regularizable=True), [hidden1.W, nested_hidden1.W, nested_hidden2.W])

    def test_persistent(self):
        """Test graph persistent."""
        def mk_graph():
            graph = G.Graph()
            with graph.as_default():
                a = G.make_variable('a', shape=(), init=10, dtype=np.int32, persistent=True)
                b = G.make_variable('b', shape=(), init=20, dtype=np.int32, persistent=False)
            return graph, a, b

        def check_persists(persist_file, save_in_session, restore_in_session):
            graph, a, b = mk_graph()
            with G.Session(graph) as session:
                session.set_variable_values({a: 11, b: 21})
                if save_in_session:
                    G.utils.save_graph_state(graph, persist_file)
                self.assertEquals(session.get_variable_values((a, b)), (11, 21))

            if not save_in_session:
                G.utils.save_graph_state(graph, persist_file)

            graph, a, b = mk_graph()
            if restore_in_session:
                with G.Session(graph) as session:
                    G.utils.restore_graph_state(graph, persist_file)
                    self.assertEquals(session.get_variable_values((a, b)), (11, 20))
            else:
                G.utils.restore_graph_state(graph, persist_file)
                self.assertEquals(graph.get_last_values((a, b)), (11, None))
                with G.Session(graph) as session:
                    self.assertEquals(session.get_variable_values((a, b)), (11, 20))

        with tempdir.TemporaryDirectory() as tmpdir:
            persists = [os.path.join(tmpdir, '%d.pkl' % i) for i in range(4)]
            check_persists(persists[0], False, False)
            check_persists(persists[1], False, True)
            check_persists(persists[2], True, False)
            check_persists(persists[3], True, True)

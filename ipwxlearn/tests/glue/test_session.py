# -*- coding: utf-8 -*-
import unittest

import numpy as np

from ipwxlearn.glue import G


class SessionTestCase(unittest.TestCase):

    def test_session_vars(self):
        """Test save/restore/init variables for a session."""

        with G.Graph().as_default() as graph:
            a = G.make_variable('a', (), 1, dtype=np.int32, persistent=True)
            b = G.make_variable('b', (), 2, dtype=np.int32, persistent=True)
            c = G.make_variable('c', (), 3, dtype=np.int32)

            with G.Session(graph):
                self.assertEqual(G.get_variable_values([a, b, c]), (1, 2, 3))
                G.set_variable_values({a: 10, b: 20, c: 30})
                self.assertEqual(G.get_variable_values([a, b, c]), (10, 20, 30))
            self.assertEqual(graph.get_last_values([a, b, c]), (10, 20, None))
            self.assertDictEqual(graph.get_last_values_as_dict([a, b, c]), {a: 10, b: 20})

            with G.Session(graph):
                self.assertEqual(G.get_variable_values([a, b, c]), (10, 20, 3))
            self.assertEqual(graph.get_last_values([a, b, c]), (10, 20, None))

            with G.Session(graph, feed_values={b: 200, c: 300}):
                self.assertEqual(G.get_variable_values([a, b, c]), (10, 200, 300))
            self.assertEqual(graph.get_last_values([a, b, c]), (10, 200, None))

            with G.Session(graph):
                self.assertEqual(G.get_variable_values([a, b, c]), (10, 200, 3))
            self.assertEqual(graph.get_last_values([a, b, c]), (10, 200, None))

            with G.Session(graph, init_variables=True):
                self.assertEqual(G.get_variable_values([a, b, c]), (1, 2, 3))
            self.assertEqual(graph.get_last_values([a, b, c]), (1, 2, None))

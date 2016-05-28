# -*- coding: utf-8 -*-
import unittest

import numpy as np

from ipwxlearn.glue import G


class FunctionTestCase(unittest.TestCase):

    def test_make_function(self):
        """Test make function."""
        graph = G.Graph()
        with graph.as_default():
            a = G.make_placeholder('a', shape=(), dtype=np.int32)
            b = G.make_placeholder('b', shape=(), dtype=np.int32)
            c = G.make_placeholder('c', shape=(), dtype=np.int32)
            fn = G.make_function(inputs=[a, b], outputs=(a + b + c), givens={c: np.array(1000, dtype=np.int32)})
            self.assertEqual(fn(1, 2), 1003)

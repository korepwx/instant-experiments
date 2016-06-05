# -*- coding: utf-8 -*-
import unittest

import numpy as np

from ipwxlearn import glue
from ipwxlearn.glue import G


class EmbeddingTestCase(unittest.TestCase):

    def test_lookup(self):
        """Test the lookup of embedding layer."""

        input_size = 10086
        output_size = 256
        W = np.random.random((input_size, output_size)).astype(glue.config.floatX)

        graph = G.Graph()
        with graph.as_default():
            input_var = G.make_placeholder('X', shape=(None,), dtype=np.int32)
            input_layer = G.layers.InputLayer(input_var, shape=(None,))
            embed_layer = G.layers.EmbeddingLayer('embed', input_layer, input_size, output_size, W=W)
            output = G.layers.get_output(embed_layer)
            get_output = G.make_function(inputs=input_var, outputs=output)

        with G.Session(graph):
            indices = np.random.randint(low=0, high=input_size, size=(1024,)).astype(np.int32)
            self.assertTrue(np.allclose(W[indices], get_output(indices)))

# -*- coding: utf-8 -*-
import unittest

from ipwxlearn.glue import G


class LayersTestCase(unittest.TestCase):

    def test_tags(self):
        graph = G.Graph()

        with graph.as_default():
            # graph.add_variable()
            # TODO: complete tags testing.
            pass

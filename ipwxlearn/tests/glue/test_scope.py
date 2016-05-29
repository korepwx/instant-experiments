# -*- coding: utf-8 -*-
import unittest

from ipwxlearn.glue import G


class ScopeTestCase(unittest.TestCase):

    def test_tags(self):
        """Test filtering variables with tags in a graph."""
        with G.Graph().as_default() as graph:
            v1 = G.make_variable('trainable', shape=(), init=0.0, trainable=True)
            self.assertIn('trainable', graph.get_variable_info('trainable').tags)
            self.assertIn('persistent', graph.get_variable_info('trainable').tags)
            self.assertIn('resumable', graph.get_variable_info(v1).tags)

            v2 = G.make_variable('trainable_but_not_persistent', shape=(), init=0.0, trainable=True, persistent=False)
            self.assertIn('trainable', graph.get_variable_info('trainable_but_not_persistent').tags)
            self.assertNotIn('persistent', graph.get_variable_info('trainable_but_not_persistent').tags)
            self.assertNotIn('resumable', graph.get_variable_info(v2).tags)

            v3 = G.make_variable('no_tags', shape=(), init=0.0)
            self.assertNotIn('trainable', graph.get_variable_info('no_tags').tags)
            self.assertNotIn('persistent', graph.get_variable_info('no_tags').tags)
            self.assertNotIn('resumable', graph.get_variable_info(v3).tags)

            self.assertEqual(graph.get_variables(trainable=True), [v1, v2])
            self.assertEqual(graph.get_variables(trainable=True, persistent=True), [v1])

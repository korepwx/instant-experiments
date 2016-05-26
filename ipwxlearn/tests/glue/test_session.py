# -*- coding: utf-8 -*-
import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from ipwxlearn.glue import G


class SessionTestCase(unittest.TestCase):

    def test_session_vars(self):
        """Test save/restore/init variables for a session."""
        graph = G.Graph()

        with graph.as_default():
            a = G.make_variable('a', (), 1, dtype=np.int32, persistent=True)
            b = G.make_variable('b', (), 2, dtype=np.int32, persistent=True)
            c = G.make_variable('c', (), 3, dtype=np.int32)

        with self.assertRaises(ValueError, msg='No graph is activated.'):
            with G.Session():
                pass

        with graph.as_default():
            with G.Session():
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

    def test_checkpoint(self):
        graph = G.Graph()

        with graph.as_default():
            a = G.make_variable('a', (), 1, dtype=np.int32, trainable=True)
            b = G.make_variable('b', (), 2, dtype=np.int32, persistent=True)
            c = G.make_variable('c', (), 3, dtype=np.int32, resumable=True)
            d = G.make_variable('d', (), 4, dtype=np.int32)

        with TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.pkl')

            with G.Session(graph, checkpoint_file=path, max_checkpoints=3) as sess:
                self.assertEqual(sess.next_checkpoint_index, 1)

                G.set_variable_values({a: 10, b: 20, c: 30, d: 40})
                sess.checkpoint()
                self.assertEqual(sess.next_checkpoint_index, 2)
                self.assertTrue(os.path.isfile('%s.1' % path))

            with G.Session(graph, checkpoint_file=path, max_checkpoints=3) as sess:
                self.assertEqual(sess.next_checkpoint_index, 2)
                self.assertEqual(G.get_variable_values([a, b, c, d]), (10, 20, 30, 4))

                G.set_variable_values({a: 11, b: 21, c: 31, d: 41})
                sess.checkpoint()
                self.assertEqual(sess.next_checkpoint_index, 3)
                self.assertTrue(os.path.isfile('%s.1' % path))
                self.assertTrue(os.path.isfile('%s.2' % path))

            with G.Session(graph) as sess:
                self.assertEqual(sess.next_checkpoint_index, 1)
                self.assertEqual(G.get_variable_values([a, b, c, d]), (11, 21, 3, 4))
                G.set_variable_values({a: 100, b: 200, c: 300, d: 400})

            with G.Session(graph, checkpoint_file=path, max_checkpoints=3) as sess:
                self.assertEqual(sess.next_checkpoint_index, 3)
                self.assertEqual(G.get_variable_values([a, b, c, d]), (11, 21, 31, 4))

                G.set_variable_values({a: 12, b: 22, c: 32, d: 42})
                sess.checkpoint()
                self.assertEqual(sess.next_checkpoint_index, 4)
                for idx in (1, 2, 3):
                    self.assertTrue(os.path.isfile('%s.%d' % (path, idx)))

                G.set_variable_values({a: 13, b: 23, c: 33, d: 43})
                sess.checkpoint()
                self.assertEqual(sess.next_checkpoint_index, 5)
                self.assertFalse(os.path.isfile('%s.1' % path))
                for idx in (2, 3, 4):
                    self.assertTrue(os.path.isfile('%s.%d' % (path, idx)))

            with G.Session(graph, checkpoint_file=path, max_checkpoints=2) as sess:
                self.assertEqual(sess.next_checkpoint_index, 5)
                self.assertEqual(G.get_variable_values([a, b, c, d]), (13, 23, 33, 4))

                sess.checkpoint()
                self.assertEqual(sess.next_checkpoint_index, 6)
                for idx in (1, 2, 3):
                    self.assertFalse(os.path.isfile('%s.%d' % (path, idx)))
                for idx in (4, 5):
                    self.assertTrue(os.path.isfile('%s.%d' % (path, idx)))

# -*- coding: utf-8 -*-
import unittest

import numpy as np

from ipwxlearn import glue
from ipwxlearn.glue import G


class InitTestCase(unittest.TestCase):

    def test_normalized_uniform(self):
        """Test normalized uniform initializer."""
        def test(shape, axis, norm, norm_type):
            graph = G.Graph()
            with graph.as_default():
                init = G.init.NormalizedUniform(axis=axis, norm=norm, norm_type=norm_type)
                X = G.make_variable('X', shape=shape, init=init, dtype=glue.config.floatX)
            with G.Session(graph) as session:
                val = session.get_variable_values(X)
                self.assertEquals(val.shape, shape, msg='test(%r, %r, %r, %r): expect shape %r, got %r.' %
                                                        (shape, axis, norm, norm_type, shape, val.shape))

                if norm_type == 'l1':
                    val_norm = np.sum(np.abs(val), axis=axis, keepdims=True)
                elif norm_type == 'l2':
                    val_norm = np.sqrt(np.sum(val ** 2, axis=axis, keepdims=True))

                self.assertTrue(np.allclose(norm, val_norm), msg='test(%r, %r, %r, %r): norm mismatch' %
                                                                 (shape, axis, norm, norm_type))

        for shape in [(51,), (51, 53), (51, 1, 53), (51, 53, 57)]:
            for axis in range(0, len(shape)):
                for norm in (0.1, 1.0, 10.0):
                    for norm_type in ('l1', 'l2'):
                        test(shape, axis, norm, norm_type)

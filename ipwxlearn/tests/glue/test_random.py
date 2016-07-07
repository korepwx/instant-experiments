# -*- coding: utf-8 -*-
from __future__ import absolute_import

import unittest

import numpy as np

from ipwxlearn import glue
from ipwxlearn.glue import G


class RandomTestCase(unittest.TestCase):

    def _do_test_distribution(self, op, shape, n=1, dtype=glue.config.floatX,
                              minimum=None, maximum=None, expected_mean=None,
                              stddev=None, *args, **kwargs):
        graph = G.Graph()
        with graph.as_default():
            gen = G.make_function(outputs=op(shape=shape, n=n, dtype=dtype, *args, **kwargs))
            with G.Session():
                x = gen()
            self.assertEquals(x.shape, tuple(shape))
            if minimum is not None:
                self.assertGreater(np.min(x), minimum - 1e-7)
            if maximum is not None:
                self.assertLess(np.min(x), maximum + 1e-7)
            if expected_mean is not None and stddev is not None:
                # if diff is larger than 3-sigma, we consider it abnormal.
                self.assertLess(np.abs(np.mean(x) - expected_mean), 3. * stddev / (np.size(x) ** .5))

    def test_binomial(self):
        """Test binomial distribution."""
        shape = [21, 33, 1, 47]
        self._do_test_distribution(G.random.binomial, shape=shape, n=1, p=.5,
                                   minimum=0., maximum=1., expected_mean=.5, stddev=np.sqrt(.25))
        self._do_test_distribution(G.random.binomial, shape=shape, n=10, p=.3,
                                   minimum=0., maximum=10., expected_mean=3., stddev=np.sqrt(2.1))

# -*- coding: utf-8 -*-
from __future__ import absolute_import

import unittest

import numpy as np

from ipwxlearn import glue
from ipwxlearn.glue import G


class RandomTestCase(unittest.TestCase):

    shape = [21, 33, 1, 47]

    def _do_test_distribution(self, op, shape, dtype=glue.config.floatX,
                              minimum=None, maximum=None, expected_mean=None,
                              expected_stddev=None, *args, **kwargs):
        graph = G.Graph()
        with graph.as_default():
            gen = G.make_function(outputs=op(shape=shape, dtype=dtype, *args, **kwargs))
            with G.Session():
                x = gen()
            self.assertEquals(x.shape, tuple(shape))
            if minimum is not None:
                self.assertGreater(np.min(x), minimum - 1e-7)
            if maximum is not None:
                self.assertLess(np.min(x), maximum + 1e-7)
            if expected_mean is not None and expected_stddev is not None:
                # if diff is larger than 3-sigma, we consider it abnormal.
                self.assertLess(np.abs(np.mean(x) - expected_mean), 3. * expected_stddev / np.sqrt(np.size(x)))

    def test_binomial(self):
        """Test binomial distribution."""
        self._do_test_distribution(G.random.binomial, shape=self.shape, n=1, p=.5,
                                   minimum=0., maximum=1., expected_mean=.5, expected_stddev=np.sqrt(.25))
        self._do_test_distribution(G.random.binomial, shape=self.shape, n=10, p=.3,
                                   minimum=0., maximum=10., expected_mean=3., expected_stddev=np.sqrt(2.1))

    def test_uniform(self):
        """Test uniform distribution."""
        self._do_test_distribution(G.random.uniform, shape=self.shape,
                                   minimum=0., maximum=1., expected_mean=0.5, expected_stddev=np.sqrt(1. / 12))
        self._do_test_distribution(G.random.uniform, shape=self.shape, low=-1., high=1.,
                                   minimum=-1., maximum=1., expected_mean=0., expected_stddev=np.sqrt(4. / 12))
        self._do_test_distribution(G.random.uniform, shape=self.shape, low=-10., high=10.,
                                   minimum=-10., maximum=10., expected_mean=0., expected_stddev=np.sqrt(400. / 12))
        self._do_test_distribution(G.random.uniform, shape=self.shape, low=-1., high=101.,
                                   minimum=-1., maximum=101., expected_mean=50., expected_stddev=np.sqrt(102.**2 / 12))
        self._do_test_distribution(G.random.uniform, shape=self.shape, low=0., high=1e-2,
                                   minimum=0., maximum=1e-2, expected_mean=5e-3, expected_stddev=np.sqrt(1e-4 / 12))

    def test_normal(self):
        """Test normal distribution."""
        self._do_test_distribution(G.random.normal, shape=self.shape, mean=0., stddev=1.,
                                   expected_mean=0., expected_stddev=1.)
        self._do_test_distribution(G.random.normal, shape=self.shape, mean=3.14, stddev=4.7,
                                   expected_mean=3.14, expected_stddev=4.7)
        self._do_test_distribution(G.random.normal, shape=self.shape, mean=-1000.0, stddev=1e-2,
                                   expected_mean=-1000.0, expected_stddev=1e-2)

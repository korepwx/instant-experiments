# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

from .graph import current_graph

__all__ = [
    'binomial'
    'uniform',
    'normal',
]


def _get_rng(seed):
    return current_graph().random_state if seed is None else RandomStreams(seed)


def binomial(shape, p, n=1, dtype=np.int32, seed=None):
    """
    Generate a random tensor according to binomial experiments.

    :param shape: Shape of the result tensor.
    :param p: Probability of each trial to be success.
    :param n: Number of trials carried out for each element.
              If n = 1, each element is just the result in a binomial experiment.
              If n > 1, each element is the total count of success in n-repeated binomial experiments.
    :param dtype: Data type of the returning tensor.
    :param seed: Specify the random seed for this operation.
                 Share the random state of current graph if not specified.
    """
    return _get_rng(seed).binomial(size=shape, n=n, p=p, dtype=dtype)


def uniform(shape, low=0.0, high=1.0, dtype=None, seed=None):
    """
    Generate a random tensor following uniform distribution.

    :param shape: Shape of the result tensor.
    :param low: Minimum value of the uniform distribution.
    :param high: Maximum value of the uniform distribution.
    :param dtype: Data type of the returning tensor.
    :param seed: Specify the random seed for this operation.
                 Share the random state of current graph if not specified.
    """
    from ipwxlearn import glue
    return _get_rng(seed).uniform(size=shape, low=low, high=high, dtype=dtype or glue.config.floatX)


def normal(shape, mean, stddev, dtype=None, seed=None):
    """
    Generate a random tensor following normal distribution.

    :param shape: Shape of the result tensor.
    :param mean: Mean of the normal distribution.
    :param stddev: Standard derivation of the normal distribution.
    :param dtype: Data type of the returning tensor.
    :param seed: Specify the random seed for this operation.
                 Share the random state of current graph if not specified.
    """
    from ipwxlearn import glue
    return _get_rng(seed).normal(size=shape, avg=mean, std=stddev, dtype=dtype or glue.config.floatX)

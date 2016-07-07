# -*- coding: utf-8 -*-
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

from .graph import current_graph

__all__ = [
    'binomial'
]


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
    rng = current_graph().random_state if seed is None else RandomStreams(seed)
    return rng.binomial(size=shape, n=n, p=p, dtype=dtype)

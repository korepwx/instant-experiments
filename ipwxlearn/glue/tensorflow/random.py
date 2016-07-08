# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

__all__ = [
    'binomial',
    'uniform',
    'normal',
]


def _expand_shape(shape, n):
    if isinstance(shape, tf.TensorShape):
        return shape.concatenate([n])
    if isinstance(shape, (list, tuple)):
        return tuple(shape) + (n,)
    return tf.concat(0, [shape, [n]])


def _shape_length(shape):
    if isinstance(shape, tf.TensorShape):
        return shape.ndims
    if isinstance(shape, (list, tuple)):
        return len(shape)
    return tf.size(shape)


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
    with tf.op_scope([], 'binomial'):
        if n > 1:
            random_shape = _expand_shape(shape, n)
        else:
            random_shape = shape
        x = tf.random_uniform(shape=random_shape, minval=0., maxval=1., seed=seed)
        x = tf.cast(x <= p, dtype=tf.as_dtype(dtype))
        if n > 1:
            x = tf.reduce_sum(x, _shape_length(shape))
        return x


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
    return tf.random_uniform(shape=shape, minval=low, maxval=high, dtype=dtype or glue.config.floatX, seed=seed)


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
    return tf.random_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype or glue.config.floatX, seed=seed)

# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

__all__ = [
    'binomial'
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
    # TODO: test this binomial function.
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

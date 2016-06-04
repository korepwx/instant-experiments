# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import six
import tensorflow as tf

from .base import Layer

__all__ = ['MaxPool2DLayer', 'AvgPool2DLayer']

#: Lookup table to translate pooling mode to TensorFlow function.
POOLING_LOOKUP = {
    'max': tf.nn.max_pool,
    'average': tf.nn.avg_pool
}


class Pool2DLayer(Layer):
    """
    2D pooling layer.

    Performs 2D mean or max-pooling over 2D convolution layer.

    :param name: Name of this pooling layer.
    :param incoming: Incoming layer as the input.
    :param pool_size: Size of the pooling region, an integer or an iterable of integers.
    :param stride: The strides between sucessive pooling regions in each dimension.
                   If not specified, just use the :param:`pool_size`.
    :param pad: Number of zeros to pad at each side.  The padded zeros will have effect in pooling.
    :param mode: Pooling mode, one of {'max', 'mean'}
    """

    def __init__(self, name, incoming, pool_size, stride=None, pad=None, mode='max'):
        super(Pool2DLayer, self).__init__(incoming=incoming, name=name)

        f = lambda v: (v,) * 2 if isinstance(v, six.integer_types) else tuple(v)
        self.pool_size = f(pool_size)
        self.stride = f(stride) if stride is not None else self.pool_size
        self.pad = f(pad) if pad is not None else (0, 0)
        self.mode = mode

        assert(len(self.pool_size) == 2)
        assert(len(self.stride) == 2)
        assert(len(self.pad) == 2)

    def get_output_shape_for(self, input_shape):
        data_shape = tuple(
            (v + 2 * self.pad[i] + self.stride[i] - 1) // self.stride[i]
            for i, v in enumerate(input_shape[1: -1])
        )
        return (input_shape[0],) + data_shape + (input_shape[-1],)

    def get_output_for(self, input, **kwargs):
        input_shape = input.get_shape().as_list()
        assert (len(input_shape) == 4)

        ksize = (1,) + self.pool_size + (1,)
        strides = (1,) + self.stride + (1,)
        pooling = POOLING_LOOKUP[self.mode]

        if any(i > 0 for i in self.pad):
            paddings = np.asarray([[0,0]] + [[p, p] for p in self.pad] + [[0,0]], dtype=np.int32)
            input = tf.pad(input, paddings, "CONSTANT")

        return pooling(input, ksize, strides, 'VALID')


class MaxPool2DLayer(Pool2DLayer):
    """2D max-pooling layer."""

    def __init__(self, name, incoming, pool_size, stride=None, pad=None):
        super(MaxPool2DLayer, self).__init__(name=name, incoming=incoming, pool_size=pool_size, stride=stride,
                                             pad=pad, mode='max')


class AvgPool2DLayer(Pool2DLayer):
    """2D average-pooling layer."""

    def __init__(self, name, incoming, pool_size, stride=None, pad=None):
        super(AvgPool2DLayer, self).__init__(name=name, incoming=incoming, pool_size=pool_size, stride=stride,
                                             pad=pad, mode='average')


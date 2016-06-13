# -*- coding: utf-8 -*-
from __future__ import absolute_import

import six
import tensorflow as tf

from ipwxlearn.glue.common.pool import PoolPadType
from .base import Layer

__all__ = [
    'PoolPadType',
    'MaxPool2DLayer',
    'AvgPool2DLayer'
]

#: Lookup table to translate pooling mode to TensorFlow function.
POOLING_FUNC_LOOKUP = {
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
    :param padding: Way to pad the input data.  See :class:`~ipwxlearn.glue.common.pool.PoolPadType`.
    :param mode: Pooling mode, one of {'max', 'average'}
    """

    def __init__(self, name, incoming, pool_size, stride=None, padding=PoolPadType.NONE, mode='max'):
        super(Pool2DLayer, self).__init__(name=name, incoming=incoming)

        f = lambda v: (v,) * 2 if isinstance(v, six.integer_types) else tuple(v)
        self.pool_size = f(pool_size)
        self.stride = f(stride) if stride is not None else self.pool_size
        self.padding = padding
        self.mode = mode

        if self.padding == PoolPadType.NONE:
            self._conv_pad = 'VALID'
        elif self.padding in (PoolPadType.BACKEND, PoolPadType.SAME):
            self._conv_pad = 'SAME'
        else:
            raise ValueError('Unsupported padding type %r.' % self.padding)
        self._conv_func = POOLING_FUNC_LOOKUP[mode]

        assert(len(self.pool_size) == 2)
        assert(len(self.stride) == 2)

    def get_output_shape_for(self, input_shape):
        if self.padding == PoolPadType.NONE:
            output_size_off = [1 - s for s in self.pool_size]
        else:
            output_size_off = [0] * len(self.pool_size)
        data_shape = tuple(
            (v + output_size_off[i] + self.stride[i] - 1) // self.stride[i]
            for i, v in enumerate(input_shape[1: -1])
        )
        assert(all(v > 0 for v in data_shape))
        return (input_shape[0],) + data_shape + (input_shape[-1],)

    def get_output_for(self, input, **kwargs):
        input_shape = input.get_shape().as_list()
        assert (len(input_shape) == 4)

        ksize = (1,) + self.pool_size + (1,)
        strides = (1,) + self.stride + (1,)

        return self._conv_func(input, ksize, strides, self._conv_pad)


class MaxPool2DLayer(Pool2DLayer):
    """2D max-pooling layer."""

    def __init__(self, name, incoming, pool_size, stride=None, padding=PoolPadType.NONE):
        super(MaxPool2DLayer, self).__init__(name=name, incoming=incoming, pool_size=pool_size, stride=stride,
                                             padding=padding, mode='max')


class AvgPool2DLayer(Pool2DLayer):
    """2D average-pooling layer."""

    def __init__(self, name, incoming, pool_size, stride=None, padding=PoolPadType.NONE):
        super(AvgPool2DLayer, self).__init__(name=name, incoming=incoming, pool_size=pool_size, stride=stride,
                                             padding=padding, mode='average')

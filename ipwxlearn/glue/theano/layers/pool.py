# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne
import six

from ipwxlearn.glue.common.pool import PoolPadType
from .base import Layer

__all__ = [
    'PoolPadType',
    'MaxPool2DLayer',
    'AvgPool2DLayer'
]

#: Lookup table to translate pooling mode to Lasagne mode string.
POOLING_MODE_LOOKUP = {'max': 'max', 'average': 'average_exc_pad'}


class Pool2DLayer(lasagne.layers.Pool2DLayer, Layer):
    """
    2D pooling layer.

    Performs 2D mean or max-pooling over 2D convolution layer.

    :param name: Name of this pooling layer.
    :param incoming: Incoming layer as the input, or a shape tuple.
    :param pool_size: Size of the pooling region, an integer or an iterable of integers.
    :param stride: The strides between sucessive pooling regions in each dimension.
                   If not specified, just use the :param:`pool_size`.
    :param padding: Way to pad the input data.  See :class:`~ipwxlearn.glue.common.pool.PoolPadType`.
    :param mode: Pooling mode, one of {'max', 'average'}
    """

    def __init__(self, name, incoming, pool_size, stride=None, padding=PoolPadType.NONE, mode='max'):
        # normalize the arguments.
        f = lambda v: (v,) * 2 if isinstance(v, six.integer_types) else tuple(v)
        pool_size = f(pool_size)
        stride = f(stride) if stride is not None else pool_size

        # construct the layer.
        if padding == PoolPadType.NONE:
            pad = (0, 0)
        elif padding in (PoolPadType.BACKEND, PoolPadType.SAME):
            pad = tuple(k // 2 for k in pool_size)
        else:
            raise ValueError('Unsupported padding type %r.' % padding)
        self.padding = padding

        super(Pool2DLayer, self).__init__(incoming=incoming, name=name, pool_size=pool_size, stride=stride,
                                          pad=pad, ignore_border=True, mode=POOLING_MODE_LOOKUP[mode])

    def _compute_border_discarding(self, input_shape):
        input_shape = tuple(input_shape)
        if self.padding == PoolPadType.SAME:
            ret = tuple(
                ((d + 2 * (k // 2) - k + s) // s - (d + s - 1) // s)
                for d, k, s in zip(input_shape[2:], self.pool_size, self.stride)
            )
        else:
            ret = (0, 0)
        return ret

    def get_output_shape_for(self, input_shape):
        discards_border = self._compute_border_discarding(input_shape)
        output_shape = super(Pool2DLayer, self).get_output_shape_for(input_shape)
        output_shape = tuple(output_shape[:2]) + \
            tuple(v - d for v, d in zip(output_shape[2:], discards_border))
        return output_shape

    def get_output_for(self, input, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = self.input_shape

        discards_border = self._compute_border_discarding(input_shape)
        output = super(Pool2DLayer, self).get_output_for(input, **kwargs)
        if discards_border[0]:
            output = output[:, :, :-1, :]
        if discards_border[1]:
            output = output[:, :, :, :-1]
        return output


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

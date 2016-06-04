# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne
import six

from .base import Layer

__all__ = ['MaxPool2DLayer', 'AvgPool2DLayer']

#: Lookup table to translate pooling mode to Lasagne mode string.
POOLING_MODE_LOOKUP = {'max': 'max', 'average': 'average_inc_pad'}


class Pool2DLayer(lasagne.layers.Pool2DLayer, Layer):
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
        if pad is not None:
            pad = (pad,) * 2 if isinstance(pad, six.integer_types) else tuple(pad)
        else:
            pad = (0, 0)
        super(Pool2DLayer, self).__init__(incoming=incoming, name=name, pool_size=pool_size, stride=stride,
                                          pad=pad, ignore_border=True, mode=POOLING_MODE_LOOKUP[mode])


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

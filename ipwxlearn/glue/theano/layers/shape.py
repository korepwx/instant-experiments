# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne

from .base import Layer

__all__ = [
    'SliceLayer'
]


class SliceLayer(lasagne.layers.SliceLayer, Layer):
    """
    Slices the input at a specific axis and at specific indices.

    :param incoming: The layer feeding into this layer, or a shape tuple.
    :param indices: If an ``int``, selects a single element from the given axis, dropping
                    the axis. If a slice, selects all elements in the given range, keeping
                    the axis.
    :param axis: Specifies the axis from which the indices are selected.
    """

    def __init__(self, incoming, indices, axis=-1):
        super(SliceLayer, self).__init__(incoming=incoming, indices=indices, axis=axis)

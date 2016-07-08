# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne

from .base import Layer

__all__ = [
    'SliceLayer',
    'ReshapeLayer',
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


class ReshapeLayer(lasagne.layers.ReshapeLayer, Layer):
    """
    Layer that reshaping its input tensor to another of the same total number of elements.

    :param incoming: The layer feeding into this layer, or a shape tuple.
    :param shape: A tuple of the target shape specification.  Each element can be one of:

                  * ``i``, a positive integer directly giving the size of the dimension.
                  * ``[i]``, a single-element list of int, denoting to use the size of the
                    ``i``th input dimension.
                  * ``-1``, denoting to infer the size for this dimension to match the total
                    number of elements in the input tensor (can only be used only once).
                  * Tensor variable directly giving the size of the dimension.
    """

    def __init__(self, incoming, shape, name=None):
        super(ReshapeLayer, self).__init__(incoming, shape, name=name)

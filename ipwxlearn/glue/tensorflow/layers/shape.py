# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf

from .base import Layer

__all__ = [
    'SliceLayer'
]


class SliceLayer(Layer):
    """
    Slices the input at a specific axis and at specific indices.

    :param incoming: The layer feeding into this layer.
    :param indices: If an ``int``, selects a single element from the given axis, dropping
                    the axis. If a slice, selects all elements in the given range, keeping
                    the axis.
    :param axis: Specifies the axis from which the indices are selected.
    """

    def __init__(self, incoming, indices, axis=-1):
        if isinstance(indices, slice) and indices.step not in (None, 1):
            raise ValueError('TensorFlow backend has supported stepped slicing yet.')

        super(SliceLayer, self).__init__(incoming=incoming, name=None)
        self.slice = indices
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        if isinstance(self.slice, int):
            del output_shape[self.axis]
        else:
            start, stop, step = self.slice.indices(input_shape[self.axis])
            output_shape[self.axis] = (stop - start + step - 1) // step
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        input_shape = input.get_shape()
        input_dim = len(input_shape)

        # check the shape of the input tensor
        axis = self.axis
        if axis < 0:
            axis += input_dim
        if input_shape[axis].value is not None:
            dim_size = input_shape[axis].value
        else:
            dim_size = tf.shape(input)[axis]

        # compute begin and size.
        begin = [0] * input_dim
        size = [-1] * input_dim
        if isinstance(self.slice, slice):
            if isinstance(dim_size, int):
                start, stop, step = self.slice.indices(dim_size)
                begin[axis] = start
                size[axis] = stop - start
            else:
                start, stop, step = self.slice.start, self.slice.stop, self.slice.step
                if start is not None:
                    if start >= 0:
                        begin[axis] = start
                    else:
                        begin[axis] = dim_size + start
                if stop is not None:
                    if stop >= 0:
                        size[axis] = stop - begin[axis]
                    else:
                        size[axis] = stop + dim_size - begin[axis]
        else:
            if self.slice < 0:
                begin[axis] = self.slice + dim_size
            else:
                begin[axis] = self.slice
            size[axis] = 1

        # if begin and size relies on dynamic tensor, we need to convert them to a single tensor,
        # which is required by tf.slice.
        if not all(isinstance(v, int) for v in begin):
            begin = tf.pack(begin)
        if not all(isinstance(v, int) for v in size):
            size = tf.pack(size)

        output = tf.slice(input, begin, size)
        if isinstance(self.slice, int):
            output = tf.squeeze(output, [axis])

        return output

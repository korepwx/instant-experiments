# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from .base import Layer

__all__ = [
    'SliceLayer',
    'ReshapeLayer',
]


class SliceLayer(Layer):
    """
    Slices the input at a specific axis and at specific indices.

    :param incoming: The layer feeding into this layer, or a shape tuple.
    :param indices: If an ``int``, selects a single element from the given axis, dropping
                    the axis. If a slice, selects all elements in the given range, keeping
                    the axis.
    :param axis: Specifies the axis from which the indices are selected.
    """

    def __init__(self, incoming, indices, axis=-1):
        if isinstance(indices, slice) and indices.step not in (None, 1):
            raise ValueError('TensorFlow backend has supported stepped slicing yet.')

        super(SliceLayer, self).__init__(name=None, incoming=incoming)
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


class ReshapeLayer(Layer):
    """
    Layer that reshaping its input tensor to another of the same total number of elements.
    Many of the code is copied from Lasagne.

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
        super(ReshapeLayer, self).__init__(incoming, name=name)
        shape = tuple(shape)
        for s in shape:
            if isinstance(s, int):
                if s == 0 or s < -1:
                    raise ValueError('"shape" integers must be positive or -1')
            elif isinstance(s, list):
                if len(s) != 1 or not isinstance(s[0], int) or s[0] < 0:
                    raise ValueError('"shape" input references must be '
                                     'single-element lists of int >= 0')
            elif isinstance(s, (tf.Tensor, tf.Variable)):
                t = s.get_shape()
                if t.ndims != 0:
                    raise ValueError(
                        'A symbolic variable in a shape specification must be '
                        'a scalar, but had %i dimensions' % t.ndims)
            else:
                raise ValueError('"shape" must be a tuple of int and/or [int]')
        if sum(s == -1 for s in shape) > 1:
            raise ValueError('"shape" cannot contain multiple -1')
        self.shape = shape
        # try computing the output shape once as a sanity check
        self.get_output_shape_for(self.input_shape)

    def get_output_shape_for(self, input_shape):
        # Initialize output shape from shape specification
        output_shape = list(self.shape)
        # First, replace all `[i]` with the corresponding input dimension, and
        # mask parts of the shapes thus becoming irrelevant for -1 inference
        masked_input_shape = list(input_shape)
        masked_output_shape = list(output_shape)
        for dim, o in enumerate(output_shape):
            if isinstance(o, list):
                if o[0] >= len(input_shape):
                    raise ValueError('specification contains [%d], but input '
                                     'shape has %d dimensions only' %
                                     (o[0], len(input_shape)))
                output_shape[dim] = input_shape[o[0]]
                masked_output_shape[dim] = input_shape[o[0]]
                if input_shape[o[0]] is None and masked_input_shape[o[0]] is None:
                    # first time we copied this unknown input size: mask
                    # it, we have a 1:1 correspondence between out[dim] and
                    # in[o[0]] and can ignore it for -1 inference even if
                    # it is unknown.
                    masked_input_shape[o[0]] = 1
                    masked_output_shape[dim] = 1
        # Secondly, replace all symbolic shapes with `None`, as we cannot
        # infer their size here.
        for dim, o in enumerate(output_shape):
            if isinstance(o, (tf.Tensor, tf.Variable)):
                output_shape[dim] = None
                masked_output_shape[dim] = None
        # From the shapes, compute the sizes of the input and output tensor
        input_size = (None if any(x is None for x in masked_input_shape)
                      else np.prod(masked_input_shape))
        output_size = (None if any(x is None for x in masked_output_shape)
                       else np.prod(masked_output_shape))
        del masked_input_shape, masked_output_shape
        # Finally, infer value for -1 if needed
        if -1 in output_shape:
            dim = output_shape.index(-1)
            if input_size is None or output_size is None:
                output_shape[dim] = None
                output_size = None
            else:
                output_size *= -1
                output_shape[dim] = input_size // output_size
                output_size *= output_shape[dim]
        # Sanity check
        if (input_size is not None) and (output_size is not None) \
                and (input_size != output_size):
            raise ValueError("%s cannot be reshaped to specification %s. "
                             "The total size mismatches." %
                             (input_shape, self.shape))
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        # Replace all `[i]` with the corresponding input dimension
        input_shape = tf.shape(input)
        output_shape = list(self.shape)
        for dim, o in enumerate(output_shape):
            if isinstance(o, list):
                output_shape[dim] = input_shape[o[0]]
        # Everything else is handled by TensorFlow
        if any(isinstance(o, list) for o in output_shape):
            return tf.reshape(input, tf.pack(output_shape))
        return tf.reshape(input, tuple(output_shape))

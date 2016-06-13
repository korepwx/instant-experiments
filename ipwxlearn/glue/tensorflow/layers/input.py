# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .base import Layer

__all__ = ['InputLayer']


class InputLayer(Layer):
    """
    This layer holds a symbolic variable that represents a network input.

    :param input_var: Input variable for this layer.  Use :method:`make_placeholder` to create such variable.
    :param shape: Shape of the input variable.
    """

    def __init__(self, input_var, shape):
        self.name = self.full_name = None
        self.shape = shape
        if any(d is not None and d <= 0 for d in self.shape):
            raise ValueError("Could not create InputLayer with a non-positive shape %s." % self.shape)

        var_shape = tuple(input_var.get_shape().as_list())
        if var_shape != self.shape:
            raise ValueError('Shape %s does not match that of input variable %s.' % (var_shape, self.shape))

        self.input_var = input_var
        self.params = []

    @Layer.output_shape.getter
    def output_shape(self):
        return self.shape

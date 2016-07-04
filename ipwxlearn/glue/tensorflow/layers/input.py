# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .base import Layer

__all__ = [
    'InputLayer',
    'make_input',
]


class InputLayer(Layer):
    """
    This layer holds a symbolic variable that represents a network input.

    :param input_var: Input variable for this layer.  Use :method:`make_placeholder` to create such variable.
    :param shape: Shape of the input variable.
    """

    def __init__(self, input_var, shape):
        from ..graph import current_graph
        self.graph = current_graph()

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


def make_input(name, data, dtype=None, **tags):
    """
    Make an InputLayer for specified data.

    This method will construct the input layer, as well as the placeholder inside
    the input layer for specified data.  The placeholder will have the shape
    (None,) + data.shape[1:], and have the same dtype as the data.

    :param name: Name of the placeholder.
    :param data: Data to put into the placeholder.
    :param dtype: Specify a data type other than the data.
    :param tags: Tags for this placeholder.

    :return: (input_layer, input_var)
    """
    from ..utils import make_placeholder_for
    input_var = make_placeholder_for(name, data, dtype=dtype, **tags)
    input_layer = InputLayer(input_var, shape=(None,) + data.shape[1:])
    return input_layer, input_var

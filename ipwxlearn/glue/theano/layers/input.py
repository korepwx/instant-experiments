# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne

from .base import Layer

__all__ = [
    'InputLayer',
    'make_input',
]


class InputLayer(lasagne.layers.InputLayer, Layer):
    """
    This layer holds a symbolic variable that represents a network input.

    :param input_var: Input variable for this layer.  Use :method:`make_placeholder` to create such variable.
    :param shape: Shape of the input variable.
    """

    def __init__(self, input_var, shape):
        from ..graph import current_graph
        self.graph = current_graph()

        lasagne.layers.InputLayer.__init__(self, shape=shape, input_var=input_var)
        _ = self.name_scope

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.input_var)


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

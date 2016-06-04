# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne

from ipwxlearn.glue.theano.layers import Layer

__all__ = ['InputLayer']


class InputLayer(lasagne.layers.InputLayer, Layer):
    """
    This layer holds a symbolic variable that represents a network input.

    :param input_var: Input variable for this layer.  Use :method:`make_placeholder` to create such variable.
    :param shape: Shape of the input variable.
    """

    def __init__(self, input_var, shape):
        super(InputLayer, self).__init__(shape=shape, input_var=input_var)

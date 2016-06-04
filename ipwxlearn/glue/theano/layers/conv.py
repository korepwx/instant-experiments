# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne

from .. import init, nonlinearities
from .base import Layer


class Conv2DLayer(lasagne.layers.Conv2DLayer, Layer):
    """
    2D convolutional layer.

    Performs a 2D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.


    """

    def __init__(self, name, incoming, num_filters, filter_size, stride=(1, 1), pad=0,
                 untie_biases=False, W=init.XavierNormal(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True):
        super(Conv2DLayer, self).__init__(
            name=name, incoming=incoming, num_filters=num_filters, filter_size=filter_size,
            stride=stride, pad=pad, untie_biases=untie_biases, W=W, b=b, nonlinearity=nonlinearity,
            flip_filters=flip_filters
        )
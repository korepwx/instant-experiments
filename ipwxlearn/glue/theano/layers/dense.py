# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne

from .base import Layer
from .. import init, nonlinearities

__all__ = ['DenseLayer']


class DenseLayer(lasagne.layers.DenseLayer, Layer):
    """
    Fully connected layer.

    :param name: Name of this layer.
    :param incoming: The layer feed into this layer, or a shape tuple.
    :param num_units: The number of units of this layer.
    :param W: Theano variable, numpy array, or an initializer.
    :param b: Theano variable, numpy array, or an initializer.
    :param nonlinearity: Nonlinear function as the layer activation, or None if the layer is linear.
    """

    def __init__(self, name, incoming, num_units, W=init.XavierNormal(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify):
        super(DenseLayer, self).__init__(
            incoming=incoming, num_units=num_units, W=W, b=b, nonlinearity=nonlinearity, name=name)

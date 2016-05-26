# -*- coding: utf-8 -*-

from ipwxlearn.glue.theano import init, nonlinearities
from ipwxlearn.glue.theano.layers.imports import DenseLayer

__all__ = [
    'SoftmaxLayer'
]


class SoftmaxLayer(DenseLayer):
    """
    Softmax layer.

    :param name: Name of this layer.
    :param incoming: The layer feed into this layer.
    :param num_units: The number of units of this layer.
    :param W: Theano variable, numpy array, or an initializer.
    :param b: Theano variable, numpy array, or an initializer.
    """

    def __init__(self, name, incoming, num_units, W=init.XavierNormal(), b=init.Constant(0.)):
        super(SoftmaxLayer, self).__init__(
            name=name, incoming=incoming, num_units=num_units, W=W, b=b, nonlinearity=nonlinearities.softmax)

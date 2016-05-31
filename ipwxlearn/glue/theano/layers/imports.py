# -*- coding: utf-8 -*-
import lasagne

from .base import _Layer
from .. import init, nonlinearities

__all__ = [
    'InputLayer',
    'DropoutLayer',
    'DenseLayer'
]


class InputLayer(lasagne.layers.InputLayer, _Layer):
    """
    This layer holds a symbolic variable that represents a network input.

    :param input_var: Input variable for this layer.  Use :method:`make_placeholder` to create such variable.
    :param shape: Shape of the input variable.
    """

    def __init__(self, input_var, shape):
        super(InputLayer, self).__init__(shape=shape, input_var=input_var)


class DropoutLayer(lasagne.layers.DropoutLayer, _Layer):
    """
    Dropout layer, which sets values to zero with probability p.

    :param name: Name of this layer.
    :param incoming: The layer feeding into this layer.
    :param p: The probability of setting a value to zero.
    :param rescale: If True, the input is rescaled with factor of 1 / (1-p) on training.
    """

    def __init__(self, name, incoming, p=0.5, rescale=True):
        super(DropoutLayer, self).__init__(incoming=incoming, p=p, rescale=rescale, name=name)


class DenseLayer(lasagne.layers.DenseLayer, _Layer):
    """
    Fully connected layer.

    :param name: Name of this layer.
    :param incoming: The layer feed into this layer.
    :param num_units: The number of units of this layer.
    :param W: Theano variable, numpy array, or an initializer.
    :param b: Theano variable, numpy array, or an initializer.
    :param nonlinearity: Nonlinear function as the layer activation, or None if the layer is linear.
    """

    def __init__(self, name, incoming, num_units, W=init.XavierNormal(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify):
        super(DenseLayer, self).__init__(
            incoming=incoming, num_units=num_units, W=W, b=b, nonlinearity=nonlinearity, name=name)

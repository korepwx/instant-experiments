# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from .base import Layer
from .. import init, nonlinearities

__all__ = [
    'DenseLayer'
]


class DenseLayer(Layer):
    """
    Fully connected layer.

    :param name: Name of this layer.
    :param incoming: The layer feed into this layer.
    :param num_units: The number of units of this layer.
    :param W: Theano variable, numpy array, or an initializer.
    :param b: Theano variable, numpy array, or an initializer.
              If set to None, will cancel the biases.
    :param nonlinearity: Nonlinear function as the layer activation, or None if the layer is linear.
    """

    def __init__(self, name, incoming, num_units, W=init.XavierNormal(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify):
        super(DenseLayer, self).__init__(name=name, incoming=incoming)
        self.nonlinearity = nonlinearity

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name='W')
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units, ), name='b', regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        input_shape = input.get_shape().as_list()
        if len(input_shape) > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            batch_size = input_shape[0] if input_shape[0] is not None else -1
            input = tf.reshape(input, [batch_size, np.prod(input_shape[1:])])

        activation = tf.matmul(input, self.W)
        if self.b is not None:
            activation = activation + self.b
        if self.nonlinearity is not None:
            activation = self.nonlinearity(activation)
        return activation

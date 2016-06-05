# -*- coding: utf-8 -*-
from __future__ import absolute_import

import six
import tensorflow as tf

from ipwxlearn.glue.common.conv import ConvPadType
from .base import Layer
from .. import init, nonlinearities

__all__ = [
    'ConvPadType',
    'Conv2DInputLayer',
    'Conv2DOutputLayer',
    'Conv2DLayer'
]


class BaseConvInputLayer(Layer):
    """
    Convolutional input layer.

    :param incoming: Predecessor layer as the input.
    :param ndim: Number of dimensions.
    """

    def __init__(self, incoming, ndim):
        super(BaseConvInputLayer, self).__init__(name=None, incoming=incoming)
        self.ndim = ndim

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        assert(self.ndim + 2 == len(input.get_shape()))
        return input


class BaseConvOutputLayer(Layer):
    """
    Convolutional output layer.

    This layer does exactly the opposite thing as convolutional input layer.

    :param incoming: Predecessor layer as the input.
    :param ndim: Number of dimensions.
    """

    def __init__(self, incoming, ndim):
        super(BaseConvOutputLayer, self).__init__(name=None, incoming=incoming)
        self.ndim = ndim

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        assert(self.ndim + 2 == len(input.get_shape()))
        return input


class Conv2DInputLayer(BaseConvInputLayer):
    """Convolutional input layer."""

    def __init__(self, incoming):
        super(Conv2DInputLayer, self).__init__(incoming, ndim=2)


class Conv2DOutputLayer(BaseConvOutputLayer):
    """Convolutional output layer."""

    def __init__(self, incoming):
        super(Conv2DOutputLayer, self).__init__(incoming, ndim=2)


class Conv2DLayer(Layer):
    """
    2D convolutional layer.

    Performs a 2D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.

    :param incoming: The incoming layer, with shape ``(batch_size, input_rows, input_columns, n_channels)``.
    :param num_filters: The number of learnable convolutional filters this layer has.
    :param filter_size: An integer or a 2-element tuple specifying the size of the filters.
    :param stride: An integer or a 2-element tuple specifying the stride of the convolution operation.
    :param padding: One of the padding types specified in :class:`~ipwxlearn.glue.common.conv.ConvPadType`.
    :param untie_biases: If False, the layer will have only one bias parameter for each channel.
                         If True, the layer will have separate bias parameters for each position
                         in each channel.
    :param W: Theano variable, numpy array, or an initializer.
    :param b: Theano variable, numpy array, or an initializer.
    :param nonlinearity: Nonlinear function as the layer activation, or None if the layer is linear.
    """

    def __init__(self, name, incoming, num_filters, filter_size, stride=(1, 1), padding=ConvPadType.VALID,
                 untie_biases=False, W=init.XavierNormal(), b=init.Constant(0.), nonlinearity=nonlinearities.rectify):
        super(Conv2DLayer, self).__init__(name, incoming)

        f = lambda v: (v,) * num_filters if isinstance(v, six.integer_types) else tuple(v)
        self.num_filters = num_filters
        self.filter_size = f(filter_size)
        self.stride = f(stride)
        self.padding = padding
        self.untie_biases = untie_biases
        self.nonlinearity = nonlinearity

        input_channel = self.input_shape[-1]
        output_shape = self.get_output_shape_for(self.input_shape)

        self.W = self.add_param(W, self.filter_size + (input_channel, num_filters,), name='W')
        if self.untie_biases:
            self.b = self.add_param(b, output_shape[1: -1] + (num_filters,), name='b')
        else:
            self.b = self.add_param(b, (num_filters,), name='b')

    def get_output_shape_for(self, input_shape):
        if self.padding == 'valid':
            output_size_off = [1 - s for s in self.filter_size]
        else:
            output_size_off = [0] * len(self.filter_size)
        data_shape = tuple(
            (v + output_size_off[i] + self.stride[i] - 1) // self.stride[i]
            for i, v in enumerate(input_shape[1: -1])
        )
        assert(all(v > 0 for v in data_shape))
        return (input_shape[0],) + data_shape + (self.num_filters,)

    def get_output_for(self, input, **kwargs):
        input_shape = input.get_shape().as_list()
        assert(len(input_shape) == 4)

        strides = (1,) + self.stride + (1,)
        activation = tf.nn.conv2d(input, filter=self.W, strides=strides, padding=self.padding.upper())
        if self.untie_biases:
            data_size = input_shape[0] if input_shape[0] else input.get_shape()[0]
            activation = activation + tf.tile(tf.expand_dims(self.b, 0), [data_size] + [1] * len(input_shape[1:]))
        else:
            activation = tf.nn.bias_add(activation, self.b)

        if self.nonlinearity:
            activation = self.nonlinearity(activation)
        return activation

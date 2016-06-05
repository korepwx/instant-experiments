# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne

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

    Theano convolution functions treat the inputs as 4-D tensors, while the dimensions are
    [sample, channel, dim1, ...].  However, as for images and many other types of data, the
    channel dimension would be the last, not the second dimension.  Because of this, we
    provide this input layer, to translate from the raw input data to the backend convolution
    data.

    :param incoming: Predecessor layer as the input.
    :param ndim: Number of dimensions.
    """

    def __init__(self, incoming, ndim):
        super(BaseConvInputLayer, self).__init__(incoming=incoming, name=None)
        self.ndim = ndim

    def get_output_shape_for(self, input_shape):
        assert(self.ndim + 2 == len(input_shape))
        return (input_shape[0], input_shape[-1]) + tuple(input_shape[1: -1])

    def get_output_for(self, input, **kwargs):
        assert(self.ndim + 2 == input.ndim)
        return input.dimshuffle((0, input.ndim - 1) + tuple(range(1, input.ndim - 1)))


class BaseConvOutputLayer(Layer):
    """
    Convolutional output layer.

    This layer does exactly the opposite thing as convolutional input layer.

    :param incoming: Predecessor layer as the input.
    :param ndim: Number of dimensions.
    """

    def __init__(self, incoming, ndim):
        super(BaseConvOutputLayer, self).__init__(incoming=incoming, name=None)
        self.ndim = ndim

    def get_output_shape_for(self, input_shape):
        assert(self.ndim + 2 == input.ndim)
        return (input_shape[0], ) + tuple(input_shape[2:]) + (input_shape[1], )

    def get_output_for(self, input, **kwargs):
        assert(self.ndim + 2 == input.ndim)
        return input.dimshuffle((0, ) + tuple(range(2, input.ndim)) + (1, ))


class Conv2DInputLayer(BaseConvInputLayer):
    """Convolutional input layer."""

    def __init__(self, incoming):
        super(Conv2DInputLayer, self).__init__(incoming, ndim=2)


class Conv2DOutputLayer(BaseConvOutputLayer):
    """Convolutional output layer."""

    def __init__(self, incoming):
        super(Conv2DOutputLayer, self).__init__(incoming, ndim=2)


class Conv2DLayer(lasagne.layers.Conv2DLayer, Layer):
    """
    2D convolutional layer.

    Performs a 2D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.

    :param incoming: The incoming layer, with shape ``(batch_size, n_channels, input_rows, input_columns)``.
                     For ordinary images where the channel appears at last, use :class:`Conv2DInputLayer`
                     to re-arrange the dimensions.
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
        super(Conv2DLayer, self).__init__(
            name=name, incoming=incoming, num_filters=num_filters, filter_size=filter_size,
            stride=stride, pad=padding, untie_biases=untie_biases, W=W, b=b, nonlinearity=nonlinearity
        )

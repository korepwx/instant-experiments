# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne
from theano import tensor as T

from .helpers import get_output
from .dense import DenseLayer
from .. import init, nonlinearities

__all__ = [
    'SoftmaxLayer',
    'get_output_with_sparse_softmax_crossentropy'
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
        # if the target num is only 2, we use sigmoid instead of softmax.
        true_num_units = num_units if num_units > 2 else 1
        activation = nonlinearities.softmax if num_units > 2 else nonlinearities.sigmoid
        super(SoftmaxLayer, self).__init__(
            name=name, incoming=incoming, num_units=true_num_units, W=W, b=b, nonlinearity=activation)

        # although we might actually use sigmoid with 1 output, we still want the output shape to be 2.
        self.num_units = num_units

    def get_output_for(self, input, sigmoid=False, **kwargs):
        output = super(SoftmaxLayer, self).get_output_for(input, **kwargs)
        if self.num_units == 2 and not sigmoid:
            output = T.concatenate([1.0 - output, output], axis=1)
        return output


def get_output_with_sparse_softmax_crossentropy(layer, labels, inputs=None, **kwargs):
    """
    Get the output tensor for given softmax layer, as well as the cross entropy given integer labels.

    :param layer: Softmax layer.
    :param labels: Integer labels as targets.
    :param inputs: Dict with some input layers as keys and numeric scalars or numpy arrays as values,
                   causing these input layers to be substituted by constant values.
    :param kwargs: Additional parameters passed to :method:`Layer.get_output`.

    :return: A tuple of two tensors: (output, loss)
    """
    if not isinstance(layer, SoftmaxLayer):
        raise TypeError('Expect a SoftmaxLayer, got %r' % layer)

    kwargs_sigmoid = kwargs.copy()
    kwargs_sigmoid['sigmoid'] = True
    output = get_output(layer, inputs=inputs, **kwargs_sigmoid)

    if layer.num_units == 2:
        output_flatten = output.flatten(ndim=1)
        # Sigmoid activation in sigmoid might goes up to 1.0 or down to 0.0,
        # which would make the grad becomes NaN.  In order to avoid this overflow
        # or underflow, we clip the output here, so that it would not cause trouble.
        output_flatten = T.clip(output_flatten, 1e-7, 1.0 - 1e-7)
        loss = lasagne.objectives.binary_crossentropy(output_flatten, labels)

        # Finally, we complete the output probability.
        output = T.concatenate([1.0 - output, output], axis=1)
    else:
        loss = lasagne.objectives.categorical_crossentropy(output, labels)
    return output, loss

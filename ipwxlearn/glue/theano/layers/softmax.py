# -*- coding: utf-8 -*-
import lasagne

from .helpers import get_output
from .imports import DenseLayer
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
        super(SoftmaxLayer, self).__init__(
            name=name, incoming=incoming, num_units=num_units, W=W, b=b, nonlinearity=nonlinearities.softmax)


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
    output = get_output(layer, inputs=inputs, **kwargs)
    # if layer.num_units == 2:
    #    loss = lasagne.objectives.binary_crossentropy(output, labels)
    # else:
    loss = lasagne.objectives.categorical_crossentropy(output, labels)
    return output, loss

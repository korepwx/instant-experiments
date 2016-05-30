# -*- coding: utf-8 -*-
import tensorflow as tf

from .dense import DenseLayer
from .helpers import get_output
from .. import init

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
            name=name, incoming=incoming, num_units=num_units, W=W, b=b, nonlinearity=None)

    def get_output_for(self, input, **kwargs):
        logits = kwargs.get('logits', False)
        output = super(SoftmaxLayer, self).get_output_for(input, **kwargs)
        if not logits:
            output = tf.nn.softmax(output)
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

    # get the softmax layer logits output.
    kwargs_logits = kwargs.copy()
    kwargs_logits['logits'] = True
    logits = get_output(layer, inputs=inputs, **kwargs_logits)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)

    # compose the real output by applying softmax to the logits.
    output = tf.nn.softmax(logits)
    return output, loss

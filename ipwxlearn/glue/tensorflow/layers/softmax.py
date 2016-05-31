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
        # if the target num is only 2, we use sigmoid instead of softmax.
        true_num_units = num_units if num_units > 2 else 1
        super(SoftmaxLayer, self).__init__(
            name=name, incoming=incoming, num_units=true_num_units, W=W, b=b, nonlinearity=None)

        # although we might actually use sigmoid with 1 output, we still want the output shape to be 2.
        self.num_units = num_units

    def get_output_from_logits(self, logits):
        if self.num_units == 2:
            output = tf.nn.sigmoid(logits)
            output = tf.concat(1, [output, 1.0 - output])
        else:
            output = tf.nn.softmax(logits)
        return output

    def get_output_for(self, input, logits=False, **kwargs):
        output = super(SoftmaxLayer, self).get_output_for(input, **kwargs)
        if not logits:
            output = self.get_output_from_logits(output)
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

    # compute loss, and compose the real output by applying softmax to the logits.
    if layer.num_units == 2:
        logits_squeeze = tf.squeeze(logits, [-1])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits_squeeze, tf.cast(labels, logits_squeeze.dtype))
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)

    return layer.get_output_from_logits(logits), loss

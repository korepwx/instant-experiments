# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf

from .base import Layer

__all__ = ['DropoutLayer']


class DropoutLayer(Layer):
    """
    Dropout layer, which sets values to zero with probability p.

    :param name: Name of this layer.
    :param incoming: The layer feeding into this layer, or a shape tuple.
    :param p: The probability of setting a value to zero.
    :param rescale: If True, the input is rescaled with factor of 1 / (1-p) on training.
    """

    def __init__(self, name, incoming, p=0.5, rescale=True):
        super(DropoutLayer, self).__init__(name=name, incoming=incoming)
        self.p = p
        self.rescale = rescale

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic or self.p == 0:
            return input
        else:
            retain_prob = 1 - self.p
            ret = tf.nn.dropout(input, retain_prob)
            if not self.rescale:
                # TODO: implement the situation when rescale is False directly.
                ret *= retain_prob
            return ret

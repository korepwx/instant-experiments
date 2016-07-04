# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne

from ipwxlearn.glue.theano.layers import Layer

__all__ = ['DropoutLayer']


class DropoutLayer(lasagne.layers.DropoutLayer, Layer):
    """
    Dropout layer, which sets values to zero with probability p.

    :param name: Name of this layer.
    :param incoming: The layer feeding into this layer, or a shape tuple.
    :param p: The probability of setting a value to zero.
    :param rescale: If True, the input is rescaled with factor of 1 / (1-p) on training.
    """

    def __init__(self, name, incoming, p=0.5, rescale=True):
        super(DropoutLayer, self).__init__(incoming=incoming, p=p, rescale=rescale, name=name)

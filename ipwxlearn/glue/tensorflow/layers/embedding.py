# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf

from .base import Layer
from .. import init

__all__ = ['EmbeddingLayer']


class EmbeddingLayer(Layer):
    """
    A layer for embeddings, which look up for desired elements from a matrix,
    according to specified integral indices.

    :param name: Name of this layer.
    :param incoming: The layer feed into this layer.
    :param input_size: The number of different embeddings / number of entities.
    :param output_size: The size of each embedding / number of features in each entity.
    :param W: Theano variable, numpy array, or an initializer, with shape ``(input_size, output_size)``.
    """

    def __init__(self, name, incoming, input_size, output_size, W=init.Normal()):
        super(EmbeddingLayer, self).__init__(incoming=incoming, name=name)

        self.input_size = input_size
        self.output_size = output_size
        self.W = self.add_param(W, (input_size, output_size), name='W')

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_size)

    def get_output_for(self, input, **kwargs):
        return tf.nn.embedding_lookup(self.W, input)

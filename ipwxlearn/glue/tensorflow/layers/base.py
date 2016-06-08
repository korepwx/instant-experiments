# -*- coding: utf-8 -*-
from __future__ import absolute_import

from collections import OrderedDict

import tensorflow as tf

__all__ = [
    'Layer',
    'MergeLayer'
]


class Layer(object):
    """
    The :class:`Layer` class represents a single layer of a neural network.

    :param name: Name that attach to this layer.
    :type name: :class:`str`
    :param incoming: The layer feeding into this layer.
    :type incoming: :class:`Layer`
    """

    def __init__(self, name, incoming):
        self.name = name
        self.input_shape = incoming.output_shape
        self.input_layer = incoming
        self.params = []
        self.get_output_kwargs = []

        if any(d is not None and d <= 0 for d in self.input_shape):
            raise ValueError("Could not create Layer %r with a non-positive shape %r." %
                             (self.name, self.input_shape))

    @property
    def output_shape(self):
        shape = self.get_output_shape_for(self.input_shape)
        assert(all(not isinstance(s, (tf.Tensor, tf.Variable)) for s in shape))
        return shape

    def get_params(self, **tags):
        """
        Return a list of TensorFlow shared variables or expressions that parameterize the layer.

        :param **tags: Tags that filter the parameters.
        :return: List of variables that parameterize the layer.
        """
        from ..graph import current_graph
        graph = current_graph()
        result = [p for p in self.params if graph.get_variable_info(p).match_tags(**tags)]
        return result

    def get_output_shape_for(self, input_shape):
        """Computes the output shape of this layer, given an input shape."""
        return input_shape

    def get_output_for(self, input, **kwargs):
        """
        Propagates the given input through this layer (and only this layer).

        :param input: TensorFlow expression that should be propagate through this layer.
        :return: TensorFlow expression as the output of this layer.
        """
        raise NotImplementedError()

    def add_param(self, spec, shape, name, **tags):
        """
        Register and possibly initialize a parameter tensor for the layer.

        :param spec: TensorFlow variable, numpy array, or initializer.
        :param shape: Shape of this parameter.
        :type shape: :class:`tuple`
        :param name: Name of this parameter variable.
        :type name: :class:`str`
        :param **tags: Tags associated with this parameter.

        :return: The resulting parameter variable.
        """
        from ipwxlearn import glue
        from ..utils import make_variable
        from ..scope import name_scope

        # parameters should be trainable and regularizable by default
        tags['trainable'] = tags.get('trainable', True)
        tags['regularizable'] = tags.get('regularizable', True)

        # create the variable for the parameter, or reuse the existing variable.
        if not isinstance(spec, tf.Variable):
            with name_scope(self.name):
                param = make_variable(name, shape, spec, dtype=glue.config.floatX, **tags)
        else:
            assert(tuple(spec.get_shape().as_list()) == shape)
            param = spec

        # okay, now add to layer parameter list.
        self.params.append(param)
        return param


class MergeLayer(Layer):
    """
    This class represents a layer that aggregates input from multiple layers.
    It should be subclassed when implementing new types of layers that obtain
    their input from multiple layers.
    """

    def __init__(self, incomings, name=None):
        incomings = list(incomings)     # Support for iterators.
        self.input_shapes = [incoming.output_shape for incoming in incomings]
        self.input_layers = [incoming for incoming in incomings]
        self.name = name
        self.params = OrderedDict()

    @Layer.output_shape.getter
    def output_shape(self):
        return self.get_output_shape_for(self.input_shapes)

    def get_output_shape_for(self, input_shapes):
        raise NotImplementedError

    def get_output_for(self, inputs, **kwargs):
        raise NotImplementedError

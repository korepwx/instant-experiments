# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne

from ipwxlearn.utils.misc import maybe_iterable_to_list
from ..graph import current_graph

__all__ = [
    'get_all_layers',
    'get_output',
    'get_all_params'
]


def get_all_layers(layer, treat_as_input=None):
    """
    Collect all layers of a network given the output layer(s).

    :param layer: Layer or an iterable of layers.
    :param treat_as_input: None or an iterable of layers.  These layers will be collected,
                           but the incoming layers of them will not be explored.
    :return: A list of layers.
    """
    return lasagne.layers.get_all_layers(layer, treat_as_input=treat_as_input)



def get_output(layer_or_layers, inputs=None, **kwargs):
    """
    Get the output tensor for given layer or layers.

    :param layer_or_layers: Layer or an iterable of layers.
    :param inputs: Dict with some input layers as keys and numeric scalars or numpy arrays as values,
                   causing these input layers to be substituted by constant values.
    :param kwargs: Additional parameters passed to :method:`Layer.get_output`.

    :return: Output tensor, or a tuple of output tensor.
    """
    return lasagne.layers.get_output(maybe_iterable_to_list(layer_or_layers), inputs=inputs, **kwargs)


def get_all_params(layer_or_layers, **tags):
    """
    Get all the parameters of layers, filtered by tags.

    Only the parameters included in current graph would be returned, even if there're more parameters
    contained in backend layers.

    :param layer_or_layers: Layer or an iterable of layers.
    :param tags: Filters on the tags.

    :return: A list of variables of the parameters.
    """
    graph = current_graph()
    backend_params = lasagne.layers.get_all_params(maybe_iterable_to_list(layer_or_layers))
    graph_params = set(graph.get_variables(**tags))
    return [var for var in backend_params if var in graph_params]
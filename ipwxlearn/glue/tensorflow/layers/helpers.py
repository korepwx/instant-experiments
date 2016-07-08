# -*- coding: utf-8 -*-
from __future__ import absolute_import

from collections import deque

import six

from ipwxlearn.utils import misc
from ipwxlearn.utils.misc import maybe_iterable_to_list

__all__ = [
    'get_all_layers',
    'get_output',
    'get_all_params',
    'with_param_tags',
]


def get_all_layers(layer, treat_as_input=None):
    """
    This function gathers all layers below one or more given :class:`Layer`
    instances, including the given layer(s).

    :param layer: Layer or list of layers.
    :param treat_as_input: Iterable of layers to be treated as input layers.

    :return: List of layers.
    """
    # We perform a depth-first search. We add a layer to the result list only
    # after adding all its incoming layers (if any) or when detecting a cycle.
    # We use a LIFO stack to avoid ever running into recursion depth limits.
    try:
        queue = deque(layer)
    except TypeError:
        queue = deque([layer])
    seen = set()
    done = set()
    result = []

    # If treat_as_input is given, we pretend we've already collected all their
    # incoming layers.
    if treat_as_input is not None:
        seen.update(treat_as_input)

    while queue:
        # Peek at the leftmost node in the queue.
        layer = queue[0]
        if layer is None:
            # Some node had an input_layer set to `None`. Just ignore it.
            queue.popleft()
        elif layer not in seen:
            # We haven't seen this node yet: Mark it and queue all incomings
            # to be processed first. If there are no incomings, the node will
            # be appended to the result list in the next iteration.
            seen.add(layer)
            if hasattr(layer, 'input_layers'):
                queue.extendleft(reversed(layer.input_layers))
            elif hasattr(layer, 'input_layer'):
                queue.appendleft(layer.input_layer)
        else:
            # We've been here before: Either we've finished all its incomings,
            # or we've detected a cycle. In both cases, we remove the layer
            # from the queue and append it to the result list.
            queue.popleft()
            if layer not in done:
                result.append(layer)
                done.add(layer)

    return result


def get_output(layer_or_layers, inputs=None, **kwargs):
    """
    Get the output tensor for given layer or layers.

    :param layer_or_layers: Layer or an iterable of layers.
    :param inputs: Dict with some input layers as keys and numeric scalars or numpy arrays as values,
                   causing these input layers to be substituted by constant values.
    :param kwargs: Additional parameters passed to :method:`Layer.get_output`.

    :return: Output tensor, or a tuple of output tensor.
    """
    from .input import InputLayer
    from .base import MergeLayer
    # obtain topological ordering of all layers the output layer(s) depend on
    treat_as_input = inputs.keys() if isinstance(inputs, dict) else []
    all_layers = get_all_layers(layer_or_layers, treat_as_input)
    # initialize layer-to-expression mapping from all input layers
    all_outputs = dict((layer, layer.input_var)
                       for layer in all_layers
                       if isinstance(layer, InputLayer) and
                       layer not in treat_as_input)
    # update layer-to-expression mapping from given input(s), if any
    if inputs is not None:
        all_outputs.update(inputs.items())
    # update layer-to-expression mapping by propagating the inputs
    for layer in all_layers:
        if layer not in all_outputs:
            try:
                if isinstance(layer, MergeLayer):
                    layer_inputs = [all_outputs[input_layer]
                                    for input_layer in layer.input_layers]
                else:
                    layer_inputs = all_outputs[layer.input_layer]
            except KeyError:
                # one of the input_layer attributes must have been `None`
                raise ValueError("get_output() was called without giving an "
                                 "input expression for the free-floating "
                                 "layer %r. Please call it with a dictionary "
                                 "mapping this layer to an input expression."
                                 % layer)
            all_outputs[layer] = layer.get_output_for(layer_inputs, **kwargs)
    # return the output(s) of the requested layer(s) only
    try:
        return [all_outputs[layer] for layer in layer_or_layers]
    except TypeError:
        return all_outputs[layer_or_layers]


def get_all_params(layer_or_layers, treat_as_input=None, **tags):
    """
    Get all the parameters of layers, filtered by tags.

    Only the parameters included in current graph would be returned, even if there're more parameters
    contained in backend layers.

    :param layer_or_layers: Layer or an iterable of layers.
    :param treat_as_input: Iterable of layers to be treated as input layers.
                           Layers that feed into input layers will not be discovered, and the parameters
                           of these input layers will also be excluded from the returning list.
    :param tags: Filters on the tags.

    :return: A list of variables of the parameters.
    """
    treat_as_input = maybe_iterable_to_list(treat_as_input)
    layers = get_all_layers(layer_or_layers, treat_as_input=treat_as_input)
    if treat_as_input is not None:
        treat_as_input = set(treat_as_input)
        layers = [l for l in layers if l not in treat_as_input]
    params = sum([l.get_params(**tags) for l in layers], [])
    return misc.unique(params)


@misc.contextmanager
def with_param_tags(*params_or_layers, **tags):
    """
    Temporarily set the tags for specified parameters.

    :param params_or_layers: Parameters, or layers containing parameters.
                             The layer parameters will not be discovered recursively,
                             thus if you want to set the tags for all layers in the
                             network, you might use :method:`get_all_layers` to discover
                             all the related layers.
    :param tags: Set/unset tags, by setting them to True/False.
    """
    from .base import Layer
    from ..graph import current_graph

    def switch_tags(target, tags):
        old = {}
        for t, v in six.iteritems(tags):
            if v and t not in target:
                old[t] = False
                target.add(t)
            elif not v and t in target:
                old[t] = True
                target.remove(t)
        return old

    # discover the tags
    graph = current_graph()
    params = []
    for o in params_or_layers:
        if isinstance(o, Layer):
            params.extend(o.get_params())
        else:
            params.append(o)
    param_info = [graph.get_variable_info(v) for v in misc.unique(params)]
    original_tags = {}

    try:
        for pi in param_info:
            original_tags[pi] = switch_tags(pi.tags, tags)
        yield
    finally:
        for pi, old in six.iteritems(original_tags):
            switch_tags(pi.tags, old)

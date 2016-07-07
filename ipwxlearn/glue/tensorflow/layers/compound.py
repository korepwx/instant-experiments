# -*- coding: utf-8 -*-
from ipwxlearn.utils import misc

from .base import Layer, MergeLayer
from .input import InputLayer

__all__ = [
    'CompoundLayer',
    'ChainLayer',
]


class CompoundLayer(MergeLayer):
    """
    Layer that is composed up by several other layers.

    A compound layer is like a layer directory, which holds several other layers inside it.
    Besides this, the compound layer also provides support for the network discovery methods
    like :method:`get_all_layers` and :method:`get_all_params`.  This is achieved by collecting
    all the incoming of these child layers, and making the compound layer itself a merge layer,
    depending on all these child incomings.

    All kinds of layers are supported by the compound layer, expect for one restriction: all
    the child layers must either be input layers, or have determined incoming layers, rather
    than just a shape hint for the incoming.

    :param children: Iterable of child layers.
    """

    class GetOutputContext(object):
        """Context to get output for each child in a compound layer."""

        def __init__(self, parent, inputs=None, input_shapes=None, **kwargs):
            self.parent = parent
            self.inputs = inputs
            self.input_shapes = input_shapes
            self.kwargs = kwargs
            self.outputs = {l: i for l, i in zip(parent.input_layers, inputs or ())}
            self.output_shapes = {l: s for l, s in zip(parent.input_layers, input_shapes or ())}
            self._children_set = set(parent.children)

        def get_output(self, child):
            """Get the output of specified layer."""
            if child not in self.outputs:
                if child not in self._children_set:
                    raise ValueError('Unable to resolve the output of non-child layer %r.' % child)
                if isinstance(child, MergeLayer):
                    output = child.get_output_for([self.get_output(l) for l in child.input_layers], **self.kwargs)
                else:
                    output = child.get_output_for(self.get_output(child.input_layer), **self.kwargs)
                self.outputs[child] = output
            return self.outputs[child]

        def get_output_shape(self, child):
            """Get the output shape of specified layer."""
            if child not in self.output_shapes:
                if child not in self._children_set:
                    raise ValueError('Unable to resolve the output shape of non-child layer %r.' % child)
                if isinstance(child, MergeLayer):
                    output_shape = child.get_output_shape_for([self.get_output_shape(l) for l in child.input_layers])
                else:
                    output_shape = child.get_output_shape_for(self.get_output_shape(child.input_layer))
                self.output_shapes[child] = output_shape
            return self.output_shapes[child]

    def __init__(self, children, name=None):
        self.children = list(children)

        # discover all the incoming layers of these child layers.
        incomings = []
        for layer in self.children:
            if not isinstance(layer, Layer):
                raise TypeError('%r is not a layer.' % layer)
            if not isinstance(layer, InputLayer):
                if isinstance(layer, MergeLayer):
                    if not all(l is not None for l in layer.input_layers):
                        raise ValueError('Not all incoming layers of %r are determined.' % layer)
                    for l in layer.input_layers:
                        if l not in self.children and l not in incomings:
                            incomings.append(l)
                else:
                    if layer.input_layer is None:
                        raise ValueError('Incoming layer of %r is not determined.' % layer)
                    if layer.input_layer not in self.children and layer.input_layer not in incomings:
                        incomings.append(layer.input_layer)

        # now initialize the merge layer
        super(CompoundLayer, self).__init__(incomings, name=name)

    def get_params(self, **tags):
        params = super(CompoundLayer, self).get_params(**tags) + sum([l.get_params(**tags) for l in self.children], [])
        return misc.unique(params)


class ChainLayer(CompoundLayer):
    """
    Layer that is composed up of a chain of other layers.

    A chain layer is a specialized compound layer, where all the child layers form a chain.
    The output of the whole chain is thus the last layer in the chain.
    """

    def get_output_shape_for(self, input_shapes):
        ctx = CompoundLayer.GetOutputContext(self, input_shapes=input_shapes)
        return ctx.get_output_shape(self.children[-1])

    def get_output_for(self, inputs, **kwargs):
        ctx = CompoundLayer.GetOutputContext(self, inputs=inputs, **kwargs)
        return ctx.get_output(self.children[-1])

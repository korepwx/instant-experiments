# -*- coding: utf-8 -*-

import lasagne
import theano

from ipwxlearn.glue.theano import name_scope, current_graph
from ipwxlearn.utils.misc import require_object_name


class _Layer(lasagne.layers.Layer):

    _layer_name_validated_ = False

    def add_param(self, spec, shape, name=None, **tags):
        # We expect the layer to have a qualified name.
        if not self._layer_name_validated_:
            if not self.name:
                raise ValueError('No name specified for the layer.')
            require_object_name(self.name)
            self._layer_name_validated_ = True

        # We don't add the parameter to the graph in the following situations:
        #
        # 1. The parameter does not have name.
        # 2. The 'spec' is already a Theano variable (which means the variable should have added to graph).
        if name is None or isinstance(spec, theano.Variable):
            return super(_Layer, self).add_param(spec, shape, name, **tags)

        # At this stage, we know that a new Theano variable should be created.
        # We call the backend method to construct the variable, and add to graph.
        require_object_name(name)
        param = super(_Layer, self).add_param(spec, shape, name, **tags)
        if hasattr(spec, '__call__'):
            initializer = spec
        else:
            spec = spec.copy()  # Copy the numpy array and store the the graph.
            initializer = lambda: spec
        for tag in self.params[param]:
            tags.setdefault(tag, True)
        with name_scope(self.name):
            current_graph().add_variable(param, initializer, name=name, **tags)

        return param


class InputLayer(lasagne.layers.InputLayer, _Layer): pass
class DropoutLayer(lasagne.layers.DropoutLayer, _Layer): pass
class DenseLayer(lasagne.layers.DenseLayer, _Layer): pass

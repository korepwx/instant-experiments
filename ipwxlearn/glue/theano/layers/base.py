# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne
import theano

from ipwxlearn.utils import misc
from ..utils import make_initializer


class _Layer(lasagne.layers.Layer):

    _layer_name_validated_ = False

    @misc.contextmanager
    def _temporary_erase_name(self):
        """Temporarily erase the name of this layer."""
        old_name = self.name
        self.name = None
        yield
        self.name = old_name

    def add_param(self, spec, shape, name=None, **tags):
        from ipwxlearn import glue
        from ipwxlearn.glue.common.scope import name_scope, current_name_scope

        # We expect the layer to have a qualified name.
        if not self._layer_name_validated_:
            if not self.name:
                raise ValueError('No name specified for the layer.')
            misc.require_object_name(self.name)
            self._layer_name_validated_ = True

        # We don't add the parameter to the graph in the following situations:
        #
        # 1. The parameter does not have name.
        # 2. The 'spec' is already a Theano variable (which means the variable should have added to graph).
        if name is None or isinstance(spec, theano.Variable):
            return super(_Layer, self).add_param(spec, shape, name, **tags)

        # At this stage, we know that a new Theano variable should be created.
        # We call the backend method to construct the variable, and add to graph.
        misc.require_object_name(name)
        with name_scope(self.name):
            full_name = current_name_scope().resolve_name(name)
            with self._temporary_erase_name():
                param = super(_Layer, self).add_param(spec, shape, full_name, **tags)
            for tag in self.params[param]:
                tags.setdefault(tag, True)
            init = make_initializer(spec, shape, dtype=glue.config.floatX)
            current_name_scope().add_variable(param, init, name, **tags)

        return param

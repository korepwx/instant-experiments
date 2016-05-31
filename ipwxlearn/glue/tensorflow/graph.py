# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf

from ipwxlearn.utils import misc
from .scope import NameScope
from ..common.graph import BaseGraph, VariableTags, VariableInfo, current_graph

__all__ = [
    'Graph',
    'VariableTags',
    'VariableInfo',
    'current_graph'
]


class Graph(BaseGraph):
    """Computation graph for TensorFlow backend."""

    def __init__(self):
        super(Graph, self).__init__()
        self.root_scope = NameScope(None)
        self._graph = tf.Graph()

    @property
    def tf_graph(self):
        """Get the backend Graph object."""
        return self._graph

    @misc.contextmanager
    def as_default(self):
        with super(Graph, self).as_default(), self._graph.as_default():
            yield self

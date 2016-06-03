# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf

from ipwxlearn.utils import misc
from ipwxlearn.utils.misc import merged_context
from .scope import NameScope
from ..common.graph import BaseGraph, VariableTags, VariableInfo, current_graph, SummaryTypes

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
        with merged_context(super(Graph, self).as_default(), self._graph.as_default()):
            yield self

    def _make_summary_op(self, stype, tag, value):
        if stype == SummaryTypes.SCALAR_SUMMARY:
            return tf.scalar_summary(tag, value)
        elif stype == SummaryTypes.HISTOGRAM_SUMMARY:
            return tf.histogram_summary(tag, value)
        elif stype == SummaryTypes.ZERO_FRACTION_SUMMARY:
            return tf.histogram_summary(tag, value)
        else:
            raise NotImplementedError('TensorFlow backend has not supported %s summary yet.' % stype)

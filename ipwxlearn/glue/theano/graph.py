# -*- coding: utf-8 -*-
from __future__ import absolute_import

from theano.tensor.shared_randomstreams import RandomStreams

from ..common.graph import BaseGraph, VariableTags, VariableInfo, current_graph, iter_graphs

__all__ = [
    'Graph',
    'VariableTags',
    'VariableInfo',
    'current_graph',
    'iter_graphs'
]


class Graph(BaseGraph):
    """Computation graph for Theano backend."""

    def __init__(self, *args, **kwargs):
        super(Graph, self).__init__(*args, **kwargs)

    def create_random_state(self, seed):
        return RandomStreams(seed)

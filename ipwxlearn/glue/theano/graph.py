# -*- coding: utf-8 -*-
from __future__ import absolute_import

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

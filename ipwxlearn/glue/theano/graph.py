# -*- coding: utf-8 -*-
from ipwxlearn.glue.common.graph import BaseGraph, VariableTags, VariableInfo, current_graph

__all__ = [
    'Graph',
    'VariableTags',
    'VariableInfo',
    'current_graph'
]


class Graph(BaseGraph):
    """Computation graph for Theano backend."""

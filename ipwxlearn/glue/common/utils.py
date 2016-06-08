# -*- coding: utf-8 -*-
import six

from .graph import VariableTags, current_graph
from .session import iter_sessions

if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl

__all__ = [
    'save_current_graph',
    'restore_current_graph'
]


def _get_current_graph_and_session():
    graph = current_graph()
    session = None
    for sess in iter_sessions():
        if sess.graph == graph:
            session = sess
            break
    return graph, session


def save_current_graph(persist_file, **tags):
    """
    Save the current graph to external persistent file.

    If there's active session opened for current graph, will save the session variables.
    Otherwise, will save the last session values stored in the graph.

    :param persist_file: Path to the persistent file.
    :param tags: Tag of variables that should be saved.  "persistent=True" is selected by default.
    """
    graph, session = _get_current_graph_and_session()

    tags.setdefault(VariableTags.PERSISTENT, True)
    vars = graph.get_variables(**tags)
    if session is not None:
        var_dict = session.get_variable_values_dict(vars)
    else:
        var_dict = graph.get_last_values_as_dict(vars)

    var_dict = {graph.get_variable_info(k).full_name: v for k, v in six.iteritems(var_dict)}
    with open(persist_file, 'wb') as f:
        pkl.dump(var_dict, f, pkl.HIGHEST_PROTOCOL)


def restore_current_graph(persist_file):
    """
    Restore the current graph from external persistent file.

    If there's active session opened fro current graph, will assign to session variables.
    Otherwise, will assign to last session values stored in the graph.

    :param persist_file: Path to the persistent file.
    """
    graph, session = _get_current_graph_and_session()
    with open(persist_file, 'rb') as f:
        var_dict = pkl.load(f)

    if session is not None:
        session.set_variable_values({graph.get_variable(k): v for k, v in six.iteritems(var_dict)})
    else:
        graph.set_last_values(var_dict)

# -*- coding: utf-8 -*-
import six

from .graph import VariableTags
from .session import iter_sessions

if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl

__all__ = [
    'get_graph_state',
    'set_graph_state',
    'save_graph_state',
    'restore_graph_state'
]


def _get_graph_session(graph):
    session = None
    for sess in iter_sessions():
        if sess.graph == graph:
            session = sess
            break
    return session


def get_graph_state(graph, **tags):
    """
    Get the graph variable values as a state dict.

    If there's active session opened for the graph, will get the session variables.
    Otherwise, will get the last session values stored in the graph.

    :param graph: Graph object.
    :param tags: Tag of variables that should be included in the state.  "persistent=True" is set by default.
    :return: dict from full name to variable values.
    """
    session = _get_graph_session(graph)
    tags.setdefault(VariableTags.PERSISTENT, True)
    vars = graph.get_variables(**tags)
    if session is not None:
        var_dict = session.get_variable_values_dict(vars)
    else:
        var_dict = graph.get_last_values_as_dict(vars)
    return {graph.get_variable_info(k).full_name: v for k, v in six.iteritems(var_dict)}


def save_graph_state(persist_file, graph, **tags):
    """
    Save graph state to persistent file.
    See :method:`get_graph_state` for more details about arguments.
    """
    if isinstance(persist_file, six.string_types):
        with open(persist_file, 'wb') as f:
            save_graph_state(f, graph, **tags)
    else:
        pkl.dump(get_graph_state(graph, **tags), persist_file, protocol=pkl.HIGHEST_PROTOCOL)


def set_graph_state(graph, state):
    """
    Set the graph variable values according to state dict.

    If there's active session opened for the graph, will assign to session variables.
    Otherwise, will assign to last session values stored in the graph.

    :param graph: Graph object.
    :param state: State dict, from full name to variable values.
    """
    session = _get_graph_session(graph)
    if session is not None:
        session.set_variable_values({graph.get_variable(k): v for k, v in six.iteritems(state)})
    else:
        graph.set_last_values(state)


def restore_graph_state(persist_file, graph):
    """
    Restore graph state from persistent file.
    See :method:`set_graph_state` for more details about arguments.
    """
    if isinstance(persist_file, six.string_types):
        with open(persist_file, 'rb') as f:
            restore_graph_state(f, graph)
    else:
        set_graph_state(graph, pkl.load(persist_file))

# -*- coding: utf-8 -*-
import contextlib
from collections import OrderedDict

import six

from ipwxlearn.glue.common.scope import NameScope, _name_scope_stack, current_name_scope
from ipwxlearn.utils.concurrent import ThreadLocalStack
from ipwxlearn.utils.misc import require_object_name

__all__ = [
    'VariableTags',
    'VariableInfo',
    'BaseGraph',
    'current_graph',
]


class VariableTags:
    #: Indicate that a variable should be tuned by training.
    TRAINABLE = 'trainable'

    #: Indicate that a variable should be included in regularization term.
    REGULARIZABLE = 'regularizable'

    #: Indicate that a variable should be included in model persistent file.
    #: Having TRAINABLE tag would implicitly set PERSISTENT tag, unless explicitly set to False.
    PERSISTENT = 'persistent'

    #: Indicate that a variable should be included in checkpoint file (so as to resume training).
    #: Having PERSISTENT tag would implicitly set RESUMABLE, unless explicitly set to False.
    RESUMABLE = 'resumable'


class VariableInfo(object):
    """
    Class to hold the backend variable object, as well as related information.

    :param var: Backend variable object.
    :param initializer: numpy array, or initializer object for the particular backend.
    :param full_name: Full name of the variable.
    :param tags: Tags assigned to this variable, if value is True.
    """

    def __init__(self, var, init, full_name, **tags):
        if tags.get(VariableTags.TRAINABLE, False):
            tags.setdefault(VariableTags.PERSISTENT, True)
        if tags.get(VariableTags.PERSISTENT, False):
            tags.setdefault(VariableTags.RESUMABLE, True)
        self.var = var
        self.init = init
        self.full_name = full_name
        self.tags = {t for t, v in six.iteritems(tags) if v}
        #: Value saved from the last session running on this graph.
        self.last_value = None

    def __repr__(self):
        tag_formatted = ', '.join(sorted(self.tags))
        if tag_formatted:
            tag_formatted = ', ' + tag_formatted
        return 'VariableInfo(%s%s)' % (self.full_name, tag_formatted)

    def has_tags(self, tags, match_all=True):
        """
        Whether or not this variable has all of the specified tags.

        :param tags: Set of tags.  If empty, will return True.
        :param match_all: If True, all the tags must exist for the method to return True.
                          If False, any of the tags existing will result in True.
        """
        if not tags:
            return True
        if match_all:
            return all(k in self.tags for k in tags)
        return any(k in self.tags for k in tags)


class BaseGraph(object):
    """
    Base class to manage all the variables defined for a computation graph.
    Backend should derive from this class, to provide actual graph implementation.

    Layers are not managed by a name scope, since there might be some situations
    that algorithms are composed directly from tensor variables, without neural
    network layers.
    """

    def __init__(self):
        self.root_scope = NameScope(None)

        #: Dict from backend variable to :class:`GlueVariable`
        self._variables = OrderedDict()
        #: Dict from full name to :class:`GlueVariable`
        self._names_map = {}

    @contextlib.contextmanager
    def as_default(self):
        """
        Lifting this graph as the default graph for current thread.
        :rtype: :class:`BaseGraph`
        """
        _graph_stack.push(self)
        _name_scope_stack.push(self.root_scope)
        yield self
        _name_scope_stack.pop()
        _graph_stack.pop()

    def add_variable(self, var, init, name, **tags):
        """
        Add backend variable to the graph.

        :param var: Backend variable object.
        :param init: numpy array, or initializer object for the particular backend.
        :param name: Name of the backend variable.
        :param tags: Tags of this variable.  See also :class:`VariableTags`.
        """
        if var in self._variables:
            raise KeyError('Backend variable %s is already added to the graph.' % var)
        require_object_name(name)
        full_name = current_name_scope().resolve_name(name)
        if full_name in self._names_map:
            raise KeyError('Full name %s is already used by %s.' % (full_name, self._names_map[full_name]))
        variable = VariableInfo(var, init, full_name, **tags)
        self._names_map[full_name] = self._variables[var] = variable

    def get_variable(self, full_name):
        """
        Get the variable according to full name or backend variable.

        :param full_name: Full name of the variable.
        :return: Backend variable object.
        """
        return self._names_map[full_name].var

    def get_variable_info(self, full_name_or_var):
        """
        Get the variable information

        :param full_name_or_var: Full name of the variable, or backend variable object.
        :rtype: :class:`VariableInfo`
        """
        if isinstance(full_name_or_var, six.string_types):
            return self._names_map[full_name_or_var]
        return self._variables[full_name_or_var]

    def iter_variables(self, tags=(), match_all=True):
        """
        Iterate the backend variables in this graph, having specified tags.

        :param tags: Set of tags.  Will match all variables if not given.
        :param match_all: If True, all of the tags must exist for the variable to be yielded.
                          If False, any of the tags existing would result in the variable to be yielded.
        """
        if not tags:
            for var in six.iterkeys(self._variables):
                yield var
        else:
            for var, variable in six.iteritems(self._variables):
                if variable.has_tags(tags, match_all=match_all):
                    yield var

    def get_variables(self, tags=(), match_all=True):
        """Get the backend variables in this graph, having specified tags."""
        return list(self.iter_variables(tags, match_all))

    @property
    def variable_info_dict(self):
        """Get the dict from backend variable object to variable information."""
        return self._variables


#: Thread local graph scope stack, with a default graph on the thread.
_graph_stack = ThreadLocalStack()


def current_graph():
    """
    Get the current active graph scope.
    :rtype: :class:`BaseGraph`
    """
    if _graph_stack.empty:
        raise ValueError('No graph is activated.')
    return _graph_stack.top
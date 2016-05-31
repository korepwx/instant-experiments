# -*- coding: utf-8 -*-
from __future__ import absolute_import

from collections import OrderedDict

import six

from ipwxlearn.utils import misc
from ipwxlearn.utils.concurrent import ThreadLocalStack
from ipwxlearn.utils.misc import require_object_full_name
from .scope import NameScope, _name_scope_stack

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
    :param init: numpy array, or initializer object of the particular backend.
    :param full_name: Full name of the variable.
    :param **tags: Tags assigned to this variable, if value is True.
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

    def match_tags(self, **tags):
        """
        Whether or not this variable matches the tags filter.

        :param **tags: Set tag to True would require the variable to have such tag, while set to False will
                       require not to have such tag.
        """
        for t, v in six.iteritems(tags):
            if (t in self.tags) != v:
                return False
        return True


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

    @misc.contextmanager
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

    def add_variable(self, var, init, full_name, **tags):
        """
        Add backend variable to the graph.

        :param var: Backend variable object.
        :param init: numpy array, or initializer object for the particular backend.
        :param full_name: Full name of the backend variable.
        :param **tags: Tags of this variable.  See also :class:`VariableTags`.
        """
        if var in self._variables:
            raise KeyError('Backend variable %s is already added to the graph.' % var)
        require_object_full_name(full_name)
        if full_name in self._names_map:
            raise KeyError('Full name %s is already used by %s.' % (full_name, self._names_map[full_name]))
        info = VariableInfo(var, init, full_name, **tags)
        self._names_map[full_name] = self._variables[var] = info

    def get_variable(self, full_name):
        """
        Get the variable according to full name or backend variable.

        :param full_name: Full name of the variable.
        :return: Backend variable object.
        """
        return self._names_map[full_name].var

    def iter_variables(self, **tags):
        """
        Iterate the backend variables in this graph, having specified tags.

        :param **tags: Tags used to filter the variables.  Set tag=True would require the variable to have such tag,
                       while set to False would require not to have such tag.
        """
        if not tags:
            for var in six.iterkeys(self._variables):
                yield var
        else:
            for var, info in six.iteritems(self._variables):
                if info.match_tags(**tags):
                    yield var

    def get_variables(self, **tags):
        """Get the backend variables in this graph, having specified tags."""
        return list(self.iter_variables(**tags))

    def get_persistent_variables(self):
        """Get all persistent variables in this graph."""
        return list(self.iter_variables(persistent=True))

    def get_variable_info(self, full_name_or_var):
        """
        Get the variable information

        :param full_name_or_var: Full name of the variable, or backend variable object.
        :rtype: :class:`VariableInfo`
        """
        if isinstance(full_name_or_var, six.string_types):
            return self._names_map[full_name_or_var]
        return self._variables[full_name_or_var]

    @property
    def variable_info_dict(self):
        """Get the dict from backend variable object to variable information."""
        return self._variables

    def get_last_values(self, full_names_or_vars):
        """
        Get the last values of given variables.

        :param full_names_or_vars: iterable full names or backend variable objects.
        :return: tuple of the values, each corresponds to one given variable.
        """
        return tuple(
            info.last_value
            for info in (self.get_variable_info(k) for k in full_names_or_vars)
        )

    def get_last_values_as_dict(self, full_names_or_vars):
        """
        Get the last values of given variables, as a dict.
        If a variable does not have last value, it would be excluded from the returning dict.

        :param full_names_or_vars: iterable full names or backend variable objects.
        :return: dict from backend variable object to value.
        """
        return {
            info.var: info.last_value
            for info in (self.get_variable_info(k) for k in full_names_or_vars)
            if info.last_value is not None
        }

    def set_last_values(self, value_dict):
        """
        Set the last values for given variables.

        :param value_dict: dict from variable full name, or backend variable object, to the value.
        """
        for k, v in six.iteritems(value_dict):
            if v is not None:
                info = self.get_variable_info(k)
                info.last_value = v


#: Thread local graph stack, with a default graph on the thread.
_graph_stack = ThreadLocalStack()


def current_graph():
    """
    Get the current active graph scope.
    :rtype: :class:`BaseGraph`
    """
    if _graph_stack.empty:
        raise ValueError('No graph is activated.')
    return _graph_stack.top
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import six

from ipwxlearn.utils import misc
from ipwxlearn.utils.concurrent import ThreadLocalStack

__all__ = [
    "NameScope",
    "current_name_scope",
    "name_scope",
    "iter_name_scopes",
]


class NameScope(object):
    """
    Class to manage variables defined in a name scope.

    :param full_name: Full name of the scope.  Full names of scopes should be
                      composed by several scope names, separated by '/'.
    """

    def __init__(self, full_name):
        assert(full_name is None or isinstance(full_name, six.string_types))

        #: Full name of this scope.
        self.full_name = full_name
        #: All the created scopes.
        self._scopes = {}

    def __repr__(self):
        return 'NameScope(%s)' % (self.full_name or '')

    def resolve_name(self, name):
        """
        Resolve the full name of :param:`name`, under this scope.

        :param name: Name in this scope.
        """
        if not self.full_name:
            return name
        return '%s/%s' % (self.full_name, name)

    def add_variable(self, var, init, name, **tags):
        """
        Add variable to the default graph, while :param:`name` would be resolved into full name.
        """
        from ipwxlearn.glue.common.graph import current_graph
        return current_graph().add_variable(var, init, self.resolve_name(name), **tags)

    def _create_sub_scope(self, name):
        return NameScope(self.resolve_name(name))

    def sub_scope(self, name):
        """
        Create or open a sub name scope with :param:`name`.

        :rtype: :class:`NameScope`
        """
        if name is None:
            return self
        if name not in self._scopes:
            self._scopes[name] = self._create_sub_scope(name)
        return self._scopes[name]

    def push_default(self):
        """Push this name scope to the default stack."""
        _name_scope_stack.push(self)

    def pop_default(self):
        """Pop this name scope from the default stack."""
        assert(_name_scope_stack.top == self)
        _name_scope_stack.pop()


#: The thread local name scope.
_name_scope_stack = ThreadLocalStack()


def current_name_scope():
    """
    Get the current active name scope.
    :rtype: :class:`NameScope`
    """
    if _name_scope_stack.empty:
        raise ValueError('No name scope is activated. You might need to activate a Graph first.')
    return _name_scope_stack.top


@misc.contextmanager
def name_scope(name_or_scope):
    """
    Context manager to open a scope for naming layers and variables.

    A dedicated name is required to define every layer and variable, in order to give
    the debugging and visualization tools a chance to produce nice outputs.

    Every name scope would be created for only once.  Successive class with identical
    names would only open the existing name scope.

    :param name_or_scope: Name of the sub scope to open, or a NameScope instance.
    """
    if isinstance(name_or_scope, NameScope):
        scope = name_or_scope
    else:
        scope = current_name_scope().sub_scope(name_or_scope)

    scope.push_default()
    yield scope
    scope.pop_default()


def iter_name_scopes():
    """Iterate all the active name scopes, from the newest scope on stack to the oldest."""
    return iter(_name_scope_stack)


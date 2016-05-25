# -*- coding: utf-8 -*-
import contextlib

from ipwxlearn.utils.concurrent import ThreadLocalStack

__all__ = [
    "NameScope",
    "current_name_scope",
    "name_scope",
]


class NameScope(object):
    """
    Class to manage variables defined in a name scope.

    :param full_name: Full name of the scope.  Full names of scopes should be
                      composed by several scope names, separated by '/'.
    """

    def __init__(self, full_name):
        #: Full name of this scope.
        self.full_name = full_name

    def __repr__(self):
        return 'NameScope(%s)' % (self.full_name or '')

    def __str__(self):
        return self.full_name

    def resolve_name(self, name):
        """
        Resolve the full name of :param:`name`, under this scope.

        :param name: Name in this scope.
        """
        if not self.full_name:
            return name
        return '%s/%s' % (self.full_name, name)

    def sub_scope(self, name):
        """
        Create or open a sub name scope with :param:`name`.

        :rtype: :class:`NameScope`
        """
        return NameScope(self.resolve_name(name))


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


@contextlib.contextmanager
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
    _name_scope_stack.push(scope)
    yield scope
    _name_scope_stack.pop()

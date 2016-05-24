# -*- coding: utf-8 -*-
import contextlib

from ipwxlearn.utils.concurrent import ThreadLocalStack
from ipwxlearn.utils.misc import require_object_name


class NameScope(object):
    """
    Class to manage variables defined in a name scope.

    Layers are not managed by a name scope, since there might be some situations
    that algorithms are composed directly from tensor variables, without neural
    network layers.
    """

    def __init__(self, full_name):
        #: Full name of this scope.
        self.full_name = full_name

        #: dict from names to sub name scopes.
        self._scopes = {}

        #: dict from names to the variables.
        self._variables = {}

    def __repr__(self):
        return 'NameScope(%s)' % repr(self.full_name)

    def __str__(self):
        return self.full_name

    def resolve_name(self, name):
        """
        Resolve the full name of :param:`name`, under this scope.
        In Theano backend, names are separated by "." in a full name.
        """
        if not self.full_name:
            return name
        return '%s.%s' % (self.full_name, name)

    def get_scope(self, name):
        """
        Create or open a sub name scope with :param:`name`.
        :rtype: :class:`NameScope`
        """
        if name not in self._scopes:
            require_object_name(name)
            self._scopes[name] = NameScope(self.resolve_name(name))
        return self._scopes[name]

    @property
    def scopes(self):
        """Get the dict of all sub scopes."""
        return self._scopes

    def add_variable(self, name, variable):
        """
        Add a :param:`variable` with :param:`name` to this scope.
        :raises KeyError: If :param:`name` already exists.
        """
        if name in self._variables:
            raise KeyError('Variable %s already exists in %s.' % (repr(name), repr(self)))
        require_object_name(name)
        self._variables[name] = variable

    def get_variable(self, name):
        """Get a variable according to :param:`name`."""
        return self._variables[name]

    @property
    def variables(self):
        """Get the dict of all variables."""
        return self._variables


class GraphScope(object):
    """Class to manage all the objects defined for a computation graph."""

    def __init__(self):
        #: The root name scope for this graph.
        self.root_scope = NameScope('')

    @contextlib.contextmanager
    def as_default(self):
        """
        Lifting this graph as the default graph.
        :rtype: :class:`GraphScope`
        """
        _graph_scope_stack.push(self)
        _name_scope_stack.push(self.root_scope)
        yield self
        _name_scope_stack.pop()
        _graph_scope_stack.pop()


#: Thread local graph scope stack, with a default graph on the thread.
_graph_scope_stack = ThreadLocalStack([GraphScope()])


def current_graph():
    """
    Get the current active graph scope.
    :rtype: :class:`GraphScope`
    """
    return _graph_scope_stack.top


#: The thread local name scope.
_name_scope_stack = ThreadLocalStack([current_graph().root_scope])


def current_name_scope():
    """
    Get the current active name scope.
    :rtype: :class:`NameScope`
    """
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
    :rtype: :class:`NameScope`
    """
    if isinstance(name_or_scope, NameScope):
        scope = name_or_scope
    else:
        scope = current_name_scope().get_scope(name_or_scope)
    _name_scope_stack.push(scope)
    yield scope
    _name_scope_stack.pop()

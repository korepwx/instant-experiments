# -*- coding: utf-8 -*-
import contextlib
from collections import OrderedDict

import six

from ipwxlearn.utils.concurrent import ThreadLocalStack
from ipwxlearn.utils.misc import require_object_name

__all__ = [
    "VariableTags",
    "GlueVariable",
    "NameScope",
    "BaseGraph",
    "current_graph",
    "current_name_scope",
    "name_scope",
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


class GlueVariable(object):
    """
    Class to hold the backend variable object, as well as related information.

    :param var: Backend variable object.
    :param initializer: Initializer for the backend variable.
    :param full_name: Full name of the variable.
    :param tags: Tags assigned to this variable, if value is True.
    """

    def __init__(self, var, initializer, full_name, **tags):
        if VariableTags.TRAINABLE in tags:
            tags.setdefault(VariableTags.PERSISTENT, True)
        if VariableTags.PERSISTENT in tags:
            tags.setdefault(VariableTags.RESUMABLE, True)
        self.var = var
        self.initializer = initializer
        self.full_name = full_name
        self.tags = {t for t, v in six.iteritems(tags) if v}

    def __repr__(self):
        tag_formatted = ', '.join(sorted(self.tags))
        if tag_formatted:
            tag_formatted = ', ' + tag_formatted
        return 'Variable(%s%s)' % (self.full_name, tag_formatted)

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

    def _extract_var_name(self, var):
        raise NotImplementedError()

    def add_variable(self, var, initializer, name=None, **tags):
        """
        Add backend variable to the graph.

        :param var: Backend variable object.
        :param initializer: Initializer for the backend variable.
        :param name: Name of the backend variable.
                     Would attempt to extract the name from the variable if not given.
        :param tags: Tags of this variable.  See also :class:`VariableTags`.
        """
        if var in self._variables:
            raise KeyError('Backend variable %s is already added to the graph.' % var)
        name = name or self._extract_var_name(var)
        require_object_name(name)
        full_name = current_name_scope().resolve_name(name)
        if full_name in self._names_map:
            raise KeyError('Full name %s is already used by %s.' % (full_name, self._names_map[full_name]))
        variable = GlueVariable(var, initializer, full_name, **tags)
        self._names_map[full_name] = self._variables[var] = variable

    def get_variable(self, name_or_var):
        """
        Get the variable according to full name or backend variable.

        :param name_or_var: Full name of the variable, or the backend variable instance.
        :rtype: :class:`GlueVariable`
        """
        if isinstance(name_or_var, six.string_types):
            return self._names_map[name_or_var]
        return self._variables[name_or_var]

    def iter_variables(self, tags=(), match_all=True):
        """
        Iterate the :class:`GlueVariable` instances in this graph, having specified tags.

        :param tags: Set of tags.  Will match all variables if not given.
        :param match_all: If True, all of the tags must exist for the variable to be yielded.
                          If False, any of the tags existing would result in the variable to be yielded.
        """
        if not tags:
            for variable in six.itervalues(self._variables):
                yield variable
        else:
            for var, variable in six.iteritems(self._variables):
                if variable.has_tags(tags, match_all=match_all):
                    yield variable

    def get_variables(self, tags=(), match_all=True):
        return list(self.iter_variables(tags, match_all))


#: Thread local graph scope stack, with a default graph on the thread.
_graph_stack = ThreadLocalStack()


def current_graph():
    """
    Get the current active graph scope.
    """
    if _graph_stack.empty:
        raise ValueError('No graph is activated.')
    return _graph_stack.top


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

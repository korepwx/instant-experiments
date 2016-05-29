# -*- coding: utf-8 -*-
from collections import OrderedDict

import six

from ipwxlearn.utils.misc import maybe_iterable_to_list, ensure_list_sealed
from .session import current_session


class BaseFunction(object):
    """
    Abstract interface for all compiled backend functions.

    Some backend might require a compiling step, which converts part or all of the computation graph
    into function on target device.  Others might not require such step.
    To keep the API consistent, we thus need to provide such abstract function interface, so that

    :param inputs: Variable, or list/dict of variables.
                   If given only a single variable, it would be the only unnamed argument of the function.
                   If given a list of variables, they would be the unnamed arguments of the function.
                   If given a dict of variables, they would be the named arguments of the function,
                   where each key is the argument name.
    :param outputs: Variable, or list of variables.
    :param updates: Operation or list of operations, so as to update the variables.
                    Although some backend might not have difference on expressions and operations,
                    it is better to have the concept of update operations.
    :param givens: Feeding values of variables to the computation graph.
    """

    def __init__(self, inputs=None, outputs=None, updates=None, givens=None):
        # check arguments.
        if inputs and not isinstance(inputs, (dict, OrderedDict)):
            inputs = ensure_list_sealed(inputs)
        if outputs:
            outputs = maybe_iterable_to_list(outputs)
        if updates:
            updates = self._merge_updates(maybe_iterable_to_list(updates, exclude_types=(dict, OrderedDict)))

        # assign to properties
        self._inputs = inputs
        self._outputs = outputs
        self._updates = updates
        self._givens = givens

        # compile the function
        self._function = self._compile()

    def _compile(self):
        """
        Derived classes should override this to actually compile the backend function.
        Returns the callable object which could be called to execute the backend function.
        """
        raise NotImplementedError()

    def _merge_updates(self, updates):
        """Merge several updates into one update, for the backend."""
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        # require there's a session on the stack.
        _ = current_session()
        args = args or ()
        kwargs = kwargs or {}
        if isinstance(self._inputs, (dict, OrderedDict)):
            if args:
                raise ValueError('Function only accepts named arguments.')
            for k, v in six.iteritems(kwargs):
                if k not in self._inputs:
                    raise ValueError('Unexpected named argument %s.' % k)
            for k, v in six.iteritems(self._inputs):
                if k not in kwargs:
                    raise ValueError('Named argument %s is required but not specified.' % k)
        else:
            if kwargs:
                raise ValueError('Function only accepts unnamed arguments.')
            if len(args) != len(self._inputs or ()):
                raise ValueError('Require %d unnamed arguments, but got %s.' % (len(self._inputs or ()), len(args)))

        return self._function(*args, **kwargs)

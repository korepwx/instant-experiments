# -*- coding: utf-8 -*-
from collections import OrderedDict

import six
import theano

from ..common.function import BaseFunction

__all__ = ['Function', 'make_function']


class Function(BaseFunction):
    """Theano compiled function."""

    def _compile(self):
        """
        Derived classes should override this to actually compile the backend function.
        Returns the callable object which could be called to execute the backend function.
        """
        if isinstance(self._inputs, (dict, OrderedDict)):
            keys = []
            inputs = []
            for k, v in six.iteritems(self._inputs):
                keys.append(k)
                inputs.append(v)
            func = theano.function(inputs=inputs, outputs=self._outputs, updates=self._updates, givens=self._givens)

            def named_call(**kwargs):
                args = tuple(kwargs[k] for k in keys)
                return func(*args)
            return named_call

        else:
            return theano.function(inputs=self._inputs or [], outputs=self._outputs, updates=self._updates,
                                   givens=self._givens)

    def _merge_updates(self, updates):
        """Merge several updates into one update, for the backend."""
        if isinstance(updates, (dict, OrderedDict)):
            return OrderedDict(updates)
        ret = OrderedDict()
        for u in updates:
            for k, v in six.iteritems(u):
                ret[k] = v
        return ret


make_function = Function

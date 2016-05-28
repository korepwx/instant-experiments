# -*- coding: utf-8 -*-
import theano

from ipwxlearn.glue.common.function import BaseFunction

__all__ = ['Function', 'make_function']


class Function(BaseFunction):
    """Theano compiled function."""

    def _compile(self):
        """
        Derived classes should override this to actually compile the backend function.
        Returns the callable object which could be called to execute the backend function.
        """
        return theano.function(inputs=self._inputs, outputs=self._outputs, updates=self._updates, givens=self._givens)

    def _merge_updates(self, updates):
        """Merge several updates into one update, for the backend."""
        if isinstance(updates, dict):
            return updates
        ret = {}
        for u in updates:
            ret.update(u)
        return ret


make_function = Function

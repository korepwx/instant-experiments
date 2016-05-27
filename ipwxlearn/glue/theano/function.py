# -*- coding: utf-8 -*-
import theano

from ipwxlearn.glue.common.function import BaseFunction

__all__ = ['make_function']


class Function(BaseFunction):
    """Theano function wrapper."""

    def _compile(self):
        return theano.function(inputs=self._inputs, outputs=self._outputs, updates=self._updates, givens=self._givens)


make_function = Function

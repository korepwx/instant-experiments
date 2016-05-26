# -*- coding: utf-8 -*-
import six

from ipwxlearn.glue.common.session import BaseSession, current_session

__all__ = [
    'Session',
    'current_session'
]


class Session(BaseSession):
    """
    Theano tensor computing session.
    """

    def _enter(self, feed_values, init_values):
        for var, value in six.iteritems(feed_values):
            var.set_value(value, borrow=False)
        for var, init in six.iteritems(init_values):
            if init is not None:
                var.set_value(init(), borrow=False)

    def _exit(self, save_vars):
        return self._extract_vars(save_vars)

    def _extract_vars(self, vars):
        from .utils import maybe_extract_scalar
        return {var: maybe_extract_scalar(var.get_value(borrow=False)) for var in vars}

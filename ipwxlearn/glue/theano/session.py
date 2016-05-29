# -*- coding: utf-8 -*-
import six

from ipwxlearn.utils.misc import maybe_iterable_to_list
from .utils import maybe_extract_scalar
from ..common.session import BaseSession, current_session

__all__ = [
    'Session',
    'current_session'
]


class Session(BaseSession):
    """Theano computing session."""

    def _enter(self, feed_values, init_values):
        for var, value in six.iteritems(feed_values):
            var.set_value(value, borrow=False)
        for var, init in six.iteritems(init_values):
            if init is not None:
                var.set_value(init(), borrow=False)

    def _exit(self, save_vars):
        return self.get_variable_values_dict(save_vars)

    def get_variable_values(self, vars):
        vars = maybe_iterable_to_list(vars)
        if not isinstance(vars, list):
            return maybe_extract_scalar(vars.get_value(borrow=False))
        return tuple(maybe_extract_scalar(v.get_value(borrow=False)) for v in vars)

    def set_variable_values(self, vars_values):
        for var, value in six.iteritems(vars_values):
            var.set_value(value, borrow=False)

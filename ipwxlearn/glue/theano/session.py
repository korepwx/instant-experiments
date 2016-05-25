# -*- coding: utf-8 -*-
import six

from ipwxlearn.glue.common.session import BaseSession


class Session(BaseSession):
    """
    Theano tensor computing session.
    """

    def _enter(self, feed_values, init_values):
        for var, value in six.iteritems(feed_values):
            var.set_value(value, borrow=False)
        for var, init in six.iteritems(init_values):
            # for Theano backend, the initializer should either be a direct value,
            # or a callable object that generates some value.
            if hasattr(init, '__call__'):
                init = init()
            var.set_value(init, borrow=False)

    def _exit(self, save_vars):
        return {var: var.get_value(borrow=False) for var in save_vars}

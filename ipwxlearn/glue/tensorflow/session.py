# -*- coding: utf-8 -*-
import sys

import six
import tensorflow as tf

from ipwxlearn.utils.misc import maybe_iterable_to_list
from ..common.session import BaseSession, current_session

__all__ = [
    'Session',
    'current_session'
]


class Session(BaseSession):
    """TensorFlow computing session."""

    def __init__(self, graph=None, feed_values=None, init_variables=False, checkpoint_file=None,
                 max_checkpoints=10):
        super(Session, self).__init__(graph=graph, feed_values=feed_values, init_variables=init_variables,
                                      checkpoint_file=checkpoint_file, max_checkpoints=max_checkpoints)
        self._session = tf.Session(graph=self.graph.tf_graph)

    @property
    def tf_session(self):
        """Get the backend session object."""
        return self._session

    def __enter__(self):
        self._session.__enter__()
        try:
            super(Session, self).__enter__()
        except:
            self._session.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            super(Session, self).__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._session.__exit__(exc_type, exc_val, exc_tb)

    def _enter(self, feed_values, init_values):
        ops = [tf.assign(var, value) for var, value in six.iteritems(feed_values)] + \
            [var.initializer for var, init in six.iteritems(init_values) if init is not None]
        self._session.run(ops)

    def _exit(self, save_vars):
        return self.get_variable_values_dict(save_vars)

    def get_variable_values(self, vars):
        vars = maybe_iterable_to_list(vars)
        if not isinstance(vars, list):
            return self._session.run([vars])[0]
        return tuple(self._session.run(vars))

    def set_variable_values(self, vars_values):
        self._session.run([tf.assign(var, value) for var, value in six.iteritems(vars_values)])

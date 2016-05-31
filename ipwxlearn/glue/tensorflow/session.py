# -*- coding: utf-8 -*-
from __future__ import absolute_import

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
            self._session.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            super(Session, self).__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._session.__exit__(None, None, None)

    def _enter(self, feed_values, init_values):
        # we have to call `initialize_all_variables`, since some variables may not be managed by us.
        self._session.run(tf.initialize_all_variables())

        # run additional assignments.
        ops = [tf.assign(var, value) for var, value in six.iteritems(feed_values)]
        if ops:
            self._session.run(ops)

    def _exit(self, save_vars):
        return self.get_variable_values_dict(save_vars)

    def get_variable_values(self, vars):
        vars = maybe_iterable_to_list(vars)
        if not isinstance(vars, list):
            return self._session.run([vars])[0]
        if vars:
            return tuple(self._session.run(vars))
        return ()

    def set_variable_values(self, vars_values):
        updates = [tf.assign(var, value) for var, value in six.iteritems(vars_values)]
        if updates:
            self._session.run(updates)

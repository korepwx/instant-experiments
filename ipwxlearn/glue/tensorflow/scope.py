# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf

from ipwxlearn.utils import misc
from .. import common
from ..common.scope import current_name_scope, iter_name_scopes

__all__ = ['NameScope', 'current_name_scope', 'name_scope', 'iter_name_scopes']


class NameScope(common.scope.NameScope):
    """TensorFlow name scope context."""

    def _create_sub_scope(self, name):
        return NameScope(self.resolve_name(name))

    @property
    def tf_scope_name(self):
        if self.full_name is None:
            return None
        return self.full_name + '/'


@misc.contextmanager
def name_scope(name_or_scope):
    if isinstance(name_or_scope, NameScope):
        scope = name_or_scope
    else:
        scope = current_name_scope().sub_scope(name_or_scope)

    if scope.tf_scope_name is None:
        yield scope
    else:
        with tf.name_scope(scope.tf_scope_name):
            scope.push_default()
            yield scope
            scope.pop_default()

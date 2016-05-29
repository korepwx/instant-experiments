# -*- coding: utf-8 -*-
import tensorflow as tf

from ipwxlearn.utils import misc
from .. import common
from ..common.scope import current_name_scope

__all__ = ['current_name_scope', 'name_scope']


class NameScope(common.scope.NameScope):
    """TensorFlow name scope context."""

    def __init__(self, scope):
        super(NameScope, self).__init__(scope.name)

    def sub_scope(self, name):
        raise NotImplementedError()


@misc.contextmanager
def name_scope(name_or_scope):
    with tf.variable_scope(name_or_scope) as scope:
        ns = NameScope(scope)
        yield ns

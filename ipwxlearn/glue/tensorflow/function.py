# -*- coding: utf-8 -*-
from __future__ import absolute_import

from collections import OrderedDict

import six
import tensorflow as tf

from ipwxlearn.glue import current_session
from ipwxlearn.utils.misc import ensure_list_sealed
from .utils import merge_updates
from ..common.function import BaseFunction

__all__ = ['Function', 'make_function']


class Function(BaseFunction):
    """TensorFlow compiled function."""

    def _compile(self):
        if self._updates is None:
            updates = []
        else:
            updates = ensure_list_sealed(self._updates)

        if isinstance(self._outputs, list):
            outputs = self._outputs or []
            fetches = outputs + updates
            direct_value = False
            output_num = len(outputs)
        elif self._outputs is not None:
            fetches = [self._outputs] + updates
            direct_value = True
            output_num = 1
        else:
            fetches = updates
            direct_value = False
            output_num = 0

        if isinstance(self._inputs, (dict, OrderedDict)):
            def get_feed_dict(**kwargs):
                givens = self._givens.copy() if self._givens else {}
                for k, v in six.iteritems(self._inputs):
                    givens[v] = kwargs[k]
                return givens
        else:
            def get_feed_dict(*args):
                givens = self._givens.copy() if self._givens else {}
                if not self._inputs:
                    return givens
                for var, value in zip(self._inputs, args):
                    givens[var] = value
                return givens

        def run_func(*args, **kwargs):
            givens = get_feed_dict(*args, **kwargs)

            # Special trick: TensorFlow don't allow us to both feed & fetch the same variable.
            # Thus we have to wrap these variables with some computation node.
            fetches2 = [tf.identity(v) if v in givens else v for v in fetches]

            ret = current_session().tf_session.run(fetches2, feed_dict=givens)
            ret = ret[: output_num]
            return tuple(ret) if not direct_value else ret[0]

        return run_func

    def _merge_updates(self, updates):
        """Merge several updates into one update, for the backend."""
        return merge_updates(updates)


make_function = Function

# -*- coding: utf-8 -*-
from __future__ import absolute_import

from collections import OrderedDict

import six
import theano

from ipwxlearn.utils.misc import ensure_list_sealed
from .summary import SummaryObject
from ..common.function import BaseFunction

__all__ = ['Function', 'make_function']


class Function(BaseFunction):
    """Theano compiled function."""

    def _compile(self):
        # deal with output list.
        outputs = []
        output_mapping = []
        direct_output = not isinstance(self._outputs, list)

        if self._outputs:
            for o in ensure_list_sealed(self._outputs):
                if isinstance(o, SummaryObject):
                    output_mapping.append(o)
                else:
                    output_mapping.append(None)
                    outputs.append(o)

        def merge_results(results):
            ret = []
            it = iter(results)
            for o in output_mapping:
                if o is None:
                    ret.append(next(it))
                else:
                    ret.append(None)
            return ret[0] if direct_output and ret else tuple(ret)

        # deal with input list or dict.
        if isinstance(self._inputs, (dict, OrderedDict)):
            keys = []
            inputs = []
            for k, v in six.iteritems(self._inputs):
                keys.append(k)
                inputs.append(v)
            func = theano.function(inputs=inputs, outputs=outputs, updates=self._updates, givens=self._givens)

            def named_call(**kwargs):
                args = tuple(kwargs[k] for k in keys)
                ret = func(*args)
                return merge_results(ret)
            return named_call

        else:
            func = theano.function(inputs=self._inputs or [], outputs=outputs, updates=self._updates,
                                   givens=self._givens)

            def unnamed_call(*args):
                ret = func(*args)
                return merge_results(ret)
            return unnamed_call

    def _merge_updates(self, updates):
        """Merge several updates into one update, for the backend."""
        if isinstance(updates, (dict, OrderedDict)):
            return OrderedDict(updates)
        ret = OrderedDict()
        for u in updates:
            for k, v in six.iteritems(u):
                ret[k] = v
        return ret


make_function = Function

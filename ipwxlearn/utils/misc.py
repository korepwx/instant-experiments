# -*- coding: utf-8 -*-
from __future__ import absolute_import

import contextlib
import copy
import re

import six


def require_object_name(name):
    """
    Check whether or not :param:`name` could be used as the name of some object.

    When defining any layer or variable, we require the user to provide a name which
    follows the restrictions of a Python variable name.  This could make sure the code
    could run under various backend.

    :raises ValueError: If :param:`name` does not following the restrictions.
    """
    if not re.match(r'^[_a-zA-Z][_a-zA-Z0-9]*$', name):
        raise ValueError('%s is not a valid object name.' % repr(name))


def require_object_full_name(full_name):
    """
    Check whether or not :param:`full_name` could be used as full name of some object.
    """
    parts = full_name.split('/')
    return parts and all(require_object_name(n) for n in parts)


def silent_try(_function, *args, **kwargs):
    """Call function with args and kwargs, without throw any error."""
    try:
        _function(*args, **kwargs)
    except Exception:
        pass


def maybe_iterable_to_list(iterable_or_else, exclude_types=()):
    """
    Convert given object to list if it is an iterable object, or keep it still if not.

    :param iterable_or_else: Iterable object or anything else.
    :param exclude_types: Don't convert the given object of these types to list, even if it is iterable.

    :return: List, or iterator_or_else itself if the given object could not be converted to list.
    """
    try:
        if not exclude_types or not isinstance(iterable_or_else, exclude_types):
            return list(iterable_or_else)
    except:
        pass
    return iterable_or_else


def ensure_list_sealed(element_or_iterable):
    """
    Ensure that given element, or a list of elements is sealed in a list.

    :param element_or_iterable: Element, or an iterable of elements.
    :return: List of elements.
    """
    if isinstance(element_or_iterable, (tuple, list)) or hasattr(element_or_iterable, '__next__'):
        return list(element_or_iterable)
    return [element_or_iterable]


#: Shortcut to the contextmanager from standard library.
contextmanager = contextlib.contextmanager


def _ensure_exit_context(context_stack, exc_type, exc_val, exc_tb):
    """
    Exit all the contexts in a recursive way will enable the Python interpreter to realize that
    more than one exception is raised during the cleanup stage.
    """
    try:
        back = context_stack.pop()
        back.__exit__(exc_type, exc_val, exc_tb)
    finally:
        if context_stack:
            _ensure_exit_context(context_stack, exc_type, exc_val, exc_tb)


class _MergedContexts(object):
    """
    contextlib.contextmanager might not execute the cleanup code if more than one exception is raised
    when leaving the context.  So we need this class to ensure every context is exited.
    """

    def __init__(self, contexts):
        self.contexts = ensure_list_sealed(contexts)

    def __enter__(self):
        entered_ctx = []
        try:
            for ctx in self.contexts:
                ctx.__enter__()
                entered_ctx.append(ctx)
        except:
            _ensure_exit_context(entered_ctx, None, None, None)
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        _ensure_exit_context(copy.copy(self.contexts), exc_type, exc_val, exc_tb)


def merged_context(*contexts):
    """Merge several contexts into one, such that every contexts would be ensured to exit."""
    return _MergedContexts(contexts)


class _AssertRaisesMessageContext(object):
    def __init__(self, owner, ctx, message):
        self.owner = owner
        self.ctx = ctx
        self.message = message

    def __enter__(self):
        self.ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        ret = self.ctx.__exit__(exc_type, exc_val, exc_tb)
        self.owner.assertEquals(str(self.ctx.exception), self.message)
        return ret


def assert_raises_message(test_case, error_type, message):
    return _AssertRaisesMessageContext(test_case, test_case.assertRaises(error_type), message)


def infinite_counter(start, step=1):
    """Iterator that counts from start to infinite number."""
    i = start
    while True:
        yield i
        i += step


class DictProxy(object):
    """Dict-like object that proxies operations to another dict-like object."""

    def __init__(self, proxied):
        self._proxied = proxied

    def __len__(self):
        return len(self._proxied)

    def __contains__(self, key):
        return key in self._proxied

    def __getitem__(self, key):
        return self._proxied[key]

    def __setitem__(self, key, value):
        self._proxied[key] = value

    def get(self, key, default=None):
        return self._proxied.get(key, default)

    def items(self):
        return self._proxied.items()

    def values(self):
        return self._proxied.values()

    def keys(self):
        return self._proxied.keys()

    if six.PY2:
        def iteritems(self):
            return self._proxied.iteritems()

        def itervalues(self):
            return self._proxied.itervalues()

        def iterkeys(self):
            return self._proxied.iterkeys()


def unique(list):
    """Deduplicate elements in a list."""
    ret = []
    for e in list:
        if e not in ret:
            ret.append(e)
    return ret


def flatten_list(root):
    """
    Flatten the given list, so that all the non-list elements in it would be
    aggregated to the root-level, with the same order as they appear in the original list.
    """
    ret = []
    try:
        stack = list(reversed(root))
    except TypeError:
        stack = [root]
    while stack:
        u = stack.pop()
        if isinstance(u, list):
            for v in reversed(u):
                stack.append(v)
        else:
            ret.append(u)
    return ret

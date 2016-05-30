# -*- coding: utf-8 -*-
import contextlib
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

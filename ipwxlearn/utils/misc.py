# -*- coding: utf-8 -*-
import re


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


def maybe_iterable_to_list(iterable_or_else):
    """
    Convert given object to list if it is an iterable object, or keep it still if not.

    :param iterable_or_else: Iterable object or anything else.
    :return: List, or iterator_or_else itself.
    """
    try:
        return list(iterable_or_else)
    except:
        return iterable_or_else

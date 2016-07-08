# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import six
import theano
from theano import tensor as T

from ..common.utils import (get_graph_state, get_graph_state_by_vars, set_graph_state, save_graph_state,
                            save_graph_state_by_vars, restore_graph_state)

__all__ = [
    'as_dtype',
    'make_variable',
    'is_variable',
    'make_placeholder',
    'make_placeholder_for',
    'get_variable_values',
    'set_variable_values',
    'get_variable_name',
    'get_graph_state',
    'get_graph_state_by_vars',
    'set_graph_state',
    'save_graph_state',
    'save_graph_state_by_vars',
    'restore_graph_state'
]


class VariableInitializer(object):

    def __init__(self, _function, *args, **kwargs):
        self.fn = _function
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.fn(*self.args, **self.kwargs)


def maybe_convert_dtype(method, dtype=None):
    def wrapper(*args, **kwargs):
        v = method(*args, **kwargs)
        if isinstance(v, np.ndarray):
            v = v.astype(dtype)
        return v
    return method if dtype is None else wrapper


def as_dtype(dtype):
    """
    Convert the specified dtype to backend dtype.

    :param dtype: String or numpy dtype.
    :return: Backend dtype.
    """
    if isinstance(dtype, six.class_types):
        if issubclass(dtype, np.dtype):
            dtype = dtype.name
        elif issubclass(dtype, np.generic):
            dtype = dtype.__name__
    return dtype


def make_initializer(init, shape, dtype=None):
    """
    Make a initializer according to given init value.

    Will return the init value itself if it is a scaler or a numpy array, or a VariableInitializer
    if init is a callable function to generate some value according to given shape.

    :param init: scalar, numpy array, or an initializer for the backend variable.
    :param shape: Shape of the variable, a tuple of integers.
    :param dtype: Data type of the variable.  Might be ignored.

    :rtype: :class:`VariableInitializer`
    """
    shape = tuple(shape)
    if isinstance(init, np.ndarray):
        if shape != init.shape:
            raise RuntimeError('initial value has shape %s, should be %s' % (init.shape, shape))
        init = np.asarray(init, dtype=dtype) if dtype else np.copy(init)
        fn = VariableInitializer(lambda: init)

    elif isinstance(init, six.integer_types + six.string_types + (float,)):
        if shape:
            raise RuntimeError('initial value is a scalar, should have shape %s' % shape)
        init = np.array([init], dtype=dtype)[0] if dtype else init
        fn = VariableInitializer(lambda: init)

    elif isinstance(init, VariableInitializer):
        # the initializer is already a VariableInitializer, just use it.
        fn = init

    elif callable(init):
        fn = VariableInitializer(maybe_convert_dtype(init, dtype), shape)

    else:
        raise TypeError('cannot initialize variable, since "init" is neither a constant nor an initializer.')

    return fn


def make_variable(name, shape, init, dtype=None, **tags):
    """
    Make a backend variable and add to current graph.

    :param name: Name of the variable.
    :param shape: Shape of the variable, a tuple of integers.
    :param init: numpy array, or an initializer for the backend variable.
    :param dtype: Data type of the variable.  Might be ignored.
    :param tags: Tags for this variable.

    :return: Backend variable object.
    """
    from .scope import current_name_scope
    shape = tuple(shape)
    full_name = current_name_scope().resolve_name(name)
    init = make_initializer(init, shape, dtype=dtype)
    var = theano.shared(init(), name=full_name)
    current_name_scope().add_variable(var, init, name, **tags)
    return var


def is_variable(x):
    """
    Check whethor or not 'x' is a backend variable.

    A variable refers to a value object in memory. The results of tensor operations are typically
    not variables.
    """
    return isinstance(x, theano.compile.SharedVariable)


def make_placeholder(name, shape, dtype, **tags):
    """
    Make a backend placeholder and add to graph.

    :param name: Name of the placeholder.
    :param shape: Shape of the placeholder, a tuple of integers or None.
    :param dtype: Data type of the placeholder.
    :param tags: Tags for this placeholder.

    :return: Backend placeholder object.
    """
    # TODO: add placeholders to the graph.
    shape = tuple(shape)
    dtype = as_dtype(dtype)
    return T.TensorType(dtype, (False,) * len(shape))(name)


def make_placeholder_for(name, data, dtype=None, **tags):
    """
    Make a placeholder for specified data array.

    The constructed placeholder will have the shape (None,) + data.shape[1:],
    if the backend supports shape on placeholders.  Furthermore, the placeholder
    will have the same dtype as the data, unless a different one is given.

    :param name: Name of the placeholder.
    :param data: Data to put into this placeholder.
    :param dtype: Specify a data type other than the data.
    :param tags: Tags for this placeholder.

    :return: Backend placeholder object.
    """
    return make_placeholder(name, shape=(None,) + data.shape[1:], dtype=dtype or data.dtype, **tags)


def maybe_extract_scalar(v):
    """Maybe extract scalar from numpy 0-dimensional array."""
    return np.asarray([v], dtype=v.dtype)[0] if v.shape == () else v


def get_variable_values(vars):
    from .session import current_session
    return current_session().get_variable_values(vars)


def set_variable_values(vars_values):
    """
    Set the values of specified variables.

    :param vars_values: Dict from backend variables to their values.
    """
    from .session import current_session
    return current_session().set_variable_values(vars_values)


def get_variable_name(var):
    """
    Get the full name of specified backend variable.
    Might return None if the variable does not have a name.
    """
    return var.name

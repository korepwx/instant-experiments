# -*- coding: utf-8 -*-
import numpy as np
import six
import theano
from theano import tensor as T

from ipwxlearn.glue.theano.graph import current_graph
from ipwxlearn.glue.theano.scope import current_name_scope

__all__ = [
    'make_variable',
    'make_placeholder',
    'get_variable_values',
    'set_variable_values'
]


class VariableInitializer(object):

    def __init__(self, _function, *args, **kwargs):
        self.fn = _function
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.fn(*self.args, **self.kwargs)


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
        init = np.asarray([init], dtype=dtype)[0] if dtype else init
        fn = VariableInitializer(lambda: init)

    elif isinstance(init, VariableInitializer):
        # the initializer is already a VariableInitializer, just use it.
        fn = init

    elif callable(init):
        fn = VariableInitializer(init, shape)

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
    shape = tuple(shape)
    full_name = current_name_scope().resolve_name(name)
    init = make_initializer(init, shape, dtype=dtype)
    var = theano.shared(init(), name=full_name)
    current_graph().add_variable(var, init, name, **tags)
    return var


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
    if isinstance(dtype, six.class_types):
        if issubclass(dtype, np.dtype):
            dtype = dtype.name
        elif issubclass(dtype, np.generic):
            dtype = dtype.__name__
    return T.TensorType(dtype, tuple(not not k for k in shape))(name)


def maybe_extract_scalar(v):
    """Maybe extract scalar from numpy 0-dimensional array."""
    return np.asarray([v], dtype=v.dtype)[0] if v.shape == () else v


def get_variable_values(vars):
    """
    Get the values of specified variables.

    :param vars: iterable backend variables.
    :return: Tuple of variable values.
    """
    if isinstance(vars, theano.compile.SharedVariable):
        return maybe_extract_scalar(vars.get_value(borrow=False))
    return tuple(get_variable_values(v) for v in vars)


def set_variable_values(vars_values):
    """
    Set the values of specified variables.

    :param vars_values: Dict from backend variables to their values.
    """
    for var, value in six.iteritems(vars_values):
        var.set_value(value, borrow=False)

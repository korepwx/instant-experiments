# -*- coding: utf-8 -*-
import numpy as np
import six
import theano
from theano import tensor as T

from ipwxlearn.glue.theano import current_graph, current_name_scope, floatX


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

    if isinstance(init, np.ndarray):
        if shape != init.shape:
            raise RuntimeError('initial value has shape %s, should be %s' % (init.shape, shape))
        init = np.asarray(init, dtype=dtype) if dtype else np.copy(init)
        var = theano.shared(init, name=full_name)

    elif isinstance(init, six.integer_types + (float, )):
        if shape:
            raise RuntimeError('initial value is a scalar, should have shape %s' % shape)
        init = np.asarray([init], dtype=dtype)[0] if dtype else init
        var = theano.shared(init, name=full_name)

    elif hasattr(init, '__call__'):
        init = (lambda: init(shape))
        zeros = np.zeros(shape, dtype=dtype) if dtype else np.zeros(shape, dtype=floatX)
        var = theano.shared(zeros, name=full_name)

    else:
        raise TypeError('cannot initialize variable, since "init" is not a numpy array or an initializer.')

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
    return T.TensorType(dtype, tuple(not not k for k in shape))(name)

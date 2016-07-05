# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf

from ipwxlearn.utils.misc import maybe_iterable_to_list
from .utils import as_dtype


# imported unary operators
log = tf.log
tanh = tf.tanh
sqrt = tf.sqrt
sin = tf.sin
cos = tf.cos
abs = tf.abs
sigmoid = tf.sigmoid
softmax = tf.nn.softmax


# imported binary operators
eq = tf.equal
neq = tf.not_equal
maximum = tf.maximum
minimum = tf.minimum


def cast(tensor, dtype):
    """Cast `tensor` to `dtype`."""
    return tf.cast(tensor, as_dtype(dtype))


def sum(input, axis=None, keepdims=False):
    return tf.reduce_sum(input, reduction_indices=axis, keep_dims=keepdims)


def mean(input, axis=None, keepdims=False):
    return tf.reduce_mean(input, reduction_indices=axis, keep_dims=keepdims)


def max(input, axis=None, keepdims=False):
    return tf.reduce_max(input, reduction_indices=axis, keep_dims=keepdims)


def argmax(input, axis):
    return tf.arg_max(input, dimension=axis)


def min(input, axis=None, keepdims=False):
    return tf.reduce_min(input, reduction_indices=axis, keep_dims=keepdims)


def argmin(input, axis):
    return tf.arg_min(input, dimension=axis)


def dot(a, b):
    # TODO: implement N-dimensinal dot product that consistent with Numpy.
    a_shape = a.get_shape().as_list()
    a_dims = len(a_shape)
    b_shape = b.get_shape().as_list()
    b_dims = len(b_shape)

    # scalar dot scalar, scalar dot tensor or tensor dot scalar: just do element-wise multiply.
    if a_dims == 0 or b_dims == 0:
        return a * b

    # vector dot vector, where we can just perform element-wise prod, and then sum them all.
    if a_dims == 1 and b_dims == 1:
        return tf.reduce_sum(a * b)

    # vector dot matrix or matrix dot vector, where we should expand the vector to matrix, and then squeeze result.
    if a_dims <= 2 and b_dims <= 2:
        if a_dims == 1:
            a = tf.expand_dims(a, dim=0)
        else:
            b = tf.expand_dims(b, dim=1)
        ret = tf.matmul(a, b)
        if a_dims == 1:
            ret = tf.squeeze(ret, [0])
        else:
            ret = tf.squeeze(ret, [1])
        return ret

    # throw exception, that we do not know how to handle the situation.
    raise TypeError('Tensor dot between shape %r and %r is not supported.' % (a_shape, b_shape))


def concat(tensors, axis):
    """
    Concatenate specified tensors along certain axis.

    :param tensors: Iterable of tensors.
    :param axis: Concatenate axis.
    :return: Concatenated tensor.
    """
    return tf.concat(axis, maybe_iterable_to_list(tensors))


def squeeze(x, dims=None):
    """
    Remove the broadcastable dimensions from the shape of a tensor.

    :param x: Tensor whose dimensions should be removed.
    :param dims: Dimensions to be removed.  If not specified, remove all broadcastable dimensions.
    """
    return tf.squeeze(x, squeeze_dims=dims)


def flatten(x, ndim=1):
    """
    Returns a view of x with ndim dimensions, whose shape for the first ndim-1 dimensions will be
    the same as x, and shape in the remaining dimension will be expanded to fit in all the data
    from x.
    """
    shape = x.get_shape()
    total_dim = len(shape)

    if total_dim == ndim:
        return x
    elif total_dim < ndim:
        raise ValueError('Attempt to flatten "x" to %r dimensions, but "x" only has %r dimensions.' %
                         (ndim, total_dim))

    if shape.is_fully_defined():
        # all the dimensions are fixed, thus we can use the static shape.
        shape = shape[:ndim - 1] + [-1]
    else:
        # the shape is dynamic, so we have to generate a dynamic flatten shape.
        shape = tf.concat(0, [tf.shape(x)[:ndim - 1], [-1]])

    return tf.reshape(x, shape)


# Operations that change the values of variables
def assign(target, value):
    return tf.assign(target, value)


def l1_reg(params):
    """
    Compute the L1 regularization term for given parameters.

    :param params: Backend variable, or a list of backend variables.
    :return: L1 loss expression.
    """
    from ipwxlearn.utils import sysops
    if isinstance(params, (tuple, list)) or hasattr(params, '__iter__'):
        return sysops.sum_(tf.reduce_sum(tf.abs(p)) for p in params)
    return tf.reduce_sum(tf.abs(params))


def l2_reg(params):
    """
    Compute the L2 regularization term for given parameters.

    :param params: Backend variable, or a list of backend variables.
    :return: L2 loss expression.
    """
    from ipwxlearn.utils import sysops
    if isinstance(params, (tuple, list)) or hasattr(params, '__iter__'):
        return sysops.sum_(tf.nn.l2_loss(p) for p in params)
    return tf.nn.l2_loss(params)

# -*- coding: utf-8 -*-
from __future__ import absolute_import

from collections import OrderedDict

from theano import tensor as T

from ipwxlearn.utils.misc import maybe_iterable_to_list
from .utils import as_dtype


# imported unary operators
log = T.log
tanh = T.tanh
sqrt = T.sqrt
sin = T.sin
cos = T.cos
abs = T.abs_
sigmoid = T.nnet.sigmoid
softmax = T.nnet.softmax

# imported binary operators
eq = T.eq
neq = T.neq
maximum = T.maximum
minimum = T.minimum


def cast(tensor, dtype):
    """Cast `tensor` to `dtype`."""
    return tensor.astype(as_dtype(dtype))


def sum(input, axis=None, keepdims=False):
    return T.sum(input, axis=axis, keepdims=keepdims)


def mean(input, axis=None, keepdims=False):
    # We require neither "dtype" nor "acc_dtype" argument, even if Theano expect us to do so
    # in order to ensure the result is not suffering from overflow or underflow.
    # However, we argue that this should be ensured by the library, not by the external user.
    return T.mean(input, axis=axis, keepdims=keepdims)


def max(input, axis=None, keepdims=False):
    return T.max(input, axis=axis, keepdims=keepdims)


def argmax(input, axis):
    return T.argmax(input, axis=axis, keepdims=False)


def min(input, axis=None, keepdims=False):
    return T.min(input, axis=axis, keepdims=keepdims)


def argmin(input, axis):
    return T.argmin(input, axis=axis, keepdims=False)


def dot(a, b):
    return T.dot(a, b)


def concat(tensors, axis):
    """
    Concatenate specified tensors along certain axis.

    :param tensors: Iterable of tensors.
    :param axis: Concatenate axis.
    :return: Concatenated tensor.
    """
    return T.concatenate(maybe_iterable_to_list(tensors), axis)


def squeeze(x, dims=None):
    """
    Remove the broadcastable dimensions from the shape of a tensor.

    :param x: Tensor whose dimensions should be removed.
    :param dims: Dimensions to be removed.  If not specified, remove all broadcastable dimensions.
    """
    if dims is None:
        return T.squeeze(x)
    else:
        ndim = x.ndim
        dims = [d + ndim if d < 0 else d for d in dims]
        return x.dimshuffle(*(i for i in range(ndim) if i not in dims))


def flatten(x, ndim=1):
    """
    Returns a view of x with ndim dimensions, whose shape for the first ndim-1 dimensions will be
    the same as x, and shape in the remaining dimension will be expanded to fit in all the data
    from x.
    """
    return T.flatten(x, ndim)


def shape(x):
    """Get the shape of x."""
    return x.shape


def reshape(x, shape):
    """Reshape x to specified shape."""
    return T.reshape(x, shape)


def transpose(x, axes=None):
    """Transpose x according to the order of axes."""
    return T.transpose(x, axes=axes)


# Operations that change the values of variables.
def assign(target, value):
    ret = OrderedDict()
    ret[target] = value
    return ret


def l1_reg(params):
    """
    Compute the L1 regularization term for given parameters.

    :param params: Backend variable, or a list of backend variables.
    :return: L1 loss expression.
    """
    from ipwxlearn.utils import sysops
    if isinstance(params, (tuple, list)) or hasattr(params, '__next__'):
        return sysops.sum_(T.sum(T.abs_(p)) for p in params)
    return T.sum(abs(params))


def l2_reg(params):
    """
    Compute the L2 regularization term for given parameters.

    :param params: Backend variable, or a list of backend variables.
    :return: L2 loss expression.
    """
    from ipwxlearn.utils import sysops
    if isinstance(params, (tuple, list)) or hasattr(params, '__next__'):
        return sysops.sum_(T.sum(p ** 2) * 0.5 for p in params)
    return T.sum(params ** 2) * 0.5

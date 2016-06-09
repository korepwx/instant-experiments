# -*- coding: utf-8 -*-
from __future__ import absolute_import

from collections import OrderedDict

from theano import tensor as T

# imported unary operators
log = T.log
tanh = T.tanh
sqrt = T.sqrt
sin = T.sin
cos = T.cos
abs = T.abs_

emo
# imported binary operators
eq = T.eq
neq = T.neq
maximum = T.maximum
minimum = T.minimum


def cast(tensor, dtype):
    """Cast `tensor` to `dtype`."""
    return tensor.astype(dtype)


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


# Operations that change the values of variables.
def assign(target, value):
    ret = OrderedDict()
    ret[target] = value
    return ret

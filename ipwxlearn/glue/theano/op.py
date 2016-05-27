# -*- coding: utf-8 -*-
from theano import tensor as T

# Unary operators on scalars.


# Binary operators on scalars.
eq = T.eq
neq = T.neq


def sum(input, axis=None, keepdims=False):
    return T.sum(input, axis=axis, keepdims=keepdims)


def mean(input, axis=None, keepdims=False):
    # We require neither "dtype" nor "acc_dtype" argument, even if Theano expect us to do so
    # in order to ensure the result is not suffering from overflow or underflow.
    # However, we argue that this should be ensured by the library, not by the external user.
    return T.mean(input, axis=axis, keepdims=keepdims)


def max(input, axis=None, keepdims=False):
    return T.max(input, axis=axis, keepdims=keepdims)


def argmax(input, axis=None, keepdims=False):
    return T.argmax(input, axis=axis, keepdims=keepdims)


def min(input, axis=None, keepdims=False):
    return T.min(input, axis=axis, keepdims=keepdims)


def argmin(input, axis=None, keepdims=False):
    return T.argmin(input, axis=axis, keepdims=keepdims)
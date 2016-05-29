# -*- coding: utf-8 -*-
import tensorflow as tf


# Unary operators on scalars.

# Binary operators on scalars.
eq = tf.equal
neq = tf.not_equal


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


# Operations that change the values of variables
def assign(target, value):
    return tf.assign(target, value)

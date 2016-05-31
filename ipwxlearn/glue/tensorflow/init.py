# -*- coding: utf-8 -*-
from __future__ import absolute_import

import math

import tensorflow as tf


class _Initializer(object):
    """Adapter from the glue initializer interface to tensorflow initializer."""

    def __init__(self, tf_init):
        self.tf_init = tf_init

    def __call__(self, shape):
        from ipwxlearn import glue
        return self.tf_init(shape, dtype=tf.as_dtype(glue.config.floatX))


class Normal(_Initializer):
    """Sample initial weights from the Gaussian distribution."""

    def __init__(self, std=0.01, mean=0.0):
        super(Normal, self).__init__(tf.random_normal_initializer(mean=mean, stddev=std))


class Uniform(_Initializer):
    """Sample initial weights from the uniform distribution."""

    def __init__(self, range=0.01):
        try:
            a, b = range
        except TypeError:
            a, b = -range, range
        super(Uniform, self).__init__(tf.random_uniform_initializer(minval=a, maxval=b))


class _Xavier(_Initializer):
    """
    Base xavier initializer.

    :param gain: Scaling factor for the weights. Set this to 1.0 for linear and sigmoid
                 units, to 'relu' or sqrt(2) for rectified linear units. Other transfer
                 functions may need different factors.
    """

    @staticmethod
    def xavier_initializer(uniform=True, seed=None, gain=1.0, dtype=None):
        def _initializer(shape, dtype=dtype):
            from ipwxlearn import glue
            dtype = dtype or glue.config.floatX

            n_inputs, n_outputs = shape[0], shape[1]
            if uniform:
                # 6 was used in the paper.
                init_range = gain * math.sqrt(6.0 / (n_inputs + n_outputs))
                return tf.random_uniform(shape, -init_range, init_range, dtype, seed=seed)
            else:
                # 3 gives us approximately the same limits as above since this repicks
                # values greater than 2 standard deviations from the mean.
                stddev = gain * math.sqrt(3.0 / (n_inputs + n_outputs))
                return tf.truncated_normal(shape, 0.0, stddev, dtype, seed=seed)
        return _initializer


class XavierNormal(_Xavier):
    """
    Xavier weight initialization with normal distribution.

    :param gain: Scaling factor for the weights. Set this to 1.0 for linear and sigmoid
                 units, to 'relu' or sqrt(2) for rectified linear units. Other transfer
                 functions may need different factors.
    """

    def __init__(self, gain=1.0):
        super(XavierNormal, self).__init__(self.xavier_initializer(uniform=False, gain=gain))


class XavierUniform(_Xavier):
    """
    Xavier weight initialization with uniform distribution.

    :param gain: Scaling factor for the weights. Set this to 1.0 for linear and sigmoid
                 units, to 'relu' or sqrt(2) for rectified linear units. Other transfer
                 functions may need different factors.
    """

    def __init__(self, gain=1.0):
        super(XavierUniform, self).__init__(self.xavier_initializer(uniform=True, gain=gain))


class Constant(_Initializer):
    """Initialize weights with constant value."""

    def __init__(self, val=0.0):
        super(Constant, self).__init__(tf.constant_initializer(val))

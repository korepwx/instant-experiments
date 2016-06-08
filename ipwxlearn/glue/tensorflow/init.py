# -*- coding: utf-8 -*-
from __future__ import absolute_import

import math

import numpy as np
import tensorflow as tf

__all__ = [
    'Normal',
    'Uniform',
    'XavierNormal',
    'XavierUniform',
    'Constant',
    'NormalizedUniform'
]


class Initializer(object):
    """Base variable initializer."""

    def __call__(self, shape):
        return self.sample(shape)

    def sample(self, shape):
        raise NotImplementedError()


class StandardInitializer(Initializer):
    """Adapter from the glue initializer interface to tensorflow initializer."""

    def __init__(self, tf_init):
        self.tf_init = tf_init

    def sample(self, shape):
        from ipwxlearn import glue
        return self.tf_init(shape, dtype=tf.as_dtype(glue.config.floatX))


class Normal(StandardInitializer):
    """Sample initial weights from the Gaussian distribution."""

    def __init__(self, std=0.01, mean=0.0):
        super(Normal, self).__init__(tf.random_normal_initializer(mean=mean, stddev=std))


class Uniform(StandardInitializer):
    """Sample initial weights from the uniform distribution."""

    def __init__(self, range=0.01):
        try:
            a, b = range
        except TypeError:
            a, b = -range, range
        super(Uniform, self).__init__(tf.random_uniform_initializer(minval=a, maxval=b))


class XavierBase(StandardInitializer):
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

            if len(shape) < 2:
                raise RuntimeError('This initializer only works with shapes of length >= 2: got %r.' % shape)
            n_inputs, n_outputs = shape[0], shape[-1]
            receptive_field_size = np.prod(shape[1: -1])    # for convolution layers.

            # TODO: check the original paper.
            if uniform:
                init_range = gain * math.sqrt(6.0 / ((n_inputs + n_outputs) * receptive_field_size))
                return tf.random_uniform(shape, -init_range, init_range, dtype, seed=seed)
            else:
                stddev = gain * math.sqrt(3.0 / ((n_inputs + n_outputs) * receptive_field_size))
                return tf.truncated_normal(shape, 0.0, stddev, dtype, seed=seed)
        return _initializer


class XavierNormal(XavierBase):
    """
    Xavier weight initialization with normal distribution.

    :param gain: Scaling factor for the weights. Set this to 1.0 for linear and sigmoid
                 units, to 'relu' or sqrt(2) for rectified linear units. Other transfer
                 functions may need different factors.
    """

    def __init__(self, gain=1.0):
        super(XavierNormal, self).__init__(self.xavier_initializer(uniform=False, gain=gain))


class XavierUniform(XavierBase):
    """
    Xavier weight initialization with uniform distribution.

    :param gain: Scaling factor for the weights. Set this to 1.0 for linear and sigmoid
                 units, to 'relu' or sqrt(2) for rectified linear units. Other transfer
                 functions may need different factors.
    """

    def __init__(self, gain=1.0):
        super(XavierUniform, self).__init__(self.xavier_initializer(uniform=True, gain=gain))


class Constant(StandardInitializer):
    """Initialize weights with constant value."""

    def __init__(self, val=0.0):
        super(Constant, self).__init__(tf.constant_initializer(val))


class NormalizedUniform(StandardInitializer):
    """
    Sample initial weights from symmetric uniform distribution, and then normalize the weights
    along given axis.

    :param axis: Norms are computed along this axis.
    :param norm: Ensure weights to have this norm.
    :param norm_type: Type of the norm, possible values are {'l1', 'l2'}.
    """

    @staticmethod
    def normalized_uniform(axis, norm, norm_type, seed=None, dtype=None):
        def _initializer(shape, dtype=dtype):
            from ipwxlearn import glue
            dtype = tf.as_dtype(dtype or glue.config.floatX)
            if dtype.is_integer:
                raise TypeError('Normalized uniform initializer cannot generate integer tensors.')
            norm_axis = axis if axis >= 0 else axis + len(shape)

            # start constructing the operation.
            with tf.op_scope([shape, norm], None, "normalized_uniform") as name:
                shape = tf.constant(tuple(shape), dtype=np.int32)
                rnd = tf.random_uniform(shape, dtype=dtype, seed=seed)

                if norm_type == 'l1':
                    rnd_norm = tf.reduce_sum(tf.abs(rnd), [norm_axis], keep_dims=True) / norm
                elif norm_type == 'l2':
                    rnd_norm = tf.sqrt(tf.reduce_sum(tf.pow(rnd, 2), [norm_axis], keep_dims=True)) / norm

                delta = tf.cast(tf.equal(rnd_norm, 0.0), dtype=dtype) * 1e-7
                return rnd / (rnd_norm + delta)

        if norm_type not in ('l1', 'l2'):
            raise ValueError('Unsupported norm type %r' % norm_type)
        return _initializer

    def __init__(self, axis=-1, norm=1.0, norm_type='l2'):
        super(NormalizedUniform, self).__init__(self.normalized_uniform(axis, norm, norm_type))

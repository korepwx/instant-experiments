# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne.init
import numpy as np

from ipwxlearn import glue

__all__ = [
    'Normal',
    'Uniform',
    'XavierNormal',
    'XavierUniform',
    'Constant',
    'NormalizedUniform'
]


class Normal(lasagne.init.Normal):
    """Sample initial weights from the Gaussian distribution."""

    def __init__(self, std=0.01, mean=0.0):
        super(Normal, self).__init__(std=std, mean=mean)


class Uniform(lasagne.init.Uniform):
    """Sample initial weights from the uniform distribution."""

    def __init__(self, range=0.01):
        super(Uniform, self).__init__(range=range)


class XavierNormal(lasagne.init.GlorotNormal):
    """
    Xavier weight initialization with normal distribution.

    :param gain: Scaling factor for the weights. Set this to 1.0 for linear and sigmoid
                 units, to 'relu' or sqrt(2) for rectified linear units. Other transfer
                 functions may need different factors.
    """

    def __init__(self, gain=1.0):
        super(XavierNormal, self).__init__(gain=gain)


class XavierUniform(lasagne.init.GlorotUniform):
    """
    Xavier weight initialization with uniform distribution.

    :param gain: Scaling factor for the weights. Set this to 1.0 for linear and sigmoid
                 units, to 'relu' or sqrt(2) for rectified linear units. Other transfer
                 functions may need different factors.
    """

    def __init__(self, gain=1.0):
        super(XavierUniform, self).__init__(gain=gain)


class Constant(lasagne.init.Constant):
    """Initialize weights with constant value."""

    def __init__(self, val=0.0):
        super(Constant, self).__init__(val=val)


class NormalizedUniform(lasagne.init.Initializer):
    """
    Sample initial weights from symmetric uniform distribution, and then normalize the weights
    along given axis.

    :param axis: Norms are computed along this axis.
    :param norm: Ensure weights to have this norm.
    :param norm_type: Type of the norm, possible values are {'l1', 'l2'}.
    """

    def __init__(self, axis=-1, norm=1.0, norm_type='l2'):
        norm_functions = {
            'l1': self._l1_norm,
            'l2': self._l2_norm,
        }
        if norm_type not in norm_functions:
            raise ValueError('Unsupported norm type %r.' % norm_type)
        self.axis = axis
        self.norm = norm
        self.norm_func = norm_functions[norm_type]

    @staticmethod
    def _l1_norm(tensor, axis):
        return np.sum(np.abs(tensor), axis=axis, keepdims=True)

    @staticmethod
    def _l2_norm(tensor, axis):
        return np.sqrt(np.sum(tensor ** 2, axis=axis, keepdims=True))

    def sample(self, shape):
        C = lambda c: np.array(c, dtype=glue.config.floatX)
        ret = np.random.random(shape).astype(glue.config.floatX)
        norm = self.norm_func(ret, self.axis) / C(self.norm)
        delta = (norm == C(0.0)) * C(1e-7)
        return ret / (norm + delta)

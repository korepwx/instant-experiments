# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne.init


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

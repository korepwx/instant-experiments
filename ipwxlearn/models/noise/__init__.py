# -*- coding: utf-8 -*-
from ipwxlearn.glue import G


class NoiseGenerator(object):
    """
    Base class for all noise generators.

    A noise generator should add noise to the input tensor.
    """

    def add_noise(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        return self.add_noise(x)


class DropoutNoise(NoiseGenerator):
    """
    Noise generator that randomly set some elements of the input tensor to zero.

    :param p: Probability to zero-out each element.
    """

    def __init__(self, p):
        self.p = p

    def add_noise(self, x):
        return G.random.binomial(shape=G.op.shape(x), p=1. - self.p, n=1, dtype=x.dtype) * x

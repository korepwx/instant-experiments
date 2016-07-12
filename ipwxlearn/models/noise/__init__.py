# -*- coding: utf-8 -*-
from ipwxlearn.glue import G

__all__ = [
    'NoiseGenerator',
    'DropoutNoise',
    'GaussianNoise',
]


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


class GaussianNoise(NoiseGenerator):
    """
    Noise generator that randomly add gaussian noise with determined mean and variance to the input tensor.

    Since this noise generator will add i.i.d. gaussian random noise to every dimension of the input tensor,
    you might need to normalize the input dimensions first.

    :param mean: Mean of the gaussian noise. (Default 0.0)
    :param stddev: Standard derivation of the gaussian noise. (Default 1.0)
    """

    def __init__(self, mean=0.0, stddev=1.0):
        self.mean = mean
        self.stddev = stddev

    def add_noise(self, x):
        return G.random.normal(shape=G.op.shape(x), mean=self.mean, stddev=self.stddev, dtyhpe=x.dtype) + x


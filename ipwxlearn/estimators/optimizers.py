# -*- coding: utf-8 -*-
from __future__ import absolute_import

from ipwxlearn.glue import G

__all__ = [
    'Optimizer',
    'SGDOptimizer',
    'MomentumOptimizer',
    'AdamOptimizer'
]


class Optimizer(object):
    """Base class for all optimizers."""

    def minimize(self, loss, params):
        """
        Derivate the update to :param:`params` so as to minimize :param:`loss`.

        :param loss: Tensor expression representing the loss.
        :param params: Tuple/list of parameters that should be minimized.
        :return: Update object to the parameters.
        """
        raise NotImplementedError()

    def maximize(self, loss, params):
        """
        Derivate the update to :param:`params` so as to maximize :param:`loss`.

        :param loss: Tensor expression representing the loss.
        :param params: Tuple/list of parameters that should be maximized.
        :return: Update object to the parameters.
        """
        return self.minimize(-loss, params)


class SGDOptimizer(Optimizer):
    """Stochastic gradient descent optimizer."""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def minimize(self, loss, params):
        return G.updates.sgd(loss, params, learning_rate=self.learning_rate)


class MomentumOptimizer(Optimizer):
    """Momentum optimizer."""

    def __init__(self, learning_rate=0.001, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def minimize(self, loss, params):
        return G.updates.momentum(loss, params, learning_rate=self.learning_rate, momentum=self.momentum)


class AdamOptimizer(Optimizer):
    """Adam optimizer."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def minimize(self, loss, params):
        return G.updates.adam(loss, params, learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2,
                              epsilon=self.epsilon)

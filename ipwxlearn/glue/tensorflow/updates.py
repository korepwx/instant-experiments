# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf

from ipwxlearn.utils.misc import maybe_iterable_to_list, ensure_list_sealed

__all__ = [
    'sgd',
    'momentum',
    'nesterov_momentum',
    'adagrad',
    'rmsprop',
    'adadelta',
    'adam'
]


def _apply_optimizer(optimizer, loss_or_grads, params, *args, **kwargs):
    loss_or_grads = maybe_iterable_to_list(loss_or_grads)
    params = ensure_list_sealed(params)

    if isinstance(loss_or_grads, list):
        if len(loss_or_grads) != len(params):
            raise ValueError('Got %r gradients, but there are %r parameters.' % (len(loss_or_grads), len(params)))
        return optimizer.apply_gradients(list(zip(loss_or_grads, params)), *args, **kwargs)

    else:
        return optimizer.minimize(loss_or_grads, var_list=params, *args, **kwargs)


def sgd(loss_or_grads, params, learning_rate):
    """
    Stochastic Gradient Descent (SGD) updates

    :param loss_or_grads: Loss tensor, or a list of tensors as the pre-computed gradients of params.
    :param params: List of parameters.
    :param learning_rate: Constant or tensor, as the learning rate.

    :return: Updates to the variables for SGD training.
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return _apply_optimizer(optimizer, loss_or_grads, params)


def momentum(loss_or_grads, params, learning_rate, momentum=0.9):
    raise NotImplementedError()


def nesterov_momentum(loss_or_grads, params, learning_rate, momentum=0.9):
    raise NotImplementedError()


def adagrad(loss_or_grads, params, learning_rate=1.0, epsilon=1e-6):
    raise NotImplementedError()


def rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    raise NotImplementedError()


def adadelta(loss_or_grads, params, learning_rate=1.0, rho=0.95, epsilon=1e-6):
    raise NotImplementedError()


def adam(loss_or_grads, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    return _apply_optimizer(optimizer, loss_or_grads, params)

# -*- coding: utf-8 -*-

"""
This module contains the functions to derive update operations for variables.

In Theano, updates to variables should be explicitly given when compiling functions.
This might not be true in other backend, but it should be nice to keep this abstraction.
"""
import lasagne

from ipwxlearn.utils.misc import maybe_iterable_to_list


def sgd(loss_or_grads, params, learning_rate):
    """
    Stochastic Gradient Descent (SGD) updates

    :param loss_or_grads: Loss tensor, or a list of tensors as the pre-computed gradients of params.
    :param params: List of parameters.
    :param learning_rate: Constant or tensor, as the learning rate.

    :return: Updates to the variables for SGD training.
    """
    return lasagne.updates.sgd(maybe_iterable_to_list(loss_or_grads), list(params), learning_rate)


def momentum(loss_or_grads, params, learning_rate, momentum=0.9):
    return lasagne.updates.momentum(maybe_iterable_to_list(loss_or_grads), list(params), learning_rate,
                                    momentum=momentum)


def nesterov_momentum(loss_or_grads, params, learning_rate, momentum=0.9):
    return lasagne.updates.nesterov_momentum(maybe_iterable_to_list(loss_or_grads), list(params), learning_rate,
                                             momentum=momentum)


def adagrad(loss_or_grads, params, learning_rate=1.0, epsilon=1e-6):
    return lasagne.updates.adagrad(maybe_iterable_to_list(loss_or_grads), list(params), learning_rate, epsilon=epsilon)


def rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    return lasagne.updates.rmsprop(maybe_iterable_to_list(loss_or_grads), list(params), learning_rate, rho=rho,
                                   epsilon=epsilon)


def adadelta(loss_or_grads, params, learning_rate=1.0, rho=0.95, epsilon=1e-6):
    return lasagne.updates.rmsprop(maybe_iterable_to_list(loss_or_grads), list(params), learning_rate, rho=rho,
                                   epsilon=epsilon)


def adam(loss_or_grads, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    return lasagne.updates.adam(maybe_iterable_to_list(loss_or_grads), list(params), learning_rate, beta1=beta1,
                                beta2=beta2, epsilon=epsilon)

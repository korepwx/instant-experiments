# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne.objectives
from theano import tensor as T

__all__ = [
    "sigmoid_cross_entropy_with_logits",
    "sparse_softmax_cross_entropy_with_logits",
    "squared_error",
    "aggregate",
]


def square_error(a, b):
    """Computes the element-wise squared difference between two tensors."""
    return lasagne.objectives.squared_error(a, b)


def aggregate(loss, weights=None, mode='mean'):
    """
    Aggregates an element- or item-wise loss to a scalar loss.

    :param loss: Tensor, the loss to aggregate.
    :param weights: Tensor, optional, the weights for each element or item.
    :param mode: One of {'mean', 'sum', 'normalized_sum'}.
    :return: Tensor scalar.
    """
    return lasagne.objectives.aggregate(loss, weights, mode)


def sigmoid_cross_entropy_with_logits(logits, targets):
    """
    Compute the cross entropy for sigmoid logits.

    :param logits: Logits of the sigmoid probability.
    :param targets: Target probability.
    """
    return lasagne.objectives.binary_crossentropy(T.nnet.sigmoid(logits), targets)


def sparse_softmax_cross_entropy_with_logits(logits, targets):
    """
    Compute the cross entropy for softmax logits, with sparse targets.

    :param logits: Logits of the sigmoid probability.
    :param targets: Target labels.
    """
    return lasagne.objectives.categorical_crossentropy(T.nnet.softmax(logits), targets)

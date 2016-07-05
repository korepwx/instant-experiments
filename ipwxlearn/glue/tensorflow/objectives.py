# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf

__all__ = [
    "sigmoid_cross_entropy_with_logits",
    "sparse_softmax_cross_entropy_with_logits",
    "squared_error",
    "aggregate",
]


def square_error(a, b):
    """Computes the element-wise squared difference between two tensors."""
    return tf.squared_difference(a, b)


def aggregate(loss, weights=None, mode='mean'):
    """
    Aggregates an element- or item-wise loss to a scalar loss.

    :param loss: Tensor, the loss to aggregate.
    :param weights: Tensor, optional, the weights for each element or item.
    :param mode: One of {'mean', 'sum', 'normalized_sum'}.
    :return: Tensor scalar.
    """
    if weights is not None:
        loss = loss * weights
    if mode == 'mean':
        return tf.reduce_mean(loss)
    elif mode == 'sum':
        return tf.reduce_sum(loss)
    elif mode == 'normalized_sum':
        if weights is None:
            raise ValueError('require weights for mode="normalized_sum"')
        return tf.reduce_sum(loss) / tf.reduce_sum(weights)
    else:
        raise ValueError('mode must be "mean", "sum" or "normalized_sum", got %r' % mode)


def sigmoid_cross_entropy_with_logits(logits, targets):
    """
    Compute the cross entropy for sigmoid logits.

    :param logits: Logits of the sigmoid probability.
    :param targets: Target probability.
    """
    return tf.nn.sigmoid_cross_entropy_with_logits(logits, targets)


def sparse_softmax_cross_entropy_with_logits(logits, targets):
    """
    Compute the cross entropy for softmax logits, with sparse targets.

    :param logits: Logits of the sigmoid probability.
    :param targets: Target labels.
    """
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)

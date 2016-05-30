# -*- coding: utf-8 -*-
import tensorflow as tf

__all__ = [
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
            raise ValueError('require weights fro mode="normalized_sum"')
        return tf.reduce_sum(loss) / tf.reduce_sum(weights)
    else:
        raise ValueError('mode must be "mean", "sum" or "normalized_sum", got %r' % mode)

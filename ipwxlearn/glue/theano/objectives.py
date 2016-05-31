# -*- coding: utf-8 -*-
from __future__ import absolute_import

import lasagne.objectives

__all__ = [
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

# -*- coding: utf-8 -*-

import lasagne.objectives

__all__ = [
    "binary_crossentropy",
    "categorical_crossentropy",
    "sparse_categorical_crossentropy",
    "squared_error",
    "aggregate",
]


def binary_crossentropy(predictions, targets):
    """
    Computes the binary cross-entropy between predictions and targets.

    :param predictions: Tensor, predictions in (0, 1).
    :param targets: Tensor, targets in [0, 1].

    :return: 1D tensor, the element-wise binary cross-entropy.
    """
    return lasagne.objectives.binary_crossentropy(predictions, targets)


def categorical_crossentropy(predictions, targets):
    """
    Computes the categorical cross-entropy between predictions and targets.

    :param predictions: 2D tensor, predictions in (0, 1).
    :param targets: 2D tensor, targets in [0, 1] matching the layout of predictions.

    :return: 1D tensor, the element-wise categorical cross-entropy.
    """
    return lasagne.objectives.categorical_crossentropy(predictions, targets)


def sparse_categorical_crossentropy(predictions, targets):
    """
    Computes the categorical cross-entropy between predictions and targets.

    :param predictions: 2D tensor, predictions in (0, 1).
    :param targets: 1D tensor, a vector of int given the correct class index per data point.

    :return: 1D tensor, the element-wise categorical cross-entropy.
    """
    return lasagne.objectives.categorical_crossentropy(predictions, targets)


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

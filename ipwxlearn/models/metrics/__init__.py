# -*- coding: utf-8 -*-
from ipwxlearn.glue import G


class ErrorMetric(object):
    """
    Base class for all error metrics.

    An error metric should accept two tensors of same shape, and compute a non-negative 1-D
    tensor indicating the element-wise distance between these two tensors.
    """

    def distance(self, a, b):
        raise NotImplementedError()

    def __call__(self, a, b):
        return self.distance(a, b)


class SquareError(ErrorMetric):
    """Squared error of two tensors."""

    def distance(self, a, b):
        square = (a - b) ** 2
        try:
            return G.op.sum(G.op.flatten(square, ndim=2), axis=1)
        except ValueError:
            # this branch indicates that a & b only has one dimension.
            return square

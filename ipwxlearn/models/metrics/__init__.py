# -*- coding: utf-8 -*-


class ErrorMetric(object):
    """
    Base class for all error metrics.

    An error metric should accept two tensors of same shape, and compute a non-negative
    tensor indicating the element-wise distance between these two tensors.
    """

    def distance(self, a, b):
        raise NotImplementedError()

    def __call__(self, a, b):
        return self.distance(a, b)


class SquareError(ErrorMetric):
    """Squared error of two tensors."""

    def distance(self, a, b):
        return (a - b) ** 2

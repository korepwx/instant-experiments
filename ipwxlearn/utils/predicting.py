# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from . import dataflow


def concatenate_predict(predict_fn, arrays, batch_size=256):
    """
    Compute predicting result by applying given arrays to predict function in mini-batches,
    then concatenate batch results by the first axis.

    :param predict_fn: Predict function.
    :param arrays: numpy.ndarray, or an iterable of numpy.ndarray.
    :param batch_size: Mini-batch size.

    :return: Concatenated predict results.
    """
    ret = []
    for args in dataflow.iterate_testing_batches(arrays, batch_size=batch_size):
        ret.append(predict_fn(*args))
    return np.concatenate(ret, axis=0)

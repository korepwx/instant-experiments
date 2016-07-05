# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from ipwxlearn.utils.misc import ensure_list_sealed


def collect_batch_predict(predict_fn, arrays, batch_size=256, mode='concat'):
    """
    Collect batch prediction.

    :param predict_fn: Predict function.
    :param arrays: numpy.ndarray, or an iterable of numpy.ndarray, as the input to the :param:`predict_fn`
    :param batch_size: Mini-batch size.
    :param mode: Way to collect the batch predicts.  One of {'concat', 'sum', 'average'}
                 All of these three operations will only merge the result along the first axis.
                 If there are more than 1 dimension in batch results, the extra dimensions will be preserved.

    :return: Merged prediction.
    """
    def weighted_average(batch_arrays):
        weight0 = len(batch_arrays[0])
        factor = weight0 / np.sum([len(arr) for arr in batch_arrays]).astype(np.float64)
        return np.sum([arr.sum(axis=0) * factor for arr in batch_arrays], axis=0) / weight0

    ret = []
    processors = {
        'concat': lambda v: np.concatenate(v, axis=0),
        'sum': lambda v: np.sum([a.sum(axis=0) for a in ret]),
        'average': weighted_average
    }
    processor = processors[mode]

    from ipwxlearn.training import dataflow
    for args in dataflow.iterate_testing_batches(ensure_list_sealed(arrays), batch_size=batch_size):
        ret.append(predict_fn(*args))
    return processor(ret)

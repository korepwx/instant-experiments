# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from ipwxlearn.utils.misc import ensure_list_sealed

__all__ = [
    'DataFlow',
    'OneShotDataFlow',
    'TrainingBatchDataFlow',
    'TestingBatchDataFlow',
    'iterate_training_batches',
    'iterate_testing_batches',
]


class DataFlow(object):
    """
    Abstract interface for all classes that creates iterators for data.

    Derived classes of this would be used in :method:`~ipwxlearn.utils.training.run_steps` and
    :class:`~ipwxlearn.utils.training.ValidationMonitor`, so as to get iterators of training or
    validation data.
    """

    def iter_epoch(self):
        """
        Get data iterator for this epoch.
        Yielding (arr1, arr2, ...) tuples.
        """
        raise NotImplementedError()

    @property
    def num_examples(self):
        """Get the total number of examples in one batch."""
        raise NotImplementedError()


class OneShotDataFlow(DataFlow):
    """Return the given array or arrays as the only data in an epoch."""

    def __init__(self, array_or_arrays):
        self.array_or_arrays = ensure_list_sealed(array_or_arrays)

    def iter_epoch(self):
        yield tuple(self.array_or_arrays)

    @property
    def num_examples(self):
        return len(self.array_or_arrays[0])


class TrainingBatchDataFlow(DataFlow):
    """General training data flow in mini-batches."""

    def __init__(self, array_or_arrays, batch_size, shuffle=True):
        self.array_or_arrays = ensure_list_sealed(array_or_arrays)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def iter_epoch(self):
        return iterate_training_batches(self.array_or_arrays, self.batch_size, self.shuffle)

    @property
    def num_examples(self):
        return len(self.array_or_arrays[0])


class TestingBatchDataFlow(DataFlow):
    """General testing data flow in mini-batches."""

    def __init__(self, array_or_arrays, batch_size):
        self.array_or_arrays = ensure_list_sealed(array_or_arrays)
        self.batch_size = batch_size

    def iter_epoch(self):
        return iterate_testing_batches(self.array_or_arrays, self.batch_size)

    @property
    def num_examples(self):
        return len(self.array_or_arrays[0])


def iterate_training_batches(array_or_arrays, batch_size, shuffle=True):
    """
    Iterate the given array or arrays in mini-batches, for training purpose.

    Yielding (array1, array2, ...) at each mini-batch.  The arrays yielded for training batches
    would be exactly batch_size, while the tail of the given data might be dropped.

    :param array_or_arrays: Numpy array, or a list of numpy arrays.
    :param batch_size: Batch size of the mini-batches.
    :param shuffle: If True, will shuffle the data before iterating the batches.
    """
    if not isinstance(array_or_arrays, (tuple, list, np.ndarray)):
        raise TypeError('Given array is neither a numpy array, or a list of numpy arrays.')

    direct_value = False
    if not isinstance(array_or_arrays, (tuple, list)):
        array_or_arrays = [array_or_arrays]
        direct_value = True

    num_examples = len(array_or_arrays[0])
    assert(num_examples >= batch_size)

    if shuffle:
        perm = np.arange(num_examples)
        array_or_arrays = [arr[perm] for arr in array_or_arrays]

    index_in_batch = 0
    while True:
        start = index_in_batch
        index_in_batch += batch_size
        if index_in_batch > num_examples:
            break
        yield_arrays = tuple(arr[start: index_in_batch] for arr in array_or_arrays)
        if direct_value:
            yield_arrays = yield_arrays[0]
        yield yield_arrays


def iterate_testing_batches(array_or_arrays, batch_size):
    """
    Iterate the given array or arrays in mini-batches, for testing purpose.

    Yielding (array1, array2, ...) at each mini-batch.  The arrays yielded for testing batches
    would be equal to or less than batch_size, while the tail of the given data might not be dropped.

    :param array_or_arrays: Numpy array, or a list of numpy arrays.
    :param batch_size: Batch size of the mini-batches.
    """
    if not isinstance(array_or_arrays, (tuple, list, np.ndarray)):
        raise TypeError('Given array is neither a numpy array, or a list of numpy arrays.')

    direct_value = False
    if not isinstance(array_or_arrays, (tuple, list)):
        array_or_arrays = [array_or_arrays]
        direct_value = True

    num_examples = len(array_or_arrays[0])
    index_in_batch = 0
    while index_in_batch < num_examples:
        start = index_in_batch
        index_in_batch += batch_size
        if index_in_batch > num_examples:
            index_in_batch = num_examples
        yield_arrays = tuple(arr[start: index_in_batch] for arr in array_or_arrays)
        if direct_value:
            yield_arrays = yield_arrays[0]
        yield yield_arrays

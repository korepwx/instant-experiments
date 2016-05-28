# -*- coding: utf-8 -*-
import numpy as np


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
    while index_in_batch < num_examples:
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

# -*- coding: utf-8 -*-
import numpy as np


class FilteredPickleSupport(object):
    _UNPICKABLE_FIELDS_ = ()

    def __getstate__(self):
        states = self.__dict__.copy()
        for k in self._UNPICKABLE_FIELDS_:
            del states[k]
        return states

    def __setstate__(self, states):
        self.__dict__.update(states)
        for k in self._UNPICKABLE_FIELDS_:
            setattr(self, k, None)


def safe_reduce(op, seq, empty_value=None):
    """
    Reduce the :param:`seq` with :param:`op`.

    :return: If :param:`seq` is empty, return :param:`empty_value`.
             If :param:`seq` contains only one element, return this element.
             Otherwise returns the reduced value.
    """
    it = iter(seq)

    # try to take the first element, or return empty value
    try:
        value = next(it)
    except StopIteration:
        return empty_value

    # then reduce the value with operator.
    try:
        while True:
            e = next(it)
            value = op(value, e)
    except StopIteration:
        pass

    return value


def option_map(o, f, *args, **kwargs):
    """Return None if :param:`o` is None, otherwise f(o, *args, **kwargs)."""
    if o is not None:
        return f(o, *args, **kwargs)


class LookaheadIterator(object):
    def __init__(self, iterator):
        self.it = iterator
        self.stack = []

    def __iter__(self):
        try:
            while True:
                if self.stack:
                    yield self.stack.pop()
                yield next(self.it)
        except StopIteration:
            pass

    def push_back(self, value):
        self.stack.append(value)


class ObjectDebuggingRepr(object):
    """Mixin for arbitrary object class to have a debugging repr output."""

    def __repr__(self):
        args = ','.join('%s=%s' % (k, repr(v))
                        for k, v in ((k, getattr(self, k))
                                     for k in sorted(dir(self)) if not k.startswith('_'))
                        if not callable(v))
        return '%s(%s)' % (self.__class__.__name__, args)


class MiniBatchIterator(object):
    """
    Base class to iterate through data in mini-batches.

    :param arrays: Arbitrary number of arrays that should be iterated in mini-batches.
                   When calling to :method:`next_batch`, mini-batches of all these arrays will be returned in a tuple.
    """

    def __init__(self, *arrays):
        if not arrays:
            raise ValueError('No array specified.')
        if len(set(len(a) for a in arrays)) > 1:
            raise ValueError('First dimension of specified arrays are not equal.')
        self._origin_arrays = self.arrays = tuple(arrays)
        self.num_examples = len(self.arrays[0])
        self.epochs_completed = 0
        self.index_in_epoch = 0

    def reset(self):
        self.arrays = self._origin_arrays
        self.epochs_completed = 0
        self.index_in_epoch = 0

    def next_batch(self, batch_size):
        raise NotImplementedError()

    def iter_batches(self, batch_size):
        while True:
            yield self.next_batch(batch_size)


class TrainingBatchIterator(MiniBatchIterator):
    """
    Iterate through training data in stochastic mini-batches.

    :param arrays: Arbitrary number of arrays that should be iterated in mini-batches.
                   When calling to :method:`next_batch`, mini-batches of all these arrays will be returned in a tuple.
    """

    def next_batch(self, batch_size):
        assert(batch_size <= self.num_examples)
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            self.epochs_completed += 1
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.arrays = tuple(arr[perm] for arr in self.arrays)
            start = 0
            self.index_in_epoch = batch_size
        end = self.index_in_epoch
        return tuple(arr[start: end] for arr in self.arrays)


class TestingBatchIterator(MiniBatchIterator):
    """
    Iterate through training data in stochastic mini-batches.

    :param arrays: Arbitrary number of arrays that should be iterated in mini-batches.
                   When calling to :method:`next_batch`, mini-batches of all these arrays will be returned in a tuple.
    """

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        if start >= self.num_examples:
            raise StopIteration('Index out of range.')
        self.index_in_epoch += batch_size
        end = self.index_in_epoch
        if end > self.num_examples:
            end = self.num_examples
        v = tuple(arr[start: end] for arr in self.arrays)
        if len(v) == 1:
            v = v[0]
        return v

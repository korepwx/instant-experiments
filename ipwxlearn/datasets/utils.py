# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os

import numpy as np
import six

if six.PY2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

__all__ = [
    'get_cache_dir',
    'cached_download',
    'split_train_valid'
]


def get_cache_dir(name, root_dir=None):
    """
    Get the cache directory for a particular data set.

    :param name: Name of the data set.
    :param root_dir: Root cache directory.  If not specified, will automatically select one according to OS.
    :return: Path to the cache directory.
    """
    if root_dir is None:
        root_dir = os.path.expanduser('~/.ipwxlearn/dataset')
    return os.path.join(root_dir, name)


def cached_download(uri, cache_file):
    """
    Download :param:`uri` with caching.

    :param uri: URI to be downloaded.
    :param cache_file: Path of the cached file.

    :return: The full path of the downloaded file.
    """
    cache_file = os.path.abspath(os.path.abspath(cache_file))
    if not os.path.isfile(cache_file):
        parent_dir = os.path.split(cache_file)[0]
        if not os.path.isdir(parent_dir):
            os.makedirs(parent_dir)

        tmp_file = '%s~' % cache_file
        try:
            urlretrieve(uri, tmp_file)
            os.rename(tmp_file, cache_file)
        finally:
            if os.path.isfile(tmp_file):
                os.remove(tmp_file)
    return cache_file


def split_train_valid(arrays, validation_split=None, valid_size=None, shuffle=True):
    """
    Split training and validation set by portion or by size.

    :param arrays: Numpy ndarray, or a tuple/list of numpy ndarrays.
    :param validation_split: Portion of the validation set.
                          Would be ignored if :param:`valid_size` is specified.
    :param valid_size: Size of the validation set.
    :param shuffle: Whether or not to shuffle before splitting? (Default True)

    :return: (train_arrays, valid_arrays)
    """
    if isinstance(arrays, (tuple, list)):
        direct_value = False
        num_examples = len(arrays[0])
    else:
        direct_value = True
        num_examples = len(arrays)

    if valid_size is None:
        if validation_split is None:
            raise ValueError('At least one of "validation_split", "valid_size" should be specified.')

        if validation_split < 0.5:
            valid_size = num_examples - int(num_examples * (1.0 - validation_split))
        else:
            valid_size = int(num_examples * validation_split)

    if valid_size <= 0 or valid_size >= num_examples:
        raise ValueError('Estimated size of validation set %r is either too small or too large.' % valid_size)

    if shuffle:
        indices = np.arange(num_examples)
        np.random.shuffle(indices)
        get_train = lambda v: v[indices[: -valid_size]]
        get_valid = lambda v: v[indices[-valid_size:]]
    else:
        get_train = lambda v: v[: -valid_size, ...]
        get_valid = lambda v: v[-valid_size:, ...]

    if direct_value:
        return (get_train(arrays), get_valid(arrays))
    else:
        return (tuple(get_train(v) for v in arrays), tuple(get_valid(v) for v in arrays))

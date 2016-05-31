# -*- coding: utf-8 -*-
import os
from ipwxlearn.utils.tempdir import TemporaryDirectory

import numpy as np


def read_data_sets(cache_dir=None, floatX=None):
    """
    Download mnist training and testing data as numpy array, with specified cache directory.

    :param cache_dir: Path to the cache directory.  If not specified, will use a temporary directory.
    :param floatX: Type of the float numbers.  If specified, X_train and X_test will be converted to this type.

    :return: X_train, y_train, X_test, y_test
    """
    from tensorflow.examples.tutorials.mnist import input_data
    if not cache_dir:
        with TemporaryDirectory() as tempdir:
            return read_data_sets(tempdir)

    cached_npz_file = os.path.join(cache_dir, 'parsed.npz')
    if not os.path.exists(cached_npz_file):
        mnist = input_data.read_data_sets(cache_dir, one_hot=False)
        X_train = np.concatenate([mnist.train.images, mnist.validation.images], axis=0)
        y_train = np.concatenate([mnist.train.labels, mnist.validation.labels], axis=0).astype(np.int32)
        X_test = mnist.test.images
        y_test = mnist.test.labels.astype(np.int32)
        np.savez_compressed(cached_npz_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    else:
        cached = np.load(cached_npz_file)
        X_train, y_train, X_test, y_test = cached['X_train'], cached['y_train'], cached['X_test'], cached['y_test']

    if floatX is not None:
        X_train = X_train.astype(floatX)
        X_test = X_test.astype(floatX)

    return X_train, y_train, X_test, y_test

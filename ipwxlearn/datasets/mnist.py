# -*- coding: utf-8 -*-
import os

import gzip

import numpy as np

from .utils import get_cache_dir, cached_download

__all__ = [
    'load_mnist'
]


def load_mnist(cache_dir=None, flatten_to_vectors=False, convert_to_float=True, dtype=None):
    """
    Download mnist training and testing data as numpy array, with specified cache directory.

    :param cache_dir: Path to the cache directory.  If not specified, will use a temporary directory.
    :param flatten_to_vectors: If True, flatten images to 1D vectors.
                               If False (default), shape the images to 3D tensors with shape (28, 28, 1),
                               where the last dimension is the greyscale channel.
    :param convert_to_float: If True (default), scale the byte pixels to 0.0~1.0 float numbers.
    :param dtype: Cast the image tensors to this type.  If not specified, will use `glue.config.floatX`
                  if :param:`convert_to_float` is specified, or keep the images in uint8 if not converting to float.

    :return: (train_X, train_y), (test_X, test_y)
    """
    from ipwxlearn import glue

    cache_dir = cache_dir or get_cache_dir('mnist')
    root_uri = 'http://yann.lecun.com/exdb/mnist/'

    def load_mnist_images(filename):
        with gzip.open(cached_download(root_uri + filename, os.path.join(cache_dir, filename)), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        if flatten_to_vectors:
            data = data.reshape(-1, 784)
        else:
            data = data.reshape(-1, 28, 28, 1)

        if convert_to_float:
            data = data / np.array(256, dtype=dtype or glue.config.floatX)
        elif dtype is not None:
            data = np.asarray(data, dtype=dtype)

        return data

    def load_mnist_labels(filename):
        with gzip.open(cached_download(root_uri + filename, os.path.join(cache_dir, filename)), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    # We can now download and read the training and test set images and labels.
    train_X = load_mnist_images('train-images-idx3-ubyte.gz')
    train_y = load_mnist_labels('train-labels-idx1-ubyte.gz')
    test_X = load_mnist_images('t10k-images-idx3-ubyte.gz')
    test_y = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    return (train_X, train_y), (test_X, test_y)

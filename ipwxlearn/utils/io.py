# -*- coding: utf-8 -*-
import gzip
import os

import six

from ipwxlearn.utils import misc

if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl


@misc.contextmanager
def file_redirected(original_file, redirected_file):
    """
    A context manager to temporarily redirect written outputs from one file to another.

    The two files must be both system files, i.e., having file descriptor in the system.
    Thus file-like objects are not supported.
    """
    dup_fileno = None
    try:
        dup_fileno = os.dup(original_file.fileno())
        os.dup2(redirected_file.fileno(), original_file.fileno())
        yield
    finally:
        if dup_fileno is not None:
            os.dup2(dup_fileno, original_file.fileno())
            os.close(dup_fileno)


def open_devnull(mode):
    """Open /dev/null with specified :param:`mode`."""
    return open(os.devnull, mode)


@misc.contextmanager
def file_muted(file):
    """
    A context manager to temporarily mute all written outputs from specified file.
    """
    with open_devnull('wb') as null_file:
        with file_redirected(file, null_file):
            yield


def load_object_compressed(path):
    """
    Load object from compressed file.

    :param path: Path to the file.
    :return: Whatever object loaded from the file.
    """
    with gzip.open(path, mode='rb') as f:
        return pkl.load(f)


def save_object_compressed(path, obj):
    """
    Save object to compressed file.

    :param path: Path to the file.
    :param obj: Object to be saved.
    """
    with gzip.open(path, mode='wb', compresslevel=9) as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)


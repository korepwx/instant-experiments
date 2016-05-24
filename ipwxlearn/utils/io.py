# -*- coding: utf-8 -*-
import contextlib
import os


@contextlib.contextmanager
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


@contextlib.contextmanager
def file_muted(file):
    """
    A context manager to temporarily mute all written outputs from specified file.
    """
    with open_devnull('wb') as null_file:
        with file_redirected(file, null_file):
            yield

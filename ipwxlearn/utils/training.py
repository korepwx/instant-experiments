# -*- coding: utf-8 -*-


class Monitor(object):
    """Base monitor class that watches training process."""


def run_steps(train_fn, train_data, monitor=None, batch_size=32, max_steps=None, shuffle=True):
    """

    :param train_fn:
    :param train_data:
    :param monitor:
    :param batch_size:
    :param max_steps:
    :param shuffle:
    :return:
    """

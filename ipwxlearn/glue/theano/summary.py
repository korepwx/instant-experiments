# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import warnings

from ipwxlearn.utils.training.monitors import EveryFewStepMonitor

__all__ = [
    'scalar_summary',
    'histogram_summary',
    'image_summary',
    'zero_fraction_summary',
    'make_summary_function',
    'SummaryMonitor'
]


def scalar_summary(tag, value):
    return None


def histogram_summary(tag, value):
    return None


def image_summary(tag, value):
    return None


def zero_fraction_summary(tag, value):
    return None


def make_summary_function(summaries=None):
    """
    Make a callable function that aggregates all the summary operations and construct a TensorBoard
    Summary object.

    :param summaries: Iterable of summary operations.  If not given, will collect all the summaries in the graph.
    """
    return (lambda: None)


class SummaryMonitor(EveryFewStepMonitor):
    """
    Monitor to save Theano summaries every few steps or duration.

    :param log_dir: Save the summaries to specified directory.
    :param summary_fn: Callable function to generate the TensorFlow summary.
    :param seconds: Save session checkpoint every this number of seconds.
    :param steps: Save session checkpoint every this number of steps.
    """

    def __init__(self, log_dir, summary_fn, seconds=None, steps=None):
        super(SummaryMonitor, self).__init__(seconds, steps)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self._log_dir = log_dir
        self._summary_fn = summary_fn

    def _end_step(self, step, loss, now_time):
        warnings.warn('Theano does not support summary yet.')

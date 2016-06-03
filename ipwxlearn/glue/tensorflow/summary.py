# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os

import tensorflow as tf

from ipwxlearn.utils.misc import ensure_list_sealed
from ipwxlearn.utils.training.monitors import EveryFewStepMonitor
from .function import make_function
from .graph import current_graph
from ..common.graph import SummaryTypes

__all__ = [
    'scalar_summary',
    'histogram_summary',
    'image_summary',
    'zero_fraction_summary',
    'make_summary_function',
    'SummaryMonitor'
]


def scalar_summary(tag, value):
    return current_graph().add_summary(SummaryTypes.SCALAR_SUMMARY, tag, value)


def histogram_summary(tag, value):
    return current_graph().add_summary(SummaryTypes.HISTOGRAM_SUMMARY, tag, value)


def image_summary(tag, value):
    return current_graph().add_summary(SummaryTypes.IMAGE_SUMMARY, tag, value)


def zero_fraction_summary(tag, value):
    return current_graph().add_summary(SummaryTypes.ZERO_FRACTION_SUMMARY, tag, value)


def make_summary_function(summaries=None):
    """
    Make a callable function that aggregates all the summary operations and construct a TensorBoard
    Summary object.

    :param summaries: Iterable of summary operations.  If not given, will collect all the summaries in the graph.
    """
    if summaries is None:
        summaries = current_graph().get_summary_operations()
    else:
        summaries = ensure_list_sealed(summaries)
    return make_function(outputs=tf.merge_summary(summaries))


class SummaryMonitor(EveryFewStepMonitor):
    """
    Monitor to save TensorFlow summaries every few steps or duration.

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
        self._writer = tf.train.SummaryWriter(logdir=self._log_dir)

    def _end_step(self, step, loss, now_time):
        self._writer.add_summary(self._summary_fn(), global_step=step)

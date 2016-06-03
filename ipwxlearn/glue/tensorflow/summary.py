# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf

from ipwxlearn.glue.common.summary import BaseSummaryWriter
from ipwxlearn.utils.misc import ensure_list_sealed
from .graph import current_graph
from .session import current_session
from .utils import get_variable_name

__all__ = [
    'scalar_summary',
    'histogram_summary',
    'zero_fraction_summary',
    'collect_variable_summaries',
    'compile_summary',
    'SummaryWriter'
]


def scalar_summary(tag, value):
    return tf.scalar_summary(tag, value)


def histogram_summary(tag, value):
    return tf.histogram_summary(tag, value)


def zero_fraction_summary(tag, value):
    return tf.scalar_summary(tag, tf.nn.zero_fraction(value))


def collect_variable_summaries():
    """
    Collect the summaries for all variables.
    Returns list of summary operations for the variables.
    """
    ret = []
    for v in current_graph().iter_variables(summary=True):
        name = get_variable_name(v)
        ret.append(histogram_summary(name, v))
        # also generate the mean/min/max statistics for this variable.
        for n, f in [('mean', tf.reduce_mean), ('min', tf.reduce_min), ('max', tf.reduce_max)]:
            ret.append(scalar_summary('%s/%s' % (name, n), f(v)))
    return ret


def compile_summary(summaries):
    """
    Compile the given summaries, so that they could be written by :class:`SummaryWriter`.

    :param summaries: Iterable of summaries.
    :return: An object that could be fed to :method:`SummaryWriter.write`
    """
    summaries = ensure_list_sealed(summaries)
    return tf.merge_summary(summaries)


class SummaryWriter(BaseSummaryWriter):
    """Summary writer for TensorFlow."""

    def __init__(self, log_dir):
        super(SummaryWriter, self).__init__(log_dir)
        self.tf_writer = tf.train.SummaryWriter(logdir=log_dir)

    def _write(self, summary, global_step, **kwargs):
        session = current_session()
        givens = kwargs.get('givens', {})
        if isinstance(summary, tf.Tensor):
            summary = session.tf_session.run(summary, feed_dict=givens)
        self.tf_writer.add_summary(summary, global_step=global_step)

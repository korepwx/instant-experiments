# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf

from ipwxlearn.glue.common.summary import BaseSummaryWriter
from ipwxlearn.utils.misc import flatten_list, maybe_iterable_to_list
from .graph import current_graph
from .session import current_session
from .utils import get_variable_name

__all__ = [
    'scalar_summary',
    'histogram_summary',
    'zero_fraction_summary',
    'collect_variable_summaries',
    'merge_summary',
    'SummaryWriter'
]


def scalar_summary(tag, value):
    return tf.scalar_summary(tag, value)


def histogram_summary(tag, value):
    return tf.histogram_summary(tag, value)


def zero_fraction_summary(tag, value):
    return tf.scalar_summary(tag, tf.nn.zero_fraction(value))


def collect_variable_summaries(vars=None):
    """
    Collect the summaries for all variables.
    Returns list of summary operations for the variables.

    :param vars: If specified, will gather summaries for these variables.
                 Otherwise will gather summaries for all summarizable variables in current graph.
    """
    ret = []
    for v in (vars or current_graph().iter_variables(summary=True)):
        name = get_variable_name(v)
        ret.append(histogram_summary(name, v))
        # also generate the mean/min/max/stddev statistics for this variable.
        v_mean = tf.reduce_mean(v)
        v_min = tf.reduce_min(v)
        v_max = tf.reduce_max(v)
        with tf.name_scope('stddev'):
            v_stddev = tf.sqrt(tf.reduce_sum(tf.square(v - v_mean)))
        ret.extend([
            scalar_summary('%s/mean' % name, v_mean),
            scalar_summary('%s/min' % name, v_min),
            scalar_summary('%s/max' % name, v_max),
            scalar_summary('%s/stddev' % name, v_stddev),
        ])
    return ret


def merge_summary(summaries):
    """
    Merge several summaries into one.

    :param summaries: Iterable of summaries.
    :return: An object that could be fed to :method:`SummaryWriter.add`
    """
    summaries = flatten_list(maybe_iterable_to_list(summaries))
    return tf.merge_summary(summaries)


class SummaryWriter(BaseSummaryWriter):
    """Summary writer for TensorFlow."""

    def __init__(self, log_dir, delete_exist=False):
        super(SummaryWriter, self).__init__(log_dir, delete_exist=delete_exist)
        self.tf_writer = tf.train.SummaryWriter(logdir=log_dir, graph=current_graph().tf_graph)

    def _write(self, summary, global_step, **kwargs):
        session = current_session()
        givens = kwargs.get('givens', {})
        if isinstance(summary, (list, tuple)):
            summary = merge_summary(summary)
        if isinstance(summary, tf.Tensor):
            summary = session.tf_session.run(summary, feed_dict=givens)
        self.tf_writer.add_summary(summary, global_step=global_step)

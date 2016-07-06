# -*- coding: utf-8 -*-
from __future__ import absolute_import

import warnings

from ipwxlearn.glue.common.summary import BaseSummaryWriter

__all__ = [
    'scalar_summary',
    'histogram_summary',
    'zero_fraction_summary',
    'collect_variable_summaries',
    'merge_summary',
    'SummaryWriter'
]


class SummaryObject(object):
    """Class to hold a summary object."""


def scalar_summary(tag, value):
    return SummaryObject()


def histogram_summary(tag, value):
    return SummaryObject()


def zero_fraction_summary(tag, value):
    return SummaryObject()


def collect_variable_summaries(vars=None):
    """
    Collect the summaries for all variables.
    Returns list of summary operations for the variables.

    :param vars: If specified, will gather summaries for these variables.
                 Otherwise will gather summaries for all summarizable variables in current graph.
    """
    return []


def merge_summary(summaries):
    """
    Merge several summaries into one.

    :param summaries: Iterable of summaries.
    :return: An object that could be fed to :method:`SummaryWriter.add`
    """
    return SummaryObject()


class SummaryWriter(BaseSummaryWriter):
    """Summary writer for Theano."""

    def _write(self, summary, global_step, **kwargs):
        warnings.warn('Theano backend has not supported summary yet.')

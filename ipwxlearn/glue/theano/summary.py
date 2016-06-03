# -*- coding: utf-8 -*-
from __future__ import absolute_import

import warnings

from ipwxlearn.glue.common.summary import BaseSummaryWriter

__all__ = [
    'scalar_summary',
    'histogram_summary',
    'zero_fraction_summary',
    'collect_variable_summaries',
    'compile_summary',
    'SummaryWriter'
]


class CompiledSummary(object):
    """Class to hold a compiled summary object."""


def scalar_summary(tag, value):
    return CompiledSummary()


def histogram_summary(tag, value):
    return CompiledSummary()


def zero_fraction_summary(tag, value):
    return CompiledSummary()


def collect_variable_summaries():
    """
    Collect the summaries for all variables.
    Returns list of summary operations for the variables.
    """
    return []


def compile_summary(summaries):
    """
    Compile the given summaries, so that they could be written by :class:`SummaryWriter`.

    :param summaries: Iterable of summaries.
    :return: An object that could be fed to :method:`SummaryWriter.add`
    """
    return CompiledSummary()


class SummaryWriter(BaseSummaryWriter):
    """Summary writer for Theano."""

    def _write(self, summary, global_step, **kwargs):
        warnings.warn('Theano backend has not supported summary yet.')

# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os


class BaseSummaryWriter(object):
    """
    Abstract class to write compiled summary object.

    :param log_dir: Directory to store the summary files.
    """

    def __init__(self, log_dir):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir

    def _write(self, summary, global_step, **kwargs):
        raise NotImplementedError()

    def write(self, summary, global_step, **kwargs):
        """
        Write a compiled summary object.

        :param summary: Summary object created by :method:`compile_summary`.
        :param global_step: Global step associated with this summary object.
        :param **kwargs: Additional named arguments passed to backend summary writer.

        :return: self
        """
        self._write(summary, global_step, **kwargs)
        return self

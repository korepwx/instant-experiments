# -*- coding: utf-8 -*-
from __future__ import absolute_import

import math
import time
from datetime import datetime, timedelta

import numpy as np

from ipwxlearn.utils.dataflow import DataFlow, OneShotDataFlow
from ..io import write_string
from ..misc import ensure_list_sealed

__all__ = [
    'Monitor',
    'MonitorChain',
    'ValidationMonitor',
    'EveryFewStepMonitor',
    'CheckpointMonitor',
    'SummaryMonitor'
]


class Monitor(object):
    """Base monitor class that watches training process."""

    def start_training(self, batch_size, steps_in_epoch, max_steps):
        """
        Tell the monitor that a training loop will start.

        :param batch_size: Size of each step (mini-batch).
        :param steps_in_epoch: Estimated number of steps in one epoch.
        :param max_steps: Hard limit of total steps.
        """

    def end_training(self):
        """Tell the monitor that a training loss has been completed."""

    def start_epoch(self, epoch):
        """
        Tell the monitor that a training epoch will start.
        :param epoch: Index of the epoch, starting from 0.
        """

    def end_epoch(self, epoch, avg_loss):
        """
        Tell the monitor that a training epoch has been completed.

        :param epoch: Index of the epoch, starting from 0.
        :param avg_loss: Average training loss of all the mini-batches in this epoch.
        """

    def start_step(self, step):
        """
        Tell the monitor that a training step (mini-batch) will start.

        :param step: Index of the step, starting from 0.
                     It is also the total number of mini-batches have ever been performed.
        """

    def end_step(self, step, loss):
        """
        Tell the monitor that a training step (mini-batch) has been completed.
        :param step: Index of the step, starting from 0.
        :param loss: Training loss of this step.
        """

    @property
    def is_inducing_stopping(self):
        """Whether or not this monitor is inducing early-stopping?"""
        return False


class MonitorChain(Monitor):
    """
    Chain of monitors, to aggregate multiple monitors into one.

    Methods of the monitors in this chain would be called one by one, in determined order.
    If any one of the monitors is inducing early-stopping, then the whole chain would do so.
    """

    def __init__(self, monitors):
        self.monitors = ensure_list_sealed(monitors)

    def start_training(self, batch_size, steps_in_epoch, max_steps):
        for m in self.monitors:
            m.start_training(batch_size, steps_in_epoch, max_steps)

    def end_training(self):
        for m in self.monitors:
            m.end_training()

    def start_epoch(self, epoch):
        for m in self.monitors:
            m.start_epoch(epoch)

    def end_epoch(self, epoch, avg_loss):
        for m in self.monitors:
            m.end_epoch(epoch, avg_loss)

    def start_step(self, step):
        for m in self.monitors:
            m.start_step(step)

    def end_step(self, step, loss):
        for m in self.monitors:
            m.end_step(step, loss)

    @property
    def is_inducing_stopping(self):
        return any(m.is_inducing_stopping for m in self.monitors)


class ValidationMonitor(Monitor):
    """
    Monitor that performs validation and early-stopping.

    This monitor computes the loss on validation set every few steps, and use the validation loss
    to determine whether or not to accept the current set of parameters.

    :param valid_fn: Callable function to perform a validation pass.
                     This function should either return a scalar which indicates the training loss,
                     or return a tuple which contains not only the training loss, but also the summary object
                     for the loss.
    :param valid_data: Numpy array, a list of numpy arrays, or a DataFlow object as the validation data.
                       If it is a DataFlow, it must yield exactly one batch of data for validation in each epoch.
    :param params: List of parameters that should be regularized by early-stopping.
                   If not specified, will select all the trainable variables in current graph.
                   To disable early-stopping on parameters, you may pass an empty list or tuple.
    :param steps: Perform validation every this number of steps.
                  If not specified, will use (valid_data_count / training_batch_size).
    :param stopping_steps: If not None, will induce early stopping if no improvement has been achieved
                           after this number of steps.
    :param log_file: Print the loss to this file.
    :param summary_writer: If specified, will try to output the summary of training loss.
    """

    def __init__(self, valid_fn, valid_data, params=None, steps=None, stopping_steps=None, log_file=None,
                 summary_writer=None):

        self._valid_fn = valid_fn
        if not isinstance(valid_data, DataFlow):
            valid_data = OneShotDataFlow(valid_data)
        self._valid_data = valid_data
        self._params = params
        self._steps = steps
        self._stopping_steps = stopping_steps
        self._log_file = log_file
        self._summary_writer = summary_writer

        # start time stamp.
        self._start_time_stamp = None
        # this monitor will do validation every this number of steps (guaranteed not None after training started).
        self._actual_steps = None
        # number of steps remaining before performing another validation.
        self._remain_steps = None
        # number of steps remaining before inducing early stopping.
        self._remain_stopping_steps = None

        # the session memo dict
        self._memo = None

    def start_training(self, batch_size, steps_in_epoch, max_steps):
        from ipwxlearn.glue import current_session

        # determine the step interval.
        if self._steps is None:
            num_examples = self._valid_data.num_examples
            # automatically determine the step interval, such that:
            #
            # 1. At least the same number of training data is used before using the validation data.
            # 2. A multiple of 10, 100 or 1000, etc, according to the step-interval selected from previous rule.
            actual_steps = (num_examples + batch_size - 1) // batch_size
            ten_base = 10 ** int(math.log(actual_steps, 10))
            self._actual_steps = ((actual_steps + ten_base - 1) // ten_base) * ten_base
        else:
            self._actual_steps = self._steps

        # reset the remaining counters.
        self._remain_steps = self._actual_steps
        if self._stopping_steps is not None:
            self._remain_stopping_steps = max(self._stopping_steps, self._actual_steps)

        # resume the previous training
        self._memo = current_session().memo.with_prefix(self.__class__.__name__)

        # set the start time stamp
        self._start_time_stamp = time.time()
        if self._log_file:
            time_str = datetime.strftime(datetime.fromtimestamp(self._start_time_stamp), '%Y-%m-%d %H:%M:%S')
            write_string(self._log_file, 'Start training at %s, max steps is %s.\n' % (time_str, max_steps))

    def _do_validation(self, step, train_loss):
        """Perform the validation and early-stopping."""
        from ipwxlearn.glue import current_graph, current_session

        # compute the validation loss.
        args = next(self._valid_data.iter_epoch())
        result = self._valid_fn(*args)
        if isinstance(result, (tuple, list)):
            loss, summary = result
        else:
            loss = result
            summary = None

        if self._summary_writer is not None and summary is not None and step is not None:
            self._summary_writer.write(summary, global_step=step)

        # do early-stopping.
        params = self._params if self._params is not None else current_graph().get_variables(trainable=True)
        session = current_session()
        best_params_updated = False
        if loss < self._memo.get('best_valid_loss', np.inf):
            best_params_updated = True
            # record the currently found best parameter.
            self._memo['best_valid_loss'] = loss
            self._memo['best_params'] = session.get_variable_values_dict(params)
            # set the flag that we've got a better parameter, so do not induce early stopping.
            if self._stopping_steps is not None:
                self._remain_stopping_steps = self._stopping_steps

        # report the loss if required
        if step is not None and self._log_file:
            best_mark = ' (*)' if (best_params_updated and params) else ''
            time_offset = str(timedelta(seconds=time.time() - self._start_time_stamp))
            if '.' in time_offset:
                time_offset = time_offset[: time_offset.find('.')]
            msg = ('Step %d: at %s, train loss %.6f, valid loss %.6f%s\n' %
                   (step, time_offset, train_loss, loss, best_mark))
            write_string(self._log_file, msg)
            self._log_file.flush()

    def end_step(self, step, loss):
        # do validation if necessary.
        if self._remain_steps <= 0:
            self._do_validation(step, loss)
            self._remain_steps = self._actual_steps

        # decrease the counter.
        self._remain_steps -= 1
        if self._remain_stopping_steps is not None:
            self._remain_stopping_steps -= 1

    def end_training(self):
        from ipwxlearn.glue import current_session
        # perform the final validation if there's some more training since the last validation.
        if self._remain_steps < self._actual_steps:
            self._do_validation(None, None)
        # restore the best ever params.
        best_params = self._memo.get('best_params', None)
        if best_params is not None:
            current_session().set_variable_values(best_params)
        # and finally, we should clear the recorded best params in the session.
        self._memo['best_params'] = self._memo['best_valid_loss'] = None

    @property
    def is_inducing_stopping(self):
        return self._remain_stopping_steps is not None and self._remain_stopping_steps <= 0


class EveryFewStepMonitor(Monitor):
    """
    Monitor to run every few steps or duration.

    :param seconds: Save session checkpoint every this number of seconds.
    :param steps: Save session checkpoint every this number of steps.
    """

    def __init__(self, seconds=None, steps=None):
        if seconds is None and steps is None:
            raise ValueError('At least either "seconds" or "steps" should be specified.')

        self._seconds = seconds
        self._steps = steps

        # last checkpoint time and step
        self._last_chk_time = None
        self._last_chk_step = None

    def start_training(self, batch_size, steps_in_epoch, max_steps):
        self._last_chk_time = time.time()
        self._last_chk_step = 0

    def _end_step(self, step, loss, now_time):
        """Run monitor after given step."""
        raise NotImplementedError()

    def end_step(self, step, loss):
        if (self._steps is not None and (step - self._last_chk_step) >= self._steps) or \
                (self._seconds is not None and (time.time() - self._last_chk_time) >= self._seconds):
            now_time = time.time()
            self._end_step(step, loss, now_time)
            self._last_chk_time = time.time()
            self._last_chk_step = step


class CheckpointMonitor(EveryFewStepMonitor):
    """
    Monitor to save session checkpoints every few steps or duration.

    :param seconds: Save session checkpoint every this number of seconds.
    :param steps: Save session checkpoint every this number of steps.
    :param log_file: Print the message that checkpoint has been saved to this file.
    """

    def __init__(self, seconds=None, steps=None, log_file=None):
        super(CheckpointMonitor, self).__init__(seconds, steps)
        self._log_file = log_file

    def _end_step(self, step, loss, now_time):
        from ipwxlearn.glue import current_session
        current_session().checkpoint()
        if self._log_file:
            time_str = datetime.strftime(datetime.fromtimestamp(now_time), '%Y-%m-%d %H:%M:%S')
            write_string(self._log_file, 'Checkpoint saved at step %d, %s.\n' % (step, time_str))
            self._log_file.flush()


class SummaryMonitor(EveryFewStepMonitor):
    """
    Monitor to save summaries every few steps or duration.

    :param writer: Backend summary writer.
    :param summary: Compiled backend summary object.
    :param seconds: Save session checkpoint every this number of seconds.
    :param steps: Save session checkpoint every this number of steps.
    """

    def __init__(self, writer, summary, seconds=None, steps=None):
        super(SummaryMonitor, self).__init__(seconds, steps)
        self._writer = writer
        self._summary = summary

    def _end_step(self, step, loss, now_time):
        self._writer.write(self._summary, step)

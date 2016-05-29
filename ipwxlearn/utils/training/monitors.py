# -*- coding: utf-8 -*-
import numpy as np
import six

from ipwxlearn.utils import dataflow
from ipwxlearn.utils.misc import ensure_list_sealed

__all__ = [
    'Monitor',
    'MonitorChain',
    'ValidationMonitor'
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

    :param valid_fn: Callable function to perform a validation pass, and return the validation loss.
    :param valid_data: Numpy array, or a list of numpy arrays, as the validation data.
    :param params: List of parameters that should be regularized by early-stopping.
                   If not specified, will select all the trainable variables in current graph.
    :param batch_size: Batch size of validation.  If not None, will do validation in batches.
                       This might be useful when the validation data is too large to be hold in device.
    :param step_interval: Perform validation every this number of steps.
                          If not specified, will use (valid_data_count / training_batch_size).
    :param stopping_steps: If not None, will induce early stopping if no improvement has been achieved
                           after this number of steps.
    :param loss_print_file: Print the loss to this file.
    """

    def __init__(self, valid_fn, valid_data, params=None, batch_size=None, step_interval=None,
                 stopping_steps=None, loss_print_file=None):
        self._valid_fn = valid_fn
        self._valid_data = ensure_list_sealed(valid_data)
        self._params = params
        self._batch_size = batch_size
        self._step_interval = step_interval
        self._stopping_steps = stopping_steps
        self._loss_print_file = loss_print_file

        # actual step interval for this monitor to do validation.
        self._actual_step_interval = None
        # number of steps remaining before performing another validation.
        self._remain_step_interval = None
        # number of steps remaining before inducing early stopping.
        self._remain_stopping_steps = None
        # best ever validation loss we've encountered
        self._best_valid_loss = None
        # best ever parameters we've encountered
        self._best_params = None

    def start_training(self, batch_size, steps_in_epoch, max_steps):
        # determine the step interval.
        if self._step_interval is None:
            num_examples = len(self._valid_data[0])
            self._actual_step_interval = (num_examples + batch_size - 1) // batch_size
        else:
            self._actual_step_interval = self._step_interval

        # reset the remaining counters.
        self._remain_step_interval = self._actual_step_interval
        if self._stopping_steps is not None:
            self._remain_stopping_steps = max(self._stopping_steps, self._actual_step_interval)

        # reset the best loss and parameters.
        self._best_valid_loss = np.inf
        self._best_params = None

    def _do_validation(self, step, train_loss):
        from ipwxlearn.glue import current_graph, current_session
        """Perform the validation and early-stopping."""
        # compute the validation loss.
        num_examples = len(self._valid_data[0])
        if self._batch_size is not None:
            loss = 0
            for args in dataflow.iterate_testing_batches(self._valid_data, self._batch_size):
                loss += self._valid_fn(*args)
        else:
            loss = self._valid_fn(*self._valid_data)
        loss /= float(num_examples)

        # do early-stopping.
        params = self._params or current_graph().get_variables(trainable=True)
        session = current_session()
        if loss < self._best_valid_loss:
            self._best_valid_loss = loss
            self._best_params = session.get_variable_values_dict(params)

        # report the loss if required
        if step is not None and self._loss_print_file:
            msg = 'Step %d: train loss %.6f, valid loss %.6f' % (step, train_loss, loss)
            if isinstance(msg, six.text_type):
                msg = msg.encode('utf-8')
            self._loss_print_file.write(msg)

    def end_step(self, step, loss):
        # decrease the counter.
        self._remain_step_interval -= 1
        if self._remain_stopping_steps is not None:
            self._remain_stopping_steps -= 1

        # do validation if necessary.
        if self._remain_step_interval <= 0:
            self._do_validation(step, loss)
            self._remain_step_interval = self._actual_step_interval

    def end_training(self):
        from ipwxlearn.glue import current_session
        # perform the final validation if there's some more training since the last validation.
        if self._remain_step_interval < self._actual_step_interval:
            self._do_validation(None, None)
        # restore the best ever params.
        if self._best_params is not None:
            current_session().set_variable_values(self._best_params)

    @property
    def is_inducing_stopping(self):
        return self._remain_stopping_steps is not None and self._remain_stopping_steps <= 0
